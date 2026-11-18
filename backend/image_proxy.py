#!/usr/bin/env python3
"""
M.A.R.K. Holo-CAD // Hardened SSRF-Safe Image Proxy & LRU Cache Service
Architecturally secure, mathematical DNS tree & routable IP validation,
Content-Type and binary signature verification, and quota-constrained LRU eviction.
"""

import hashlib
import io
import ipaddress
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_CACHE_DIR = os.path.join(BASE_DIR, "data", "cache_images")
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# Quota & Cache Budget Constants (Environment-configurable hysteresis model)
MAX_CACHE_BYTES = max(10 * 1024 * 1024, int(os.environ.get("HOLO_CAD_PROXY_MAX_CACHE_BYTES", str(50 * 1024 * 1024))))
MAX_CACHE_FILES = max(20, int(os.environ.get("HOLO_CAD_PROXY_MAX_CACHE_FILES", "200")))
MAX_CACHE_AGE_SECONDS = max(3600, int(os.environ.get("HOLO_CAD_PROXY_MAX_AGE_S", str(7 * 86400))))
LOW_WATERMARK_RATIO = float(os.environ.get("HOLO_CAD_PROXY_LOW_WATERMARK", "0.75"))
MAX_IMAGE_DOWNLOAD_BYTES = max(1024 * 1024, int(os.environ.get("HOLO_CAD_PROXY_MAX_IMG_BYTES", str(15 * 1024 * 1024))))
PROXY_TIMEOUT_SECONDS = max(3, int(os.environ.get("HOLO_CAD_PROXY_TIMEOUT_S", "8")))
PROXY_STREAM_CHUNK_BYTES = max(4096, int(os.environ.get("HOLO_CAD_PROXY_CHUNK_BYTES", "65536")))
PROXY_USER_AGENT = os.environ.get(
    "HOLO_CAD_PROXY_USER_AGENT",
    "StarkHoloCAD-SSRFProxy/2.0 (starklab-proxy@local; free research)",
)


def _parse_allowed_domain_suffixes() -> List[Tuple[str, ...]]:
    raw = os.environ.get("HOLO_CAD_ALLOWED_DOMAINS", "wikipedia.org,wikimedia.org")
    suffixes: List[Tuple[str, ...]] = []
    for entry in raw.split(","):
        cleaned = entry.strip().strip(".").lower()
        if not cleaned:
            continue
        parts = tuple(p for p in cleaned.split(".") if p)
        if len(parts) >= 2:
            suffixes.append(parts)
    return suffixes or [("wikipedia", "org"), ("wikimedia", "org")]


ALLOWED_DOMAIN_SUFFIXES: List[Tuple[str, ...]] = _parse_allowed_domain_suffixes()

ALLOWED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
}

_CACHE_WRITE_LOCK = threading.Lock()
_REMOTE_DL_SEM = threading.Semaphore(max(1, int(os.environ.get("HOLO_CAD_PROXY_CONCURRENCY", "3"))))
_REMOTE_RATE_LOCK = threading.Lock()
_REMOTE_NEXT_SLOT_TIME = 0.0
_REMOTE_MIN_INTERVAL_S = max(0.0, float(os.environ.get("HOLO_CAD_PROXY_INTERVAL_S", "0.08")))


def _wait_proxy_rate_slot() -> None:
    global _REMOTE_NEXT_SLOT_TIME
    with _REMOTE_RATE_LOCK:
        now = time.monotonic()
        wait_s = max(0.0, _REMOTE_NEXT_SLOT_TIME - now)
        slot_start = max(_REMOTE_NEXT_SLOT_TIME, now)
        _REMOTE_NEXT_SLOT_TIME = slot_start + _REMOTE_MIN_INTERVAL_S
    if wait_s > 0.0:
        time.sleep(wait_s)


def _extend_proxy_cooldown(cooldown_s: float) -> None:
    global _REMOTE_NEXT_SLOT_TIME
    if cooldown_s <= 0.0:
        return
    with _REMOTE_RATE_LOCK:
        target = time.monotonic() + cooldown_s
        if target > _REMOTE_NEXT_SLOT_TIME:
            _REMOTE_NEXT_SLOT_TIME = target


class SSRFSecurityError(ValueError):
    """Raised when an outbound URL violates SSRF architectural constraints."""
    pass


class InvalidImageMediaError(ValueError):
    """Raised when upstream payload fails binary image integrity or MIME validation."""
    pass


def validate_hostname_tree(hostname: str) -> bool:
    """
    Validates that a hostname strictly belongs to authorized DNS subtrees
    without naive string matching or regex loopholes.
    """
    if not hostname:
        return False
    clean_host = hostname.rstrip(".").lower()
    labels = clean_host.split(".")
    if len(labels) < 2:
        return False

    for lbl in labels:
        if not (1 <= len(lbl) <= 63):
            return False
        if lbl.startswith("-") or lbl.endswith("-"):
            return False
        if not all(c.isalnum() or c == "-" for c in lbl):
            return False

    for suffix in ALLOWED_DOMAIN_SUFFIXES:
        n = len(suffix)
        if len(labels) >= n and tuple(labels[-n:]) == suffix:
            return True

    return False


def is_globally_routable_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    Checks that the destination IP address is globally routable on public Internet.
    Rejects loopback, link-local, private LAN, carrier NAT, multicast, and cloud metadata.
    """
    return (
        not ip.is_loopback
        and not ip.is_private
        and not ip.is_link_local
        and not ip.is_reserved
        and not ip.is_multicast
        and not ip.is_unspecified
        and ip.is_global
    )


def _validate_url_syntax(url: str) -> Tuple[urllib.parse.SplitResult, str, int]:
    """Validates URL scheme, absence of credentials, allowed DNS subtree, and port without network I/O."""
    if not url or not isinstance(url, str):
        raise SSRFSecurityError("URL must be a non-empty string.")

    parsed = urllib.parse.urlsplit(url.strip())

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFSecurityError(f"Prohibited URL scheme '{scheme}'. Only HTTP and HTTPS are permitted.")

    if parsed.username or parsed.password:
        raise SSRFSecurityError("Embedded credentials in URL are prohibited.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise SSRFSecurityError("URL lacks a valid destination hostname.")

    if not validate_hostname_tree(hostname):
        raise SSRFSecurityError(f"SSRF violation: Host '{hostname}' is not in permitted domain hierarchy.")

    target_port = parsed.port or (443 if scheme == "https" else 80)
    if target_port not in (80, 443):
        raise SSRFSecurityError(f"Prohibited network port '{target_port}'. Only ports 80 and 443 are allowed.")

    return parsed, hostname, target_port


def _validate_routable_dns(hostname: str, target_port: int) -> None:
    """Resolves DNS and verifies all A/AAAA targets are globally routable (retries once on EAI_AGAIN)."""
    addrinfo = None
    last_err = None
    for attempt in range(2):
        try:
            addrinfo = socket.getaddrinfo(hostname, target_port, type=socket.SOCK_STREAM)
            break
        except socket.gaierror as e:
            last_err = e
            if attempt == 0 and getattr(e, "errno", None) == socket.EAI_AGAIN:
                time.sleep(0.35)
                continue
            raise SSRFSecurityError(f"DNS resolution failed for '{hostname}': {e}")

    if not addrinfo:
        if last_err is not None:
            raise SSRFSecurityError(f"DNS resolution failed for '{hostname}': {last_err}")
        raise SSRFSecurityError(f"No routable addresses discovered for host '{hostname}'.")

    for _, _, _, _, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            raise SSRFSecurityError(f"Malformed IP address '{ip_str}' encountered during resolution.")

        if not is_globally_routable_ip(ip):
            raise SSRFSecurityError(
                f"SSRF violation: Host '{hostname}' resolved to non-routable / internal IP address {ip_str}."
            )


def validate_safe_image_url(url: str) -> urllib.parse.SplitResult:
    """
    Performs full RFC 3986 parse, DNS subtree authorization, and pre-flight
    socket-level IP resolution to prevent SSRF and DNS rebinding attacks.
    """
    parsed, hostname, target_port = _validate_url_syntax(url)
    _validate_routable_dns(hostname, target_port)
    return parsed


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Intercepts HTTP 3xx redirects and enforces SSRF validation on target locations."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_safe_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def detect_and_validate_media_type(upstream_content_type: str, data: bytes) -> str:
    """
    Validates HTTP Content-Type header against binary magic signatures and
    verifies image integrity using PIL for raster formats.
    """
    if not data:
        raise InvalidImageMediaError("Image payload contains 0 bytes.")

    claimed_mime = upstream_content_type.split(";")[0].strip().lower()

    detected_mime: Optional[str] = None
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        detected_mime = "image/jpeg"
    elif len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        detected_mime = "image/png"
    elif len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        detected_mime = "image/gif"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        detected_mime = "image/webp"
    elif b"<svg" in data[:1024].lower():
        detected_mime = "image/svg+xml"

    if not detected_mime:
        raise InvalidImageMediaError("Payload does not match any recognized binary image format signatures.")

    if detected_mime not in ALLOWED_MIME_TYPES:
        raise InvalidImageMediaError(f"Detected MIME '{detected_mime}' is not in allowed image media whitelist.")

    if claimed_mime and claimed_mime not in ALLOWED_MIME_TYPES and claimed_mime != "application/octet-stream":
        raise InvalidImageMediaError(f"Upstream Content-Type '{claimed_mime}' is prohibited.")

    if claimed_mime and claimed_mime != "application/octet-stream":
        if claimed_mime == "image/svg+xml" and detected_mime != "image/svg+xml":
            raise InvalidImageMediaError("Content-Type claimed SVG but binary signature is raster.")
        if detected_mime == "image/svg+xml" and claimed_mime != "image/svg+xml":
            raise InvalidImageMediaError(f"Content-Type claimed '{claimed_mime}' but payload contains SVG markup.")

    if detected_mime in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except Exception as e:
            raise InvalidImageMediaError(f"PIL verification failed for {detected_mime}: {e}")

    return detected_mime


def evict_disk_cache(
    cache_dir: str = IMAGE_CACHE_DIR,
    max_bytes: int = MAX_CACHE_BYTES,
    max_files: int = MAX_CACHE_FILES,
    max_age_s: int = MAX_CACHE_AGE_SECONDS,
) -> None:
    """
    Performs dynamic Least-Recently-Used (LRU) cache eviction with low-watermark hysteresis.
    Eliminates thrashing by draining to LOW_WATERMARK_RATIO capacity upon breach.
    """
    if not os.path.exists(cache_dir):
        return

    now = time.time()
    entries = []
    total_bytes = 0

    for entry in os.scandir(cache_dir):
        if not entry.is_file() or entry.name.endswith(".meta") or ".tmp." in entry.name:
            continue
        try:
            st = entry.stat()
        except OSError:
            continue

        if now - st.st_mtime > max_age_s:
            try:
                os.remove(entry.path)
                meta_expired = entry.path + ".meta"
                if os.path.exists(meta_expired):
                    os.remove(meta_expired)
                continue
            except OSError:
                pass

        entries.append((entry.path, st.st_size, st.st_atime))
        total_bytes += st.st_size

    target_bytes = int(max_bytes * LOW_WATERMARK_RATIO)
    target_files = int(max_files * LOW_WATERMARK_RATIO)

    current_files = len(entries)
    if total_bytes > max_bytes or current_files > max_files:
        entries.sort(key=lambda x: x[2])
        for path, size, _ in entries:
            if total_bytes <= target_bytes and current_files <= target_files:
                break
            try:
                os.remove(path)
                total_bytes -= size
                current_files -= 1
                meta_path = path + ".meta"
                if os.path.exists(meta_path):
                    try:
                        os.remove(meta_path)
                    except OSError:
                        pass
            except OSError:
                pass


def _find_cached_entry(cache_dir: str, url_hash: str) -> Optional[Tuple[bytes, str]]:
    """Deterministic O(1) lookup across permitted extensions."""
    if not os.path.exists(cache_dir):
        return None
    for mime, ext in ALLOWED_MIME_TYPES.items():
        candidate_path = os.path.join(cache_dir, f"{url_hash}{ext}")
        if os.path.isfile(candidate_path):
            try:
                with open(candidate_path, "rb") as f:
                    data = f.read()
                os.utime(candidate_path, None)
                meta_path = candidate_path + ".meta"
                resolved_mime = mime
                if os.path.isfile(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta_val = mf.read().strip()
                        if meta_val in ALLOWED_MIME_TYPES:
                            resolved_mime = meta_val
                return data, resolved_mime
            except OSError:
                continue
    return None


def fetch_or_cache_proxied_image(url: str, cache_dir: str = IMAGE_CACHE_DIR) -> Tuple[bytes, str]:
    """
    Fetches an image with full SSRF protection, caches it to disk with SHA-256 key,
    and returns (image_bytes, content_type). Checks verified disk cache before DNS I/O.
    """
    parsed, hostname, target_port = _validate_url_syntax(url)

    canonical_url = f"{parsed.scheme.lower()}://{hostname}{parsed.path}"
    if parsed.query:
        canonical_url += f"?{parsed.query}"

    url_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:32]

    cached = _find_cached_entry(cache_dir, url_hash)
    if cached is not None:
        return cached

    _validate_routable_dns(hostname, target_port)

    opener = urllib.request.build_opener(SafeRedirectHandler())
    req = urllib.request.Request(canonical_url, headers={"User-Agent": PROXY_USER_AGENT})

    claimed_content_type = ""
    data = b""
    max_retries = 2
    for attempt in range(max_retries + 1):
        _wait_proxy_rate_slot()
        try:
            with _REMOTE_DL_SEM:
                with opener.open(req, timeout=PROXY_TIMEOUT_SECONDS) as resp:
                    claimed_content_type = resp.headers.get("Content-Type", "")
                    chunks = []
                    bytes_read = 0
                    while True:
                        chunk = resp.read(PROXY_STREAM_CHUNK_BYTES)
                        if not chunk:
                            break
                        bytes_read += len(chunk)
                        if bytes_read > MAX_IMAGE_DOWNLOAD_BYTES:
                            raise InvalidImageMediaError(
                                f"Image payload exceeds maximum limit of {MAX_IMAGE_DOWNLOAD_BYTES} bytes."
                            )
                        chunks.append(chunk)
                    data = b"".join(chunks)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_retries:
                retry_after = None
                try:
                    ra_hdr = e.headers.get("Retry-After") if e.headers else None
                    if ra_hdr:
                        retry_after = min(5.0, max(0.5, float(ra_hdr)))
                except Exception:
                    retry_after = None
                cooldown = retry_after if retry_after is not None else min(5.0, 1.0 * (2 ** attempt))
                _extend_proxy_cooldown(cooldown)
                continue
            raise

    verified_mime = detect_and_validate_media_type(claimed_content_type, data)
    ext = ALLOWED_MIME_TYPES.get(verified_mime, ".jpg")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{url_hash}{ext}")
    tmp_suffix = f".tmp.{os.getpid()}.{threading.get_ident()}"
    tmp_img = cache_path + tmp_suffix
    tmp_meta = cache_path + ".meta" + tmp_suffix

    with _CACHE_WRITE_LOCK:
        try:
            with open(tmp_img, "wb") as f:
                f.write(data)
            os.replace(tmp_img, cache_path)

            with open(tmp_meta, "w", encoding="utf-8") as mf:
                mf.write(verified_mime)
            os.replace(tmp_meta, cache_path + ".meta")
            os.utime(cache_path, None)
        except Exception as e:
            print(f"[!] Warning: Could not write cache file: {e}")
        finally:
            for orphan in (tmp_img, tmp_meta):
                if os.path.exists(orphan):
                    try:
                        os.remove(orphan)
                    except OSError:
                        pass

        evict_disk_cache(cache_dir)

    return data, verified_mime