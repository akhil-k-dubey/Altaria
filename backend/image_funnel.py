#!/usr/bin/env python3
"""
M.A.R.K. Holo-CAD // Zero-Hardcoding 5-Stage Reference Image Funnel & 3D Self-Verification
=========================================================================================
Eliminates "garbage-in, garbage-out" 2D -> 3D TripoSR failures via a progressive funnel:
  Tiered Acquisition -> Stage 1 (Metadata Filter, 0 ms)
                     -> Stage 2 (Otsu / Sobel / Lab / Shannon Entropy CV Signals)
                     -> Stage 3 (CLIP openai/clip-vit-base-patch32 Centroid Bayes + MMR Diversity)
                     -> Stage 4 (Single-Call Qwen2.5-VL Multi-Image Structured Judge)
                     -> TripoSR Mesh Synthesis
                     -> Stage 5 (PCA + Component Diagnostics + Vectorized Z-Buffer Turntable + VLM Self-Verify)
  + Learned per-query user candidate preferences (data/candidate_preferences.json).

STRICTLY ZERO HARDCODING:
- Zero topic keyword dictionaries, zero per-object templates, zero hardcoded image URLs.
"""

import io
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
import scipy.special
import scipy.stats
import skimage.color
import skimage.filters
import torch
import trimesh
from PIL import Image, ImageDraw

from grammar_decoder import extract_json_with_bracket_depth
from image_proxy import fetch_or_cache_proxied_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFS_PATH = os.path.join(BASE_DIR, "data", "candidate_preferences.json")
os.makedirs(os.path.dirname(PREFS_PATH), exist_ok=True)

USER_AGENT = os.environ.get(
    "HOLO_CAD_USER_AGENT",
    "StarkHoloCAD-Funnel/3.2 (https://commons.wikimedia.org; free academic 3D CAD research)",
)
HEADERS = {"User-Agent": USER_AGENT}

# Stage 1..5 Environment-Configurable Parameters
STAGE1_MIN_WIDTH = max(128, int(os.environ.get("HOLO_CAD_S1_MIN_WIDTH", "640")))
STAGE1_MIN_ASPECT = float(os.environ.get("HOLO_CAD_S1_MIN_ASPECT", "0.4"))
STAGE1_MAX_ASPECT = float(os.environ.get("HOLO_CAD_S1_MAX_ASPECT", "2.5"))

STAGE2_DIM = max(96, int(os.environ.get("HOLO_CAD_S2_DIM", "192")))
MIN_PHOTO_MARGIN = float(os.environ.get("HOLO_CAD_MIN_PHOTO_MARGIN", "0.0"))
MIN_SUBJ_VS_PHOTO_MARGIN = float(os.environ.get("HOLO_CAD_MIN_SUBJ_PHOTO_MARGIN", "0.0"))
CLIP_DEDUP_COS = float(os.environ.get("HOLO_CAD_CLIP_DEDUP_COS", "0.94"))
CLIP_MMR_LAMBDA = float(os.environ.get("HOLO_CAD_CLIP_MMR_LAMBDA", "0.25"))

MIN_PCA_FLATNESS = float(os.environ.get("HOLO_CAD_MIN_PCA_FLATNESS", "0.06"))
MAX_PLANAR_SHEET_PLANARITY = float(os.environ.get("HOLO_CAD_MAX_PLANARITY", "0.62"))
MAX_PLANAR_SHEET_FLATNESS = float(os.environ.get("HOLO_CAD_MAX_PLANAR_FLATNESS", "0.21"))
MIN_FILAMENT_CROSS_FLATNESS = float(os.environ.get("HOLO_CAD_MIN_CROSS_FLATNESS", "0.45"))
MIN_MAIN_COMPONENT_FRAC = float(os.environ.get("HOLO_CAD_MIN_MAIN_COMP_FRAC", "0.35"))

GATE_TIMEOUT_MS = max(0, int(os.environ.get("HOLO_CAD_GATE_TIMEOUT_MS", "2000")))
SOFT_VETO_MARGIN_THRESHOLD = float(os.environ.get("HOLO_CAD_SOFT_VETO_MARGIN", "4.0"))
LEARNED_AUTO_MIN_MATCH = float(os.environ.get("HOLO_CAD_LEARNED_MIN_MATCH", "6.0"))
LEARNED_AUTO_MIN_PHOTO = float(os.environ.get("HOLO_CAD_LEARNED_MIN_PHOTO", "6.0"))

# Format/medium tokens only — never domain/topic nouns.
# Matched as delimiter-separated tokens (with optional plural 's') so words like
# "airplane" or "planetary" are never falsely rejected by "plan".
NON_PHOTO_MEDIA_TOKENS = frozenset({
    "diagram", "schematic", "cutaway", "drawing", "logo", "map", "graph",
    "plan", "chart", "infographic", "blueprint", "icon", "flowchart",
    "poster", "sketch", "svg", "clipart", "silhouette", "vector",
    "watermark", "banner", "illustration", "illustrated", "painting",
    "engraving", "lithograph", "etching", "woodcut", "cartoon", "cgi",
    "wireframe", "bioart",
})

# Grammatical & meta-media stopwords used solely to validate Commons category title overlap.
_CATEGORY_STOPWORDS = frozenset({
    "category", "of", "the", "a", "an", "in", "on", "at", "by", "for", "from",
    "to", "with", "without", "and", "or", "as", "into", "photo", "photos",
    "photograph", "photographs", "photography", "picture", "pictures",
    "image", "images", "pic", "pics", "media", "file", "files", "view",
    "views", "part", "parts", "type", "types", "other", "misc", "various",
})

ALLOWED_RASTER_MIMES = frozenset({"image/jpeg", "image/png"})
DISALLOWED_EXTENSIONS = (".svg", ".gif", ".tif", ".tiff", ".pdf", ".webp", ".xcf", ".webm", ".ogv")

# Lazy CLIP singleton (runs on CPU to keep 100% GPU VRAM free for Qwen2.5-VL + TripoSR)
_CLIP_LOCK = threading.Lock()
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_FAILED = False
_PREFS_LOCK = threading.Lock()

# Cooperative Wikimedia HTTP rate-limiter (prevents HTTP 429 bursts across concurrent workers)
_WIKI_HTTP_SEM = threading.Semaphore(max(1, int(os.environ.get("HOLO_CAD_WIKI_CONCURRENCY", "1"))))
_WIKI_RATE_LOCK = threading.Lock()
_WIKI_NEXT_SLOT_TIME = 0.0
_WIKI_MIN_INTERVAL_S = max(0.0, float(os.environ.get("HOLO_CAD_WIKI_INTERVAL_S", "0.35")))


def slugify(text: str) -> str:
    clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(text).lower())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "assembly"


# =========================================================================
# LEARNED USER VOTE / PREFERENCE STORE (ZERO HARDCODING)
# =========================================================================
def load_candidate_preferences() -> Dict[str, dict]:
    with _PREFS_LOCK:
        if not os.path.exists(PREFS_PATH):
            return {}
        try:
            with open(PREFS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def save_candidate_preference(query: str, candidate: dict, part_id: Optional[str] = None) -> dict:
    q_key = slugify(query)
    entry = {
        "query": query,
        "part_id": part_id or "__master__",
        "title": candidate.get("title", ""),
        "thumbUrl": candidate.get("thumbUrl", ""),
        "originalUrl": candidate.get("originalUrl") or candidate.get("thumbUrl", ""),
        "mime": candidate.get("mime", "image/jpeg"),
        "caption": candidate.get("caption", ""),
        "updatedAt": int(time.time()),
    }
    with _PREFS_LOCK:
        prefs = {}
        if os.path.exists(PREFS_PATH):
            try:
                with open(PREFS_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        prefs = loaded
            except Exception:
                prefs = {}
        prefs_key = f"{q_key}::{slugify(part_id)}" if part_id else q_key
        prefs[prefs_key] = entry
        tmp_path = f"{PREFS_PATH}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
        os.replace(tmp_path, PREFS_PATH)
    return entry


def get_candidate_preference(query: str, part_id: Optional[str] = None) -> Optional[dict]:
    prefs = load_candidate_preferences()
    q_key = slugify(query)
    if part_id:
        p_key = f"{q_key}::{slugify(part_id)}"
        return prefs.get(p_key)
    return prefs.get(q_key)


def serialize_candidate(c: dict) -> dict:
    """Strips internal transient PIL/tensor objects before JSON / SSE emission."""
    return {k: v for k, v in c.items() if not k.startswith("_")}


# =========================================================================
# TIERED CANDIDATE ACQUISITION (Wikipedia Lead -> Curated Categories -> Search)
# =========================================================================
def _wait_wiki_rate_slot() -> None:
    global _WIKI_NEXT_SLOT_TIME
    with _WIKI_RATE_LOCK:
        now = time.monotonic()
        wait_s = max(0.0, _WIKI_NEXT_SLOT_TIME - now)
        slot_start = max(_WIKI_NEXT_SLOT_TIME, now)
        _WIKI_NEXT_SLOT_TIME = slot_start + _WIKI_MIN_INTERVAL_S
    if wait_s > 0.0:
        time.sleep(wait_s)


def _extend_wiki_cooldown(cooldown_s: float) -> None:
    global _WIKI_NEXT_SLOT_TIME
    if cooldown_s <= 0.0:
        return
    with _WIKI_RATE_LOCK:
        target = time.monotonic() + cooldown_s
        if target > _WIKI_NEXT_SLOT_TIME:
            _WIKI_NEXT_SLOT_TIME = target


def _http_get_json(url: str, timeout: float = 6.0, retries: int = 2) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries + 1):
        _wait_wiki_rate_slot()
        try:
            with _WIKI_HTTP_SEM:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                retry_after = None
                try:
                    ra_hdr = e.headers.get("Retry-After") if e.headers else None
                    if ra_hdr:
                        retry_after = min(4.0, max(0.5, float(ra_hdr)))
                except Exception:
                    retry_after = None
                cooldown = retry_after if retry_after is not None else min(4.0, 1.0 * (2 ** attempt))
                _extend_wiki_cooldown(cooldown)
                continue
            raise
    return {}


def _infer_mime_from_filename(fname: str) -> str:
    ext_lower = os.path.splitext(fname.lower())[1]
    if ext_lower == ".png":
        return "image/png"
    if ext_lower == ".gif":
        return "image/gif"
    if ext_lower == ".svg":
        return "image/svg+xml"
    if ext_lower == ".webp":
        return "image/webp"
    if ext_lower in (".tif", ".tiff"):
        return "image/tiff"
    if ext_lower == ".pdf":
        return "application/pdf"
    return "image/jpeg"


def _build_commons_candidate(page: dict, source_tier: int, rank: int, source_label: str) -> Optional[dict]:
    imageinfo = page.get("imageinfo") or []
    if not imageinfo:
        return None
    info = imageinfo[0]
    orig_url = info.get("url")
    thumb_url = info.get("thumburl") or orig_url
    if not thumb_url:
        return None

    mime = str(info.get("mime", "")).lower()
    raw_title = str(page.get("title", "")).replace("File:", "").replace("_", " ").strip()
    orig_w = int(info.get("width") or info.get("thumbwidth") or 0)
    orig_h = int(info.get("height") or info.get("thumbheight") or 0)
    thumb_w = int(info.get("thumbwidth") or orig_w)
    thumb_h = int(info.get("thumbheight") or orig_h)

    return {
        "title": raw_title,
        "canonicalFile": str(page.get("title", "")).strip(),
        "thumbUrl": thumb_url,
        "originalUrl": orig_url or thumb_url,
        "mime": mime,
        "caption": raw_title[:137] + "..." if len(raw_title) > 140 else raw_title,
        "width": thumb_w,
        "height": thumb_h,
        "origWidth": orig_w,
        "origHeight": orig_h,
        "sourceTier": source_tier,
        "sourceLabel": source_label,
        "tierRank": rank,
    }


def _fetch_wikidata_commons_category(qid: str) -> Optional[str]:
    if not qid:
        return None
    try:
        wd_url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={urllib.parse.quote(qid)}&property=P373&format=json"
        wd_data = _http_get_json(wd_url, timeout=3.5)
        p373 = ((wd_data.get("claims") or {}).get("P373") or [{}])[0].get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(p373, str) and p373.strip():
            return f"Category:{p373.strip()}"
    except Exception:
        pass
    return None


def _fetch_wikipedia_lead_and_category(query: str) -> Tuple[List[dict], Dict, Optional[str]]:
    encoded = urllib.parse.quote(query.strip())
    candidates = []
    summary = {"title": query, "description": "", "extract": "", "thumbnail": None, "originalImage": None, "pageUrl": ""}
    commons_cat = None

    # 1. Fast Wikipedia REST v1 summary (returns title, summary, lead image, AND wikibase_item directly)
    wiki_title_for_qid = None
    rest_qid = None
    try:
        rest_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        rest_data = _http_get_json(rest_url, timeout=4.5)
        if rest_data.get("type") != "disambiguation" and rest_data.get("title"):
            title = rest_data["title"]
            wiki_title_for_qid = title
            rest_qid = rest_data.get("wikibase_item")
            orig = (rest_data.get("originalimage") or {}).get("source")
            thumb = (rest_data.get("thumbnail") or {}).get("source")
            summary["title"] = title
            summary["description"] = rest_data.get("description", "")
            summary["extract"] = rest_data.get("extract", "")
            summary["thumbnail"] = thumb or orig
            summary["originalImage"] = orig or thumb
            summary["pageUrl"] = (rest_data.get("content_urls") or {}).get("desktop", {}).get("page", "")
            if orig or thumb:
                orig_obj = rest_data.get("originalimage") or {}
                thumb_obj = rest_data.get("thumbnail") or {}
                w = int(orig_obj.get("width") or thumb_obj.get("width") or 0)
                h = int(orig_obj.get("height") or thumb_obj.get("height") or 0)
                img_ref = orig or thumb
                fname = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(img_ref).path)).replace("_", " ")
                mime = _infer_mime_from_filename(fname)
                candidates.append({
                    "title": fname or f"{title} Lead Image",
                    "canonicalFile": f"File:{fname}" if fname else f"wiki:{title}:0",
                    "thumbUrl": thumb or orig,
                    "originalUrl": orig or thumb,
                    "mime": mime,
                    "caption": f"Wikipedia Lead Image ({title})",
                    "width": int(thumb_obj.get("width") or w),
                    "height": int(thumb_obj.get("height") or h),
                    "origWidth": w,
                    "origHeight": h,
                    "sourceTier": 1,
                    "sourceLabel": "wikipedia_lead",
                    "tierRank": 0,
                })
    except Exception:
        pass

    if rest_qid:
        commons_cat = _fetch_wikidata_commons_category(str(rest_qid))

    # 2. Action API fallback ONLY if REST v1 missed the title/QID
    if not wiki_title_for_qid or not rest_qid:
        try:
            if wiki_title_for_qid:
                q_url = (
                    "https://en.wikipedia.org/w/api.php?action=query"
                    f"&titles={urllib.parse.quote(wiki_title_for_qid)}&prop=pageprops&ppprop=wikibase_item&format=json"
                )
            else:
                q_url = (
                    "https://en.wikipedia.org/w/api.php?action=query&generator=search"
                    f"&gsrsearch={encoded}&gsrlimit=2&prop=pageimages|pageprops|extracts"
                    "&exintro=1&explaintext=1&exsentences=3&piprop=original|thumbnail"
                    "&pithumbsize=1024&ppprop=wikibase_item&format=json"
                )

            data = _http_get_json(q_url, timeout=5.0)
            pages = list((data.get("query") or {}).get("pages", {}).values())
            pages.sort(key=lambda p: p.get("index", 999))

            for idx, page in enumerate(pages):
                title = page.get("title", query)
                if not candidates:
                    orig = (page.get("original") or {}).get("source")
                    thumb = (page.get("thumbnail") or {}).get("source")
                    if idx == 0 and not summary.get("extract"):
                        summary["title"] = title
                        summary["extract"] = page.get("extract", "")
                        summary["thumbnail"] = thumb or orig
                        summary["originalImage"] = orig or thumb
                        summary["pageUrl"] = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

                    img_url = thumb or orig
                    if img_url:
                        orig_obj = page.get("original") or {}
                        thumb_obj = page.get("thumbnail") or {}
                        w = int(orig_obj.get("width") or thumb_obj.get("width") or 0)
                        h = int(orig_obj.get("height") or thumb_obj.get("height") or 0)
                        fname = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(orig or img_url).path)).replace("_", " ")
                        mime = _infer_mime_from_filename(fname)
                        candidates.append({
                            "title": fname or f"{title} Lead Image",
                            "canonicalFile": f"File:{fname}" if fname else f"wiki:{title}:{idx}",
                            "thumbUrl": thumb or orig,
                            "originalUrl": orig or thumb,
                            "mime": mime,
                            "caption": f"Wikipedia Lead Image ({title})",
                            "width": int(thumb_obj.get("width") or w),
                            "height": int(thumb_obj.get("height") or h),
                            "origWidth": w,
                            "origHeight": h,
                            "sourceTier": 1,
                            "sourceLabel": "wikipedia_lead",
                            "tierRank": idx,
                        })

                qid = (page.get("pageprops") or {}).get("wikibase_item")
                if qid and not commons_cat:
                    commons_cat = _fetch_wikidata_commons_category(str(qid))
        except Exception as e:
            print(f"[!] Wikipedia Tier-1 acquisition warning for '{query}': {e}")

    return candidates, summary, commons_cat


def _split_alnum_tokens(text: str) -> List[str]:
    buf = [c.lower() if c.isalnum() else " " for c in str(text or "")]
    return [tok for tok in "".join(buf).split() if tok]


def _extract_content_tokens(text: str) -> set:
    tokens = set()
    for tok in _split_alnum_tokens(text):
        if len(tok) < 2 or tok in _CATEGORY_STOPWORDS or tok in NON_PHOTO_MEDIA_TOKENS:
            continue
        tokens.add(tok)
        if len(tok) > 3 and tok.endswith("s"):
            stem = tok[:-1]
            if stem not in _CATEGORY_STOPWORDS and stem not in NON_PHOTO_MEDIA_TOKENS:
                tokens.add(stem)
    return tokens


def _category_matches_query(query: str, category_title: str) -> bool:
    """
    Guards Commons namespace-14 category search fallback against unrelated hijacks
    (e.g. 'Category:Iconic photographs', 'Category:People', 'Category:Microscope slides and coverslips')
    using pure lexical content-token intersection (zero domain lists).
    """
    q_raw = [
        tok for tok in _split_alnum_tokens(query)
        if len(tok) >= 2 and tok not in _CATEGORY_STOPWORDS and tok not in NON_PHOTO_MEDIA_TOKENS
    ]
    if not q_raw:
        return False
    q_forms = set()
    for t in q_raw:
        q_forms.add(t)
        if len(t) > 3 and t.endswith("s"):
            q_forms.add(t[:-1])

    c_toks = _extract_content_tokens(category_title)
    if not c_toks:
        return False

    matched_q = sum(
        1 for t in q_raw
        if (t in c_toks) or (len(t) > 3 and t.endswith("s") and t[:-1] in c_toks)
    )
    if len(q_raw) == 1:
        return matched_q == 1
    if len(q_raw) >= 2:
        # Require at least 2 distinct query tokens in the category title,
        # OR exact single-topic category where all category tokens appear in the query.
        return matched_q >= 2 or (matched_q >= 1 and c_toks <= q_forms)
    return bool(q_forms & c_toks)


def _fetch_commons_category_members(category_title: str, limit: int = 80, tier_rank_offset: int = 0) -> List[dict]:
    encoded_cat = urllib.parse.quote(category_title)
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers"
        f"&gcmtitle={encoded_cat}&gcmtype=file&gcmlimit={min(100, limit)}"
        "&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=640&format=json"
    )
    out = []
    try:
        data = _http_get_json(url, timeout=5.5)
        pages = list((data.get("query") or {}).get("pages", {}).values())
        for idx, p in enumerate(pages):
            cand = _build_commons_candidate(p, source_tier=2, rank=tier_rank_offset + idx, source_label=f"commons_cat:{category_title}")
            if cand:
                out.append(cand)
    except Exception as e:
        print(f"[!] Commons category fetch warning ({category_title}): {e}")
    return out


def _search_commons_categories(query: str, limit: int = 3) -> List[str]:
    encoded = urllib.parse.quote(query.strip())
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&list=search"
        f"&srsearch={encoded}&srnamespace=14&srlimit={limit}&format=json"
    )
    try:
        data = _http_get_json(url, timeout=4.5)
        return [
            item["title"]
            for item in (data.get("query") or {}).get("search", [])
            if item.get("title") and _category_matches_query(query, item["title"])
        ]
    except Exception:
        return []


def _fetch_commons_search_page(query: str, offset: int = 0, limit: int = 50) -> List[dict]:
    encoded = urllib.parse.quote(query.strip())
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={encoded}&gsrnamespace=6&gsrlimit={min(50, limit)}&gsroffset={offset}"
        "&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=640&format=json"
    )
    out = []
    try:
        data = _http_get_json(url, timeout=6.0)
        pages = list((data.get("query") or {}).get("pages", {}).values())
        pages.sort(key=lambda p: p.get("index", 9999))
        for p in pages:
            rank = int(p.get("index", offset + len(out)))
            cand = _build_commons_candidate(p, source_tier=3, rank=rank, source_label="commons_search")
            if cand:
                out.append(cand)
    except Exception as e:
        print(f"[!] Commons search warning ('{query}', offset={offset}): {e}")
    return out


def _candidate_tier_sort_key(cand: dict, prefer_search: bool = False) -> Tuple[int, int]:
    tier = int(cand.get("sourceTier", 3))
    rank = int(cand.get("tierRank", 9999))
    if prefer_search:
        prio = 0 if tier <= 1 else (1 if tier == 3 else 2)
        return (prio, rank)
    return (tier, rank)


def acquire_candidates(
    query: str,
    max_candidates: int = 200,
    exclude_categories: Optional[set] = None,
    exclude_wiki_titles: Optional[set] = None,
    prefer_search: bool = False,
) -> Tuple[List[dict], Dict]:
    """
    Gentle tiered Wikimedia acquisition (zero hardcoding):
      Tier 1: Wikipedia lead images & summary + Wikidata curated category (P373)
      Tier 2: Curated Wikimedia Commons category files (overlap-validated on fallback)
      Tier 3: Wikimedia Commons search (prioritized ahead of Tier 2 when prefer_search=True)
    """
    clean_query = " ".join(str(query or "").split())
    if not clean_query:
        return [], {}

    excl_cats = {str(c).strip().lower() for c in (exclude_categories or ()) if c}
    excl_titles = {str(t).strip().lower() for t in (exclude_wiki_titles or ()) if t}

    all_raw: List[dict] = []
    wiki_summary: Dict = {}
    cat_titles: List[str] = []

    wiki_cands, wiki_summary, wd_cat = _fetch_wikipedia_lead_and_category(clean_query)
    resolved_wtitle = str(wiki_summary.get("title") or "").strip().lower()
    if resolved_wtitle and resolved_wtitle in excl_titles:
        wiki_cands = []
        wd_cat = None
        wiki_summary["thumbnail"] = None
        wiki_summary["originalImage"] = None
    all_raw.extend(wiki_cands)

    search0_limit = 50 if max_candidates >= 40 else max(25, max_candidates)
    search0_cands = _fetch_commons_search_page(clean_query, 0, search0_limit)
    all_raw.extend(search0_cands)

    if wd_cat and wd_cat.strip().lower() not in excl_cats:
        cat_titles.append(wd_cat)
    elif (not wd_cat or wd_cat.strip().lower() in excl_cats) and not (prefer_search and len(search0_cands) >= 10):
        if max_candidates > 50 or len(all_raw) < 20:
            for c in _search_commons_categories(clean_query, 3):
                if c.strip().lower() not in excl_cats and c not in cat_titles:
                    cat_titles.append(c)
                    break

    wiki_summary["commonsCategory"] = cat_titles[0] if cat_titles else None

    if cat_titles:
        cat_limit = 12 if prefer_search else min(60, max(20, max_candidates // 2))
        all_raw.extend(_fetch_commons_category_members(cat_titles[0], limit=cat_limit, tier_rank_offset=0))

    if max_candidates > 70 and len(search0_cands) >= 40:
        all_raw.extend(_fetch_commons_search_page(clean_query, 50, 50))
    if max_candidates > 140 and len(all_raw) < max_candidates - 20:
        all_raw.extend(_fetch_commons_search_page(clean_query, 100, 50))

    dedup: Dict[str, dict] = {}
    for c in all_raw:
        key = (c.get("canonicalFile") or c.get("thumbUrl") or "").lower().strip()
        if not key:
            continue
        existing = dedup.get(key)
        if existing is None or _candidate_tier_sort_key(c, prefer_search) < _candidate_tier_sort_key(existing, prefer_search):
            dedup[key] = c

    merged = list(dedup.values())
    merged.sort(key=lambda c: _candidate_tier_sort_key(c, prefer_search))
    return merged[:max_candidates], wiki_summary


# =========================================================================
# STAGE 1: METADATA FILTERS (0 ms, FREE)
# =========================================================================
def _has_non_photo_filename_token(title_or_url: str) -> Optional[str]:
    stem = os.path.splitext(str(title_or_url).lower())[0]
    tokens = _split_alnum_tokens(stem)
    for tok in tokens:
        if tok in NON_PHOTO_MEDIA_TOKENS:
            return tok
        if len(tok) > 3 and tok.endswith("s") and tok[:-1] in NON_PHOTO_MEDIA_TOKENS:
            return tok
    return None


def stage1_metadata_filter(
    candidates: List[dict], max_survivors: int = 60, prefer_search: bool = False
) -> List[dict]:
    """
    Zero-vision metadata sieve (env-configurable bounds):
      - JPEG/PNG only (kills SVGs, GIFs, PDFs, WebPs)
      - Rejects diagram/schematic/cutaway/drawing/logo/map/graph/plan tokens
      - Enforces aspect ratio in [STAGE1_MIN_ASPECT, STAGE1_MAX_ASPECT]
      - Enforces width >= STAGE1_MIN_WIDTH
      - Preserves Tier 1 lead, and guarantees Tier 3 search diversity alongside Tier 2 categories
    """
    survivors = []
    for cand in candidates:
        mime = str(cand.get("mime", "")).lower()
        title = str(cand.get("title", ""))
        orig_url = str(cand.get("originalUrl") or cand.get("thumbUrl") or "").lower()
        url_path = urllib.parse.urlparse(orig_url).path

        if mime not in ALLOWED_RASTER_MIMES:
            continue
        if any(url_path.endswith(ext) or title.lower().endswith(ext) for ext in DISALLOWED_EXTENSIONS):
            continue
        if _has_non_photo_filename_token(title) or _has_non_photo_filename_token(os.path.basename(url_path)):
            continue

        orig_w = int(cand.get("origWidth") or cand.get("width") or 0)
        orig_h = int(cand.get("origHeight") or cand.get("height") or 0)
        if orig_w < STAGE1_MIN_WIDTH or orig_h <= 0:
            continue
        aspect = orig_w / float(orig_h)
        if not (STAGE1_MIN_ASPECT <= aspect <= STAGE1_MAX_ASPECT):
            continue

        tier = max(1, int(cand.get("sourceTier", 3)))
        c_copy = dict(cand)
        c_copy["aspectRatio"] = round(aspect, 3)
        c_copy["stage1Score"] = round(1.0 / float(tier), 4)
        survivors.append(c_copy)

    survivors.sort(key=lambda c: _candidate_tier_sort_key(c, prefer_search))
    if not prefer_search and len(survivors) > max_survivors and max_survivors >= 12:
        t1 = [c for c in survivors if int(c.get("sourceTier", 3)) <= 1]
        t2 = [c for c in survivors if int(c.get("sourceTier", 3)) == 2]
        t3 = [c for c in survivors if int(c.get("sourceTier", 3)) >= 3]
        if t3 and len(t1) + len(t2) > max_survivors - 4:
            t2_cap = max(8, ((max_survivors - len(t1)) * 2) // 3)
            kept = t1 + t2[:t2_cap]
            rem_slots = max(0, max_survivors - len(kept))
            kept.extend(t3[:rem_slots])
            if len(kept) < max_survivors:
                kept.extend(t2[t2_cap : t2_cap + (max_survivors - len(kept))])
            return kept[:max_survivors]

    return survivors[:max_survivors]


# =========================================================================
# STAGE 2: FAST SCIENTIFIC CV SIGNALS (Otsu / Sobel / CIE Lab / Shannon Entropy)
# =========================================================================
def compute_cv_signals(pil_img: Image.Image) -> dict:
    """
    Computes information-theoretic and perceptual CV signals at STAGE2_DIM^2 resolution:
      - Otsu inter-class variance separability (eta) + valley density -> distinguishes bimodal ink/line-art from shaded photos
      - Sobel gradient magnitude + Laplacian variance -> sharp structure vs flat noise
      - CIE L*a*b* chroma C* & outer-border Gaussian uniformity -> predicts clean rembg cutout vs monochrome scan
      - Shannon color entropy (8x8x8 cube / 9 bits) & luminance entropy (/ 6 bits) -> exact information content
      - Unweighted geometric mean across (tonal continuity, information richness, border cleanliness)
    """
    dim = STAGE2_DIM
    border_w = max(2, dim // 16)
    has_transparent_border = False

    if pil_img.mode == "RGBA":
        rgba_small = pil_img.resize((dim, dim), Image.Resampling.BILINEAR)
        arr_rgba = np.asarray(rgba_small, dtype=np.float32) / 255.0
        alpha = arr_rgba[:, :, 3]
        b_alpha = np.concatenate([
            alpha[:border_w, :].ravel(),
            alpha[-border_w:, :].ravel(),
            alpha[:, :border_w].ravel(),
            alpha[:, -border_w:].ravel(),
        ])
        opaque_frac = float(np.mean(alpha > 0.5))
        if float(np.mean(b_alpha)) < 0.25 and 0.008 < opaque_frac < 0.98:
            has_transparent_border = True
        rgb_np = np.clip(arr_rgba[:, :, :3] * alpha[:, :, None] + 0.5 * (1.0 - alpha[:, :, None]), 0.0, 1.0)
    else:
        rgb_small = pil_img.convert("RGB").resize((dim, dim), Image.Resampling.BILINEAR)
        rgb_np = np.clip(np.asarray(rgb_small, dtype=np.float32) / 255.0, 0.0, 1.0)

    # 1. Perceptual CIE L*a*b* and Rec.709 luminance
    lab = skimage.color.rgb2lab(rgb_np)
    lum = np.clip(0.2126 * rgb_np[:, :, 0] + 0.7152 * rgb_np[:, :, 1] + 0.0722 * rgb_np[:, :, 2], 0.0, 1.0)
    lum_std = float(np.std(lum))
    chroma_lab = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
    mean_chroma_lab = float(np.mean(chroma_lab))

    # 2. Sobel edge density & Laplacian focus variance
    sobel_grad = skimage.filters.sobel(lum)
    edge_density = float(np.mean(sobel_grad > 0.10))
    laplacian_var = float(np.var(scipy.ndimage.laplace(lum)))

    # 3. Otsu inter-class variance ratio eta = sigma_B^2 / sigma_T^2 & threshold valley fraction
    if float(np.ptp(lum)) > 1e-4:
        tau = float(skimage.filters.threshold_otsu(lum))
        fg_mask = lum > tau
        w0 = float(np.mean(~fg_mask))
        w1 = 1.0 - w0
        if w0 > 1e-6 and w1 > 1e-6:
            mu0 = float(np.mean(lum[~fg_mask]))
            mu1 = float(np.mean(lum[fg_mask]))
            sigma_b2 = w0 * w1 * ((mu0 - mu1) ** 2)
            sigma_t2 = max(1e-8, float(np.var(lum)))
            otsu_eta = float(np.clip(sigma_b2 / sigma_t2, 0.0, 1.0))
        else:
            otsu_eta = 0.0
        otsu_valley_frac = float(np.mean(np.abs(lum - tau) <= 0.15))
    else:
        otsu_eta = 0.0
        otsu_valley_frac = 0.0

    bimodal_extreme_frac = float(np.mean((lum > 0.88) | (lum < 0.14)))
    midtone_frac = float(np.mean((lum >= 0.20) & (lum <= 0.80)))

    # 4. Outer border uniformity & Lab border-center contrast (Gaussian dispersion, no magic slope)
    if has_transparent_border:
        b_alpha_clear = float(np.mean(b_alpha < 0.15))
        border_uniformity = float(np.clip(b_alpha_clear, 0.0, 1.0))
        border_lab_delta = float(np.mean(np.abs(lab[border_w:-border_w, border_w:-border_w, 0] - 50.0)))
    else:
        top = lab[:border_w, :, :].reshape(-1, 3)
        bot = lab[-border_w:, :, :].reshape(-1, 3)
        left = lab[border_w:-border_w, :border_w, :].reshape(-1, 3)
        right = lab[border_w:-border_w, -border_w:, :].reshape(-1, 3)
        border_lab = np.concatenate([top, bot, left, right], axis=0)
        border_disp = float(np.linalg.norm(np.std(border_lab, axis=0)))
        # Exponential Gaussian kernel scaled by standard CIE Lab JND ~18 units
        border_uniformity = float(np.exp(-((border_disp / 18.0) ** 2)))
        inner_mean = np.mean(lab[border_w:-border_w, border_w:-border_w, :].reshape(-1, 3), axis=0)
        border_mean = np.mean(border_lab, axis=0)
        border_lab_delta = float(np.linalg.norm(inner_mean - border_mean))

    # 5. Exact Shannon Color Entropy (512 bins -> max 9 bits) & Luminance Entropy (64 bins -> max 6 bits)
    q_rgb = np.clip((rgb_np * 7.999).astype(np.int32), 0, 7)
    bins = (q_rgb[:, :, 0] << 6) | (q_rgb[:, :, 1] << 3) | q_rgb[:, :, 2]
    counts = np.bincount(bins.ravel(), minlength=512).astype(np.float64)
    color_entropy = float(scipy.stats.entropy(counts[counts > 0], base=2))
    entropy_norm = float(np.clip(color_entropy / 9.0, 0.0, 1.0))

    q_lum = np.clip((lum * 63.999).astype(np.int32), 0, 63)
    l_counts = np.bincount(q_lum.ravel(), minlength=64).astype(np.float64)
    lum_entropy = float(scipy.stats.entropy(l_counts[l_counts > 0], base=2))
    lum_entropy_norm = float(np.clip(lum_entropy / 6.0, 0.0, 1.0))

    c_max = np.max(rgb_np, axis=2)
    c_min = np.min(rgb_np, axis=2)
    sat = np.where(c_max > 1e-4, (c_max - c_min) / np.maximum(1e-4, c_max), 0.0)
    mean_saturation = float(np.mean(sat))
    colorful_frac = float(np.mean(sat > 0.08))

    # Information-theoretic / Otsu rejection rules (avoid false rejection on real studio silver hardware)
    is_flat_blank = lum_std < 0.015 or color_entropy < 1.5
    is_bimodal_line_art = (
        not has_transparent_border
        and otsu_eta > 0.80
        and (
            (otsu_valley_frac < 0.18 and edge_density > 0.06 and lum_entropy < 3.4)
            or (
                otsu_valley_frac < 0.12
                and (
                    (edge_density > 0.045 and lum_entropy < 3.6)
                    or (bimodal_extreme_frac > 0.48 and midtone_frac < 0.26)
                )
            )
        )
    )
    is_monochrome_drawing = (
        not has_transparent_border
        and mean_chroma_lab < 3.5
        and otsu_eta > 0.72
        and otsu_valley_frac < 0.16
        and color_entropy < 3.4
    )

    rejected = bool(is_flat_blank or is_bimodal_line_art or is_monochrome_drawing)
    reject_reason = None
    if is_flat_blank:
        reject_reason = "flat_or_blank"
    elif is_bimodal_line_art:
        reject_reason = "bimodal_line_art"
    elif is_monochrome_drawing:
        reject_reason = "monochrome_drawing"

    # Unweighted geometric mean across 3 bounded [0, 1] physical quality axes
    f_tone = float(np.clip(1.0 - otsu_eta * (1.0 - otsu_valley_frac), 1e-3, 1.0))
    f_info = float(max(1e-3, math.sqrt(max(1e-6, entropy_norm * lum_entropy_norm))))
    f_border = float(max(1e-3, border_uniformity))
    cv_score = float((f_tone * f_info * f_border) ** (1.0 / 3.0))

    return {
        "edgeDensity": round(edge_density, 4),
        "laplacianVar": round(laplacian_var, 5),
        "otsuSeparability": round(otsu_eta, 4),
        "otsuValleyFrac": round(otsu_valley_frac, 4),
        "bimodalExtremeFrac": round(bimodal_extreme_frac, 4),
        "midtoneFrac": round(midtone_frac, 4),
        "borderUniformity": round(border_uniformity, 4),
        "borderLabDelta": round(border_lab_delta, 2),
        "colorEntropy": round(color_entropy, 3),
        "lumEntropy": round(lum_entropy, 3),
        "meanChromaLab": round(mean_chroma_lab, 2),
        "meanSaturation": round(mean_saturation, 4),
        "colorfulFrac": round(colorful_frac, 4),
        "hasTransparentBorder": has_transparent_border,
        "cvScore": round(cv_score, 4),
        "rejected": rejected,
        "rejectReason": reject_reason,
    }


def _fetch_and_score_cv(cand: dict) -> Optional[dict]:
    url = cand.get("thumbUrl") or cand.get("originalUrl")
    if not url:
        return None
    try:
        data, _ = fetch_or_cache_proxied_image(url)
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            pil_copy = im.copy()
        metrics = compute_cv_signals(pil_copy)
        out = dict(cand)
        out["cvMetrics"] = metrics
        out["_pil_image"] = pil_copy.convert("RGB")
        out["_pil_rgba"] = pil_copy if pil_copy.mode == "RGBA" else pil_copy.convert("RGB")
        tier = max(1, int(cand.get("sourceTier", 3)))
        # Mild monotonic provenance damping keeping score strictly in [0, 1]
        out["stage2Score"] = round(metrics["cvScore"] / (1.0 + 0.05 * (tier - 1)), 4)
        return out
    except Exception:
        return None


def stage2_cv_filter(survivors: List[dict], max_downloads: int = 32, top_k: int = 12) -> List[dict]:
    """Downloads Stage-1 survivors concurrently via SSRF proxy and scores scientific CV signals."""
    if not survivors:
        return []

    to_fetch = survivors[:max_downloads]
    scored = []
    relaxed_pool = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_fetch_and_score_cv, c) for c in to_fetch]
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            relaxed_pool.append(res)
            if not res["cvMetrics"]["rejected"]:
                scored.append(res)

    pool_to_rank = scored if scored else [c for c in relaxed_pool if c["cvMetrics"].get("rejectReason") != "flat_or_blank"]
    if not pool_to_rank:
        pool_to_rank = relaxed_pool

    pool_to_rank.sort(key=lambda c: c.get("stage2Score", 0.0), reverse=True)
    return pool_to_rank[:top_k]


# =========================================================================
# STAGE 3: CLIP SEMANTIC FILTER (Centroid Ensembles + Bayes Gate + MMR)
# =========================================================================
def _get_clip_components():
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_FAILED
    if _CLIP_FAILED:
        return None, None
    with _CLIP_LOCK:
        if _CLIP_MODEL is not None and _CLIP_PROCESSOR is not None:
            return _CLIP_MODEL, _CLIP_PROCESSOR
        try:
            from transformers import CLIPModel, CLIPProcessor
            model_id = "openai/clip-vit-base-patch32"
            try:
                proc = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
                mdl = CLIPModel.from_pretrained(model_id, local_files_only=True)
            except Exception:
                proc = CLIPProcessor.from_pretrained(model_id)
                mdl = CLIPModel.from_pretrained(model_id)
            mdl = mdl.to("cpu")
            mdl.eval()
            _CLIP_PROCESSOR = proc
            _CLIP_MODEL = mdl
            print("[✓] CLIP ViT-B/32 semantic filter ready on CPU (0 MiB GPU VRAM used)")
            return _CLIP_MODEL, _CLIP_PROCESSOR
        except Exception as e:
            print(f"[!] Warning: Could not load CLIP ViT-B/32 ({e}); skipping Stage 3.")
            _CLIP_FAILED = True
            return None, None


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms


def stage3_clip_filter(
    survivors: List[dict],
    query: str,
    top_k: int = 4,
    part_context: Optional[str] = None,
) -> List[dict]:
    """
    CLIP (openai/clip-vit-base-patch32) dual-axis prompt-centroid Bayes filter:
      1) L2-normalized prompt ensemble centroids for [subject_photo, photo_medium, non_photo_medium]
      2) Exact CLIP temperature-scaled softmax posteriors with zero-bias Bayes gate (photoMargin >= 0.0)
      3) Never backfills verified photo pool with diagrams/engravings
      4) Maximal Marginal Relevance (MMR) selection using pairwise CLIP embedding cosine similarity
    """
    if not survivors:
        return []

    clip_model, clip_proc = _get_clip_components()
    if clip_model is None or clip_proc is None:
        return survivors[:top_k]

    imgs = []
    valid_cands = []
    for c in survivors:
        pil_im = c.get("_pil_image")
        if pil_im is None:
            continue
        imgs.append(pil_im)
        valid_cands.append(c)

    if not imgs:
        return survivors[:top_k]

    clean_q = " ".join(str(query).split())
    clean_ctx = " ".join(str(part_context).split()) if part_context else ""
    is_subpart = bool(clean_ctx and clean_ctx.lower() != clean_q.lower())
    if is_subpart:
        subj_prompts = [
            f"a real camera photograph showing the isolated {clean_q} component",
            f"a close-up or studio macro photograph focusing specifically on {clean_q} of {clean_ctx}",
            f"a documentary photograph of an individual physical {clean_q} assembly",
        ]
    else:
        subj_prompts = [
            f"a real camera photograph of a complete {clean_q}",
            f"a studio or museum photograph showing an entire {clean_q}",
            f"a documentary photograph of an authentic physical {clean_q}",
        ]
    photo_prompts = [
        "a real camera photograph of a physical object",
        "an authentic studio or macro camera photograph with natural lighting",
        "a candid photograph of a real-world object or specimen",
    ]
    non_photo_prompts = [
        "a digital illustration, 3D CGI rendering, medical textbook drawing, painting, or vector artwork",
        "a schematic diagram, cutaway chart, labeled infographic, blueprint, or close-up cropped detail",
        "a vintage engraving, lithograph, etching, woodcut, or inked anatomical plate",
    ]
    all_prompts = subj_prompts + photo_prompts + non_photo_prompts
    n_subj = len(subj_prompts)
    n_photo = len(photo_prompts)

    try:
        with torch.no_grad():
            inputs = clip_proc(text=all_prompts, images=imgs, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            img_embeds = _l2_normalize_rows(outputs.image_embeds.cpu().numpy())
            txt_embeds = _l2_normalize_rows(outputs.text_embeds.cpu().numpy())
            logit_scale = float(clip_model.logit_scale.exp().item())

        # Ensemble L2-normalized class centroids: (3, D)
        c_subj = txt_embeds[:n_subj].mean(axis=0, keepdims=True)
        c_photo = txt_embeds[n_subj : n_subj + n_photo].mean(axis=0, keepdims=True)
        c_non = txt_embeds[n_subj + n_photo :].mean(axis=0, keepdims=True)
        centroids = _l2_normalize_rows(np.vstack([c_subj, c_photo, c_non]))

        cos_sim = img_embeds @ centroids.T  # (N, 3): [subj, photo, non_photo]
        logits = cos_sim * logit_scale      # exact CLIP logit scale

        photo_logits = logits[:, 1]
        non_logits = logits[:, 2]
        subj_logits = logits[:, 0]

        photo_margin = photo_logits - non_logits
        subj_margin = subj_logits - non_logits
        subj_vs_photo = subj_logits - photo_logits

        photo_probs = scipy.special.softmax(np.stack([photo_logits, non_logits], axis=1), axis=1)[:, 0]
        subj_probs = scipy.special.softmax(np.stack([subj_logits, non_logits], axis=1), axis=1)[:, 0]
        probs_3way = scipy.special.softmax(logits, axis=1)

        scored = []
        for idx, c in enumerate(valid_cands):
            c_out = dict(c)
            c_out["_clip_embed"] = img_embeds[idx]
            p_marg = float(photo_margin[idx])
            s_marg = float(subj_margin[idx])
            s_vs_p = float(subj_vs_photo[idx])
            p_prob = float(photo_probs[idx])
            s_prob = float(subj_probs[idx])
            s_vs_p_prob = float(probs_3way[idx, 0])
            cos_marg_subj = float(cos_sim[idx, 0] - cos_sim[idx, 2])
            cos_s_vs_p = float(cos_sim[idx, 0] - cos_sim[idx, 1])
            is_photo_pass = bool(p_marg >= MIN_PHOTO_MARGIN)
            is_subj_match = bool(s_vs_p >= MIN_SUBJ_VS_PHOTO_MARGIN)

            # Symmetric geometric Bayes likelihood x contrastive subject-vs-generic tie-breaker
            geom_prob = math.sqrt(max(1e-6, p_prob * s_prob)) * (0.35 + 0.65 * s_vs_p_prob)
            base_score = geom_prob * (1.0 + max(-0.5, cos_marg_subj + 1.25 * cos_s_vs_p))
            if is_photo_pass and is_subj_match:
                clip_rank_score = base_score
            elif is_photo_pass:
                clip_rank_score = base_score * 0.65
            else:
                clip_rank_score = base_score * 0.35

            c_out["clipScore"] = {
                "posLogit": round(float(subj_logits[idx]), 3),
                "photoLogit": round(float(photo_logits[idx]), 3),
                "negLogit": round(float(non_logits[idx]), 3),
                "margin": round(s_marg, 3),
                "subjMargin": round(s_marg, 3),
                "photoMargin": round(p_marg, 3),
                "subjVsPhotoMargin": round(s_vs_p, 3),
                "photoProb": round(p_prob, 4),
                "subjProb": round(s_prob, 4),
                "subjVsPhotoProb": round(s_vs_p_prob, 4),
                "cosSubjMargin": round(cos_marg_subj, 4),
                "cosSubjVsPhoto": round(cos_s_vs_p, 4),
                "isPhotoPass": is_photo_pass,
                "isSubjMatch": is_subj_match,
                "combinedScore": round(clip_rank_score, 4),
            }
            c_out["stage3Score"] = round(clip_rank_score, 4)
            scored.append(c_out)

        # Strict authenticity preservation: prefer contrastive subject matches over generic scenes
        valid_subject_photos = [
            x for x in scored
            if x["clipScore"]["isPhotoPass"]
            and x["clipScore"]["subjMargin"] >= 0.0
            and x["clipScore"]["isSubjMatch"]
        ]
        if valid_subject_photos:
            active_pool = valid_subject_photos
        else:
            fallback_subj = [
                x for x in scored
                if x["clipScore"]["isPhotoPass"] and x["clipScore"]["subjMargin"] >= 0.0
            ]
            if fallback_subj:
                active_pool = fallback_subj
            else:
                photo_only = [x for x in scored if x["clipScore"]["isPhotoPass"]]
                active_pool = photo_only if photo_only else scored

        active_pool.sort(key=lambda x: x["stage3Score"], reverse=True)

        # Maximal Marginal Relevance (MMR) greedy selection with cosine duplicate exclusion
        chosen: List[dict] = []
        remaining = list(active_pool)
        while remaining and len(chosen) < top_k:
            if not chosen:
                chosen.append(remaining.pop(0))
                continue

            best_idx = -1
            best_mmr = -1e9
            deferred_indices = []
            for r_i, item in enumerate(remaining):
                emb = item["_clip_embed"]
                max_sim = max(float(np.dot(emb, k_item["_clip_embed"])) for k_item in chosen)
                if max_sim >= CLIP_DEDUP_COS:
                    deferred_indices.append(r_i)
                    continue
                mmr_val = float(item["stage3Score"]) - CLIP_MMR_LAMBDA * max_sim
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = r_i

            if best_idx >= 0:
                chosen.append(remaining.pop(best_idx))
            else:
                break

        return chosen
    except Exception as e:
        print(f"[!] Stage 3 CLIP inference warning: {e}")
        return survivors[:top_k]


# =========================================================================
# STAGE 4: VLM MULTI-IMAGE JUDGE (SINGLE QWEN2.5-VL STRUCTURED CALL)
# =========================================================================
def build_stage4_judge_prompt(query: str, count: int, part_context: Optional[str] = None) -> str:
    clean_q = " ".join(str(query).split())
    clean_ctx = " ".join(str(part_context).split()) if part_context else ""
    is_subpart = bool(clean_ctx and clean_ctx.lower() != clean_q.lower())
    target_desc = f"{clean_q} (isolated sub-assembly of {clean_ctx})" if is_subpart else clean_q

    if is_subpart:
        scope_hint = (
            f"  * subject_scope: MUST be 'whole_parent_system' if a complete '{clean_ctx}' device/organ is visible rather than an isolated/close-up '{clean_q}'; 'unrelated_or_wrong' if an unrelated person/portrait, empty room, microscopic tissue slide, or wrong object; 'isolated_target' ONLY if '{clean_q}' itself fills the frame.\n"
        )
        rubric_lines = (
            "- scores (integers 0..10 — keep consistent with your reason, media_type, and subject_scope):\n"
            f"  * completeness: 9-10 if the isolated '{clean_q}' component itself fills the frame cleanly; 5-7 if partially shown; 0-4 if '{clean_q}' is merely a tiny attached feature inside a full '{clean_ctx}' overview.\n"
            f"  * single_subject: 8-10 ONLY if '{clean_q}' is standalone/isolated or framed as a dedicated close-up; 0-4 if a full whole '{clean_ctx}' device/organism occupies the frame.\n"
            "  * clean_background: 8-10 if plain studio/paper/solid backdrop; 5-7 if clean workshop/lab/museum display; 1-4 if cluttered environment or poster.\n"
            "  * is_photograph: 8-10 if media_type='real_camera_photograph' (including studio/lab/museum photos of real objects, specimens, or physical 3D anatomical/engineering models); 0-2 if 'illustration_or_drawing', 'engraving_or_lithograph', 'cgi_or_3d_render', or 'schematic_or_diagram'.\n"
            f"  * matches_query: 8-10 ONLY if subject_scope='isolated_target' focusing cleanly on '{clean_q}'; 0-4 if subject_scope='whole_parent_system' (shows whole '{clean_ctx}') or 'unrelated_or_wrong'.\n"
            f"    - SUB-ASSEMBLY FOCUS RULE: Reject whole-machine overviews! Score matches_query 0-4 and set subject_scope='whole_parent_system' whenever an entire '{clean_ctx}' is shown instead of isolated '{clean_q}'.\n"
        )
    else:
        scope_hint = (
            f"  * subject_scope: 'target_subject' if a physical '{clean_q}' (real object, preserved specimen, or physical 3D anatomical/engineering model) is the clear visual subject, or 'unrelated_or_wrong' if it depicts an unrelated person/portrait, room, microscopic slide, or wrong object.\n"
        )
        rubric_lines = (
            "- scores (integers 0..10 — keep consistent with your reason, media_type, and subject_scope):\n"
            "  * completeness: 9-10 if entire whole target object end-to-end is visible; 5-7 if partially cropped; 0-3 if close-up fragment or wrong subject.\n"
            "  * single_subject: 8-10 if standalone isolated subject (or clean stand/mount); 4-6 if shown within a calm lab/museum display; 0-3 if busy cluttered scene or crowd.\n"
            "  * clean_background: 8-10 if plain studio/paper/solid backdrop; 5-7 if clean workshop/lab/museum display; 1-4 if cluttered environment or poster.\n"
            "  * is_photograph: 8-10 if media_type='real_camera_photograph' (including studio/lab/museum photos of real objects, specimens, or physical 3D anatomical/engineering models); 0-2 if 'illustration_or_drawing', 'engraving_or_lithograph', 'cgi_or_3d_render', or 'schematic_or_diagram'.\n"
            f"  * matches_query: 8-10 if subject_scope='target_subject' clearly showing '{clean_q}' (or a physical 3D museum/anatomical model of '{clean_q}'); 5-6 if closely related physical cutaway/mount showing it clearly; 0-2 if subject_scope='unrelated_or_wrong'.\n"
        )

    return (
        f"Strictly vet Images 1..{count} as potential real-world 3D-modeling reference photographs for '{target_desc}'.\n"
        f"CRITICAL ANTI-SYCOPHANCY RULE: Some or ALL candidates may be illustrations, medical/botanical plates, engravings, diagrams, CGI renders, rooms, wrong objects, or full whole-system overviews when a specific sub-assembly is requested.\n"
        f"Describe ONLY what is literally in the pixels, classify its exact medium and scope honestly, and keep all numeric scores 100% consistent with that description (NEVER give high scores to an illustration or wrong subject).\n"
        f"STEP 1: Evaluate ALL {count} images independently in 'evaluations' (image=1 .. image={count}):\n"
        "- reason: Describe concrete visual pixels ONLY — exact object literally visible, isolated close-up vs whole device, real camera photo vs artwork/plate/engraving/CGI/diagram, background setting.\n"
        "- media_type: Classify strictly into one of ['real_camera_photograph', 'illustration_or_drawing', 'engraving_or_lithograph', 'cgi_or_3d_render', 'schematic_or_diagram']. Studio, museum, bench, or lab camera photographs of real physical objects, preserved specimens, or physical 3D anatomical/engineering models MUST be classified as 'real_camera_photograph'.\n"
        f"{scope_hint}"
        f"{rubric_lines}"
        "- total: exact sum of the 5 integer scores.\n"
        f"STEP 2: Set 'best_image' (1..{count}) to the highest-scoring real photograph of '{target_desc}' (or the highest total if all failed), and explain in 'best_reason'. Reply with raw JSON only."
    )


def stage4_vlm_judge(
    survivors: List[dict],
    query: str,
    vlm_judge_fn: Optional[Callable[[List[Image.Image], str], str]] = None,
    max_candidates: int = 4,
    part_context: Optional[str] = None,
) -> List[dict]:
    """
    Executes ONE multi-image Qwen2.5-VL judge call over Stage-3 survivors (including N=1 validation!),
    parses structured rubric scores via bracket-depth JSON extraction, flags VLM-rejected media/mismatches,
    and performs principled hierarchical lexicographic sorting without magic penalties.
    """
    if not survivors:
        return []
    candidates = [dict(c) for c in survivors[:max_candidates]]
    if vlm_judge_fn is None:
        return candidates

    pil_list = []
    for c in candidates:
        im = c.get("_pil_image")
        if im is None:
            im = Image.new("RGB", (384, 384), (128, 128, 128))
        else:
            im = im.copy()
            im.thumbnail((384, 384), Image.Resampling.LANCZOS)
        pil_list.append(im)

    prompt = build_stage4_judge_prompt(query, len(pil_list), part_context=part_context)
    try:
        raw_reply = vlm_judge_fn(pil_list, prompt)
        parsed = extract_json_with_bracket_depth(raw_reply)
        evals = parsed.get("evaluations", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
        best_image_idx = parsed.get("best_image") if isinstance(parsed, dict) else None

        eval_map = {}
        clean_q_s = " ".join(str(query).split()).lower()
        clean_ctx_s = " ".join(str(part_context).split()).lower() if part_context else ""
        is_subpart_eval = bool(clean_ctx_s and clean_ctx_s != clean_q_s)
        non_photo_media = {
            "illustration_or_drawing",
            "engraving_or_lithograph",
            "cgi_or_3d_render",
            "schematic_or_diagram",
        }
        for ev in evals:
            if not isinstance(ev, dict):
                continue
            img_idx = ev.get("image")
            if isinstance(img_idx, int) and 1 <= img_idx <= len(candidates):
                raw_scores = ev.get("scores") if isinstance(ev.get("scores"), dict) else {}
                scores = {}
                for k in ("completeness", "single_subject", "clean_background", "is_photograph", "matches_query"):
                    if k in raw_scores:
                        scores[k] = max(0, min(10, int(round(float(raw_scores.get(k, 0.0))))))
                media_type = str(ev.get("media_type") or "").strip().lower()
                subject_scope = str(ev.get("subject_scope") or "").strip().lower()
                if scores:
                    if media_type in non_photo_media and "is_photograph" in scores:
                        scores["is_photograph"] = min(scores["is_photograph"], 2)
                    if subject_scope == "unrelated_or_wrong" and "matches_query" in scores:
                        scores["matches_query"] = min(scores["matches_query"], 2)
                    elif is_subpart_eval and subject_scope == "whole_parent_system" and "matches_query" in scores:
                        scores["matches_query"] = min(scores["matches_query"], 4)
                total = float(sum(scores.values())) if scores else float(ev.get("total", 0.0))
                eval_map[img_idx - 1] = {
                    "scores": scores,
                    "total": round(total, 2),
                    "reason": str(ev.get("reason", ""))[:160],
                    "mediaType": media_type or None,
                    "subjectScope": subject_scope or None,
                }

        for idx, c in enumerate(candidates):
            ev = eval_map.get(idx)
            is_best_pick = isinstance(best_image_idx, int) and (best_image_idx - 1) == idx
            if ev:
                scores = ev["scores"]
                c["judgeScores"] = scores
                c["judgeTotal"] = ev["total"]
                c["judgeReason"] = ev["reason"]
                if ev.get("mediaType"):
                    c["judgeMediaType"] = ev["mediaType"]
                if ev.get("subjectScope"):
                    c["judgeSubjectScope"] = ev["subjectScope"]
                c["isVlmEvaluated"] = True

                photo_sc = float(scores.get("is_photograph", 0))
                match_sc = float(scores.get("matches_query", 0))
                is_rejected = bool(scores and (photo_sc < 5.0 or match_sc < 5.0))
                c["isVlmRejected"] = is_rejected
                c["isVlmPick"] = bool(is_best_pick and not is_rejected)
            else:
                c["judgeScores"] = {}
                c["judgeTotal"] = 0.0
                c["judgeReason"] = "Not evaluated in VLM response"
                c["isVlmEvaluated"] = False
                c["isVlmRejected"] = False
                c["isVlmPick"] = False

        # Hierarchical lexicographic ordering (zero magic linear weights/penalties):
        #  1. Validated by VLM (not rejected)
        #  2. Passed CLIP authentic-photo Bayes gate
        #  3. Explicitly evaluated by VLM
        #  4. Higher VLM rubric sum (out of 50) & stronger subject+photo fit
        #  5. VLM explicit best_image tie-breaker
        #  6. Stage-3 CLIP score tie-breaker
        def _rank_key(cand_item: dict):
            clip_s = cand_item.get("clipScore") or {}
            js = cand_item.get("judgeScores") or {}
            return (
                not bool(cand_item.get("isVlmRejected", False)),
                bool(clip_s.get("isPhotoPass", False)),
                bool(cand_item.get("isVlmEvaluated", False)),
                float(cand_item.get("judgeTotal", 0.0)),
                float(js.get("matches_query", 0.0)) + float(js.get("is_photograph", 0.0)),
                bool(cand_item.get("isVlmPick", False)),
                float(cand_item.get("stage3Score", 0.0)),
            )

        candidates.sort(key=_rank_key, reverse=True)
        for i, cand_item in enumerate(candidates):
            cand_item["stage4RankKey"] = round(float(cand_item.get("judgeTotal", 0.0)), 3) if not cand_item.get("isVlmRejected") else round(-1.0 * (i + 1), 3)

        if candidates and not candidates[0].get("isVlmRejected") and candidates[0].get("isVlmEvaluated"):
            for idx_c, cand_c in enumerate(candidates):
                cand_c["isVlmPick"] = (idx_c == 0)
        elif candidates and not any(c.get("isVlmPick") for c in candidates):
            if not candidates[0].get("isVlmRejected"):
                candidates[0]["isVlmPick"] = True
    except Exception as e:
        print(f"[!] Stage 4 VLM Judge parse/inference fallback ({query}): {e}")
        for c in candidates:
            c["isVlmEvaluated"] = False
            c["isVlmRejected"] = False
        candidates.sort(key=lambda c: c.get("stage3Score", 0.0), reverse=True)

    return candidates


def compute_candidate_telemetry(candidates: List[dict]) -> dict:
    """
    Summarizes Stage-3 / Stage-4 / user-preference signals across candidates
    strictly from quantitative vetting metrics (zero keyword heuristics).
    """
    cands = [c for c in (candidates or []) if isinstance(c, dict)]
    if not cands:
        return {
            "candidateCount": 0,
            "verifiedCount": 0,
            "fallbackCount": 0,
            "hasUserPreferred": False,
            "topScore": None,
            "secondScore": None,
            "scoreMargin": None,
            "topMatchScore": None,
            "topPhotoScore": None,
            "topIsRejected": False,
            "topClipPhotoProb": None,
            "topClipSubjMargin": None,
        }

    top = cands[0]
    fallback_count = sum(1 for c in cands if c.get("isGateFallback"))
    verified_count = sum(1 for c in cands if not c.get("isGateFallback") and not c.get("isVlmRejected"))

    top_js = top.get("judgeScores") if isinstance(top.get("judgeScores"), dict) else {}
    top_clip = top.get("clipScore") if isinstance(top.get("clipScore"), dict) else {}

    top_eval_ok = (
        isinstance(top.get("judgeTotal"), (int, float))
        and top.get("isVlmEvaluated", bool(top.get("judgeScores")))
        and not top.get("isVlmRejected")
    )
    top_score = float(top["judgeTotal"]) if top_eval_ok else None
    other_vlm_totals = [
        float(c["judgeTotal"])
        for c in cands[1:]
        if isinstance(c.get("judgeTotal"), (int, float))
        and c.get("isVlmEvaluated", bool(c.get("judgeScores")))
        and not c.get("isVlmRejected")
    ]
    second_score = max(other_vlm_totals) if other_vlm_totals else None
    score_margin = (
        round(top_score - second_score, 3)
        if top_score is not None and second_score is not None
        else None
    )

    top_match = float(top_js["matches_query"]) if "matches_query" in top_js else None
    top_photo = float(top_js["is_photograph"]) if "is_photograph" in top_js else None
    clip_photo_prob = float(top_clip["photoProb"]) if "photoProb" in top_clip else None
    clip_subj_margin = float(top_clip["subjMargin"]) if "subjMargin" in top_clip else None

    return {
        "candidateCount": len(cands),
        "verifiedCount": verified_count,
        "fallbackCount": fallback_count,
        "hasUserPreferred": bool(top.get("userPreferred")),
        "topScore": round(top_score, 3) if top_score is not None else None,
        "secondScore": round(second_score, 3) if second_score is not None else None,
        "scoreMargin": score_margin,
        "topMatchScore": top_match,
        "topPhotoScore": top_photo,
        "topIsRejected": bool(top.get("isVlmRejected", False)),
        "topClipPhotoProb": round(clip_photo_prob, 4) if clip_photo_prob is not None else None,
        "topClipSubjMargin": round(clip_subj_margin, 3) if clip_subj_margin is not None else None,
    }


def determine_operating_mode(
    candidates: List[dict],
    has_user_capture: bool = False,
    part_index: int = 0,
    gate_timeout_ms: int = GATE_TIMEOUT_MS,
) -> dict:
    """
    Selects the situation-aware operating mode purely from capture state and
    quantitative Stage-3/Stage-4 telemetry (never topic strings):
      - direct_capture: User uploaded or webcam-captured image skips web gate.
      - autonomous_subparts: Internal sub-assemblies 1..N-1 run autonomously.
      - rescue_webcam_prompt: Zero plausible candidates survived Stages 1-4.
      - learned_auto: Cached user vote confirmed valid -> bypasses wait gate.
      - soft_veto_overlap: Small Stage-4 margin, Stage-3 fallback pool, or moderate
        match/photo confidence -> 2.0s Human Gate with hover pause & webcam reject.
      - human_gate_spotlight: Clear Stage-4 web picks -> 2.0s Human Gate spotlight.
    """
    telemetry = compute_candidate_telemetry(candidates)
    timeout_val = max(0, int(gate_timeout_ms))

    if has_user_capture and part_index == 0:
        return {
            "mode": "direct_capture",
            "gateRequired": False,
            "gateTimeoutMs": 0,
            "allowWebcamRescue": False,
            "telemetry": telemetry,
            "reason": "Direct camera or uploaded image bypasses web candidate ambiguity.",
        }

    if part_index > 0:
        return {
            "mode": "autonomous_subparts",
            "gateRequired": False,
            "gateTimeoutMs": 0,
            "allowWebcamRescue": False,
            "telemetry": telemetry,
            "reason": "Internal sub-assembly synthesis proceeds autonomously.",
        }

    cands = [c for c in (candidates or []) if isinstance(c, dict)]
    if not cands:
        return {
            "mode": "rescue_webcam_prompt",
            "gateRequired": False,
            "gateTimeoutMs": 0,
            "allowWebcamRescue": True,
            "telemetry": telemetry,
            "reason": "Zero photographic candidates survived Stages 1-4.",
        }

    top = cands[0]
    top_js = top.get("judgeScores") if isinstance(top.get("judgeScores"), dict) else {}
    top_clip = top.get("clipScore") if isinstance(top.get("clipScore"), dict) else {}

    if top.get("userPreferred"):
        pref_rejected = bool(top.get("isVlmRejected", False))
        if top_js:
            m_sc = float(top_js.get("matches_query", 0.0))
            p_sc = float(top_js.get("is_photograph", 0.0))
            if m_sc < LEARNED_AUTO_MIN_MATCH or p_sc < LEARNED_AUTO_MIN_PHOTO:
                pref_rejected = True
        elif top_clip and (not top_clip.get("isPhotoPass", False) or float(top_clip.get("subjMargin", 0.0)) < 0.0):
            pref_rejected = True

        if not pref_rejected:
            return {
                "mode": "learned_auto",
                "gateRequired": False,
                "gateTimeoutMs": 0,
                "allowWebcamRescue": True,
                "telemetry": telemetry,
                "reason": "Previously learned user preference satisfies vetting quality checks.",
            }
        return {
            "mode": "soft_veto_overlap",
            "gateRequired": timeout_val > 0,
            "gateTimeoutMs": timeout_val,
            "allowWebcamRescue": True,
            "telemetry": telemetry,
            "reason": "Saved preference has borderline VLM/CLIP confidence — opening 2s review gate.",
        }

    uncertain = False
    reasons = []
    if top.get("isGateFallback") or top.get("isVlmRejected"):
        uncertain = True
        reasons.append("Stage-3 CLIP photo fallback active")
    if telemetry["verifiedCount"] == 0:
        uncertain = True
        reasons.append("no high-confidence Stage-4 survivor")
    if (
        len(cands) >= 2
        and telemetry["scoreMargin"] is not None
        and telemetry["scoreMargin"] <= SOFT_VETO_MARGIN_THRESHOLD
    ):
        uncertain = True
        reasons.append(f"narrow top-2 VLM score margin ({telemetry['scoreMargin']:.1f})")
    if top_js:
        if float(top_js.get("matches_query", 10.0)) < 7.0 or float(top_js.get("is_photograph", 10.0)) < 7.0:
            uncertain = True
            reasons.append("moderate VLM subject/photo confidence")

    if uncertain:
        return {
            "mode": "soft_veto_overlap",
            "gateRequired": timeout_val > 0,
            "gateTimeoutMs": timeout_val,
            "allowWebcamRescue": True,
            "telemetry": telemetry,
            "reason": "; ".join(reasons) or "Candidate scores show overlap or fallback status.",
        }

    return {
        "mode": "human_gate_spotlight",
        "gateRequired": timeout_val > 0,
        "gateTimeoutMs": timeout_val,
        "allowWebcamRescue": True,
        "telemetry": telemetry,
        "reason": "Stage 1-4 candidates vetted — 2s human spotlight ready before 3D build.",
    }


# =========================================================================
# STAGE 5: MESH DIAGNOSTICS + VECTORIZED Z-BUFFER TURNTABLE + VLM VERIFY
# =========================================================================
def compute_mesh_diagnostics(mesh: trimesh.Trimesh) -> dict:
    """
    Rotation-invariant spectral & topological mesh diagnostics using PCA covariance
    and connected-component analysis (catches 45-degree tilted flat cards & floating debris).
    """
    n_verts = int(len(mesh.vertices)) if mesh is not None else 0
    n_faces = int(len(mesh.faces)) if mesh is not None else 0
    if mesh is None or n_verts < 10 or n_faces < 4:
        return {
            "vertexCount": n_verts,
            "faceCount": n_faces,
            "componentCount": 0,
            "largestComponentFrac": 0.0,
            "pcaFlatness": 0.0,
            "pcaCrossFlatness": 0.0,
            "pcaPlanarity": 0.0,
            "pcaElongation": 1.0,
            "solidity": 0.0,
            "isWatertight": False,
        }

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centered = verts - verts.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)  # ascending: lambda_0 <= lambda_1 <= lambda_2
    eigvals = np.maximum(1e-12, np.sort(eigvals))
    s0 = math.sqrt(eigvals[0])
    s1 = math.sqrt(eigvals[1])
    s2 = math.sqrt(eigvals[2])
    pca_flatness = float(s0 / s2)
    pca_cross_flatness = float(s0 / s1)
    pca_planarity = float((s1 - s0) / max(1e-9, s2))
    pca_elongation = float(s2 / s1)

    try:
        comps = mesh.split(only_watertight=False)
        comp_count = len(comps) if comps else 1
        max_comp_v = max((len(c.vertices) for c in comps), default=n_verts)
        largest_comp_frac = float(max_comp_v) / float(max(1, n_verts))
    except Exception:
        comp_count = 1
        largest_comp_frac = 1.0

    solidity = 0.0
    try:
        hull_vol = abs(float(mesh.convex_hull.volume))
        mesh_vol = abs(float(mesh.volume))
        if hull_vol > 1e-9:
            solidity = float(np.clip(mesh_vol / hull_vol, 0.0, 1.0))
    except Exception:
        solidity = 0.0

    return {
        "vertexCount": n_verts,
        "faceCount": n_faces,
        "componentCount": int(comp_count),
        "largestComponentFrac": round(largest_comp_frac, 4),
        "pcaFlatness": round(pca_flatness, 4),
        "pcaCrossFlatness": round(pca_cross_flatness, 4),
        "pcaPlanarity": round(pca_planarity, 4),
        "pcaElongation": round(pca_elongation, 4),
        "solidity": round(solidity, 4),
        "isWatertight": bool(mesh.is_watertight),
    }


def _compute_smooth_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vn = np.zeros_like(verts, dtype=np.float32)
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8
    return vn / norms


def _render_single_turntable_view(
    verts: np.ndarray,
    faces: np.ndarray,
    v_normals: Optional[np.ndarray],
    yaw_deg: float,
    pitch_deg: float,
    panel_size: int = 240,
    label: str = "",
) -> Image.Image:
    """
    Zero-dependency chunked vectorized Z-buffered Gouraud CPU triangle rasterizer.
    Exact depth visibility without X11/OpenGL/pyrender overhead (< 45 ms/view for 20k+ faces).
    """
    bg_col = (14, 18, 26)
    if len(verts) == 0 or len(faces) == 0:
        return Image.new("RGB", (panel_size, panel_size), bg_col)

    ry = math.radians(yaw_deg)
    rx = math.radians(pitch_deg)
    Ry = np.array(
        [[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]],
        dtype=np.float32,
    )
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(rx), -math.sin(rx)], [0.0, math.sin(rx), math.cos(rx)]],
        dtype=np.float32,
    )
    rot = Rx @ Ry
    V = verts @ rot.T
    if v_normals is None or len(v_normals) != len(verts):
        v_normals = _compute_smooth_vertex_normals(verts, faces)
    N = v_normals @ rot.T

    tris = V[faces]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    fn = np.cross(v1 - v0, v2 - v0)
    radial_dot = float(np.mean(np.sum(tris.mean(axis=1) * fn, axis=1)))
    if radial_dot > 0.05:
        keep = fn[:, 2] > -0.02
        if np.any(keep):
            tris = tris[keep]
            faces_k = faces[keep]
        else:
            faces_k = faces
    else:
        faces_k = faces

    # Fit projected extents cleanly inside 82% of panel_size (never clip tall/wide projections)
    xy_ext = float(np.max(np.abs(V[:, :2]))) + 1e-6
    scale = (0.41 * panel_size) / xy_ext
    px = tris[:, :, 0] * scale + 0.5 * panel_size
    py = -tris[:, :, 1] * scale + 0.5 * panel_size
    pz = tris[:, :, 2]

    key_light = np.array([0.48, 0.70, 0.53], dtype=np.float32)
    rim_light = np.array([-0.58, 0.22, 0.78], dtype=np.float32)
    N_o = np.where(N[:, 2:3] >= 0.0, N, -N)
    diff = np.clip(N_o @ key_light, 0.0, 1.0)
    rim = np.clip(N_o @ rim_light, 0.0, 1.0)
    fresnel = (1.0 - np.abs(N_o[:, 2])) ** 2.0 * 0.26
    v_shade = np.clip(0.15 + 0.70 * diff + 0.15 * rim + fresnel, 0.0, 1.0)
    v_col = np.stack([v_shade * 204.0 + 24.0, v_shade * 214.0 + 24.0, v_shade * 224.0 + 24.0], axis=1)

    x0, x1, x2 = px[:, 0], px[:, 1], px[:, 2]
    y0, y1, y2 = py[:, 0], py[:, 1], py[:, 2]
    det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    xmin = np.maximum(0, np.ceil(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(np.int32))
    xmax = np.minimum(panel_size - 1, np.floor(np.maximum(np.maximum(x0, x1), x2) + 0.5).astype(np.int32))
    ymin = np.maximum(0, np.ceil(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(np.int32))
    ymax = np.minimum(panel_size - 1, np.floor(np.maximum(np.maximum(y0, y1), y2) + 0.5).astype(np.int32))
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    valid = (np.abs(det) > 1e-3) & (bw > 0) & (bh > 0)

    W = panel_size
    zbuf_flat = np.full(W * W, -1e9, dtype=np.float32)
    cbuf_flat = np.full((W * W, 3), bg_col, dtype=np.uint8)

    idx_all = np.nonzero(valid)[0]
    area = bw[idx_all] * bh[idx_all]
    big_mask = area > 4096
    small_idx = idx_all[~big_mask]
    big_idx = idx_all[big_mask]

    # Batch small/medium triangles cleanly in bounded fragment chunks (~120k fragments/chunk)
    if len(small_idx) > 0:
        small_areas = area[~big_mask]
        chunk_start = 0
        while chunk_start < len(small_idx):
            cum = np.cumsum(small_areas[chunk_start:])
            over = np.nonzero(cum > 120000)[0]
            chunk_end = chunk_start + max(1, int(over[0])) if len(over) > 0 else len(small_idx)

            c_idx = small_idx[chunk_start:chunk_end]
            counts = small_areas[chunk_start:chunk_end]
            chunk_start = chunk_end

            tri_rep = np.repeat(c_idx, counts)
            starts = np.cumsum(counts) - counts
            local = np.arange(len(tri_rep), dtype=np.int32) - np.repeat(starts, counts)
            w_rep = bw[tri_rep]
            gx_i = xmin[tri_rep] + (local % w_rep)
            gy_i = ymin[tri_rep] + (local // w_rep)
            gx = gx_i.astype(np.float32) + 0.5
            gy = gy_i.astype(np.float32) + 0.5
            d = det[tri_rep]
            w0 = ((x1[tri_rep] - gx) * (y2[tri_rep] - gy) - (y1[tri_rep] - gy) * (x2[tri_rep] - gx)) / d
            w1 = ((x2[tri_rep] - gx) * (y0[tri_rep] - gy) - (y2[tri_rep] - gy) * (x0[tri_rep] - gx)) / d
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
            if not np.any(inside):
                continue

            tr = tri_rep[inside]
            w0_i, w1_i, w2_i = w0[inside], w1[inside], w2[inside]
            z = w0_i * pz[tr, 0] + w1_i * pz[tr, 1] + w2_i * pz[tr, 2]
            pix = gy_i[inside] * W + gx_i[inside]

            prev_best = zbuf_flat[pix]
            win = z > prev_best
            if not np.any(win):
                continue

            pix_w = pix[win]
            z_w = z[win]
            order = np.argsort(z_w, kind="quicksort")
            pix_s = pix_w[order]
            z_s = z_w[order]
            tr_s = tr[win][order]
            w0s = w0_i[win][order, None]
            w1s = w1_i[win][order, None]
            w2s = w2_i[win][order, None]
            # Keep exact single furthest-in-front (+Z) winner per unique pixel
            _, rev_pos = np.unique(pix_s[::-1], return_index=True)
            uniq_last = len(pix_s) - 1 - rev_pos
            f_u = faces_k[tr_s[uniq_last]]
            cols_u = (
                w0s[uniq_last] * v_col[f_u[:, 0]]
                + w1s[uniq_last] * v_col[f_u[:, 1]]
                + w2s[uniq_last] * v_col[f_u[:, 2]]
            )
            pix_u = pix_s[uniq_last]
            zbuf_flat[pix_u] = z_s[uniq_last]
            cbuf_flat[pix_u] = np.clip(cols_u, 0, 255).astype(np.uint8)

    for i in big_idx:
        xa, xb = int(xmin[i]), int(xmax[i]) + 1
        ya, yb = int(ymin[i]), int(ymax[i]) + 1
        gx = np.arange(xa, xb, dtype=np.float32)[None, :] + 0.5
        gy = np.arange(ya, yb, dtype=np.float32)[:, None] + 0.5
        d = det[i]
        w0 = ((x1[i] - gx) * (y2[i] - gy) - (y1[i] - gy) * (x2[i] - gx)) / d
        w1 = ((x2[i] - gx) * (y0[i] - gy) - (y2[i] - gy) * (x0[i] - gx)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
        if not np.any(inside):
            continue
        z = (w0 * pz[i, 0] + w1 * pz[i, 1] + w2 * pz[i, 2])[inside]
        yg, xg = np.nonzero(inside)
        pix = (ya + yg) * W + (xa + xg)
        win = z > zbuf_flat[pix]
        if np.any(win):
            pix_w = pix[win]
            zbuf_flat[pix_w] = z[win]
            f = faces_k[i]
            col = (
                w0[inside][win, None] * v_col[f[0]]
                + w1[inside][win, None] * v_col[f[1]]
                + w2[inside][win, None] * v_col[f[2]]
            )
            cbuf_flat[pix_w] = np.clip(col, 0, 255).astype(np.uint8)

    img = Image.fromarray(cbuf_flat.reshape((W, W, 3)), mode="RGB")
    if label:
        draw = ImageDraw.Draw(img)
        draw.rectangle([4, 4, 114, 20], fill=(8, 14, 22))
        draw.text((8, 7), label, fill=(220, 232, 245))
    return img


def render_mesh_turntable_strip(
    mesh: trimesh.Trimesh,
    views_deg: Tuple[float, ...] = (-34.0, 26.0, 128.0),
    pitch_deg: float = 17.0,
    panel_size: int = 240,
) -> Image.Image:
    """Renders 3 Z-buffered turntable views of a generated 3D Trimesh into one labelled strip."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    v_normals = None
    if len(verts) > 0:
        verts = verts - verts.mean(axis=0)
        scale = float(np.max(np.ptp(verts, axis=0))) + 1e-6
        verts = verts / scale
        v_normals = _compute_smooth_vertex_normals(verts, faces)

    panels = [
        _render_single_turntable_view(
            verts, faces, v_normals, yaw_deg=ang, pitch_deg=pitch_deg, panel_size=panel_size, label=f"VIEW {i + 1} ({int(ang):+d}°)"
        )
        for i, ang in enumerate(views_deg)
    ]

    gap = 6
    strip = Image.new("RGB", (panel_size * len(panels) + gap * (len(panels) - 1), panel_size), (6, 10, 16))
    for i, p in enumerate(panels):
        strip.paste(p, (i * (panel_size + gap), 0))
    return strip


def stage5_verify_mesh(
    mesh: trimesh.Trimesh,
    query: str,
    vlm_verify_fn: Optional[Callable[[Image.Image, str], str]] = None,
) -> dict:
    """
    Self-verification loop:
      1) Computes PCA rotation-invariant flatness, elongation, and connected-component integrity
      2) Renders 3 Z-buffered turntable views and asks VLM if it plausibly represents {query}
      3) Fails CLOSED if geometry is degenerate/fragmented or if VLM JSON verification fails
    """
    diag = compute_mesh_diagnostics(mesh)
    if mesh is None or diag["vertexCount"] < 50 or diag["faceCount"] < 50:
        return {
            "plausible": False,
            "confidence": 0.0,
            "shape_summary": "degenerate mesh",
            "reason": "Generated mesh has too few vertices/faces.",
            "diagnostics": diag,
        }

    if diag["pcaFlatness"] < MIN_PCA_FLATNESS:
        return {
            "plausible": False,
            "confidence": 1.0,
            "shape_summary": "collapsed thin pancake",
            "reason": f"PCA spectral flatness ratio ({diag['pcaFlatness']:.3f} < {MIN_PCA_FLATNESS:.2f}) indicates 2D card collapse.",
            "diagnostics": diag,
        }

    is_collapsed_sheet = (
        diag["pcaPlanarity"] > MAX_PLANAR_SHEET_PLANARITY
        and diag["pcaFlatness"] < MAX_PLANAR_SHEET_FLATNESS
    ) or (
        diag["pcaFlatness"] < 0.12
        and diag["pcaCrossFlatness"] < MIN_FILAMENT_CROSS_FLATNESS
    )
    if is_collapsed_sheet:
        return {
            "plausible": False,
            "confidence": 1.0,
            "shape_summary": "collapsed planar card",
            "reason": (
                f"PCA spectral planarity ({diag['pcaPlanarity']:.3f}, flatness={diag['pcaFlatness']:.3f}, "
                f"cross={diag['pcaCrossFlatness']:.3f}) indicates 2D planar slab collapse."
            ),
            "diagnostics": diag,
        }

    if diag["largestComponentFrac"] < MIN_MAIN_COMPONENT_FRAC:
        return {
            "plausible": False,
            "confidence": 1.0,
            "shape_summary": "fragmented cloud",
            "reason": f"Main connected component holds only {diag['largestComponentFrac'] * 100:.0f}% of vertices across {diag['componentCount']} fragments.",
            "diagnostics": diag,
        }

    if vlm_verify_fn is None:
        return {
            "plausible": True,
            "confidence": 6.0,
            "shape_summary": "unverified 3D manifold",
            "reason": "VLM verifier not supplied; PCA thickness and manifold coherence valid.",
            "diagnostics": diag,
        }

    clean_q = " ".join(str(query).split())
    strip = render_mesh_turntable_strip(mesh)
    prompt = (
        f"Image 1 displays 3 turntable angles (VIEW 1, VIEW 2, VIEW 3) of a reconstructed untextured gray 3D mesh for '{clean_q}'.\n"
        f"Quantitative mesh diagnostics: pcaFlatness={diag['pcaFlatness']:.3f}, crossSection={diag['pcaCrossFlatness']:.3f}, "
        f"planarity={diag['pcaPlanarity']:.3f}, elongation={diag['pcaElongation']:.2f}, "
        f"components={diag['componentCount']}, mainComponent={diag['largestComponentFrac']*100:.0f}%, solidity={diag['solidity']:.2f}.\n"
        f"Inspect its 3D form across ALL THREE angles and determine if it is a valid, recognizable 3D reconstruction of '{clean_q}':\n"
        "- shape_summary: State the overall 3D form, symmetry, and thickness across the three views (in your own words).\n"
        f"- reason: Explain whether both the front silhouette AND side/rear depth match what a real '{clean_q}' should look like, OR whether it collapsed into a flat sheet/thin wafer/wrong shape/melted blob.\n"
        f"- plausible: true ONLY if the mesh has genuine multi-view 3D depth and proportions faithful to a whole '{clean_q}' (ignoring minor single-view grain). false if it is a flat board, thin relief missing the main 3D body, fragmented debris, or wrong object.\n"
        "- confidence: 0..10.\n"
        "Return raw JSON only."
    )

    try:
        raw = vlm_verify_fn(strip, prompt)
        parsed = extract_json_with_bracket_depth(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Non-dict JSON from Stage 5 verifier")
        return {
            "plausible": bool(parsed.get("plausible", False)),
            "confidence": float(parsed.get("confidence", 0.0)),
            "shape_summary": str(parsed.get("shape_summary", ""))[:120],
            "reason": str(parsed.get("reason", ""))[:180],
            "diagnostics": diag,
        }
    except Exception as e:
        print(f"[!] Stage 5 VLM verification parse failure ({query}): {e}")
        return {
            "plausible": False,
            "confidence": 0.0,
            "shape_summary": "verification parse error",
            "reason": f"Verifier parse failure (failing closed): {e}",
            "diagnostics": diag,
        }


# =========================================================================
# END-TO-END FUNNEL ORCHESTRATOR
# =========================================================================
def run_candidate_funnel(
    query: str,
    vlm_judge_fn: Optional[Callable[[List[Image.Image], str], str]] = None,
    max_raw: int = 200,
    max_stage1: int = 60,
    max_downloads: int = 32,
    stage2_top_k: int = 12,
    stage3_top_k: int = 4,
    part_id: Optional[str] = None,
    part_context: Optional[str] = None,
    prefer_search: bool = False,
    on_stage: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """
    Executes Stages 0 -> 4 and returns ranked candidates plus funnel telemetry.
    Applies any saved user preference vote for (query, part_id) at rank 0 if found.
    """
    t0 = time.time()
    notify = on_stage or (lambda st, d: None)

    notify("acquire", {"query": query, "message": "Acquiring candidates from Wikipedia Lead, Commons Categories & Search..."})
    raw_cands, wiki_summary = acquire_candidates(query, max_candidates=max_raw, prefer_search=prefer_search)

    s1 = stage1_metadata_filter(raw_cands, max_survivors=max_stage1, prefer_search=prefer_search)
    notify("stage1", {"query": query, "raw": len(raw_cands), "survived": len(s1)})

    s2 = stage2_cv_filter(s1, max_downloads=max_downloads, top_k=stage2_top_k)
    notify("stage2", {"query": query, "tested": min(max_downloads, len(s1)), "survived": len(s2)})

    s3 = stage3_clip_filter(s2, query=query, top_k=stage3_top_k, part_context=part_context)
    notify("stage3", {"query": query, "survived": len(s3)})

    ranked = stage4_vlm_judge(
        s3,
        query=query,
        vlm_judge_fn=vlm_judge_fn,
        max_candidates=stage3_top_k,
        part_context=part_context,
    )
    notify("stage4", {"query": query, "ranked": len(ranked)})

    # Reorder/promote previously learned user vote if present for this query
    pref = get_candidate_preference(query, part_id=part_id)
    if pref and (pref.get("thumbUrl") or pref.get("originalUrl")):
        pref_url = pref.get("thumbUrl") or pref.get("originalUrl")
        matched_idx = next(
            (i for i, c in enumerate(ranked) if c.get("thumbUrl") == pref_url or c.get("originalUrl") == pref_url),
            None,
        )
        if matched_idx is not None:
            preferred = ranked.pop(matched_idx)
            preferred["userPreferred"] = True
            ranked.insert(0, preferred)
        else:
            try:
                data, _ = fetch_or_cache_proxied_image(pref_url)
                with Image.open(io.BytesIO(data)) as im:
                    pil_copy = im.copy()
                pref_cand = {
                    "title": pref.get("title", "User Preferred Reference"),
                    "thumbUrl": pref.get("thumbUrl") or pref_url,
                    "originalUrl": pref.get("originalUrl") or pref_url,
                    "mime": pref.get("mime", "image/jpeg"),
                    "caption": pref.get("caption") or f"User locked preference for {query}",
                    "width": pil_copy.width,
                    "height": pil_copy.height,
                    "origWidth": pil_copy.width,
                    "origHeight": pil_copy.height,
                    "sourceTier": 0,
                    "sourceLabel": "user_preference_cache",
                    "userPreferred": True,
                    "_pil_image": pil_copy.convert("RGB"),
                    "_pil_rgba": pil_copy if pil_copy.mode == "RGBA" else pil_copy.convert("RGB"),
                }
                ranked.insert(0, pref_cand)
            except Exception:
                pass

    elapsed = round(time.time() - t0, 2)
    return {
        "query": query,
        "wikiSummary": wiki_summary,
        "candidates": ranked,
        "stats": {
            "rawCount": len(raw_cands),
            "stage1Count": len(s1),
            "stage2Count": len(s2),
            "stage3Count": len(s3),
            "stage4Count": len(ranked),
            "elapsedSec": elapsed,
        },
    }
