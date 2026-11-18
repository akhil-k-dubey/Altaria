#!/usr/bin/env python3
"""
Dedicated 3D Generative Mesh Engine - Tony Stark Holographic CAD
================================================================
Reconstructs continuous high-density 3D geometry (10k-50k vertices)
directly from vetted 2D imagery via TripoSR, verified by Stage 5 turntable
self-verification with automatic candidate #2/#3 retry -> webcam -> spherical
harmonic manifold fallback.
ZERO hardcoded keywords, zero regex, zero primitive cone/cylinder approximations.
"""

import contextlib
import hashlib
import io
import json
import os
import threading
import time
import urllib.parse
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import scipy.special
import torch
import trimesh
import trimesh.repair
import trimesh.smoothing
from PIL import Image, ImageOps

from image_funnel import serialize_candidate, slugify as _funnel_slugify, stage5_verify_mesh
from image_proxy import IMAGE_CACHE_DIR, fetch_or_cache_proxied_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES_DIR = os.path.join(BASE_DIR, "data", "meshes")
TEXTURES_DIR = os.path.join(BASE_DIR, "data", "textures")
os.makedirs(MESHES_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
os.makedirs(TEXTURES_DIR, exist_ok=True)

TRIPOSR_REPO_ID = os.environ.get("HOLO_CAD_TRIPOSR_MODEL", "stabilityai/TripoSR")
TRIPOSR_CHUNK_SIZE = max(4096, int(os.environ.get("HOLO_CAD_TRIPOSR_CHUNK_SIZE", "16384")))
MAX_TRIPOSR_INPUT_DIM = max(256, int(os.environ.get("HOLO_CAD_TRIPOSR_MAX_IMG_DIM", "1024")))
REMBG_MODEL_NAME = os.environ.get("HOLO_CAD_REMBG_MODEL", "isnet-general-use")
FOREGROUND_RATIO = float(os.environ.get("HOLO_CAD_FOREGROUND_RATIO", "0.85"))
DEFAULT_MESH_RESOLUTION = max(32, int(os.environ.get("HOLO_CAD_MESH_RES", "192")))
HARMONIC_FALLBACK_RESOLUTION = max(32, int(os.environ.get("HOLO_CAD_HARMONIC_RES", "64")))
DECIMATED_TARGET_POINTS = max(64, int(os.environ.get("HOLO_CAD_DECIMATED_POINTS", "2000")))
MAX_VERIFY_TRIALS = max(1, int(os.environ.get("HOLO_CAD_CANDIDATE_K", "3")))

# Global singletons & GPU lock for VRAM-safe execution on constrained GPUs
_TRIPOSR_MODEL = None
_REMBG_SESSION = None
_MODEL_INIT_LOCK = threading.Lock()
TRIPOSR_LOCK = threading.Lock()


def slugify(text: str) -> str:
    """Delegate directly to canonical character-loop slugifier."""
    return _funnel_slugify(text)


def _resolve_active_torch_device() -> str:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return f"cuda:{idx}"
    return "cpu"


def _open_image_preserving_mode(data_or_path) -> Image.Image:
    """
    Opens image bytes or file path, applies EXIF orientation cleanly, and preserves
    native RGB vs RGBA mode (NEVER coerces opaque RGB images to fake RGBA).
    """
    with Image.open(data_or_path) as im:
        im.load()
        transposed = ImageOps.exif_transpose(im)
        if transposed.mode in ("RGB", "RGBA"):
            return transposed.copy()
        if "A" in transposed.getbands():
            return transposed.convert("RGBA")
        return transposed.convert("RGB")


def load_image_from_source(source: str) -> Image.Image:
    """
    Loads an image from a remote URL (strictly via SSRF-hardened proxy),
    a local /api/cache-image/... or /api/proxy-image?... route, or a verified local path.
    Preserves RGB/RGBA channels accurately so rembg runs whenever needed.
    """
    if not source or not isinstance(source, str):
        raise ValueError("Image source is empty")

    src = source.strip()

    if src.startswith("/api/proxy-image"):
        parsed = urllib.parse.urlsplit(src)
        q_params = urllib.parse.parse_qs(parsed.query)
        remote_url = q_params.get("url", [""])[0]
        if not remote_url:
            raise ValueError("Missing 'url' query parameter in /api/proxy-image source")
        data, _ = fetch_or_cache_proxied_image(remote_url)
        return _open_image_preserving_mode(io.BytesIO(data))

    if src.startswith("/api/cache-image/"):
        raw_name = urllib.parse.unquote(os.path.basename(urllib.parse.urlsplit(src).path))
        real_cache = os.path.realpath(IMAGE_CACHE_DIR)
        cand_path = os.path.realpath(os.path.join(real_cache, raw_name))
        if os.path.commonpath([real_cache, cand_path]) != real_cache or not os.path.isfile(cand_path):
            raise FileNotFoundError(f"Cached image not found: {src}")
        return _open_image_preserving_mode(cand_path)

    if src.startswith("http://") or src.startswith("https://"):
        data, _ = fetch_or_cache_proxied_image(src)
        return _open_image_preserving_mode(io.BytesIO(data))

    if not os.path.isfile(src):
        raise FileNotFoundError(f"Local image file does not exist: {src}")
    return _open_image_preserving_mode(src)


def _has_meaningful_cutout_alpha(pil_img: Image.Image) -> bool:
    if pil_img.mode != "RGBA":
        return False
    alpha_arr = np.asarray(pil_img, dtype=np.uint8)[:, :, 3]
    h, w = alpha_arr.shape
    border = max(1, min(h, w) // 20)
    b_alpha = np.concatenate([
        alpha_arr[:border, :].ravel(),
        alpha_arr[-border:, :].ravel(),
        alpha_arr[:, :border].ravel(),
        alpha_arr[:, -border:].ravel(),
    ])
    opaque_frac = float(np.mean(alpha_arr > 128))
    return float(np.mean(b_alpha)) < 240.0 and 0.008 < opaque_frac < 0.98


def get_rembg_session():
    """Lazily initialize foreground segmentation model on CPU (isnet-general-use -> u2netp fallback)."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        with _MODEL_INIT_LOCK:
            if _REMBG_SESSION is None:
                try:
                    import rembg
                    try:
                        _REMBG_SESSION = rembg.new_session(REMBG_MODEL_NAME)
                        print(f"[*] Initialized {REMBG_MODEL_NAME} foreground segmentation session")
                    except Exception as first_err:
                        if REMBG_MODEL_NAME != "u2netp":
                            print(f"[!] Foreground model '{REMBG_MODEL_NAME}' fallback -> u2netp ({first_err})")
                            _REMBG_SESSION = rembg.new_session("u2netp")
                        else:
                            raise
                except Exception as e:
                    print(f"[!] Warning: Could not initialize rembg session: {e}")
                    _REMBG_SESSION = False
    return _REMBG_SESSION if _REMBG_SESSION is not False else None


def get_triposr_model():
    """Lazily load TripoSR neural 3D reconstruction model onto GPU with memory safety."""
    global _TRIPOSR_MODEL
    if _TRIPOSR_MODEL is None:
        with _MODEL_INIT_LOCK:
            if _TRIPOSR_MODEL is None:
                try:
                    import sys
                    backend_path = os.path.dirname(os.path.abspath(__file__))
                    if backend_path not in sys.path:
                        sys.path.insert(0, backend_path)

                    from tsr.system import TSR

                    target_dev = _resolve_active_torch_device()
                    print(f"[*] Initializing TripoSR neural 3D micro-topography engine on {target_dev}...")
                    model = TSR.from_pretrained(
                        TRIPOSR_REPO_ID,
                        config_name="config.yaml",
                        weight_name="model.ckpt",
                    )
                    model.to(target_dev)
                    model.renderer.chunk_size = TRIPOSR_CHUNK_SIZE
                    _TRIPOSR_MODEL = model
                    print("[✓] TripoSR 3D neural reconstruction engine ready")
                except Exception as e:
                    print(f"[!] Warning: Could not initialize TripoSR: {e}")
                    _TRIPOSR_MODEL = False

    return _TRIPOSR_MODEL if _TRIPOSR_MODEL is not False else None


def _extract_rgba_vertex_colors(mesh: trimesh.Trimesh, expected_count: int) -> Optional[np.ndarray]:
    try:
        vis = getattr(mesh, "visual", None)
        raw_vc = getattr(vis, "vertex_colors", None) if vis is not None else None
        if raw_vc is None:
            return None
        vc = np.asarray(raw_vc)
        if vc.ndim != 2 or len(vc) != expected_count or vc.shape[1] < 3:
            return None
        if np.issubdtype(vc.dtype, np.floating):
            vc = np.clip(vc * 255.0, 0, 255).astype(np.uint8)
        else:
            vc = np.clip(vc, 0, 255).astype(np.uint8)
        if vc.shape[1] == 3:
            vc = np.concatenate([vc, np.full((len(vc), 1), 255, dtype=np.uint8)], axis=1)
        elif vc.shape[1] > 4:
            vc = vc[:, :4]
        return vc
    except Exception:
        return None


def _clean_reconstructed_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Applies exact SO(3) camera-to-Three.js frame alignment, preserves vertex colors,
    prunes tiny floating debris (<2% fragments when main >=45%), runs Humphrey
    volume-preserving surface smoothing (removing staircase marching-cubes noise
    while preserving ridges), fixes normals, and normalizes unit bounding scale.
    """
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh

    # TripoSR view-0 camera c2w has right=+Y_tsr, up=+Z_tsr, -lookat=+X_tsr.
    # Permutation (Y_tsr, Z_tsr, X_tsr) has det=+1 (pure SO(3) rotation, zero mirror).
    verts = np.asarray(mesh.vertices, dtype=np.float64)[:, [1, 2, 0]].copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v_colors = _extract_rgba_vertex_colors(mesh, len(verts))
    cleaned = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_colors, process=False)

    # Remove degenerate/duplicate faces cleanly
    try:
        cleaned.update_faces(cleaned.unique_faces())
        cleaned.update_faces(cleaned.nondegenerate_faces())
        cleaned.remove_unreferenced_vertices()
    except Exception:
        pass

    # Drop isolated floating speckles that distort centroid & bounding extents
    try:
        comps = cleaned.split(only_watertight=False)
        if comps and len(comps) > 1:
            total_v = max(1, len(cleaned.vertices))
            comps.sort(key=lambda c: len(c.vertices), reverse=True)
            if len(comps[0].vertices) >= 0.45 * total_v:
                kept = [c for c in comps if len(c.vertices) >= 0.02 * total_v]
                if kept:
                    cleaned = kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)
    except Exception:
        pass

    # Prune unreferenced vertices BEFORE Humphrey filter so degree-0 vertices never produce NaN/warnings
    try:
        cleaned.remove_unreferenced_vertices()
    except Exception:
        pass

    if len(cleaned.vertices) >= 12 and len(cleaned.faces) >= 8:
        prev_verts = np.array(cleaned.vertices, dtype=np.float64, copy=True)
        prev_colors = _extract_rgba_vertex_colors(cleaned, len(prev_verts))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trimesh.smoothing.filter_humphrey(cleaned, alpha=0.1, beta=0.5, iterations=2)
            if not np.all(np.isfinite(cleaned.vertices)):
                cleaned.vertices = prev_verts
            if prev_colors is not None and _extract_rgba_vertex_colors(cleaned, len(cleaned.vertices)) is None:
                cleaned.visual.vertex_colors = prev_colors
        except Exception:
            cleaned.vertices = prev_verts

    try:
        trimesh.repair.fix_normals(cleaned)
    except Exception:
        pass

    if len(cleaned.vertices) > 0:
        cleaned.vertices -= cleaned.bounding_box.centroid
        max_extent = float(np.max(cleaned.bounding_box.extents))
        if max_extent > 1e-6:
            cleaned.vertices /= max_extent

    return cleaned


def _resolve_vram_safe_resolutions(requested_res: int) -> List[int]:
    req = max(32, int(requested_res))
    ladder = [req]
    for rung in (192, 160, 128, 96):
        if rung < req and rung not in ladder:
            ladder.append(rung)
    return ladder


def reconstruct_triposr_mesh(
    pil_img: Image.Image, resolution: int = DEFAULT_MESH_RESOLUTION
) -> trimesh.Trimesh:
    """
    Reconstructs continuous, high-density 3D topography with neural albedo vertex colors
    from a 2D reference image using TripoSR FP16 autocast under strict VRAM serialization
    with automatic OOM step-down recovery.
    """
    model = get_triposr_model()
    if model is None:
        raise RuntimeError("TripoSR model not available")

    from tsr.utils import remove_background, resize_foreground

    work_img = pil_img.copy()
    if max(work_img.size) > MAX_TRIPOSR_INPUT_DIM:
        work_img.thumbnail((MAX_TRIPOSR_INPUT_DIM, MAX_TRIPOSR_INPUT_DIM), Image.Resampling.LANCZOS)

    session = get_rembg_session()
    if not _has_meaningful_cutout_alpha(work_img):
        rgb_img = work_img.convert("RGB")
        if session is not None:
            try:
                work_img = remove_background(rgb_img, rembg_session=session, force=True)
            except Exception as e:
                print(f"[!] Foreground segmentation warning: {e}")
                work_img = rgb_img
        else:
            work_img = rgb_img

    if work_img.mode == "RGBA":
        alpha_arr = np.asarray(work_img, dtype=np.uint8)[:, :, 3]
        fg_frac = float(np.mean(alpha_arr > 32))
        if 0.008 <= fg_frac <= 0.995:
            work_img = resize_foreground(work_img, FOREGROUND_RATIO)
            arr = np.asarray(work_img, dtype=np.float32) / 255.0
            alpha = arr[:, :, 3:4]
            rgb = arr[:, :, :3] * alpha + (1.0 - alpha) * 0.5
            input_image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        else:
            # Guard against degenerate all-empty / all-solid alpha masks
            input_image = work_img.convert("RGB")
    else:
        input_image = work_img.convert("RGB")

    target_dev = _resolve_active_torch_device()
    use_cuda = target_dev.startswith("cuda")
    amp_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=torch.float16))
        if use_cuda
        else contextlib.nullcontext
    )
    res_ladder = _resolve_vram_safe_resolutions(resolution)

    with TRIPOSR_LOCK:
        try:
            with torch.no_grad(), amp_ctx():
                scene_codes = model([input_image], device=target_dev)

            raw_mesh = None
            last_err = None
            for idx_r, try_res in enumerate(res_ladder):
                try:
                    with torch.no_grad(), amp_ctx():
                        meshes = model.extract_mesh(scene_codes, True, resolution=try_res)
                    raw_mesh = meshes[0]
                    break
                except Exception as ex:
                    is_oom = (
                        (isinstance(ex, torch.cuda.OutOfMemoryError) if use_cuda else False)
                        or "out of memory" in str(ex).lower()
                    )
                    last_err = ex
                    if is_oom and idx_r + 1 < len(res_ladder):
                        print(
                            f"[!] TripoSR CUDA OOM at resolution={try_res}, stepping down to {res_ladder[idx_r + 1]}..."
                        )
                        if use_cuda:
                            torch.cuda.empty_cache()
                        continue
                    raise

            if raw_mesh is None:
                raise RuntimeError(f"TripoSR extract_mesh produced no mesh: {last_err}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _clean_reconstructed_mesh(raw_mesh)


def generate_high_density_harmonic_mesh(
    name_seed: str, resolution: int = HARMONIC_FALLBACK_RESOLUTION
) -> trimesh.Trimesh:
    """
    Precision parametric CAD fallback proxy.
    Synthesizes precision, watertight, movie-grade 3D CAD manifolds across all domains
    (neural computing layers, neuro-anatomical brain structures, cardiac chambers, propulsion).
    Purges primitive harmonic lumps permanently.
    """
    try:
        from cad_geometry import generate_parametric_cad_mesh
        return generate_parametric_cad_mesh(part_name=name_seed)
    except Exception:
        from cad_geometry import generate_cad_precision_machined_solid
        return generate_cad_precision_machined_solid()


def generate_voxel_decimated_points(
    mesh: trimesh.Trimesh, target_points: int = DECIMATED_TARGET_POINTS
) -> np.ndarray:
    """
    Subsamples a high-density 3D mesh into a feature-aware light-field constellation
    up to target_points (~55% sharp dihedral crest/ridge vertices + ~45% uniform voxel/surface samples).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts) <= target_points:
        return verts.copy()

    extents = mesh.bounding_box.extents
    volume = float(np.prod(extents))
    if volume <= 1e-9:
        return verts[:target_points].copy()

    crest_quota = int(round(target_points * 0.55))
    crest_pts = np.empty((0, 3), dtype=np.float32)
    try:
        adj_angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float32)
        adj_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
        if len(adj_angles) > 0 and len(adj_edges) == len(adj_angles):
            vert_sharp = np.zeros(len(verts), dtype=np.float32)
            np.maximum.at(vert_sharp, adj_edges[:, 0], adj_angles)
            np.maximum.at(vert_sharp, adj_edges[:, 1], adj_angles)
            thresh = float(np.quantile(adj_angles, 0.72)) if len(adj_angles) > 8 else 0.15
            thresh = max(0.08, thresh)
            cand_idx = np.nonzero(vert_sharp >= thresh)[0]
            if len(cand_idx) > 0:
                order = np.argsort(-vert_sharp[cand_idx], kind="quicksort")
                cand_sorted = cand_idx[order]
                max_ext = max(1e-4, float(np.max(extents)))
                cell_inv = (max(16.0, np.sqrt(target_points) * 1.15)) / max_ext
                seen_cells = set()
                chosen = []
                for vi in cand_sorted:
                    p = verts[vi]
                    key = (int(round(p[0] * cell_inv)), int(round(p[1] * cell_inv)), int(round(p[2] * cell_inv)))
                    if key in seen_cells:
                        continue
                    seen_cells.add(key)
                    chosen.append(vi)
                    if len(chosen) >= crest_quota:
                        break
                if chosen:
                    crest_pts = verts[np.asarray(chosen, dtype=np.int64)]
    except Exception:
        crest_pts = np.empty((0, 3), dtype=np.float32)

    uniform_quota = max(1, target_points - len(crest_pts))
    pitch = max(1e-3, (volume / max(1, uniform_quota)) ** (1.0 / 3.0))
    uniform_pts = np.empty((0, 3), dtype=np.float32)
    try:
        voxel_grid = mesh.voxelized(pitch=pitch)
        v_pts = np.asarray(voxel_grid.points, dtype=np.float32)
        if len(v_pts) > 0:
            uniform_pts = v_pts
    except Exception:
        uniform_pts = np.empty((0, 3), dtype=np.float32)

    if len(uniform_pts) == 0:
        try:
            samples, _ = trimesh.sample.sample_surface(mesh, uniform_quota)
            uniform_pts = np.asarray(samples, dtype=np.float32)
        except Exception:
            rng = np.random.default_rng(seed=len(verts))
            idx = rng.choice(len(verts), size=min(uniform_quota, len(verts)), replace=False)
            uniform_pts = verts[idx]

    if len(uniform_pts) > uniform_quota:
        rng = np.random.default_rng(seed=len(uniform_pts))
        indices = rng.choice(len(uniform_pts), size=uniform_quota, replace=False)
        uniform_pts = uniform_pts[indices]

    if len(crest_pts) > 0:
        combined = np.concatenate([crest_pts, uniform_pts], axis=0)
    else:
        combined = uniform_pts

    if len(combined) > target_points:
        combined = combined[:target_points]

    return np.asarray(combined, dtype=np.float32)


def _atomic_export_ply(trimesh_obj, out_path: str) -> None:
    tmp_path = f"{out_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        trimesh_obj.export(tmp_path, file_type="ply")
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def export_mesh_and_points(
    mesh: trimesh.Trimesh, slug: str, version_tag: Optional[str] = None
) -> str:
    """Atomically exports PLY mesh + voxel-decimated points constellation for Three.js streaming."""
    file_stem = f"{slug}_{version_tag}" if version_tag else slug
    out_path = os.path.join(MESHES_DIR, f"{file_stem}.ply")
    _atomic_export_ply(mesh, out_path)

    try:
        decimated_pts = generate_voxel_decimated_points(mesh, target_points=DECIMATED_TARGET_POINTS)
        pts_stem = f"{file_stem}_points"
        pts_ply = os.path.join(MESHES_DIR, f"{pts_stem}.ply")
        pts_json = os.path.join(MESHES_DIR, f"{pts_stem}.json")
        _atomic_export_ply(trimesh.PointCloud(decimated_pts), pts_ply)

        tmp_json = f"{pts_json}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump({"count": len(decimated_pts), "points": decimated_pts.flatten().tolist()}, f)
        os.replace(tmp_json, pts_json)
    except Exception as pe:
        print(f"[!] Warning: Could not generate decimated points for '{file_stem}': {pe}")

    return f"/api/mesh/{file_stem}.ply"


def synthesize_verified_part_mesh(
    part_name: str,
    verify_query: str,
    candidates: Optional[List[Dict]] = None,
    fallback_image_path: Optional[str] = None,
    vlm_verify_fn: Optional[Callable[[Image.Image, str], str]] = None,
    resolution: int = DEFAULT_MESH_RESOLUTION,
    auto_retry: bool = True,
    version_tag: Optional[str] = None,
    on_attempt: Optional[Callable[[dict], None]] = None,
    topology: Optional[str] = None,
    role: Optional[str] = None,
    blueprint_title: Optional[str] = None,
) -> dict:
    """
    Stage 5 Self-Verification Synthesis Loop:
      1. Try Candidate #1 -> TripoSR -> 3-view Z-buffer turntable -> VLM plausibility verdict.
      2. If rejected and auto_retry=True -> automatically retry Candidate #2..#K.
      3. Fail all -> flag webcam fallback needed AND emit spherical harmonic manifold fallback
         so synthesis never dead-ends.
    """
    slug = slugify(part_name)
    notify = on_attempt or (lambda d: None)

    trials: List[Dict] = []
    if fallback_image_path and os.path.exists(fallback_image_path):
        trials.append({
            "title": "Captured Camera / Uploaded Image",
            "thumbUrl": fallback_image_path,
            "originalUrl": fallback_image_path,
            "caption": "User provided visual capture",
            "sourceTier": 0,
            "sourceLabel": "user_capture",
        })

    if candidates:
        for c in candidates:
            if len(trials) >= MAX_VERIFY_TRIALS:
                break
            url = c.get("thumbUrl") or c.get("originalUrl")
            if url:
                trials.append(c)

    last_verdict = {
        "plausible": False,
        "confidence": 0.0,
        "shape_summary": "none",
        "reason": "No verified photographic candidate available for 3D reconstruction.",
    }
    chosen_mesh = None
    chosen_cand = None
    chosen_idx = -1

    for idx, cand in enumerate(trials):
        img_src = cand.get("thumbUrl") or cand.get("originalUrl")
        notify({
            "phase": "attempt",
            "attempt": idx + 1,
            "maxAttempts": len(trials),
            "title": cand.get("title", "candidate"),
        })

        try:
            img = load_image_from_source(img_src)
            print(
                f"[*] Stage 5 Attempt #{idx + 1}/{len(trials)} TripoSR on "
                f"'{cand.get('title', img_src)}' for '{part_name}'..."
            )
            mesh = reconstruct_triposr_mesh(img, resolution=resolution)
        except Exception as e:
            print(f"[!] TripoSR attempt #{idx + 1} failed for '{part_name}': {e}")
            last_verdict = {
                "plausible": False,
                "confidence": 0.0,
                "shape_summary": "reconstruction error",
                "reason": str(e)[:140],
            }
            continue

        verdict = stage5_verify_mesh(mesh, query=verify_query or part_name, vlm_verify_fn=vlm_verify_fn)
        last_verdict = verdict
        print(
            f"[*] Stage 5 Verdict (#{idx + 1}, '{part_name}'): "
            f"plausible={verdict['plausible']} conf={verdict['confidence']} reason='{verdict['reason']}'"
        )

        notify({
            "phase": "verdict",
            "attempt": idx + 1,
            "maxAttempts": len(trials),
            "title": cand.get("title", "candidate"),
            "verdict": verdict,
        })

        if verdict.get("plausible", False) or not auto_retry:
            chosen_mesh = mesh
            chosen_cand = cand
            chosen_idx = idx
            break

        print(
            f"[!] Stage 5 REJECTED Candidate #{idx + 1} for '{part_name}': "
            f"{verdict.get('reason')} -> retrying next candidate..."
        )

    used_harmonic_fallback = False
    webcam_fallback_needed = False

    if chosen_mesh is None or len(chosen_mesh.vertices) == 0:
        used_harmonic_fallback = True
        webcam_fallback_needed = True
        print(
            f"[!] Stage 5 fallback activated for '{part_name}' after {len(trials)} attempt(s). "
            "Synthesizing precision parametric CAD geometry."
        )
        try:
            from cad_geometry import generate_parametric_cad_mesh
            chosen_mesh = generate_parametric_cad_mesh(
                part_name=part_name,
                topology=topology,
                role=role,
                blueprint_title=blueprint_title,
            )
        except Exception as ce:
            print(f"[!] Warning: CAD generator fallback issue for '{part_name}': {ce}")
            from cad_geometry import generate_cad_precision_machined_solid
            chosen_mesh = generate_cad_precision_machined_solid()
        chosen_cand = None

    mesh_url = export_mesh_and_points(chosen_mesh, slug, version_tag=version_tag)
    print(f"[✓] Exported 3D hologram mesh for '{part_name}': {len(chosen_mesh.vertices)} vertices -> {mesh_url}")

    return {
        "meshUrl": mesh_url,
        "selectedCandidate": serialize_candidate(chosen_cand) if chosen_cand else None,
        "attemptIndex": chosen_idx,
        "attemptsMade": len(trials) if used_harmonic_fallback else (chosen_idx + 1),
        "verified": bool(not used_harmonic_fallback and last_verdict.get("plausible", False)),
        "verification": last_verdict,
        "usedHarmonicFallback": used_harmonic_fallback,
        "webcamFallbackNeeded": webcam_fallback_needed,
    }


def generate_mesh_from_real_reference(
    part_name: str,
    image_source: Optional[str] = None,
    topology: Optional[str] = None,
    resolution: int = DEFAULT_MESH_RESOLUTION,
    role: Optional[str] = None,
    blueprint_title: Optional[str] = None,
) -> str:
    """Backwards-compatible direct single-source mesh generator."""
    slug = slugify(part_name)
    img = None

    if image_source:
        try:
            img = load_image_from_source(image_source)
        except Exception as e:
            print(f"[!] Warning: Could not load image from {image_source}: {e}")

    if img is None:
        try:
            from reference_service import fetch_wikimedia_commons_images
            results = fetch_wikimedia_commons_images(part_name, limit=1)
            if results and results[0].get("thumbUrl"):
                img = load_image_from_source(results[0]["thumbUrl"])
        except Exception as e:
            print(f"[!] Live reference search failed for '{part_name}': {e}")

    mesh = None
    if img is not None:
        try:
            mesh = reconstruct_triposr_mesh(img, resolution=resolution)
        except Exception as te:
            print(f"[!] TripoSR neural reconstruction issue for '{part_name}': {te}")
            mesh = None

    if mesh is None or len(mesh.vertices) == 0:
        try:
            from cad_geometry import generate_parametric_cad_mesh
            mesh = generate_parametric_cad_mesh(
                part_name=part_name,
                topology=topology,
                role=role,
                blueprint_title=blueprint_title,
            )
        except Exception:
            from cad_geometry import generate_cad_precision_machined_solid
            mesh = generate_cad_precision_machined_solid()

    return export_mesh_and_points(mesh, slug)


def generate_continuous_polygon_mesh(
    part_name: str,
    part_role: str = "",
    image_source: Optional[str] = None,
    topology: Optional[str] = None,
    blueprint_title: Optional[str] = None,
) -> str:
    """Primary API entry point called by legacy synchronous callers."""
    return generate_mesh_from_real_reference(
        part_name,
        image_source=image_source,
        topology=topology,
        role=part_role,
        blueprint_title=blueprint_title,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generator_3d.py <part_name> [image_url_or_path]")
        sys.exit(1)
    name = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"[*] Generating 3D micro-topography for '{name}'...")
    res = generate_mesh_from_real_reference(name, image_source=url)
    print("Mesh URL:", res)
