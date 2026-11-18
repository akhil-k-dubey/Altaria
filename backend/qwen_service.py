#!/usr/bin/env python3
"""
M.A.R.K. Holo-CAD // Dedicated 3D Generative CAD & Progressive Streaming Engine
Combines:
1. Local Qwen2.5-VL Cognitive AI with Grammar-Constrained Decoding (llguidance).
2. SSE (Server-Sent Events) Progressive Token & Part Streaming.
3. Multi-Stage Candidate Image Funnel, TripoSR 3D Reconstruction & Stage-5 Self-Verification.
4. Pure Computational Geometry Multi-Body Assembly Validation & Auto-Repair.
"""

import base64
import colorsys
import hashlib
import http.server
import io
import json
import math
import os
import socket
import socketserver
import sys
import time
import urllib.parse
from threading import Lock, Thread
from typing import List, Callable, Optional
from PIL import Image, ImageOps

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    LogitsProcessorList,
    TextIteratorStreamer,
)
from qwen_vl_utils import process_vision_info
from pydantic import BaseModel, Field

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator_3d import (
    synthesize_verified_part_mesh,
    MESHES_DIR,
    TEXTURES_DIR,
)
from reference_service import (
    _candidate_identity_keys,
    get_reference_for_topic_and_parts,
)
from grammar_decoder import (
    build_cad_grammar_processor,
    build_grammar_processor,
    extract_json_with_bracket_depth,
    IncrementalBlueprintParser,
    make_vlm_judge_schema,
    QUERY_EXPAND_JSON_SCHEMA,
    VLM_VERIFY_JSON_SCHEMA,
)
from candidate_gate import (
    apply_gate_action,
    create_gate,
    remove_gate,
)
from image_funnel import (
    determine_operating_mode,
    save_candidate_preference,
    serialize_candidate,
    slugify,
)
from geometry_validator import validate_and_repair_blueprint
from image_proxy import (
    fetch_or_cache_proxied_image,
    IMAGE_CACHE_DIR,
    SSRFSecurityError,
    InvalidImageMediaError,
)

# Serialize GPU calls across concurrent threads on constrained VRAM
QWEN_INFER_LOCK = Lock()

PORT = int(os.environ.get("HOLO_CAD_PORT", "8000"))
MODEL_ID = os.environ.get("HOLO_CAD_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
DEFAULT_MESH_RESOLUTION = max(32, int(os.environ.get("HOLO_CAD_MESH_RES", "192")))
DEFAULT_UPLOAD_MAX_DIM = max(256, int(os.environ.get("HOLO_CAD_UPLOAD_MAX_DIM", "768")))
DEFAULT_ESCAPE_HATCH_K = max(1, int(os.environ.get("HOLO_CAD_CANDIDATE_K", "3")))
DEFAULT_CAD_MAX_TOKENS = max(1024, int(os.environ.get("HOLO_CAD_MAX_TOKENS", "4096")))
DEFAULT_STAGE4_BASE_TOKENS = max(96, int(os.environ.get("HOLO_CAD_STAGE4_BASE_TOKENS", "192")))
DEFAULT_STAGE4_PER_IMG_TOKENS = max(96, int(os.environ.get("HOLO_CAD_STAGE4_PER_IMG_TOKENS", "192")))
DEFAULT_STAGE4_MAX_TOKENS = max(512, int(os.environ.get("HOLO_CAD_STAGE4_MAX_TOKENS", "2048")))
DEFAULT_STAGE5_MAX_TOKENS = max(160, int(os.environ.get("HOLO_CAD_STAGE5_TOKENS", "320")))
DEFAULT_QUERY_EXPAND_MAX_TOKENS = max(64, int(os.environ.get("HOLO_CAD_QUERY_EXPAND_TOKENS", "128")))
MAX_JSON_BODY_BYTES = int(os.environ.get("HOLO_CAD_MAX_BODY_BYTES", str(20 * 1024 * 1024)))
FALLBACK_POS_VOLUME = float(os.environ.get("HOLO_CAD_FALLBACK_POS_VOL", "0.25"))
FALLBACK_EXPLODE_SCALE = float(os.environ.get("HOLO_CAD_FALLBACK_EXPLODE_SCALE", "1.35"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUEPRINTS_DIR = os.path.join(BASE_DIR, "data", "blueprints")
os.makedirs(BLUEPRINTS_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

_CPU_COUNT = os.cpu_count() or 4
torch.set_num_threads(max(1, _CPU_COUNT // 2))
torch.set_num_interop_threads(max(1, min(4, _CPU_COUNT // 4)))

print("=" * 68)
print("   M.A.R.K. HOLO-CAD // DEDICATED 3D GENERATIVE CAD ENGINE")
print("=" * 68)
print(f"[*] Base Directory:      {BASE_DIR}")
print(f"[*] Local Blueprints Dir: {BLUEPRINTS_DIR}")
print(f"[*] Local Meshes Dir:     {MESHES_DIR}")
print(f"[*] Cognitive Model:     {MODEL_ID}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Target Device:       {device}")

t_start = time.time()
print("[*] Loading Qwen2.5-VL model weights into memory...")
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print("[*] 4-Bit NF4 Quantization enabled: 100% of model fitted on GPU with 0 offload.")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to("cpu")
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"[✓] Model and processor ready in {time.time() - t_start:.2f}s!")


def _get_model_device() -> torch.device:
    """Resolve the active execution device dynamically even under device_map='auto'."""
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(device)


CAD_SYSTEM_PROMPT = """You are a spatial reverse-engineering and structural decomposition AI.
Your goal is to reconstruct the complete, authentic 3D blueprint of any given system, organism, structure, artifact, or mechanism as it exists in real life with strict domain-native accuracy.

CRITICAL RULES FOR REALISTIC 3D DECOMPOSITION:
1. AUTHENTIC DOMAIN TERMINOLOGY & ENCLOSURE ROLE (NEVER blanket-stamp hollow_shell):
   Decompose the requested system into its real, cohesive, interlocking physical sub-assemblies using terminology strictly authentic to THAT subject:
   - ENCLOSED PROPULSION & MACHINERY: Part 1 MUST be outer nacelle/casing with "topology": "hollow_shell". Internal stages use "cylindrical_core" for central drive shafts/spools, "rotary_disc" for bladed turbine/compressor rotors, "conical_taper" for supersonic exit nozzles.
   - BIOLOGICAL / ANATOMICAL ORGANS (Human Brain, Human Heart, visceral anatomy): NEVER assign "rotary_disc" or non-zero "rotSpeed" (organs do NOT spin!). Assign "hollow_shell" to outer enveloping cortex/capsule, "contoured_solid" to deep nuclei/ventricles/atria/lobes, "branched_conduit" to brainstem/vascular conduits/valves.
   - COMPUTATIONAL & NEURAL ARCHITECTURES (Deep Neural Networks, Transformers, Attention Graphs): NEVER assign "rotary_disc" or non-zero "rotSpeed"! Assign "structural_frame" for input/output layer neuron arrays, "branched_conduit" for hidden layers / synaptic interconnects, "contoured_solid" for processing hubs.
   - OPEN STRUCTURAL ASSEMBLIES (instruments, vehicles, machines, or frameworks whose parts mount visibly ON or AROUND a frame/stand/backbone rather than sealed inside one outer wall): Part 1 MUST be the primary structural chassis, stand, or carrier frame with "topology": "structural_frame" or "contoured_solid" (NOT "hollow_shell"), and Parts 2..N MUST be the major mounted components positioned cleanly relative to it (NONE set to "hollow_shell").
   - NEVER append generic machinery words like "Housing" or "Casing" onto biological, natural, artistic, or optical/non-enclosed subjects.
   - If an image is provided with no text topic, visually identify the exact real-world subject first and set "title" and all parts strictly to that identified subject.

2. TRUE NORMALIZED 3D COORDINATES & DISASSEMBLY VECTORS:
   - Keep all "assembledPos" coordinates compact within [-1.2, 1.2] around origin [0, 0, 0] (normalized unit scale where a full part spans ~0.6-1.0), using realistic multi-axis offsets matching how parts physically meet or stack. For OPEN structural assemblies with no outer casing, separate mounted parts cleanly (~0.65-1.35 apart) rather than clustering all parts at [0, 0, 0].
   - Assign each part a distinct outward 3D "explodedVec" (vector magnitude ~0.8 to 2.0) along realistic disassembly directions so parts separate cleanly without colliding or stacking along a single line.
   - Set "rotSpeed" to [0, 0, 0] for static, biological, fixed, or structural components; use a small non-zero vector ONLY for parts that physically rotate continuously during operation.

3. IDENTIFIERS, SUBTITLE & PROXY HINTS:
   - Use short snake_case strings for "id" (e.g. "primary_stand", "objective_turret" — NEVER numeric strings like "1").
   - Assign "topology" strictly from: ("hollow_shell", "rotary_disc", "cylindrical_core", "conical_taper", "branched_conduit", "structural_frame", "contoured_solid"). Only an enclosing outermost Part 1 may use "hollow_shell".
   - Format "subtitle" cleanly as "<N> SUB-ASSEMBLIES LOADED".

JSON object keys and structure:
{
  "title": "<UPPERCASE REAL-WORLD SUBJECT TITLE>",
  "subtitle": "<N SUB-ASSEMBLIES LOADED>",
  "overview": "<One dense factual sentence describing real-world physical structure and operation>",
  "parts": [
    {
      "id": "<unique_snake_case_id>",
      "name": "<Domain-Authentic Component Name>",
      "role": "<CORE FUNCTION OR ROLE>",
      "topology": "<transient proxy class>",
      "color": "<Distinct high-contrast #RRGGBB hex color>",
      "assembledPos": [<x_float>, <y_float>, <z_float>],
      "explodedVec": [<dx_float>, <dy_float>, <dz_float>],
      "rotSpeed": [<rx_float>, <ry_float>, <rz_float>],
      "summary": [
        "<1. Precise physical or functional definition.>",
        "<2. Mechanism of action and spatial relationship to adjacent components.>"
      ],
      "principle": "<Governing physical, chemical, or physiological law>"
    }
  ]
}

Use perceptually distinct #RRGGBB hex colors across all parts (never repeat colors across siblings).
Respond strictly with valid RAW JSON containing finite floating-point numbers (no markdown fences or commentary)."""


# =========================================================================
# ARCHITECTURAL SCHEMA TYPING (PYDANTIC) - ZERO HARDCODED PALETTES/VECTORS
# =========================================================================
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _is_valid_hex_color(value: object) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    return len(s) == 7 and s[0] == "#" and all(c in _HEX_CHARS for c in s[1:])


def _procedural_part_color(index: int, seed_text: str) -> str:
    """Derive a distinct high-contrast #RRGGBB color via golden-angle spacing in HSV space."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    base_hue = ((digest[0] << 8) | digest[1]) / 65536.0
    hue = (base_hue + index * _GOLDEN_RATIO_CONJUGATE) % 1.0
    sat = 0.78 + 0.14 * ((index % 3) / 2.0)
    val = 0.90 + 0.08 * (((index + 1) % 2))
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def _spherical_dispersion_unit(index: int, total: int) -> List[float]:
    """Fibonacci-lattice direction on the unit sphere so fallback offsets never stack."""
    n = max(1, total)
    z = 1.0 - 2.0 * (index + 0.5) / n
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    theta = 2.0 * math.pi * _GOLDEN_RATIO_CONJUGATE * index
    return [
        round(radius * math.cos(theta), 4),
        round(z, 4),
        round(radius * math.sin(theta), 4),
    ]


def _sanitize_vec3(raw: object) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    vals: List[float] = []
    for i in range(3):
        try:
            f = float(raw[i])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        vals.append(round(f, 4))
    return vals


def _to_public_image_url(source: Optional[str]) -> Optional[str]:
    """Rewrite local IMAGE_CACHE_DIR paths to servable /api/cache-image/... HTTP URLs."""
    if not source or not isinstance(source, str):
        return source
    real_cache = os.path.realpath(IMAGE_CACHE_DIR)
    try:
        if os.path.isabs(source):
            real_src = os.path.realpath(source)
            if os.path.commonpath([real_cache, real_src]) == real_cache:
                return f"/api/cache-image/{urllib.parse.quote(os.path.basename(real_src))}"
    except Exception:
        pass
    return source


def _sanitize_candidate_urls(cand: Optional[dict]) -> Optional[dict]:
    if not isinstance(cand, dict):
        return cand
    out = dict(cand)
    if out.get("thumbUrl"):
        out["thumbUrl"] = _to_public_image_url(out["thumbUrl"])
    if out.get("originalUrl"):
        out["originalUrl"] = _to_public_image_url(out["originalUrl"])
    return out


def _normalize_blueprint_urls(bp: dict) -> dict:
    """Ensure cached/loaded blueprints never leak absolute local filesystem paths."""
    if not isinstance(bp, dict):
        return bp
    for key in ("references", "candidates"):
        if isinstance(bp.get(key), list):
            bp[key] = [_sanitize_candidate_urls(c) for c in bp[key]]
    for p in bp.get("parts") or []:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("referenceImage"), dict):
            p["referenceImage"] = _sanitize_candidate_urls(p["referenceImage"])
        if isinstance(p.get("candidates"), list):
            p["candidates"] = [_sanitize_candidate_urls(c) for c in p["candidates"]]
    return bp


def _build_verify_query(topic: str, part_name: str, part_index: int) -> str:
    """Single canonical formatting for Stage-5 verification queries across synthesis & selection."""
    t = " ".join(str(topic or "").split())
    pn = " ".join(str(part_name or "").split())
    if part_index == 0 and t:
        return t
    if pn and t and pn.lower() != t.lower():
        return f"{pn} of {t}"
    return pn or t


def _atomic_write_json(path: str, payload: object) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class BlueprintPartSchema(BaseModel):
    id: str = Field(default="")
    name: str = Field(default="")
    role: str = Field(default="")
    topology: str = Field(default="contoured_solid")
    color: str = Field(default="")
    assembledPos: List[float] = Field(default_factory=list)
    explodedVec: List[float] = Field(default_factory=list)
    rotSpeed: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    summary: List[str] = Field(default_factory=list)
    principle: str = Field(default="")


class BlueprintSchemaModel(BaseModel):
    title: str = Field(default="")
    subtitle: str = Field(default="")
    overview: str = Field(default="")
    parts: List[BlueprintPartSchema] = Field(default_factory=list)


def normalize_single_blueprint_part(
    part_raw: dict,
    idx: int,
    seen_ids: set,
    seen_colors: set,
    expected_total: int = 4,
) -> dict:
    """Normalizes one streamed or finalized part while tracking sibling uniqueness."""
    p = dict(part_raw)
    total = max(1, expected_total)

    base_id = slugify(str(p.get("id") or p.get("name") or f"part_{idx + 1}"))
    unique_id = base_id
    suffix = 2
    while unique_id in seen_ids:
        unique_id = f"{base_id}_{suffix}"
        suffix += 1
    seen_ids.add(unique_id)
    p["id"] = unique_id

    p["name"] = str(p.get("name") or f"Component {idx + 1}").strip()
    p["role"] = str(p.get("role") or "Sub-assembly").strip()
    p["topology"] = str(p.get("topology") or "contoured_solid").strip()
    p["principle"] = str(p.get("principle") or "").strip()

    raw_sum = p.get("summary")
    if isinstance(raw_sum, list):
        p["summary"] = [str(s).strip() for s in raw_sum if str(s).strip()]
    elif isinstance(raw_sum, str) and raw_sum.strip():
        p["summary"] = [raw_sum.strip()]
    else:
        p["summary"] = []

    raw_color = str(p.get("color") or "").strip()
    if _is_valid_hex_color(raw_color) and raw_color.lower() not in seen_colors:
        chosen_color = raw_color.lower()
    else:
        color_attempt = idx
        chosen_color = _procedural_part_color(color_attempt, p["name"] or unique_id)
        while chosen_color.lower() in seen_colors:
            color_attempt += 1
            chosen_color = _procedural_part_color(color_attempt, p["name"] or unique_id)
    seen_colors.add(chosen_color.lower())
    p["color"] = chosen_color

    pos = _sanitize_vec3(p.get("assembledPos"))
    if pos is not None:
        p["assembledPos"] = pos
    elif idx == 0:
        p["assembledPos"] = [0.0, 0.0, 0.0]
    else:
        unit_dir = _spherical_dispersion_unit(idx, total)
        pack_radius = (FALLBACK_POS_VOLUME / max(1, total)) ** (1.0 / 3.0)
        p["assembledPos"] = [round(c * pack_radius, 4) for c in unit_dir]

    exp = _sanitize_vec3(p.get("explodedVec"))
    if exp is not None:
        p["explodedVec"] = exp
    else:
        unit_dir = _spherical_dispersion_unit(idx, total)
        disp_radius = FALLBACK_EXPLODE_SCALE * (1.0 + (FALLBACK_POS_VOLUME / max(1, total)) ** (1.0 / 3.0))
        p["explodedVec"] = [round(c * disp_radius, 4) for c in unit_dir]

    rot = _sanitize_vec3(p.get("rotSpeed"))
    p["rotSpeed"] = rot if rot is not None else [0.0, 0.0, 0.0]

    # Domain sanity repair: biological organs and neural layers do not spin like mechanical rotors
    text_check = f"{p['name']} {p['role']}".lower()
    if any(k in text_check for k in ("brain", "cortex", "ganglia", "thalamus", "ventricle", "atrium", "valve", "neural", "layer")):
        if p["topology"] == "rotary_disc":
            p["topology"] = "structural_frame" if "layer" in text_check else "contoured_solid"
        p["rotSpeed"] = [0.0, 0.0, 0.0]

    return p


def normalize_blueprint_parts(parts: list) -> list:
    """Normalizes part IDs, colors, summaries, and finite 3D vectors without fixed palettes or constants."""
    if not parts:
        return parts

    raw_dicts = [dict(p) for p in parts if isinstance(p, dict)]
    total = len(raw_dicts)
    normalized = []
    seen_ids = set()
    seen_colors = set()

    for idx, raw_p in enumerate(raw_dicts):
        normalized.append(
            normalize_single_blueprint_part(raw_p, idx, seen_ids, seen_colors, expected_total=total)
        )

    return normalized


def _finalize_blueprint_dict(data: dict) -> dict:
    parts = normalize_blueprint_parts(data.get("parts", []))
    data["parts"] = parts
    title = str(data.get("title") or "").strip()
    if not title:
        title = str(parts[0].get("name", "SPATIAL ASSEMBLY")).upper() if parts else "SPATIAL ASSEMBLY"
    data["title"] = title
    subtitle = str(data.get("subtitle") or "").strip()
    if not subtitle or subtitle.isdigit() or len(subtitle) < 4:
        subtitle = f"{len(parts)} SUB-ASSEMBLIES LOADED"
    data["subtitle"] = subtitle
    data["overview"] = str(data.get("overview") or f"Spatial 3D deconstruction of {title}.").strip()
    return data


def extract_json_blueprint(text: str) -> dict:
    """
    Architectural schema validation using Pydantic + bracket-depth extraction.
    Zero regex scraping or brittle string splitting.
    """
    cleaned = text.strip()
    try:
        model_obj = BlueprintSchemaModel.model_validate_json(cleaned)
        data = model_obj.model_dump()
        if len(data.get("parts", [])) > 0:
            return _finalize_blueprint_dict(data)
    except Exception:
        pass

    try:
        raw_obj = extract_json_with_bracket_depth(cleaned)
        model_obj = BlueprintSchemaModel.model_validate(raw_obj)
        data = model_obj.model_dump()
        if len(data.get("parts", [])) == 0:
            raise ValueError("Blueprint JSON contained zero parts")
        return _finalize_blueprint_dict(data)
    except Exception as e:
        raise ValueError(f"Could not parse valid CAD blueprint schema: {e}")


# =========================================================================
# QWEN2.5-VL STAGE 4 JUDGE & STAGE 5 SELF-VERIFICATION HELPERS
# =========================================================================
def _run_qwen_structured_multimodal(
    images: List[Image.Image],
    prompt: str,
    schema: Optional[dict] = None,
    schema_label: str = "structured_vlm",
    max_new_tokens: int = 256,
) -> str:
    """Thread-safe, VRAM-locked single-turn Qwen2.5-VL inference for Stage 4 / Stage 5."""
    content_list = []
    for idx, im in enumerate(images):
        rgb_im = im.convert("RGB") if im.mode != "RGB" else im
        content_list.append({"type": "text", "text": f"Image {idx + 1}:"})
        content_list.append({"type": "image", "image": rgb_im})
    content_list.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You are a precise vision verification agent. Respond strictly in raw JSON only."},
        {"role": "user", "content": content_list},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(_get_model_device())

    logits_processors = LogitsProcessorList()
    if schema is not None:
        g_proc = build_grammar_processor(processor.tokenizer, schema=schema, label=schema_label)
        if g_proc is not None:
            logits_processors.append(g_proc)

    input_len = inputs["input_ids"].shape[1]
    with QWEN_INFER_LOCK:
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    logits_processor=logits_processors if len(logits_processors) > 0 else None,
                )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gen_tokens = out_ids[0, input_len:]
    return processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def run_qwen_stage4_judge(
    pil_images: List[Image.Image], prompt: str, is_subpart: Optional[bool] = None
) -> str:
    n_imgs = max(1, len(pil_images))
    subpart_flag = is_subpart if is_subpart is not None else ("whole_parent_system" in prompt)
    token_budget = min(
        DEFAULT_STAGE4_MAX_TOKENS,
        DEFAULT_STAGE4_BASE_TOKENS + n_imgs * DEFAULT_STAGE4_PER_IMG_TOKENS,
    )
    return _run_qwen_structured_multimodal(
        images=pil_images,
        prompt=prompt,
        schema=make_vlm_judge_schema(n_imgs, is_subpart=subpart_flag),
        schema_label=f"stage4_vlm_judge_{'sub' if subpart_flag else 'whole'}_{n_imgs}",
        max_new_tokens=token_budget,
    )


def run_qwen_stage5_verify(turntable_pil: Image.Image, prompt: str) -> str:
    return _run_qwen_structured_multimodal(
        images=[turntable_pil],
        prompt=prompt,
        schema=VLM_VERIFY_JSON_SCHEMA,
        schema_label="stage5_vlm_verify",
        max_new_tokens=DEFAULT_STAGE5_MAX_TOKENS,
    )


_QUERY_META_MEDIA_WORDS = frozenset({
    "photo", "photos", "photograph", "photographs", "photography",
    "picture", "pictures", "pic", "pics", "image", "images",
    "diagram", "diagrams", "illustration", "illustrations",
    "drawing", "drawings", "schematic", "schematics",
})


def run_qwen_query_expand(topic: str, part_name: str, part_desc: str = "") -> List[str]:
    """Uses Qwen2.5-VL with QUERY_EXPAND_JSON_SCHEMA to generate concrete subject search terms."""
    clean_topic = " ".join(str(topic or "").split())
    clean_part = " ".join(str(part_name or "").split())
    clean_desc = " ".join(str(part_desc or "").split())[:200]
    is_subpart = bool(clean_part and clean_topic and clean_part.lower() != clean_topic.lower())
    focus_note = (
        f"identifying the isolated/standalone sub-component '{clean_part}' belonging to '{clean_topic}' "
        f"(do NOT return whole-{clean_topic} searches; use alternative anatomical, technical, or apparatus names)"
        if is_subpart
        else f"identifying the physical object, specimen, anatomy, or apparatus '{clean_part or clean_topic}'"
    )
    prompt = (
        f"Generate 2 to 4 concise Wikimedia Commons search phrases {focus_note}. "
        "STRICT FORMAT RULE: Return concrete physical SUBJECT nouns/terms ONLY (such as canonical medical/engineering terms, "
        "model names, or specimen phrases). NEVER include media words like 'photo', 'photograph', 'photography', "
        "'picture', 'image', 'diagram', 'drawing', or 'illustration' inside the phrases."
    )
    if clean_desc:
        prompt += f" Context: {clean_desc}."
    prompt += " Return strictly raw JSON matching {\"queries\": [...]}."
    try:
        raw = _run_qwen_structured_multimodal(
            images=[],
            prompt=prompt,
            schema=QUERY_EXPAND_JSON_SCHEMA,
            schema_label="query_expand",
            max_new_tokens=DEFAULT_QUERY_EXPAND_MAX_TOKENS,
        )
        parsed = extract_json_with_bracket_depth(raw)
        qs = parsed.get("queries", []) if isinstance(parsed, dict) else []
        cleaned = []
        seen = set()
        for q in qs:
            words = [w for w in str(q or "").split() if w.strip(".,;:!?\"'()").lower() not in _QUERY_META_MEDIA_WORDS]
            s = " ".join(words).strip()
            if s and len(s) >= 3 and s.lower() not in seen:
                seen.add(s.lower())
                cleaned.append(s)
        return cleaned[:4]
    except Exception as e:
        print(f"[!] Qwen query expansion fallback warning ({clean_part}): {e}")
        return []


# =========================================================================
# PROGRESSIVE STREAMING ENGINE (SSE + TOKEN STREAMER + FUNNEL + SELF-VERIFY)
# =========================================================================
def synthesize_cad_assembly_stream(
    prompt_text: str,
    image_b64: Optional[str] = None,
    emit_event: Optional[Callable[[str, dict], None]] = None,
    is_disconnected_fn: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Progressively streams tokens, runs the 5-Stage Reference Funnel,
    reconstructs meshes with Stage-5 turntable self-verification retries,
    and executes pure geometric multi-body validation over SSE.
    """
    if emit_event is None:
        emit_event = lambda event_name, data: None

    raw_prompt = " ".join(str(prompt_text or "").split())
    user_image_path = None
    content_list = []

    if image_b64:
        try:
            b64_payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
            raw_bytes = base64.b64decode(b64_payload)
            with Image.open(io.BytesIO(raw_bytes)) as opened_im:
                img = ImageOps.exif_transpose(opened_im).convert("RGB")
            img.thumbnail((DEFAULT_UPLOAD_MAX_DIM, DEFAULT_UPLOAD_MAX_DIM), Image.Resampling.LANCZOS)
            content_list.append({"type": "image", "image": img})
            img_sig = hashlib.sha256(raw_bytes).hexdigest()[:12]
            upload_stem = slugify(raw_prompt) if raw_prompt else f"capture_{img_sig}"
            user_image_path = os.path.join(IMAGE_CACHE_DIR, f"upload_{upload_stem}.png")
            img.save(user_image_path)
        except Exception as e:
            print(f"[!] Warning: Image parse error: {e}")

    if not raw_prompt and not content_list:
        raise ValueError("A non-empty prompt or valid image is required for synthesis.")

    if raw_prompt:
        user_query = f"Deconstruct into realistic 3D spatial sub-assemblies: {raw_prompt}"
        status_label = raw_prompt.upper()
    else:
        user_query = (
            "Visually identify the exact real-world subject in this image and "
            "deconstruct it into its authentic 3D spatial sub-assemblies."
        )
        status_label = "VISUAL CAPTURE"
    content_list.append({"type": "text", "text": user_query})

    messages = [
        {"role": "system", "content": CAD_SYSTEM_PROMPT},
        {"role": "user", "content": content_list},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(_get_model_device())

    emit_event("status", {
        "stage": "cognition",
        "message": f"J.A.R.V.I.S. reasoning on {status_label}..."
    })

    logits_processors = LogitsProcessorList()
    grammar_proc = build_cad_grammar_processor(processor.tokenizer)
    if grammar_proc:
        logits_processors.append(grammar_proc)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=DEFAULT_CAD_MAX_TOKENS,
        do_sample=False,
        logits_processor=logits_processors if len(logits_processors) > 0 else None,
    )

    gen_error = []
    def run_generation():
        with QWEN_INFER_LOCK:
            try:
                with torch.no_grad():
                    model.generate(**generation_kwargs)
            except Exception as e:
                print(f"[!] Generation error: {e}")
                gen_error.append(e)
                try:
                    streamer.end()
                except Exception:
                    pass
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    gen_thread = Thread(target=run_generation, daemon=True)
    gen_thread.start()

    incremental_parser = IncrementalBlueprintParser()
    full_token_chunks = []
    stream_seen_ids = set()
    stream_seen_colors = set()

    print(f"[*] Qwen VLM streaming live tokens on '{raw_prompt or status_label}'...")
    for token_chunk in streamer:
        emit_event("token", {"token": token_chunk})
        full_token_chunks.append(token_chunk)

        parsed_events = incremental_parser.feed(token_chunk)
        for ev_type, ev_payload in parsed_events:
            if ev_type == "part_discovered" and isinstance(ev_payload.get("part"), dict):
                ev_payload["part"] = normalize_single_blueprint_part(
                    ev_payload["part"],
                    int(ev_payload.get("partIndex", len(stream_seen_ids))),
                    stream_seen_ids,
                    stream_seen_colors,
                    expected_total=max(4, int(ev_payload.get("partIndex", 0)) + 2),
                )
            emit_event(ev_type, ev_payload)
            if ev_type == "part_discovered":
                print(f"[+] Materializing sub-assembly #{ev_payload['partIndex']}: '{ev_payload['part'].get('name')}'")

    gen_thread.join()
    if gen_error:
        raise gen_error[0]
    gen_text = "".join(full_token_chunks)

    blueprint = extract_json_blueprint(gen_text)
    effective_query = raw_prompt or str(blueprint.get("title") or "").strip()
    if not effective_query and blueprint.get("parts"):
        effective_query = str(blueprint["parts"][0].get("name") or "").strip()
    if not effective_query:
        effective_query = "spatial_capture"

    slug = slugify(effective_query)
    cache_path = os.path.join(BLUEPRINTS_DIR, f"{slug}.json")
    blueprint["queryPrompt"] = effective_query

    # 1. 5-Stage Reference Image Funnel (Acquire -> Stage 1..4)
    def on_funnel_stage(stage_name: str, detail: dict):
        emit_event("funnel_stage", {"stage": stage_name, **detail})
        if stage_name == "acquire":
            emit_event("status", {
                "stage": "funnel",
                "message": detail.get("message") or f"Acquiring candidate references for {effective_query}..."
            })
        elif stage_name == "stage1":
            emit_event("status", {
                "stage": "funnel",
                "message": f"FUNNEL S1 METADATA: {detail.get('survived', 0)}/{detail.get('raw', 0)} photos survived..."
            })
        elif stage_name == "stage2":
            emit_event("status", {
                "stage": "funnel",
                "message": f"FUNNEL S2 CV SIGNALS: {detail.get('survived', 0)} clean-bg photos passed to CLIP..."
            })
        elif stage_name == "stage3":
            emit_event("status", {
                "stage": "funnel",
                "message": f"FUNNEL S3 CLIP: Top {detail.get('survived', 0)} sent to Qwen2.5-VL judge..."
            })
        elif stage_name == "stage4":
            emit_event("status", {
                "stage": "funnel",
                "message": f"FUNNEL S4 VLM JUDGE: Ranked {detail.get('ranked', 0)} photographic candidates."
            })

    emit_event("status", {
        "stage": "references",
        "message": f"Running 5-stage photographic vetting funnel for {effective_query}..."
    })
    ref_bundle = get_reference_for_topic_and_parts(
        effective_query,
        blueprint.get("parts", []),
        vlm_judge_fn=run_qwen_stage4_judge,
        query_expand_fn=run_qwen_query_expand,
        on_stage=on_funnel_stage,
    )

    topic_candidates = [
        _sanitize_candidate_urls(c)
        for c in (ref_bundle.get("candidates") or ref_bundle.get("images") or [])
    ]
    part_candidates_map = ref_bundle.get("partCandidates", {})
    blueprint_parts = blueprint.get("parts", [])
    p0_id = blueprint_parts[0].get("id") if blueprint_parts else None
    p0_pool = (
        [_sanitize_candidate_urls(c) for c in part_candidates_map[p0_id] if isinstance(c, dict)]
        if (p0_id and part_candidates_map.get(p0_id))
        else list(topic_candidates)
    )
    escape_cands = (p0_pool or topic_candidates)[:DEFAULT_ESCAPE_HATCH_K]

    mode_info = determine_operating_mode(
        p0_pool if p0_pool else topic_candidates,
        has_user_capture=bool(user_image_path),
        part_index=0,
    )

    blueprint["wikiSummary"] = ref_bundle.get("summary", {})
    blueprint["references"] = topic_candidates
    blueprint["candidates"] = topic_candidates
    blueprint["funnelStats"] = ref_bundle.get("funnelStats", {})
    blueprint["operatingMode"] = mode_info

    # Surface Top-K Escape Hatch thumbnails & operating mode immediately before 3D synthesis
    emit_event("candidates_ready", {
        "query": effective_query,
        "candidates": escape_cands,
        "stats": blueprint["funnelStats"],
        "operatingMode": mode_info,
    })
    emit_event("references_ready", {
        "wikiSummary": blueprint["wikiSummary"],
        "references": blueprint["references"],
        "candidates": escape_cands,
        "operatingMode": mode_info,
    })

    # Pre-Generation 2.0s Human Gate (when situation telemetry requires or permits human review)
    part0_gate_rejected = False
    deferred_gate_vote_cand = None
    gate_chosen_url = ""
    any_webcam_fallback_needed = False
    failed_attempts_max = 0
    first_failed_reason = ""

    if mode_info.get("gateRequired") and escape_cands and not user_image_path:
        gate = create_gate(
            query=effective_query,
            candidates=escape_cands,
            mode=mode_info["mode"],
            timeout_ms=int(mode_info.get("gateTimeoutMs", 0) or 0),
            default_index=0,
        )
        try:
            emit_event("status", {
                "stage": "human_gate",
                "message": f"HUMAN GATE ({mode_info['mode'].upper()}): Confirm reference for {effective_query}...",
            })
            emit_event("candidate_gate", {
                "gateId": gate.gate_id,
                "query": effective_query,
                "partIndex": 0,
                "partId": p0_id,
                "partName": blueprint_parts[0].get("name") if blueprint_parts else effective_query,
                "candidates": escape_cands,
                "mode": mode_info["mode"],
                "timeoutMs": gate.timeout_ms,
                "allowWebcamRescue": bool(mode_info.get("allowWebcamRescue", True)),
                "reason": mode_info.get("reason", ""),
                "telemetry": mode_info.get("telemetry", {}),
            })

            resolution = gate.wait(is_disconnected_fn=is_disconnected_fn)
            outcome = str(resolution.get("action") or "timeout")
            sel_idx = int(resolution.get("selectedIndex") or 0)
            sel_cand = resolution.get("selectedCandidate")
            sel_cand_pub = _sanitize_candidate_urls(sel_cand) if isinstance(sel_cand, dict) else None

            emit_event("candidate_gate_resolved", {
                "gateId": gate.gate_id,
                "query": effective_query,
                "outcome": outcome,
                "selectedIndex": sel_idx,
                "selectedCandidate": sel_cand_pub,
                "mode": mode_info["mode"],
                "reason": resolution.get("rejectReason", ""),
            })

            if outcome == "disconnected":
                print(f"[*] SSE client disconnected during candidate gate for '{effective_query}' — aborting GPU synthesis.")
                return blueprint

            if outcome == "reject":
                part0_gate_rejected = True
                any_webcam_fallback_needed = True
                first_failed_reason = resolution.get("rejectReason") or "User rejected web candidates at pre-generation gate."
                emit_event("status", {
                    "stage": "human_gate",
                    "message": f"CANDIDATES REJECTED // Switching {effective_query} to camera rescue...",
                })
            elif outcome == "select" and 0 <= sel_idx < len(escape_cands):
                chosen_raw = dict(escape_cands[sel_idx])
                chosen_raw["userPreferred"] = True
                deferred_gate_vote_cand = dict(chosen_raw)
                gate_chosen_url = (chosen_raw.get("thumbUrl") or chosen_raw.get("originalUrl") or "").strip()

                def _mark_choice_in_place(c_list: List[dict]) -> List[dict]:
                    updated = []
                    for c in c_list:
                        cd = dict(c)
                        u1 = (cd.get("thumbUrl") or "").strip()
                        u2 = (cd.get("originalUrl") or "").strip()
                        if gate_chosen_url and (u1 == gate_chosen_url or u2 == gate_chosen_url):
                            cd["userPreferred"] = True
                        else:
                            cd.pop("userPreferred", None)
                        updated.append(cd)
                    return updated

                topic_candidates = _mark_choice_in_place(topic_candidates)
                escape_cands = _mark_choice_in_place(escape_cands)
                if p0_id and p0_id in part_candidates_map:
                    part_candidates_map[p0_id] = _mark_choice_in_place(part_candidates_map[p0_id])
                blueprint["references"] = topic_candidates
                blueprint["candidates"] = topic_candidates
                emit_event("status", {
                    "stage": "synthesis",
                    "message": f"HUMAN GATE LOCKED #{sel_idx + 1} for {effective_query} — reconstructing 3D mesh...",
                })
        finally:
            remove_gate(gate.gate_id)

    elif mode_info.get("mode") == "rescue_webcam_prompt" and not user_image_path:
        part0_gate_rejected = True
        any_webcam_fallback_needed = True
        first_failed_reason = mode_info.get("reason") or "Zero photographic candidates survived Stages 1-4."
        emit_event("candidate_gate_resolved", {
            "gateId": None,
            "query": effective_query,
            "outcome": "rescue_webcam",
            "selectedIndex": -1,
            "selectedCandidate": None,
            "mode": mode_info["mode"],
            "reason": first_failed_reason,
        })

    # 2. TripoSR Reconstruction + Stage 5 Turntable Self-Verification Retry Loop
    total_parts = len(blueprint_parts)
    runtime_used_ref_keys: set = set()

    for idx, part in enumerate(blueprint_parts):
        if is_disconnected_fn is not None:
            try:
                if is_disconnected_fn():
                    print(f"[*] Client disconnected prior to part #{idx} — halting remaining GPU synthesis.")
                    return blueprint
            except Exception:
                pass

        part_name = part.get("name", f"Component {idx + 1}")
        p_id = part.get("id")

        emit_event("mesh_progress", {
            "partIndex": idx,
            "partId": p_id,
            "partName": part_name,
            "status": "generating",
            "progress": int((idx / max(1, total_parts)) * 100),
        })

        if idx == 0 and part0_gate_rejected:
            p_cands = []
        elif p_id in part_candidates_map:
            p_cands = list(part_candidates_map[p_id])
        elif idx == 0:
            p_cands = list(topic_candidates)
        else:
            p_cands = []

        if idx > 0 and runtime_used_ref_keys:
            p_cands = [
                c for c in p_cands
                if c.get("userPreferred") or not (_candidate_identity_keys(c) & runtime_used_ref_keys)
            ]

        part["candidates"] = [
            _sanitize_candidate_urls(serialize_candidate(c))
            for c in p_cands[:DEFAULT_ESCAPE_HATCH_K]
        ]
        synth_cands = list(p_cands)
        if idx == 0 and gate_chosen_url:
            hit_i = next(
                (
                    ci for ci, c in enumerate(synth_cands)
                    if (c.get("thumbUrl") or "").strip() == gate_chosen_url
                    or (c.get("originalUrl") or "").strip() == gate_chosen_url
                ),
                None,
            )
            if hit_i is not None and hit_i > 0:
                synth_cands.insert(0, synth_cands.pop(hit_i))
            elif hit_i is None and deferred_gate_vote_cand is not None:
                synth_cands.insert(0, dict(deferred_gate_vote_cand))
        part.pop("primitives", None)

        verify_query = _build_verify_query(effective_query, part_name, idx)

        def on_verify_attempt(info: dict, _idx=idx, _pid=p_id, _pname=part_name):
            emit_event("verification_update", {
                "partIndex": _idx,
                "partId": _pid,
                "partName": _pname,
                **info,
            })
            if info.get("phase") == "verdict" and not (info.get("verdict") or {}).get("plausible", True):
                att = info.get("attempt", 1)
                tot = info.get("maxAttempts", DEFAULT_ESCAPE_HATCH_K)
                if att < tot:
                    emit_event("status", {
                        "stage": "self_verify",
                        "message": f"STAGE 5 REJECTED #{att} ({_pname}): Retrying candidate #{att + 1}..."
                    })

        # Master outer body (idx==0) tries uploaded visual capture first if supplied
        fb_img = None if (idx == 0 and part0_gate_rejected) else (user_image_path if idx == 0 else None)
        synth_res = synthesize_verified_part_mesh(
            part_name=part_name,
            verify_query=verify_query,
            candidates=synth_cands,
            fallback_image_path=fb_img,
            vlm_verify_fn=run_qwen_stage5_verify,
            resolution=DEFAULT_MESH_RESOLUTION,
            auto_retry=True,
            on_attempt=on_verify_attempt,
            topology=part.get("topology"),
            role=part.get("role"),
            blueprint_title=effective_query,
        )

        part["meshUrl"] = synth_res["meshUrl"]
        part["textureUrl"] = None
        part["normalMapUrl"] = None
        if synth_res.get("selectedCandidate"):
            part["referenceImage"] = _sanitize_candidate_urls(synth_res["selectedCandidate"])
            if not synth_res.get("usedHarmonicFallback"):
                runtime_used_ref_keys.update(_candidate_identity_keys(synth_res["selectedCandidate"]))
        elif part["candidates"]:
            part["referenceImage"] = part["candidates"][0]
            if not synth_res.get("usedHarmonicFallback"):
                runtime_used_ref_keys.update(_candidate_identity_keys(part["candidates"][0]))
        part["verification"] = synth_res.get("verification")
        part["verified3D"] = synth_res.get("verified", False)
        part["usedHarmonicFallback"] = synth_res.get("usedHarmonicFallback", False)

        # Commit deferred human-gate candidate vote ONLY after TripoSR succeeds on that exact choice
        if (
            idx == 0
            and deferred_gate_vote_cand is not None
            and not synth_res.get("usedHarmonicFallback")
            and synth_res.get("attemptIndex") == 0
        ):
            vote_cand = _sanitize_candidate_urls(serialize_candidate(deferred_gate_vote_cand))
            vote_cand["userPreferred"] = True
            if isinstance(part.get("referenceImage"), dict):
                part["referenceImage"]["userPreferred"] = True
            try:
                save_candidate_preference(effective_query, vote_cand, part_id=None)
                if p_id:
                    save_candidate_preference(effective_query, vote_cand, part_id=p_id)
            except Exception as pe:
                print(f"[!] Could not save human-gate preference for '{effective_query}': {pe}")
        elif idx == 0 and deferred_gate_vote_cand is not None and synth_res.get("attemptIndex") != 0:
            for c_list in (part.get("candidates"), blueprint.get("candidates"), blueprint.get("references")):
                if isinstance(c_list, list):
                    for item_c in c_list:
                        if isinstance(item_c, dict):
                            item_c.pop("userPreferred", None)

        if synth_res.get("webcamFallbackNeeded"):
            any_webcam_fallback_needed = True
            failed_attempts_max = max(failed_attempts_max, int(synth_res.get("attemptsMade") or 0))
            if not first_failed_reason:
                first_failed_reason = (synth_res.get("verification") or {}).get("reason", "")

        emit_event("mesh_ready", {
            "partIndex": idx,
            "partId": p_id,
            "partName": part_name,
            "meshUrl": part.get("meshUrl"),
            "textureUrl": None,
            "normalMapUrl": None,
            "referenceImage": part.get("referenceImage"),
            "candidates": part.get("candidates"),
            "verification": part.get("verification"),
            "verified3D": part.get("verified3D"),
            "usedHarmonicFallback": part.get("usedHarmonicFallback"),
        })

    if any_webcam_fallback_needed:
        att_n = failed_attempts_max if failed_attempts_max > 0 else DEFAULT_ESCAPE_HATCH_K
        fallback_msg = (
            first_failed_reason
            if (part0_gate_rejected and failed_attempts_max == 0 and first_failed_reason)
            else f"3D verification uncertain after {att_n} attempt(s) — show me one you can photograph or switch a HUD candidate thumbnail."
        )
        emit_event("webcam_fallback_needed", {
            "query": effective_query,
            "attempts": failed_attempts_max,
            "maxAttempts": max(failed_attempts_max, DEFAULT_ESCAPE_HATCH_K),
            "reason": first_failed_reason or "Candidates failed Stage 5 3D plausibility verification.",
            "message": fallback_msg,
        })

    # 3. Pure Mathematical 3D Geometry Validation & Multi-Body Relaxation
    emit_event("status", {
        "stage": "validation",
        "message": "Executing 3D Multi-Body Non-Penetration & Relaxation Solver..."
    })
    blueprint, val_report = validate_and_repair_blueprint(blueprint, mesh_dir=MESHES_DIR)
    _normalize_blueprint_urls(blueprint)
    emit_event("geometry_validated", {
        "parts": blueprint.get("parts", []),
        "report": val_report
    })

    _atomic_write_json(cache_path, blueprint)
    print(f"[✓] Saved generative 3D blueprint to {cache_path}")

    emit_event("complete", {"blueprint": blueprint})
    return blueprint


def synthesize_cad_assembly(prompt_text: str, image_b64: Optional[str] = None) -> dict:
    """Synchronous REST wrapper around the streaming engine."""
    return synthesize_cad_assembly_stream(prompt_text, image_b64, emit_event=None)


# =========================================================================
# HTTP REQUEST HANDLER & STREAMING TRANSPORT (SSE)
# =========================================================================
class _PayloadTooLargeError(ValueError):
    pass


def _resolve_local_cache_url(cand: dict) -> dict:
    """Convert public /api/cache-image/... URLs back to local disk paths for backend reconstruction."""
    out = dict(cand)
    prefix = "/api/cache-image/"
    for key in ("thumbUrl", "originalUrl"):
        val = out.get(key)
        if isinstance(val, str) and val.startswith(prefix):
            fname = os.path.basename(urllib.parse.unquote(val[len(prefix):]))
            real_base = os.path.realpath(IMAGE_CACHE_DIR)
            real_target = os.path.realpath(os.path.join(real_base, fname))
            if os.path.commonpath([real_base, real_target]) == real_base and os.path.isfile(real_target):
                out[key] = real_target
    return out


class HoloCADHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self._cache_control_sent = False
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def send_response(self, code, message=None):
        self._cache_control_sent = False
        super().send_response(code, message)

    def send_header(self, keyword, value):
        if str(keyword).lower() == "cache-control":
            self._cache_control_sent = True
        super().send_header(keyword, value)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if not getattr(self, "_cache_control_sent", False):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def _send_json(self, status_code: int, payload_obj: object, send_body: bool = True) -> None:
        raw = json.dumps(payload_obj).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if send_body:
            self.wfile.write(raw)

    def _read_json_body(self) -> dict:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            raise ValueError("Missing Content-Length header")
        try:
            content_length = int(raw_len)
        except (TypeError, ValueError):
            raise ValueError("Invalid Content-Length header")
        if content_length < 0:
            raise ValueError("Negative Content-Length header")
        if content_length > MAX_JSON_BODY_BYTES:
            raise _PayloadTooLargeError(f"Request body exceeds limit ({MAX_JSON_BODY_BYTES} bytes)")
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            raise ValueError("Incomplete request body")
        parsed = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object")
        return parsed

    def _serve_directory_file(self, route_prefix: str, base_dir: str, req_path: str, send_body: bool) -> None:
        rel = req_path[len(route_prefix):]
        filename = os.path.basename(urllib.parse.unquote(rel))
        if not filename or filename in (".", ".."):
            self.send_response(404)
            self.end_headers()
            return

        real_base = os.path.realpath(base_dir)
        real_target = os.path.realpath(os.path.join(real_base, filename))
        if os.path.commonpath([real_base, real_target]) != real_base or not os.path.isfile(real_target):
            self.send_response(404)
            self.end_headers()
            return

        lower = filename.lower()
        if lower.endswith(".json"):
            mime = "application/json"
        elif lower.endswith(".png"):
            mime = "image/png"
        elif lower.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif lower.endswith(".webp"):
            mime = "image/webp"
        else:
            mime = "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(os.path.getsize(real_target)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            with open(real_target, "rb") as f:
                self.wfile.write(f.read())

    def _handle_proxy_image(self, query_str: str, send_body: bool) -> None:
        query_params = urllib.parse.parse_qs(query_str)
        target_url = query_params.get("url", [""])[0]
        if not target_url:
            self._send_json(400, {"error": "Missing 'url' query parameter"}, send_body=send_body)
            return
        try:
            data, mime = fetch_or_cache_proxied_image(target_url)
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400, immutable")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if send_body:
                self.wfile.write(data)
        except SSRFSecurityError as e:
            self._send_json(403, {"error": "SSRF violation", "detail": str(e)}, send_body=send_body)
        except InvalidImageMediaError as e:
            self._send_json(415, {"error": "Invalid media type", "detail": str(e)}, send_body=send_body)
        except Exception as e:
            self._send_json(502, {"error": "Proxy fetch failed", "detail": str(e)}, send_body=send_body)

    def _build_health_payload(self) -> dict:
        gpu_name = None
        if device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA GPU"
        return {
            "status": "ready",
            "cognitive_model": MODEL_ID,
            "generative_3d_engine": "TripoSR + Stage-5 Turntable Self-Verification",
            "device": str(_get_model_device()),
            "gpu_name": gpu_name,
            "quantization": "4-bit NF4" if device == "cuda" else "bfloat16",
            "threads": torch.get_num_threads(),
            "default_mesh_resolution": DEFAULT_MESH_RESOLUTION,
            "escape_hatch_candidates": DEFAULT_ESCAPE_HATCH_K,
            "candidate_gate_enabled": True,
            "streaming_transport": "SSE (Server-Sent Events) via /api/synthesize-stream",
        }

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/mesh/"):
            self._serve_directory_file("/api/mesh/", MESHES_DIR, parsed.path, send_body=False)
            return
        if parsed.path.startswith("/api/texture/"):
            self._serve_directory_file("/api/texture/", TEXTURES_DIR, parsed.path, send_body=False)
            return
        if parsed.path.startswith("/api/cache-image/"):
            self._serve_directory_file("/api/cache-image/", IMAGE_CACHE_DIR, parsed.path, send_body=False)
            return
        if parsed.path == "/api/proxy-image":
            self._handle_proxy_image(parsed.query, send_body=False)
            return
        if parsed.path in (
            "/api/health",
            "/api/blueprints",
            "/api/reference",
            "/api/synthesize-stream",
            "/api/select-candidate",
            "/api/confirm-candidate",
        ):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            return
        super().do_HEAD()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/proxy-image":
            self._handle_proxy_image(parsed.query, send_body=True)
            return

        if parsed.path == "/api/reference":
            query_params = urllib.parse.parse_qs(parsed.query)
            query = query_params.get("query", [""])[0]
            if not query:
                self._send_json(400, {"error": "Missing 'query' parameter"})
                return
            data = get_reference_for_topic_and_parts(
                query,
                vlm_judge_fn=run_qwen_stage4_judge,
                query_expand_fn=run_qwen_query_expand,
            )
            self._send_json(200, data)
            return

        if parsed.path == "/api/health":
            self._send_json(200, self._build_health_payload())
            return

        if parsed.path.startswith("/api/mesh/"):
            self._serve_directory_file("/api/mesh/", MESHES_DIR, parsed.path, send_body=True)
            return

        if parsed.path.startswith("/api/texture/"):
            self._serve_directory_file("/api/texture/", TEXTURES_DIR, parsed.path, send_body=True)
            return

        if parsed.path.startswith("/api/cache-image/"):
            self._serve_directory_file("/api/cache-image/", IMAGE_CACHE_DIR, parsed.path, send_body=True)
            return

        if parsed.path == "/api/blueprints":
            blueprints = []
            bp_entries = []
            for fname in os.listdir(BLUEPRINTS_DIR):
                if fname.endswith(".json"):
                    fpath = os.path.join(BLUEPRINTS_DIR, fname)
                    try:
                        bp_entries.append((os.path.getmtime(fpath), fname, fpath))
                    except OSError:
                        pass
            bp_entries.sort(key=lambda item: item[0], reverse=True)

            for _, fname, fpath in bp_entries:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        bp = json.load(f)
                    if not isinstance(bp, dict):
                        continue
                    modified = False
                    if "geometryValidation" not in bp:
                        bp, _ = validate_and_repair_blueprint(bp, mesh_dir=MESHES_DIR)
                        modified = True
                    _normalize_blueprint_urls(bp)
                    if modified:
                        _atomic_write_json(fpath, bp)
                    blueprints.append(bp)
                except Exception as e:
                    print(f"[!] Blueprint load issue with {fname}: {e}")
            self._send_json(200, blueprints)
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/synthesize-stream":
            try:
                payload = self._read_json_body()
            except _PayloadTooLargeError as e:
                self._send_json(413, {"error": str(e)})
                return
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON request: {e}"})
                return

            prompt = str(payload.get("prompt") or "").strip()
            image_b64 = payload.get("image") or None
            if not prompt and not image_b64:
                self._send_json(400, {"error": "Prompt or image is required"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            client_disconnected = False

            def sse_emitter(event_name: str, event_data: dict):
                nonlocal client_disconnected
                if client_disconnected:
                    return
                msg = f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"
                try:
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    client_disconnected = True

            def is_client_disconnected() -> bool:
                nonlocal client_disconnected
                if client_disconnected:
                    return True
                try:
                    if getattr(self.wfile, "closed", False):
                        client_disconnected = True
                        return True
                    sock = getattr(self, "connection", None)
                    if sock is not None:
                        peek = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                        if peek == b"":
                            client_disconnected = True
                            return True
                except BlockingIOError:
                    return False
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    client_disconnected = True
                    return True
                except Exception:
                    return client_disconnected
                return client_disconnected

            try:
                synthesize_cad_assembly_stream(
                    prompt,
                    image_b64,
                    sse_emitter,
                    is_disconnected_fn=is_client_disconnected,
                )
            except Exception as e:
                print(f"[!] Streaming synthesis error: {e}")
                sse_emitter("error", {"error": str(e)})

            self.close_connection = True
            return

        if parsed.path == "/api/confirm-candidate":
            try:
                payload = self._read_json_body()
            except _PayloadTooLargeError as e:
                self._send_json(413, {"error": str(e)})
                return
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON request: {e}"})
                return

            gate_id = str(payload.get("gateId") or "").strip()
            action = str(payload.get("action") or "").strip().lower()
            if not gate_id:
                self._send_json(400, {"error": "Missing 'gateId'"})
                return
            if action not in ("select", "pause", "resume", "reject"):
                self._send_json(400, {"error": f"Unsupported action: {action!r}"})
                return

            cand_idx = payload.get("candidateIndex")
            if cand_idx is not None:
                try:
                    cand_idx = int(cand_idx)
                except (TypeError, ValueError):
                    self._send_json(400, {"error": "Invalid 'candidateIndex'"})
                    return

            cand_url = None
            cand_obj = payload.get("candidate")
            if isinstance(cand_obj, dict):
                cand_url = cand_obj.get("thumbUrl") or cand_obj.get("originalUrl")
            if not cand_url and isinstance(payload.get("candidateUrl"), str):
                cand_url = payload.get("candidateUrl")

            timeout_ms = payload.get("timeoutMs")
            if timeout_ms is not None:
                try:
                    timeout_ms = int(timeout_ms)
                except (TypeError, ValueError):
                    self._send_json(400, {"error": "Invalid 'timeoutMs'"})
                    return

            reason = str(payload.get("reason") or "").strip()
            try:
                snapshot = apply_gate_action(
                    gate_id=gate_id,
                    action=action,
                    candidate_index=cand_idx,
                    candidate_url=cand_url,
                    timeout_ms=timeout_ms,
                    reason=reason,
                )
            except ValueError as ve:
                self._send_json(400, {"error": str(ve)})
                return

            if snapshot is None:
                self._send_json(404, {"error": "Candidate gate not found or already expired", "gateId": gate_id})
                return
            self._send_json(200, snapshot)
            return

        if parsed.path == "/api/synthesize":
            try:
                payload = self._read_json_body()
            except _PayloadTooLargeError as e:
                self._send_json(413, {"error": str(e)})
                return
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON request: {e}"})
                return

            prompt = str(payload.get("prompt") or "").strip()
            image_b64 = payload.get("image") or None
            if not prompt and not image_b64:
                self._send_json(400, {"error": "Prompt or image is required"})
                return

            try:
                blueprint = synthesize_cad_assembly(prompt, image_b64)
                self._send_json(200, blueprint)
            except Exception as e:
                print(f"[!] Synthesis error: {e}")
                self._send_json(500, {"error": str(e)})
            return

        if parsed.path == "/api/select-candidate":
            try:
                payload = self._read_json_body()
            except _PayloadTooLargeError as e:
                self._send_json(413, {"error": str(e)})
                return
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON request: {e}"})
                return

            try:
                query = str(payload.get("query") or payload.get("blueprintTitle") or "").strip()
                try:
                    part_index = int(payload.get("partIndex", 0))
                except (TypeError, ValueError):
                    self._send_json(400, {"error": "Invalid partIndex"})
                    return
                if part_index < 0:
                    self._send_json(400, {"error": "partIndex must be non-negative"})
                    return

                part_id = str(payload.get("partId") or "").strip() or None
                part_name = str(payload.get("partName") or query or "Component").strip()
                cand = payload.get("candidate")

                if not isinstance(cand, dict) or not (cand.get("thumbUrl") or cand.get("originalUrl")):
                    self._send_json(400, {"error": "Valid candidate with thumbUrl/originalUrl required"})
                    return

                cand_public = _sanitize_candidate_urls(serialize_candidate(cand))
                cand_public["userPreferred"] = True
                synth_cand = _resolve_local_cache_url(cand_public)

                # 1. Reconstruct 3D mesh from user-selected candidate (+ Stage 5 check)
                verify_q = _build_verify_query(query, part_name, part_index)
                v_tag = f"v{int(time.time() * 1000)}"
                res = synthesize_verified_part_mesh(
                    part_name=part_name,
                    verify_query=verify_q,
                    candidates=[synth_cand],
                    vlm_verify_fn=run_qwen_stage5_verify,
                    resolution=DEFAULT_MESH_RESOLUTION,
                    auto_retry=False,
                    version_tag=v_tag,
                    topology=payload.get("topology"),
                    role=payload.get("role"),
                    blueprint_title=query,
                )

                # 2. Only memorize vote when candidate actually reconstructed real non-fallback geometry
                pref_saved = False
                if query and not res.get("usedHarmonicFallback"):
                    if part_index == 0:
                        save_candidate_preference(query, cand_public, part_id=None)
                        pref_saved = True
                    if part_id:
                        save_candidate_preference(query, cand_public, part_id=part_id)
                        pref_saved = True

                # 3. Update saved blueprint JSON and re-run geometry validation on updated mesh
                updated_parts = None
                updated_validation = None
                bp_candidates = []
                for cand_key in (query, payload.get("blueprintTitle")):
                    if cand_key and str(cand_key).strip():
                        bp_candidates.append(os.path.join(BLUEPRINTS_DIR, f"{slugify(str(cand_key))}.json"))

                bp_path = next((p for p in bp_candidates if os.path.isfile(p)), None)
                if bp_path:
                    try:
                        with open(bp_path, "r", encoding="utf-8") as bf:
                            bp = json.load(bf)
                        parts = bp.get("parts", [])
                        if 0 <= part_index < len(parts):
                            parts[part_index]["meshUrl"] = res["meshUrl"]
                            parts[part_index]["referenceImage"] = cand_public
                            parts[part_index]["verification"] = res.get("verification")
                            parts[part_index]["verified3D"] = res.get("verified", False)
                            parts[part_index]["usedHarmonicFallback"] = res.get("usedHarmonicFallback", False)
                            bp, updated_validation = validate_and_repair_blueprint(bp, mesh_dir=MESHES_DIR)
                            _normalize_blueprint_urls(bp)
                            updated_parts = bp.get("parts")
                            _atomic_write_json(bp_path, bp)
                    except Exception as ue:
                        print(f"[!] Could not update blueprint cache after candidate vote: {ue}")

                resp_obj = {
                    "query": query,
                    "partIndex": part_index,
                    "partId": part_id,
                    "partName": part_name,
                    "meshUrl": res["meshUrl"],
                    "referenceImage": cand_public,
                    "verification": res.get("verification"),
                    "verified3D": res.get("verified", False),
                    "usedHarmonicFallback": res.get("usedHarmonicFallback", False),
                    "preferenceSaved": pref_saved,
                }
                if updated_parts is not None:
                    resp_obj["parts"] = updated_parts
                if updated_validation is not None:
                    resp_obj["geometryValidation"] = updated_validation

                self._send_json(200, resp_obj)
            except Exception as e:
                print(f"[!] Candidate selection error: {e}")
                self._send_json(500, {"error": str(e)})
            return

        self.send_response(404)
        self.end_headers()


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    port = PORT
    open_browser = False
    for arg in sys.argv[1:]:
        if arg == "--open":
            open_browser = True
        elif not arg.startswith("-"):
            try:
                arg_port = int(arg)
                if 1 <= arg_port <= 65535:
                    port = arg_port
            except ValueError:
                pass

    server = ThreadedServer(("0.0.0.0", port), HoloCADHandler)
    url = f"http://localhost:{port}"
    print(f"[RUNNING] M.A.R.K. Holo-CAD Progressive Streaming Engine active at: {url}")
    if open_browser:
        import webbrowser
        from threading import Timer
        Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
        server.server_close()


if __name__ == "__main__":
    main()
