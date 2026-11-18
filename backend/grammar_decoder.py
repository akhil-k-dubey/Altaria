#!/usr/bin/env python3
"""
Grammar-Constrained Decoding Engine for M.A.R.K. Holo-CAD
Uses llguidance (Rust-accelerated grammar engine) with JSON Schema to guarantee
that Qwen2.5-VL outputs 100% syntactically and structurally valid CAD blueprints.
Prevents malformed JSON, missing brackets, extra markdown, and token waste upfront.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import LogitsProcessor

try:
    import llguidance
    HAS_LLGUIDANCE = True
except ImportError:
    HAS_LLGUIDANCE = False

MIN_BLUEPRINT_PARTS = max(1, int(os.environ.get("HOLO_CAD_MIN_PARTS", "2")))
MAX_BLUEPRINT_PARTS = max(MIN_BLUEPRINT_PARTS, int(os.environ.get("HOLO_CAD_MAX_PARTS", "8")))
MIN_SUMMARY_LINES = max(1, int(os.environ.get("HOLO_CAD_MIN_SUMMARY_LINES", "1")))
MAX_SUMMARY_LINES = max(MIN_SUMMARY_LINES, int(os.environ.get("HOLO_CAD_MAX_SUMMARY_LINES", "4")))
TOPOLOGY_CLASSES = [
    "hollow_shell",
    "rotary_disc",
    "cylindrical_core",
    "conical_taper",
    "branched_conduit",
    "structural_frame",
    "contoured_solid",
]

# Canonical Blueprint JSON Schema for Holographic Spatial Assembly (zero hardcoded domain nouns)
BLUEPRINT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "minLength": 2,
            "description": "System title in uppercase",
        },
        "subtitle": {
            "type": "string",
            "minLength": 2,
            "description": "Summary subtitle such as 'N SUB-ASSEMBLIES LOADED'",
        },
        "overview": {
            "type": "string",
            "minLength": 8,
            "description": "Concise factual sentence describing real-world structure and function",
        },
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9_]{1,40}$",
                        "description": "Unique snake_case identifier",
                    },
                    "name": {
                        "type": "string",
                        "minLength": 2,
                        "description": "Domain-authentic component human-readable name",
                    },
                    "role": {
                        "type": "string",
                        "minLength": 2,
                        "description": "Core physical, biological, or structural role",
                    },
                    "topology": {
                        "type": "string",
                        "enum": TOPOLOGY_CLASSES,
                    },
                    "color": {
                        "type": "string",
                        "pattern": "^#[0-9a-fA-F]{6}$",
                        "description": "High-contrast #RRGGBB hex color code",
                    },
                    "assembledPos": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -2.5, "maximum": 2.5},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "3D position [x, y, z] in normalized unit space",
                    },
                    "explodedVec": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -3.0, "maximum": 3.0},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "3D disassembly vector [dx, dy, dz]",
                    },
                    "rotSpeed": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -0.25, "maximum": 0.25},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "3D continuous spin speed [rx, ry, rz] (zero for static parts)",
                    },
                    "summary": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 4},
                        "minItems": MIN_SUMMARY_LINES,
                        "maxItems": MAX_SUMMARY_LINES,
                        "description": "Concise domain-accurate explanatory points",
                    },
                    "principle": {
                        "type": "string",
                        "minLength": 3,
                        "description": "Governing physical, chemical, or physiological law",
                    },
                },
                "required": [
                    "id",
                    "name",
                    "role",
                    "topology",
                    "color",
                    "assembledPos",
                    "explodedVec",
                    "rotSpeed",
                    "summary",
                    "principle",
                ],
                "additionalProperties": False,
            },
            "minItems": MIN_BLUEPRINT_PARTS,
            "maxItems": MAX_BLUEPRINT_PARTS,
        },
    },
    "required": ["title", "subtitle", "overview", "parts"],
    "additionalProperties": False,
}


JUDGE_MEDIA_TYPES = [
    "real_camera_photograph",
    "illustration_or_drawing",
    "engraving_or_lithograph",
    "cgi_or_3d_render",
    "schematic_or_diagram",
]

JUDGE_WHOLE_SUBJECT_SCOPES = [
    "target_subject",
    "unrelated_or_wrong",
]

JUDGE_SUBPART_SUBJECT_SCOPES = [
    "isolated_target",
    "whole_parent_system",
    "unrelated_or_wrong",
]

JUDGE_SUBJECT_SCOPES = JUDGE_SUBPART_SUBJECT_SCOPES


def _make_vlm_judge_item_schema(max_images: int, is_subpart: bool = False) -> dict:
    scope_enum = JUDGE_SUBPART_SUBJECT_SCOPES if is_subpart else JUDGE_WHOLE_SUBJECT_SCOPES
    return {
        "type": "object",
        "properties": {
            "image": {"type": "integer", "minimum": 1, "maximum": max(1, int(max_images))},
            "reason": {"type": "string"},
            "media_type": {"type": "string", "enum": JUDGE_MEDIA_TYPES},
            "subject_scope": {"type": "string", "enum": scope_enum},
            "scores": {
                "type": "object",
                "properties": {
                    "completeness": {"type": "integer", "minimum": 0, "maximum": 10},
                    "single_subject": {"type": "integer", "minimum": 0, "maximum": 10},
                    "clean_background": {"type": "integer", "minimum": 0, "maximum": 10},
                    "is_photograph": {"type": "integer", "minimum": 0, "maximum": 10},
                    "matches_query": {"type": "integer", "minimum": 0, "maximum": 10},
                },
                "required": [
                    "completeness",
                    "single_subject",
                    "clean_background",
                    "is_photograph",
                    "matches_query",
                ],
                "additionalProperties": False,
            },
            "total": {"type": "integer", "minimum": 0, "maximum": 50},
        },
        "required": ["image", "reason", "media_type", "subject_scope", "scores", "total"],
        "additionalProperties": False,
    }


def make_vlm_judge_schema(num_images: int = 4, is_subpart: bool = False) -> dict:
    """Returns a dynamically parameterized JSON Schema for N-image VLM judging."""
    n = max(1, int(num_images))
    return {
        "type": "object",
        "properties": {
            "evaluations": {
                "type": "array",
                "items": _make_vlm_judge_item_schema(n, is_subpart=is_subpart),
                "minItems": n,
                "maxItems": n,
            },
            "best_image": {"type": "integer", "minimum": 1, "maximum": n},
            "best_reason": {"type": "string"},
        },
        "required": ["evaluations", "best_image", "best_reason"],
        "additionalProperties": False,
    }


VLM_JUDGE_JSON_SCHEMA = make_vlm_judge_schema(6)

QUERY_EXPAND_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
            "description": "Concise Wikimedia Commons photographic search phrases for the component",
        },
    },
    "required": ["queries"],
    "additionalProperties": False,
}


VLM_VERIFY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "shape_summary": {"type": "string"},
        "reason": {"type": "string"},
        "plausible": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 10},
    },
    "required": ["shape_summary", "reason", "plausible", "confidence"],
    "additionalProperties": False,
}


def extract_json_with_bracket_depth(text: str):
    """
    Pure bracket-depth + JSONDecoder.raw_decode extractor (zero regex).
    Safely skips preambles/markdown wrappers and respects quoted escapes.
    Supports both JSON objects {...} and arrays [...].
    """
    if not text or not isinstance(text, str):
        raise ValueError("Empty input for JSON bracket-depth extraction")

    decoder = json.JSONDecoder()
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        if ch in "{[":
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            depth = 0
            in_str = False
            escaped = False
            for j in range(i, n):
                c = text[j]
                if in_str:
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == '"':
                        in_str = False
                    continue

                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            try:
                                obj, _ = decoder.raw_decode(text, i)
                                return obj
                            except json.JSONDecodeError:
                                break
        i += 1

    stripped = text.strip()
    for idx, ch in enumerate(stripped):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(stripped, idx)
                return obj
            except json.JSONDecodeError:
                continue

    raise ValueError("No balanced JSON object or array found in text")


class HFTokenizerWrapper:
    """Adapts HuggingFace Tokenizer to llguidance TokenizerWrapper interface."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.special_token_ids = list(tokenizer.all_special_ids)
        self.is_tokenizer_wrapper = True

        vocab = tokenizer.get_vocab()
        sorted_tokens = [None] * len(vocab)
        for k, v in vocab.items():
            if v < len(sorted_tokens):
                try:
                    sorted_tokens[v] = tokenizer.convert_tokens_to_string([k]).encode("utf-8")
                except Exception:
                    sorted_tokens[v] = k.encode("utf-8", errors="ignore")
        self.tokens = [t if t is not None else b"" for t in sorted_tokens]

    def __call__(self, s):
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="ignore")
        return self.tok.encode(s, add_special_tokens=False)


class LLMatcherLogitsProcessor(LogitsProcessor):
    """
    Logits processor enforcing JSON Schema grammar constraints at each token step.
    Disallows any token that would create invalid JSON or deviate from the schema.
    """

    def __init__(self, lltok, grammar, eos_token_id=None):
        self.matcher = llguidance.LLMatcher(lltok, grammar)
        self.eos_token_id = eos_token_id
        self.started = False
        self.curr_len = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input_ids.shape[1]
        if not self.started:
            self.started = True
            self.curr_len = seq_len
        elif seq_len > self.curr_len:
            for idx in range(self.curr_len, seq_len):
                tok_val = int(input_ids[0, idx].item())
                self.matcher.consume_token(tok_val)
            self.curr_len = seq_len

        neg_inf = float(torch.finfo(scores.dtype).min) if scores.dtype.is_floating_point else -1e9

        if self.matcher.is_stopped() or self.matcher.is_accepting():
            if self.eos_token_id is not None and 0 <= self.eos_token_id < scores.shape[1]:
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[:, self.eos_token_id] = True
                scores[:, ~mask[0]] = neg_inf
                return scores

        bias = self.matcher.compute_logit_bias()
        arr = np.frombuffer(bias, dtype=np.uint8).copy()
        mask_t = torch.from_numpy(arr != 0).to(device=scores.device, dtype=torch.bool)

        if scores.shape[1] > mask_t.shape[0]:
            extra_len = scores.shape[1] - mask_t.shape[0]
            extra_mask = torch.zeros(extra_len, dtype=torch.bool, device=scores.device)
            mask_t = torch.cat([mask_t, extra_mask])
        elif scores.shape[1] < mask_t.shape[0]:
            mask_t = mask_t[: scores.shape[1]]

        if mask_t.any():
            scores[:, ~mask_t] = neg_inf
        return scores


_LLTOK_CACHE: Dict[int, object] = {}
_GRAMMAR_CACHE: Dict[Tuple[int, str], object] = {}


def _get_lltokenizer(tokenizer):
    key = id(tokenizer)
    if key not in _LLTOK_CACHE:
        wrapper = HFTokenizerWrapper(tokenizer)
        _LLTOK_CACHE[key] = llguidance.LLTokenizer(wrapper)
    return _LLTOK_CACHE[key]


def build_grammar_processor(tokenizer, schema=None, label: str = "schema"):
    """
    Builds a grammar-constrained LogitsProcessor for any JSON Schema,
    caching compiled tokenizer & grammar tables across calls for near-0ms startup.
    """
    if not HAS_LLGUIDANCE:
        print("[!] llguidance is not available; falling back to post-generation schema validation.")
        return None

    if schema is None:
        schema = BLUEPRINT_JSON_SCHEMA

    schema_str = json.dumps(schema, sort_keys=False)
    try:
        lltok = _get_lltokenizer(tokenizer)
        cache_key = (id(tokenizer), schema_str)
        grammar = _GRAMMAR_CACHE.get(cache_key)
        if grammar is None:
            grammar = llguidance.LLMatcher.grammar_from_json_schema(schema_str)
            _GRAMMAR_CACHE[cache_key] = grammar
        processor = LLMatcherLogitsProcessor(lltok, grammar, eos_token_id=tokenizer.eos_token_id)
        print(f"[✓] Grammar-constrained decoding active: llguidance '{label}' loaded.")
        return processor
    except Exception as e:
        print(f"[!] Warning: Could not initialize llguidance grammar processor ({label}): {e}")
        return None


def build_cad_grammar_processor(tokenizer, schema=None):
    """Backwards-compatible wrapper for CAD blueprint schema decoding."""
    return build_grammar_processor(tokenizer, schema=schema, label="cad_blueprint")


def _locate_top_level_key_and_array(text: str, target_key: str) -> Tuple[int, int]:
    """
    Pure lexical state-machine scanner (zero regex, zero naive substring matching).
    Locates a top-level object key ("parts") at brace depth 1 and returns
    (key_quote_start_index, array_open_bracket_index).
    Immune to target_key appearing inside string values like "overview".
    """
    n = len(text)
    i = 0
    brace_depth = 0
    bracket_depth = 0

    while i < n:
        ch = text[i]
        if ch == '"':
            str_start = i
            i += 1
            chars: List[str] = []
            escaped = False
            while i < n:
                c = text[i]
                if escaped:
                    chars.append(c)
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    break
                else:
                    chars.append(c)
                i += 1
            if i >= n:
                return -1, -1
            str_val = "".join(chars)
            i += 1  # advance past closing quote

            if brace_depth == 1 and bracket_depth == 0 and str_val == target_key:
                k = i
                while k < n and text[k] in " \t\r\n":
                    k += 1
                if k < n and text[k] == ":":
                    k += 1
                    while k < n and text[k] in " \t\r\n":
                        k += 1
                    if k < n and text[k] == "[":
                        return str_start, k
                    return str_start, -1
            continue

        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        i += 1

    return -1, -1


class IncrementalBlueprintParser:
    """
    Decodes structured blueprint sub-assemblies progressively from streaming token chunks.
    Uses a top-level lexical scanner + json.JSONDecoder.raw_decode without regex.
    """

    def __init__(self):
        self.buffer = ""
        self.header_emitted = False
        self.parts_discovered: List[dict] = []
        self.decoder = json.JSONDecoder()
        self.cursor = 0
        self._parts_key_pos = -1
        self._parts_bracket_pos = -1

    def feed(self, chunk: str) -> List[Tuple[str, dict]]:
        self.buffer += chunk
        events: List[Tuple[str, dict]] = []

        if self._parts_key_pos == -1 or self._parts_bracket_pos == -1:
            self._parts_key_pos, self._parts_bracket_pos = _locate_top_level_key_and_array(
                self.buffer, "parts"
            )

        # 1. Emit top-level header once the top-level "parts" key is reached
        if not self.header_emitted and self._parts_key_pos != -1:
            prefix = self.buffer[: self._parts_key_pos].rstrip(", \t\r\n") + "}"
            try:
                h = json.loads(prefix)
                if isinstance(h, dict):
                    raw_sub = str(h.get("subtitle", "") or "").strip()
                    safe_sub = raw_sub if (len(raw_sub) >= 4 and not raw_sub.isdigit()) else "MATERIALIZING SUB-ASSEMBLIES..."
                    events.append(("header", {
                        "title": h.get("title", "SPATIAL ASSEMBLY"),
                        "subtitle": safe_sub,
                        "overview": h.get("overview", ""),
                    }))
                    self.header_emitted = True
            except Exception:
                pass

        # 2. Progressively decode completed sub-assemblies inside the top-level "parts" array
        if self._parts_bracket_pos != -1:
            if self.cursor <= self._parts_bracket_pos:
                self.cursor = self._parts_bracket_pos + 1

            while self.cursor < len(self.buffer):
                while self.cursor < len(self.buffer) and self.buffer[self.cursor] in " \t\r\n,":
                    self.cursor += 1

                if self.cursor < len(self.buffer) and self.buffer[self.cursor] == "{":
                    try:
                        part_obj, end_pos = self.decoder.raw_decode(self.buffer, self.cursor)
                        part_idx = len(self.parts_discovered)
                        self.parts_discovered.append(part_obj)
                        events.append(("part_discovered", {
                            "partIndex": part_idx,
                            "part": part_obj,
                        }))
                        self.cursor = end_pos
                    except json.JSONDecodeError:
                        break
                else:
                    break

        return events