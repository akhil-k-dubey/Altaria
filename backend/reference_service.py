#!/usr/bin/env python3
"""
M.A.R.K. Holo-CAD // Real-World Diagnostic Reference Service
=============================================================
Zero-Hardcoding, Zero-Regex 5-Stage Candidate Funnel Coordinator:
- Tiered Acquisition: Wikipedia canonical lead & summary > curated Commons Categories > Commons search
- Stage 1: Metadata sieve (JPEG/PNG only, non-photo filenames rejected, AR & min width bounds)
- Stage 2: Scientific CV signals (Otsu separability, Sobel/Laplace sharpness, CIE Lab chroma, Shannon entropy)
- Stage 3: CLIP ViT-B/32 centroid Bayes filter + MMR cosine diversity
- Stage 4: Single-call Qwen2.5-VL multi-image judge + optional VLM semantic query expansion
- Strictly zero whole-object photo cloning onto internal sub-parts.
"""

import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from image_funnel import (
    acquire_candidates,
    determine_operating_mode,
    get_candidate_preference,
    run_candidate_funnel,
    serialize_candidate,
    slugify as _funnel_slugify,
    stage1_metadata_filter,
    stage2_cv_filter,
    stage3_clip_filter,
    stage4_vlm_judge,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_CACHE_DIR = os.path.join(BASE_DIR, "data", "references")
os.makedirs(REF_CACHE_DIR, exist_ok=True)

FUNNEL_CACHE_VERSION = int(os.environ.get("HOLO_CAD_FUNNEL_CACHE_VER", "11"))
TOPIC_MAX_RAW = max(40, int(os.environ.get("HOLO_CAD_TOPIC_MAX_RAW", "140")))
TOPIC_MAX_STAGE1 = max(16, int(os.environ.get("HOLO_CAD_TOPIC_MAX_S1", "50")))
TOPIC_MAX_DOWNLOADS = max(8, int(os.environ.get("HOLO_CAD_TOPIC_MAX_DL", "30")))
TOPIC_STAGE2_TOP_K = max(4, int(os.environ.get("HOLO_CAD_TOPIC_S2_TOPK", "12")))
TOPIC_STAGE3_TOP_K = max(2, int(os.environ.get("HOLO_CAD_TOPIC_S3_TOPK", "4")))
PART_CANDIDATE_LIMIT = max(1, int(os.environ.get("HOLO_CAD_CANDIDATE_K", "3")))
PART_POOL_WORKERS = max(1, int(os.environ.get("HOLO_CAD_PART_WORKERS", "2")))
PART_ACQUIRE_TIMEOUT_S = max(10, int(os.environ.get("HOLO_CAD_PART_TIMEOUT_S", "90")))

MIN_JUDGE_MATCH_SCORE = float(os.environ.get("HOLO_CAD_MIN_JUDGE_MATCH", "5.0"))
MIN_JUDGE_PHOTO_SCORE = float(os.environ.get("HOLO_CAD_MIN_JUDGE_PHOTO", "5.0"))
MIN_CLIP_SUBJ_MARGIN = float(os.environ.get("HOLO_CAD_MIN_CLIP_MARGIN", "0.0"))

USER_AGENT = os.environ.get(
    "HOLO_CAD_USER_AGENT",
    "StarkHoloCAD-Diagnostic/3.2 (https://commons.wikimedia.org; free academic 3D CAD research)",
)
HEADERS = {"User-Agent": USER_AGENT}


def slugify(text: str) -> str:
    """Delegate directly to canonical character-state slugifier."""
    return _funnel_slugify(text)


def normalize_phrase(text: str) -> str:
    """Collapses underscores, hyphens, and whitespace into single spaces without regex."""
    cleaned_chars = [
        " " if ch in ("_", "-", "\t", "\r", "\n") else ch
        for ch in str(text or "")
    ]
    return " ".join("".join(cleaned_chars).split())


def _candidate_identity_keys(cand: Optional[dict]) -> set:
    """Normalizes canonicalFile and thumbnail/original URL stems (zero regex) to block cross-part cloning."""
    keys = set()
    if not isinstance(cand, dict):
        return keys
    cf = str(cand.get("canonicalFile") or "").lower().strip()
    if cf:
        keys.add(cf)
        if cf.startswith("file:"):
            keys.add(cf[5:].strip().replace("_", " "))
    for url_key in ("thumbUrl", "originalUrl"):
        u = str(cand.get(url_key) or "").strip()
        if not u:
            continue
        keys.add(u.lower())
        path = urllib.parse.unquote(urllib.parse.urlparse(u).path or "")
        fname = os.path.basename(path).replace("_", " ").strip().lower()
        if fname:
            px_pos = fname.find("px-")
            if px_pos > 0 and fname[:px_pos].isdigit():
                fname = fname[px_pos + 3:]
            keys.add(fname)
    return keys


def _is_verified_photo_candidate(cand: dict) -> bool:
    """
    Returns True iff candidate passed both Stage-3 CLIP authenticity (if scored)
    and Stage-4 Qwen2.5-VL rubric gates (is_photograph >= MIN, matches_query >= MIN).
    Zero fake positive defaults when VLM actually flagged a rejection.
    """
    if cand.get("userPreferred"):
        return True
    if cand.get("isVlmRejected"):
        return False

    clip_s = cand.get("clipScore")
    if isinstance(clip_s, dict):
        if not clip_s.get("isPhotoPass", False):
            return False
        if float(clip_s.get("subjMargin", 0.0)) < MIN_CLIP_SUBJ_MARGIN:
            return False

    js = cand.get("judgeScores")
    if isinstance(js, dict) and js:
        if float(js.get("is_photograph", 0.0)) < MIN_JUDGE_PHOTO_SCORE:
            return False
        if float(js.get("matches_query", 0.0)) < MIN_JUDGE_MATCH_SCORE:
            return False

    return True


def fetch_wikipedia_summary(topic: str, _allow_opensearch: bool = True) -> Dict:
    """Fetch official encyclopedic summary and high-res lead image from Wikipedia REST API."""
    clean_topic = normalize_phrase(topic)
    if not clean_topic:
        return {"title": topic, "extract": "", "thumbnail": None, "originalImage": None}

    encoded_topic = urllib.parse.quote(clean_topic)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
    req = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "title": data.get("title", clean_topic),
                "description": data.get("description", ""),
                "extract": data.get("extract", ""),
                "thumbnail": (data.get("thumbnail") or {}).get("source"),
                "originalImage": (data.get("originalimage") or {}).get("source"),
                "pageUrl": ((data.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
            }
    except Exception:
        if _allow_opensearch:
            try:
                search_url = (
                    f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_topic}"
                    "&limit=1&namespace=0&format=json"
                )
                req2 = urllib.request.Request(search_url, headers=HEADERS)
                with urllib.request.urlopen(req2, timeout=4) as resp2:
                    data2 = json.loads(resp2.read().decode("utf-8"))
                    if isinstance(data2, list) and len(data2) >= 2 and data2[1]:
                        return fetch_wikipedia_summary(str(data2[1][0]), _allow_opensearch=False)
            except Exception:
                pass
        return {"title": clean_topic, "extract": "", "thumbnail": None, "originalImage": None}


def fetch_wikimedia_commons_images(query: str, limit: int = 8) -> List[Dict]:
    """Fetches vetted photographic candidates via Stage 0-3 of the funnel."""
    safe_limit = max(1, int(limit))
    raw, _ = acquire_candidates(query, max_candidates=max(50, safe_limit * 8))
    s1 = stage1_metadata_filter(raw, max_survivors=max(24, safe_limit * 4))
    s2 = stage2_cv_filter(s1, max_downloads=max(16, safe_limit * 2), top_k=max(8, safe_limit))
    s3 = stage3_clip_filter(s2, query=query, top_k=safe_limit)
    return [serialize_candidate(c) for c in s3 if _is_verified_photo_candidate(c)]


def _interleave_stage1_lists(lists: List[List[Dict]], limit: int = 36) -> List[Dict]:
    """Round-robin merges Stage-1 candidate lists across queries while deduplicating normalized files."""
    merged: List[Dict] = []
    seen_keys: set = set()
    max_len = max((len(lst) for lst in lists), default=0)
    for pos in range(max_len):
        for lst in lists:
            if pos >= len(lst):
                continue
            cand = lst[pos]
            ikeys = _candidate_identity_keys(cand)
            if not ikeys:
                fallback_k = str(cand.get("canonicalFile") or cand.get("thumbUrl") or "").lower().strip()
                if fallback_k:
                    ikeys = {fallback_k}
            if ikeys & seen_keys:
                continue
            seen_keys.update(ikeys)
            merged.append(cand)
            if len(merged) >= limit:
                return merged
    return merged


def _acquire_stage0_to_2_for_queries(
    queries: List[str],
    max_downloads: int = 16,
    top_k: int = 8,
    exclude_categories: Optional[set] = None,
    exclude_wiki_titles: Optional[set] = None,
    prefer_search: bool = False,
) -> List[Dict]:
    s1_buckets: List[List[Dict]] = []
    seen_q: set = set()
    for q in queries:
        clean_q = normalize_phrase(q)
        if not clean_q or clean_q.lower() in seen_q:
            continue
        seen_q.add(clean_q.lower())
        raw, _ = acquire_candidates(
            clean_q,
            max_candidates=min(40, TOPIC_MAX_STAGE1),
            exclude_categories=exclude_categories,
            exclude_wiki_titles=exclude_wiki_titles,
            prefer_search=prefer_search,
        )
        s1 = stage1_metadata_filter(
            raw, max_survivors=max(12, TOPIC_MAX_STAGE1 // 3), prefer_search=prefer_search
        )
        if s1:
            s1_buckets.append(s1)
    if not s1_buckets:
        return []
    merged_s1 = _interleave_stage1_lists(s1_buckets, limit=max(24, max_downloads * 2))
    return stage2_cv_filter(merged_s1, max_downloads=max_downloads, top_k=top_k)


def _prepare_part_stage0_to_2(
    topic: str,
    part: Dict,
    exclude_categories: Optional[set] = None,
    exclude_wiki_titles: Optional[set] = None,
) -> Tuple[str, str, List[Dict]]:
    p_name = str(part.get("name") or "").strip()
    p_id = str(part.get("id") or slugify(p_name))
    clean_pname = normalize_phrase(p_name)
    clean_topic = normalize_phrase(topic)
    if not clean_pname:
        return p_id, "", []

    is_sub = bool(clean_pname.lower() != clean_topic.lower())
    if not is_sub:
        raw, _ = acquire_candidates(
            clean_topic,
            max_candidates=TOPIC_MAX_STAGE1,
            exclude_categories=None,
            exclude_wiki_titles=None,
            prefer_search=False,
        )
        s1 = stage1_metadata_filter(raw, max_survivors=max(16, TOPIC_MAX_STAGE1 // 2), prefer_search=False)
        s2 = stage2_cv_filter(s1, max_downloads=16, top_k=max(10, PART_CANDIDATE_LIMIT * 3))
        return p_id, clean_pname, s2

    # Run context-anchored query first so homonyms never drag off-topic images or flood Wikimedia;
    # only widen to standalone/reversed queries when Stage-1 survivors are sparse.
    anchored = clean_topic.lower() in clean_pname.lower()
    q0_text = clean_pname if anchored else f"{clean_pname} {clean_topic}".strip()
    s1_buckets: List[List[Dict]] = []
    seen_q: set = set()
    total_s1 = 0

    def _run_sub_query(q_text: str, pref_search: bool) -> None:
        nonlocal total_s1
        q_norm = q_text.lower().strip()
        if not q_norm or q_norm in seen_q:
            return
        seen_q.add(q_norm)
        raw, _ = acquire_candidates(
            q_text,
            max_candidates=32,
            exclude_categories=exclude_categories,
            exclude_wiki_titles=exclude_wiki_titles,
            prefer_search=pref_search,
        )
        s1_part = stage1_metadata_filter(raw, max_survivors=16, prefer_search=pref_search)
        if s1_part:
            s1_buckets.append(s1_part)
            total_s1 += len(s1_part)

    _run_sub_query(q0_text, True)
    if not anchored and total_s1 < 12:
        _run_sub_query(clean_pname, True)
    if not anchored and total_s1 < 10:
        _run_sub_query(f"{clean_topic} {clean_pname}".strip(), True)

    if not s1_buckets:
        return p_id, clean_pname, []

    merged_s1 = _interleave_stage1_lists(s1_buckets, limit=28)
    s2 = stage2_cv_filter(merged_s1, max_downloads=14, top_k=max(10, PART_CANDIDATE_LIMIT * 4))
    return p_id, clean_pname, s2


def _apply_learned_preference(query: str, cand_list: List[Dict], part_id: Optional[str] = None) -> List[Dict]:
    pref = get_candidate_preference(query, part_id=part_id)
    if not pref:
        return cand_list
    pref_url = pref.get("thumbUrl") or pref.get("originalUrl")
    if not pref_url:
        return cand_list

    reordered = [dict(c) for c in cand_list]
    match_idx = next(
        (i for i, c in enumerate(reordered) if c.get("thumbUrl") == pref_url or c.get("originalUrl") == pref_url),
        None,
    )
    if match_idx is not None:
        item = reordered.pop(match_idx)
        item["userPreferred"] = True
        reordered.insert(0, item)
    else:
        reordered.insert(0, {
            "title": pref.get("title", "User Preferred Reference"),
            "thumbUrl": pref.get("thumbUrl") or pref_url,
            "originalUrl": pref.get("originalUrl") or pref_url,
            "mime": pref.get("mime", "image/jpeg"),
            "caption": pref.get("caption") or f"User locked reference ({query})",
            "sourceTier": 0,
            "sourceLabel": "user_preference_cache",
            "userPreferred": True,
        })
    return reordered


def _filter_and_judge_part_s2(
    s2_list: List[Dict],
    part_query: str,
    judge_query: str,
    part_context: str,
    vlm_judge_fn: Optional[Callable],
    top_k: int,
) -> List[Dict]:
    if not s2_list:
        return []
    clip_query = judge_query if (part_context and part_context.lower() != judge_query.lower()) else part_query
    s3_raw = stage3_clip_filter(s2_list, query=clip_query, top_k=top_k, part_context=part_context)
    s3_photos = [
        c for c in s3_raw
        if c.get("clipScore", {}).get("isPhotoPass", False)
        and float(c.get("clipScore", {}).get("subjMargin", 0.0)) >= MIN_CLIP_SUBJ_MARGIN
        and float(c.get("clipScore", {}).get("subjVsPhotoMargin", 0.0)) >= -0.5
    ]
    if not s3_photos:
        return []

    s4_list = stage4_vlm_judge(
        s3_photos,
        query=judge_query,
        vlm_judge_fn=vlm_judge_fn,
        max_candidates=top_k,
        part_context=part_context,
    )
    return [serialize_candidate(c) for c in s4_list if _is_verified_photo_candidate(c)]


def get_reference_for_topic_and_parts(
    topic: str,
    parts: Optional[List[Dict]] = None,
    vlm_judge_fn: Optional[Callable] = None,
    query_expand_fn: Optional[Callable[[str, str, str], List[str]]] = None,
    on_stage: Optional[Callable[[str, dict], None]] = None,
) -> Dict:
    """
    Runs the full Zero-Hardcoding Candidate Funnel for the topic and each assembly part.
    When an initial search produces zero verified photographic candidates (and query_expand_fn is
    provided), asks Qwen2.5-VL for alternative search terms and re-runs Stages 0-4.
    Never copies whole-object images onto internal sub-parts that have no verified photos.
    """
    clean_topic = normalize_phrase(topic)
    if not clean_topic:
        return {
            "funnelVersion": FUNNEL_CACHE_VERSION,
            "topic": "",
            "summary": {"title": "", "extract": "", "thumbnail": None, "originalImage": None},
            "images": [],
            "candidates": [],
            "partReferences": {},
            "partCandidates": {},
            "funnelStats": {},
            "operatingMode": determine_operating_mode([]),
        }

    slug = slugify(clean_topic)
    cache_path = os.path.join(REF_CACHE_DIR, f"{slug}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("funnelVersion") == FUNNEL_CACHE_VERSION
                and "summary" in cached
                and "candidates" in cached
                and (not parts or len(cached.get("partCandidates", {})) >= len(parts))
            ):
                cached["candidates"] = _apply_learned_preference(clean_topic, cached.get("candidates", []))
                cached["images"] = cached["candidates"]
                cached["operatingMode"] = determine_operating_mode(cached["candidates"])
                if cached.get("partCandidates"):
                    for p_id, pc_list in cached["partCandidates"].items():
                        upd = _apply_learned_preference(clean_topic, pc_list, part_id=p_id)
                        cached["partCandidates"][p_id] = upd
                        if upd:
                            cached.setdefault("partReferences", {})[p_id] = upd[0]
                return cached
        except Exception:
            pass

    # 1. Master topic funnel
    topic_funnel = run_candidate_funnel(
        clean_topic,
        vlm_judge_fn=vlm_judge_fn,
        max_raw=TOPIC_MAX_RAW,
        max_stage1=TOPIC_MAX_STAGE1,
        max_downloads=TOPIC_MAX_DOWNLOADS,
        stage2_top_k=TOPIC_STAGE2_TOP_K,
        stage3_top_k=TOPIC_STAGE3_TOP_K,
        on_stage=on_stage,
    )

    summary = topic_funnel.get("wikiSummary") or {}
    if not summary.get("extract"):
        rest_sum = fetch_wikipedia_summary(clean_topic)
        if rest_sum.get("extract"):
            summary = rest_sum

    primary_pool = list(topic_funnel.get("candidates", []))
    topic_verified_raw = [
        c for c in primary_pool if _is_verified_photo_candidate(c)
    ]

    # Dynamic Qwen query expansion if primary topic search returned zero verified photographs
    expanded_pool: List[Dict] = []
    if not topic_verified_raw and query_expand_fn is not None:
        try:
            expanded_qs = query_expand_fn(clean_topic, clean_topic, str(summary.get("extract", "")))
            if expanded_qs:
                s2_exp = _acquire_stage0_to_2_for_queries(
                    expanded_qs,
                    max_downloads=max(12, TOPIC_MAX_DOWNLOADS // 2),
                    top_k=TOPIC_STAGE2_TOP_K,
                )
                if s2_exp:
                    s3_exp = stage3_clip_filter(s2_exp, query=clean_topic, top_k=TOPIC_STAGE3_TOP_K)
                    s4_exp = stage4_vlm_judge(
                        s3_exp,
                        query=clean_topic,
                        vlm_judge_fn=vlm_judge_fn,
                        max_candidates=TOPIC_STAGE3_TOP_K,
                    )
                    expanded_pool = list(s4_exp)
                    topic_verified_raw = [c for c in s4_exp if _is_verified_photo_candidate(c)]
        except Exception as eq_err:
            print(f"[!] Topic query expansion warning for '{clean_topic}': {eq_err}")

    topic_candidates = [serialize_candidate(c) for c in topic_verified_raw]

    # Master-only backfill up to PART_CANDIDATE_LIMIT (strictly NEVER cloned onto internal parts[1:])
    if len(topic_candidates) < PART_CANDIDATE_LIMIT:
        seen_urls = {
            (c.get("thumbUrl") or c.get("originalUrl") or "").strip()
            for c in topic_candidates
            if (c.get("thumbUrl") or c.get("originalUrl"))
        }
        fallback_pool = [
            c for c in (primary_pool + expanded_pool)
            if (c.get("clipScore") or {}).get("isPhotoPass", False)
        ]
        # Prefer non-VLM-rejected Stage-3 photo survivors ahead of borderline Stage-4 picks
        fallback_pool.sort(
            key=lambda c: (not bool(c.get("isVlmRejected", False)), float(c.get("stage3Score", 0.0))),
            reverse=True,
        )
        for fb_cand in fallback_pool:
            if len(topic_candidates) >= PART_CANDIDATE_LIMIT:
                break
            ser_fb = serialize_candidate(fb_cand)
            fb_key = (ser_fb.get("thumbUrl") or ser_fb.get("originalUrl") or "").strip()
            if not fb_key or fb_key in seen_urls:
                continue
            seen_urls.add(fb_key)
            ser_fb["isGateFallback"] = True
            topic_candidates.append(ser_fb)

    topic_candidates = _apply_learned_preference(clean_topic, topic_candidates)
    operating_mode = determine_operating_mode(topic_candidates)

    # 2. Per-part concurrent acquisition + CV -> CLIP -> VLM Judge (strict zero cross-cloning)
    part_references: Dict[str, Dict] = {}
    part_candidates: Dict[str, List[Dict]] = {}
    claimed_keys: set = set()
    parent_wiki_titles: set = {clean_topic.lower()}
    parent_categories: set = set()
    if isinstance(summary, dict):
        if summary.get("title"):
            parent_wiki_titles.add(str(summary["title"]).strip().lower())
        if summary.get("commonsCategory"):
            parent_categories.add(str(summary["commonsCategory"]).strip().lower())
        claimed_keys.update(_candidate_identity_keys({
            "thumbUrl": summary.get("thumbnail"),
            "originalUrl": summary.get("originalImage"),
        }))
    for tc in primary_pool:
        if int(tc.get("sourceTier", 3)) == 1:
            claimed_keys.update(_candidate_identity_keys(tc))
        sl = str(tc.get("sourceLabel") or "")
        if sl.startswith("commons_cat:"):
            parent_categories.add(sl[len("commons_cat:"):].strip().lower())

    if parts:
        first_part = parts[0]
        p0_id = str(first_part.get("id") or slugify(str(first_part.get("name", "part_1"))))
        p0_cands = _apply_learned_preference(clean_topic, [dict(c) for c in topic_candidates], part_id=p0_id)
        part_candidates[p0_id] = p0_cands[:PART_CANDIDATE_LIMIT]
        if part_candidates[p0_id]:
            part_references[p0_id] = part_candidates[p0_id][0]
            for reserved_c in part_candidates[p0_id]:
                claimed_keys.update(_candidate_identity_keys(reserved_c))

        sub_parts = list(enumerate(parts))[1:]
        sub_s2_map: Dict[int, Tuple[str, str, List[Dict]]] = {}
        if sub_parts:
            with ThreadPoolExecutor(max_workers=min(PART_POOL_WORKERS, len(sub_parts))) as pool:
                fut_map = {
                    idx: pool.submit(
                        _prepare_part_stage0_to_2,
                        clean_topic,
                        p,
                        parent_categories,
                        parent_wiki_titles,
                    )
                    for idx, p in sub_parts
                }
                for idx, p in sub_parts:
                    p_name = str(p.get("name") or "").strip()
                    p_id = str(p.get("id") or slugify(p_name or f"part_{idx + 1}"))
                    try:
                        sub_s2_map[idx] = fut_map[idx].result(timeout=PART_ACQUIRE_TIMEOUT_S)
                    except Exception as pe:
                        print(f"[!] Per-part reference acquisition warning for '{p_name}' ({p_id}): {pe}")
                        sub_s2_map[idx] = (p_id, p_name, [])

        for idx, p in sub_parts:
            p_name = str(p.get("name") or "").strip()
            fallback_pid = str(p.get("id") or slugify(p_name or f"part_{idx + 1}"))
            p_id, resolved_name, s2_list = sub_s2_map.get(idx, (fallback_pid, p_name, []))
            judge_name = resolved_name or clean_topic
            part_query = (
                f"{judge_name} ({clean_topic})"
                if judge_name.lower() != clean_topic.lower()
                else clean_topic
            )

            s2_distinct = [c for c in s2_list if not (_candidate_identity_keys(c) & claimed_keys)]
            ser_list = _filter_and_judge_part_s2(
                s2_distinct,
                part_query=part_query,
                judge_query=judge_name,
                part_context=clean_topic,
                vlm_judge_fn=vlm_judge_fn,
                top_k=PART_CANDIDATE_LIMIT,
            )

            # Retry with Qwen semantic query expansion before admitting empty sub-part photo set
            if not ser_list and query_expand_fn is not None and judge_name:
                try:
                    raw_sum = p.get("summary")
                    desc = " ".join(raw_sum) if isinstance(raw_sum, list) else str(p.get("role") or "")
                    exp_queries = query_expand_fn(clean_topic, judge_name, desc)
                    if exp_queries:
                        seen_files: set = set(claimed_keys)
                        for c in s2_list:
                            seen_files.update(_candidate_identity_keys(c))
                        s2_extra = _acquire_stage0_to_2_for_queries(
                            exp_queries,
                            max_downloads=14,
                            top_k=max(8, PART_CANDIDATE_LIMIT * 3),
                            exclude_categories=parent_categories,
                            exclude_wiki_titles=parent_wiki_titles,
                            prefer_search=True,
                        )
                        s2_new = [
                            c for c in s2_extra
                            if not (_candidate_identity_keys(c) & seen_files)
                        ]
                        if s2_new:
                            ser_list = _filter_and_judge_part_s2(
                                s2_new,
                                part_query=part_query,
                                judge_query=judge_name,
                                part_context=clean_topic,
                                vlm_judge_fn=vlm_judge_fn,
                                top_k=PART_CANDIDATE_LIMIT,
                            )
                except Exception as pe:
                    print(f"[!] Per-part query expansion warning for '{judge_name}': {pe}")

            ser_list = _apply_learned_preference(clean_topic, ser_list, part_id=p_id)
            part_candidates[p_id] = ser_list[:PART_CANDIDATE_LIMIT]
            if ser_list:
                part_references[p_id] = ser_list[0]
                if not ser_list[0].get("userPreferred"):
                    claimed_keys.update(_candidate_identity_keys(ser_list[0]))

    bundle = {
        "funnelVersion": FUNNEL_CACHE_VERSION,
        "topic": clean_topic,
        "summary": summary,
        "images": topic_candidates,
        "candidates": topic_candidates,
        "partReferences": part_references,
        "partCandidates": part_candidates,
        "funnelStats": topic_funnel.get("stats", {}),
        "operatingMode": operating_mode,
    }

    tmp_path = f"{cache_path}.tmp.{os.getpid()}.{int(time.time() * 1000) % 100000}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        os.replace(tmp_path, cache_path)
    except Exception as e:
        print(f"[!] Warning: Could not cache reference bundle: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

    return bundle


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reference_service.py <subject_query>")
        sys.exit(1)
    t = " ".join(sys.argv[1:])
    print(f"[*] Testing Stage 0-3 funnel for '{t}'...")
    res = get_reference_for_topic_and_parts(t)
    print("Stats:", res.get("funnelStats"))
    print("Summary:", (res.get("summary") or {}).get("title"))
    for idx, im in enumerate(res.get("candidates", [])):
        print(f" #{idx+1} [T{im.get('sourceTier')}] {im.get('title')} clip={im.get('clipScore')} cv={im.get('cvMetrics')}")
