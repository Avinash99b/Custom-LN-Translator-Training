#!/usr/bin/env python3
"""
optimized_dataset_validator.py

Accuracy-first light novel chapter alignment pipeline.

Strategy
--------
1) Load JP / EN chapter files.
2) Embed chapters in batches with a multilingual model.
3) Rank EN candidates for each JP chapter using top-K similarity.
4) Use cheap deterministic rules first:
   - same chapter number
   - reciprocal best match
   - margin over the next candidate
   - rough length ratio sanity
5) Call the LLM only when the pair is genuinely ambiguous.
6) Verify with a stronger sample that uses head + middle + tail, not just the opening.
7) Try fallback EN candidates when the best match is rejected.
8) Write JP-Output / EN-Output and a structured JSON report.
9) Deduplicate EN assignments post-pass — flag conflicts before writing.

Fix log (vs previous version)
------------------------------
FIX-1  should_call_llm: removed unconditional `return True` tail; function now
       returns False for pairs that pass all gate conditions.

FIX-2  write_output_pair missing in auto-accept + audit-pass branch:
       process_one now calls write_output_pair before returning in every PASS path.

FIX-3  Fallback margin was computed as (best_candidate.score - fb_score) — i.e.
       the drop from the original best — not the fallback's own margin over its
       next competitor.  Now computed correctly from the top-K list.

FIX-4  requests.Session was shared across ThreadPoolExecutor workers.
       Session is now created per-call inside AzureChatClient.ask() so each
       thread gets its own connection pool.  (The Session object on the instance
       is kept only for backwards-compat but is not used in threaded paths.)

FIX-5  DEFAULT_BATCH_SIZE raised from 1 to 16 (16 forward passes instead of
       200 serial ones for a 200-chapter dataset).

FIX-6  Audit threshold (0.75) vs pass threshold (0.55) were inconsistent.
       A borderline-audit pair could be rejected even though it legitimately
       passed.  audit_rejection now requires BOTH low confidence AND
       same_story=False OR major_drift=True.  Plain low-confidence-but-same-story
       is downgraded to SKIP rather than hard FAIL.

FIX-7  LLM prompt no longer sends the raw similarity score or margin; those
       numbers anchored the model's judgment before it read the text.

FIX-8  validate_candidate was dead code; process_one duplicated its logic
       inline.  All decision logic now lives exclusively in validate_candidate
       which process_one calls directly.

FIX-9  EN deduplication: after all chapters are processed, resolve_en_conflicts
       detects multiple JP chapters assigned to the same EN chapter, keeps the
       highest-confidence match, and downgrades the rest to CONFLICT status so
       they are not written to output.

Notes
-----
- This script is tuned for high recall and high precision on chapter-aligned
  light novel datasets where chapter numbering is usually consistent.
- Raw embedding scores are used as ranking signals, not as absolute truth.
- Full similarity matrix output is optional; the default is top-K only.

Environment
-----------
AZURE_OPENAI_API_KEY      required
AZURE_OPENAI_ENDPOINT     optional (defaults to Azure-compatible endpoint)
AZURE_OPENAI_DEPLOYMENT   optional (defaults to gpt-4o-mini)

Usage
-----
python optimized_dataset_validator.py /path/to/novel_root \
  --jp-dir JP-Working \
  --en-dir EN-Working \
  --output-jp JP-Output \
  --output-en EN-Output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# =============================================================================
# Defaults
# =============================================================================

MODEL_NAME = "BAAI/bge-m3"

DEFAULT_ENDPOINT = "https://bathu-mpkwvf7h-eastus2.cognitiveservices.azure.com/openai/v1/chat/completions"
DEFAULT_DEPLOYMENT = "gpt-4o-mini"

DEFAULT_TOP_K = 5
DEFAULT_WORKERS = 2
DEFAULT_BATCH_SIZE = 16  # FIX-5: was 1; raised to 16 for proper GPU/CPU batching

DEFAULT_HEAD_LINES = 24
DEFAULT_MIDDLE_LINES = 18
DEFAULT_TAIL_LINES = 24

# Intentionally conservative — rank/margin/reciprocal signals matter more than
# any single absolute score for translated chapter text.
AUTO_ACCEPT_SCORE = 0.74
AUTO_ACCEPT_MARGIN = 0.07

# Gates for deciding whether to call the LLM.
LLM_MIN_SCORE = 0.66
LLM_MIN_MARGIN = 0.015
LLM_LENGTH_RATIO_LOW = 0.18
LLM_LENGTH_RATIO_HIGH = 6.0
LLM_CHAPTER_GAP_HARD = 20
LLM_CHAPTER_GAP_OVERRIDE_SCORE = 0.62

# Final audit controls.
AUDIT_MODE_DEFAULT = "borderline"  # all | borderline | random | none
AUDIT_RANDOM_RATE_DEFAULT = 0.08
AUDIT_BORDERLINE_SCORE = 0.78
AUDIT_BORDERLINE_MARGIN = 0.04

NEARBY_WINDOW = 2
MAX_FALLBACK_TRIES = 3

STATE_FILE = "arrangement_state.json"
TOPK_FILE = "chapter_similarity_topk.json"
FULL_MATRIX_FILE = "chapter_similarity_matrix.json"

OUTPUT_LOCK = Lock()


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ChapterFile:
    name: str
    path: Path
    text: str
    chapter_num: int | None
    language: str


@dataclass
class Candidate:
    jp_name: str
    en_name: str
    score: float
    rank: int
    margin: float          # margin over the *next* candidate in the top-K list
    reciprocal: bool
    chapter_gap: int | None
    length_ratio: float


# =============================================================================
# General helpers
# =============================================================================

def natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def extract_chapter_number(name: str) -> int | None:
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def split_paragraphs(text: str) -> list[str]:
    """
    Split on blank lines so paragraph structure is preserved.
    This keeps scene-break semantics intact when slicing HEAD/MIDDLE/TAIL,
    which split_nonempty_lines destroyed by stripping all whitespace first.
    Falls back to non-empty line splitting for texts with no blank lines.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) >= 4:
        return paragraphs
    # Fallback: treat every non-empty line as its own unit
    return [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]


def save_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_output_name(jp: ChapterFile) -> str:
    if jp.chapter_num is not None:
        return f"{jp.chapter_num}.txt"
    return jp.name


def length_ratio(jp_text: str, en_text: str) -> float:
    return len(en_text) / max(len(jp_text), 1)


def chapter_gap(jp_num: int | None, en_num: int | None) -> int | None:
    if jp_num is None or en_num is None:
        return None
    return abs(jp_num - en_num)


# =============================================================================
# Sampling for LLM prompts
# =============================================================================

def sample_story_sections(
    text: str,
    head_lines: int = DEFAULT_HEAD_LINES,
    middle_lines: int = DEFAULT_MIDDLE_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> str:
    """
    Give the model a representative slice: head + middle + tail.
    Uses paragraph-aware splitting so scene breaks are not lost.
    """
    units = split_paragraphs(text)
    if not units:
        return ""

    total = len(units)
    head = units[: min(head_lines, total)]
    tail = units[max(0, total - tail_lines):]

    middle: list[str] = []
    if total > head_lines + tail_lines + 6:
        mid_start = max(head_lines, (total // 2) - (middle_lines // 2))
        mid_end = min(total - tail_lines, mid_start + middle_lines)
        if mid_end > mid_start:
            middle = units[mid_start:mid_end]

    parts: list[str] = []
    if head:
        parts.append("---HEAD---")
        parts.extend(head)
    if middle:
        parts.append("---MIDDLE---")
        parts.extend(middle)
    if tail:
        parts.append("---TAIL---")
        parts.extend(tail)

    return "\n".join(parts).strip()


# =============================================================================
# File loading
# =============================================================================

def load_chapter_folder(folder: Path, language: str) -> list[ChapterFile]:
    files = sorted(folder.glob("*.txt"), key=lambda p: natural_key(p.name))
    return [
        ChapterFile(
            name=p.name,
            path=p,
            text=read_text(p),
            chapter_num=extract_chapter_number(p.name),
            language=language,
        )
        for p in files
    ]


# =============================================================================
# Embeddings / ranking
# =============================================================================

def batched_encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def build_similarity_maps(
    jp_files: list[ChapterFile],
    en_files: list[ChapterFile],
    jp_embeddings: np.ndarray,
    en_embeddings: np.ndarray,
    top_k: int,
    write_full_matrix: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, float]] | None]:
    """
    Returns:
      topk_map[jp_name] -> list of candidate dicts sorted by descending score.
        Each dict includes:
          en_name, score, rank,
          margin  — drop from THIS candidate's score to the NEXT candidate.
      full_matrix[jp_name][en_name] -> score, if write_full_matrix=True
    """
    topk_map: dict[str, list[dict[str, Any]]] = {}
    full_matrix: dict[str, dict[str, float]] | None = {} if write_full_matrix else None

    if len(jp_embeddings) == 0 or len(en_embeddings) == 0:
        return topk_map, full_matrix

    jp_names = [x.name for x in jp_files]
    en_names = [x.name for x in en_files]

    # Normalized embeddings → dot product == cosine similarity.
    scores = jp_embeddings @ en_embeddings.T

    for i, jp_name in enumerate(jp_names):
        row = scores[i]

        if write_full_matrix:
            full_matrix[jp_name] = {
                en_names[j]: round(float(row[j]), 6) for j in range(len(en_names))
            }

        k = min(top_k, len(en_names))
        if k <= 0:
            topk_map[jp_name] = []
            continue

        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]

        sorted_scores = [float(row[int(j)]) for j in top_idx]

        candidates: list[dict[str, Any]] = []
        for rank, (j, score) in enumerate(zip(top_idx, sorted_scores), start=1):
            # FIX-3: margin = drop to the *next* entry in this sorted list,
            # not the global second-best.  For the last candidate, margin = 0.
            next_score = sorted_scores[rank] if rank < len(sorted_scores) else score
            margin = score - next_score
            candidates.append(
                {
                    "en_name": en_names[int(j)],
                    "score": round(score, 6),
                    "rank": rank,
                    "margin": round(margin, 6),
                }
            )

        topk_map[jp_name] = candidates

    return topk_map, full_matrix


def build_reverse_best(topk_map: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """en_name -> jp_name that considers this EN chapter its best match."""
    reverse: dict[str, str] = {}
    for jp_name, candidates in topk_map.items():
        if candidates:
            reverse[candidates[0]["en_name"]] = jp_name
    return reverse


# =============================================================================
# LLM client
# =============================================================================

def extract_json_from_response(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response")

    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError(f"Could not extract JSON from model response: {raw[:300]!r}")


def normalize_llm_result(data: dict[str, Any]) -> dict[str, Any]:
    same_story = data.get("same_story")
    if not isinstance(same_story, bool):
        same_story = str(same_story).strip().lower() in {"true", "1", "yes"}

    major_drift = data.get("major_drift")
    if not isinstance(major_drift, bool):
        major_drift = str(major_drift).strip().lower() in {"true", "1", "yes"}

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "no reason provided")).strip()
    if not reason:
        reason = "no reason provided"

    return {
        "same_story": same_story,
        "confidence": confidence,
        "reason": reason[:500],
        "major_drift": major_drift,
    }


class AzureChatClient:
    def __init__(self, endpoint: str, deployment: str, api_key: str):
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_key = api_key

    def _url(self) -> str:
        if self.endpoint.endswith("/chat/completions"):
            return self.endpoint
        if self.endpoint.endswith("/openai/v1"):
            return self.endpoint + "/chat/completions"
        if self.endpoint.endswith("/openai/v1/"):
            return self.endpoint + "chat/completions"
        return self.endpoint + "/openai/v1/chat/completions"

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 700,
        timeout: int = 300,
    ) -> dict[str, Any]:
        payload = {
            "model": self.deployment,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        url = self._url()
        delay = 1.0

        # FIX-4: create a fresh Session per call so that concurrent threads each
        # have their own connection pool.  requests.Session is not thread-safe
        # when shared across threads making concurrent requests.
        session = requests.Session()
        try:
            while True:
                resp = session.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.api_key,
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=timeout,
                )

                if resp.status_code == 429:
                    time.sleep(delay)
                    delay = min(delay * 2, 10.0)
                    continue

                resp.raise_for_status()
                body = resp.json()

                try:
                    raw = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    raise RuntimeError(
                        f"Unexpected LLM response structure: {e}\n"
                        f"Body: {json.dumps(body)[:600]}"
                    ) from e

                parsed = normalize_llm_result(extract_json_from_response(raw))
                parsed["raw_response"] = raw
                return parsed
        finally:
            session.close()


# FIX-7: Removed similarity score and margin from the user prompt.
# Those numbers anchored the model's verdict before it read any text.
SYSTEM_PROMPT_MAIN = """You are a bilingual light novel chapter alignment specialist.

Your task is to determine whether two chapter samples are the SAME chapter.

Rules:
- Focus on chapter title, opening scene, opening characters, opening dialogue, and the first major event.
- Use all provided sample sections together.
- Do NOT compare every line.
- Do NOT reject because one version is longer, shorter, expanded, condensed, or lightly localized.
- Do NOT reject because translator notes, afterwords, or status panels are present.
- Do NOT trust chapter numbers inside the text as a hard rule; they are only weak hints.
- Treat them as the same if they clearly start from the same chapter/story segment even if wording differs.

Mark as different only if:
- the titles clearly refer to different chapters, or
- the opening scene clearly differs, or
- the opening characters / setting do not match, or
- one sample obviously begins at a different point in the story.

Return ONLY valid JSON:
{
  "same_story": true,
  "confidence": 0.0,
  "reason": "short explanation",
  "major_drift": false
}
"""

SYSTEM_PROMPT_AUDIT = """You are a strict bilingual chapter verification auditor.

Your task is to confirm whether the two chapter samples are clearly the SAME chapter.

Be stricter than the first pass:
- If the evidence is weak, lower confidence.
- If the opening scene or core chapter identity does not clearly match, mark different.
- If one sample looks like a wrong chapter match, say so.

Return ONLY valid JSON:
{
  "same_story": true,
  "confidence": 0.0,
  "reason": "short explanation",
  "major_drift": false
}
"""


def llm_verify_pair(
    client: AzureChatClient,
    jp: ChapterFile,
    en: ChapterFile,
    audit: bool = False,
) -> dict[str, Any]:
    # FIX-7: score and margin removed from the prompt entirely.
    jp_sample = sample_story_sections(jp.text)
    en_sample = sample_story_sections(en.text)
    system_prompt = SYSTEM_PROMPT_AUDIT if audit else SYSTEM_PROMPT_MAIN

    user_prompt = f"""Japanese chapter: {jp.name}
English chapter: {en.name}

Length ratio (EN chars / JP chars): {length_ratio(jp.text, en.text):.4f}
Chapter gap: {chapter_gap(jp.chapter_num, en.chapter_num) if chapter_gap(jp.chapter_num, en.chapter_num) is not None else "unknown"}

Japanese sample:
{jp_sample}

---
English sample:
{en_sample}
"""

    return client.ask(system_prompt, user_prompt, max_tokens=700, timeout=300)


# =============================================================================
# Decision helpers
# =============================================================================

def reciprocal_match(reverse_best: dict[str, str], jp_name: str, en_name: str) -> bool:
    return reverse_best.get(en_name) == jp_name


def strong_auto_accept(candidate: Candidate) -> bool:
    """
    Fast deterministic accept for obvious matches.
    Conservative — prefers same chapter number, reciprocal best, non-trivial margin.
    """
    if candidate.chapter_gap != 0:
        return False
    if not candidate.reciprocal:
        return False
    if candidate.score < AUTO_ACCEPT_SCORE:
        return False
    if candidate.margin < AUTO_ACCEPT_MARGIN:
        return False
    if not (LLM_LENGTH_RATIO_LOW <= candidate.length_ratio <= LLM_LENGTH_RATIO_HIGH):
        return False
    return True


def should_call_llm(candidate: Candidate) -> bool:
    """
    FIX-1: The original version ended with an unconditional `return True`,
    making all early-return False paths dead code.  The function now correctly
    returns False when none of the ambiguity conditions are met.
    """
    if strong_auto_accept(candidate):
        return False

    # Conditions that warrant an LLM call:
    if candidate.score < LLM_MIN_SCORE:
        return True
    if candidate.margin < LLM_MIN_MARGIN:
        return True
    if (
        candidate.chapter_gap is not None
        and candidate.chapter_gap > LLM_CHAPTER_GAP_HARD
        and candidate.score < LLM_CHAPTER_GAP_OVERRIDE_SCORE
    ):
        return True
    if candidate.chapter_gap != 0:
        return True
    if not candidate.reciprocal:
        return True
    if candidate.length_ratio < LLM_LENGTH_RATIO_LOW or candidate.length_ratio > LLM_LENGTH_RATIO_HIGH:
        return True

    # All gates passed — no LLM needed.
    return False  # FIX-1


def should_audit(result: dict[str, Any], audit_mode: str, random_rate: float) -> bool:
    if result.get("status") != "PASS":
        return False

    if audit_mode == "all":
        return True
    if audit_mode == "none":
        return False
    if audit_mode == "random":
        key = str(result.get("jp_chapter", ""))
        seed = sum(ord(c) for c in key) % 1000
        return (seed / 1000.0) < random_rate
    if audit_mode == "borderline":
        return (
            float(result.get("semantic_score", 0.0)) < AUDIT_BORDERLINE_SCORE
            or float(result.get("margin", 0.0)) < AUDIT_BORDERLINE_MARGIN
            or not bool(result.get("llm_same_story", True))
        )
    return False


def pick_fallback_candidates(
    topk_for_jp: list[dict[str, Any]],
    en_map: dict[str, ChapterFile],
    current_en_name: str,
    max_tries: int = MAX_FALLBACK_TRIES,
) -> list[dict[str, Any]]:
    """Prefer remaining top-K alternatives; bias toward nearby chapter numbers."""
    fallback: list[dict[str, Any]] = []
    seen = {current_en_name}

    for cand in topk_for_jp[1:]:
        if len(fallback) >= max_tries:
            break
        if cand["en_name"] in seen:
            continue
        if cand["en_name"] in en_map:
            fallback.append(cand)
            seen.add(cand["en_name"])

    current_num = extract_chapter_number(current_en_name)
    if current_num is not None and len(fallback) < max_tries:
        nearby: list[tuple[int, float, dict[str, Any]]] = []
        for cand in topk_for_jp:
            en_name = cand["en_name"]
            if en_name in seen:
                continue
            en_num = extract_chapter_number(en_name)
            if en_num is None:
                continue
            gap = abs(en_num - current_num)
            if 1 <= gap <= NEARBY_WINDOW:
                nearby.append((gap, -float(cand["score"]), cand))
        nearby.sort(key=lambda x: (x[0], x[1], x[2]["en_name"]))
        for _, __, cand in nearby:
            if len(fallback) >= max_tries:
                break
            if cand["en_name"] not in seen and cand["en_name"] in en_map:
                fallback.append(cand)
                seen.add(cand["en_name"])

    return fallback[:max_tries]


def compute_final_confidence(semantic_score: float, llm_confidence: float, same_story: bool) -> float:
    semantic_score = max(0.0, min(1.0, semantic_score))
    llm_confidence = max(0.0, min(1.0, llm_confidence))
    combined = 0.68 * semantic_score + 0.32 * llm_confidence
    if not same_story:
        combined *= 0.20
    return round(combined, 6)


def make_result(
    jp: ChapterFile,
    en: ChapterFile,
    status: str,
    reason: str,
    semantic_score: float,
    margin: float,
    llm_result: dict[str, Any] | None,
    final_confidence: float | None,
    output_written: bool,
    output_chapter: str | None,
    fallback_used: bool = False,
    fallback_from: str | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "jp_chapter": jp.name,
        "en_chapter": en.name,
        "jp_chapter_num": jp.chapter_num,
        "en_chapter_num": en.chapter_num,
        "chapter_gap": chapter_gap(jp.chapter_num, en.chapter_num),
        "status": status,
        "reason": reason,
        "semantic_score": round(float(semantic_score), 6),
        "margin": round(float(margin), 6),
        "length_ratio": round(length_ratio(jp.text, en.text), 6),
        "jp_length": len(jp.text),
        "en_length": len(en.text),
        "output_chapter": output_chapter,
        "output_written": output_written,
        "fallback_used": fallback_used,
        "fallback_from": fallback_from,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_same_story": None,
        "llm_confidence": None,
        "llm_reason": None,
        "llm_major_drift": None,
        "llm_raw_response": None,
        "final_confidence": None,
    }

    if llm_result is not None:
        base.update(
            {
                "llm_same_story": llm_result["same_story"],
                "llm_confidence": round(float(llm_result["confidence"]), 6),
                "llm_reason": llm_result["reason"],
                "llm_major_drift": llm_result["major_drift"],
                "llm_raw_response": llm_result.get("raw_response", ""),
            }
        )

    if final_confidence is not None:
        base["final_confidence"] = round(float(final_confidence), 6)

    return base


def _apply_audit(
    jp: ChapterFile,
    en: ChapterFile,
    client: AzureChatClient,
    semantic_score: float,
    margin: float,
    primary_llm: dict[str, Any] | None,
    primary_conf: float,
    output_jp: Path,
    output_en: Path,
    fallback_used: bool = False,
    fallback_from: str | None = None,
) -> dict[str, Any]:
    """
    FIX-6: Run the audit LLM call and apply a consistent rejection policy.

    The old code hard-rejected on audit_conf < 0.75 even when same_story=True
    and major_drift=False — contradicting the primary pass decision.
    The new policy:
      - Hard FAIL only when the auditor explicitly disagrees
        (same_story=False OR major_drift=True).
      - Low-confidence-but-still-agrees → SKIP (needs human review).
      - Auditor agrees with reasonable confidence → PASS confirmed.
    """
    try:
        audit = llm_verify_pair(client, jp, en, audit=True)
        audit_conf = compute_final_confidence(semantic_score, audit["confidence"], audit["same_story"])
    except Exception as e:
        return make_result(
            jp, en, "ERROR", f"audit_failed ({e})",
            semantic_score, margin, primary_llm, primary_conf,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    if audit["major_drift"] or not audit["same_story"]:
        # Auditor actively disagrees → hard reject.
        return make_result(
            jp, en, "FAIL", f"audit_rejected ({audit['reason']})",
            semantic_score, margin, audit, audit_conf,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    if audit_conf < 0.75:
        # Auditor does not disagree but is uncertain → soft skip for human review.
        return make_result(
            jp, en, "SKIP", f"audit_uncertain ({audit['reason']})",
            semantic_score, margin, audit, audit_conf,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    # Audit passed — write output.
    write_output_pair(output_jp, output_en, jp, en)
    return make_result(
        jp, en, "PASS", f"audit_passed ({audit['reason']})",
        semantic_score, margin, audit, audit_conf,
        True, make_output_name(jp),
        fallback_used=fallback_used, fallback_from=fallback_from,
    )


# FIX-8: validate_candidate is the single source of truth for the decision
# logic.  process_one calls it directly instead of duplicating it inline.
def validate_candidate(
    client: AzureChatClient,
    jp: ChapterFile,
    en: ChapterFile,
    semantic_score: float,
    margin: float,
    reciprocal: bool,
    audit_mode: str,
    audit_random_rate: float,
    output_jp: Path,
    output_en: Path,
    fallback_used: bool = False,
    fallback_from: str | None = None,
) -> dict[str, Any]:
    cand = Candidate(
        jp_name=jp.name,
        en_name=en.name,
        score=semantic_score,
        rank=1,
        margin=margin,
        reciprocal=reciprocal,   # FIX-8 / was FIX for dead reciprocal field
        chapter_gap=chapter_gap(jp.chapter_num, en.chapter_num),
        length_ratio=length_ratio(jp.text, en.text),
    )

    # ------------------------------------------------------------------ #
    # Deterministic auto-accept path                                       #
    # ------------------------------------------------------------------ #
    if strong_auto_accept(cand):
        initial_result = make_result(
            jp, en, "PASS", "auto_accept_same_number_reciprocal",
            semantic_score, margin, None, semantic_score,
            False, make_output_name(jp),
            fallback_used=fallback_used, fallback_from=fallback_from,
        )
        if should_audit(initial_result, audit_mode, audit_random_rate):
            return _apply_audit(
                jp, en, client, semantic_score, margin,
                None, semantic_score, output_jp, output_en,
                fallback_used=fallback_used, fallback_from=fallback_from,
            )
        # FIX-2: write output before returning in the non-audit auto-accept path.
        write_output_pair(output_jp, output_en, jp, en)
        initial_result["output_written"] = True
        return initial_result

    # ------------------------------------------------------------------ #
    # Pre-filter: skip without LLM when all conditions are clearly bad    #
    # ------------------------------------------------------------------ #
    if not should_call_llm(cand):
        return make_result(
            jp, en, "SKIP", "prefilter_skip",
            semantic_score, margin, None, None,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    # ------------------------------------------------------------------ #
    # LLM path                                                             #
    # ------------------------------------------------------------------ #
    try:
        llm = llm_verify_pair(client, jp, en, audit=False)
    except Exception as e:
        return make_result(
            jp, en, "ERROR", str(e),
            semantic_score, margin, None, None,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    final_conf = compute_final_confidence(semantic_score, llm["confidence"], llm["same_story"])

    if llm["major_drift"]:
        return make_result(
            jp, en, "FAIL", f"major_drift_detected ({llm['reason']})",
            semantic_score, margin, llm, final_conf,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    if not llm["same_story"]:
        return make_result(
            jp, en, "FAIL", f"story_mismatch ({llm['reason']})",
            semantic_score, margin, llm, final_conf,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    if final_conf < 0.55:
        return make_result(
            jp, en, "SKIP", f"low_confidence ({llm['reason']})",
            semantic_score, margin, llm, final_conf,
            False, None,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    # LLM says same story and confidence is acceptable.
    initial_result = make_result(
        jp, en, "PASS", "same_story",
        semantic_score, margin, llm, final_conf,
        False, make_output_name(jp),
        fallback_used=fallback_used, fallback_from=fallback_from,
    )

    if should_audit(initial_result, audit_mode, audit_random_rate):
        return _apply_audit(
            jp, en, client, semantic_score, margin,
            llm, final_conf, output_jp, output_en,
            fallback_used=fallback_used, fallback_from=fallback_from,
        )

    # FIX-2: write output in the non-audit pass path.
    write_output_pair(output_jp, output_en, jp, en)
    initial_result["output_written"] = True
    return initial_result


def write_output_pair(output_jp: Path, output_en: Path, jp: ChapterFile, en: ChapterFile) -> None:
    output_name = make_output_name(jp)
    with OUTPUT_LOCK:
        (output_jp / output_name).write_text(jp.text, encoding="utf-8")
        (output_en / output_name).write_text(en.text, encoding="utf-8")


# =============================================================================
# FIX-9: Post-pass EN deduplication
# =============================================================================

def resolve_en_conflicts(
    results: list[dict[str, Any]],
    output_jp: Path,
    output_en: Path,
) -> list[dict[str, Any]]:
    """
    Multiple JP chapters can legitimately rank the same EN chapter as their
    best match — especially when chapters are split differently between editions.
    After all validation passes are done, find any EN chapter assigned to more
    than one JP chapter, keep the highest-confidence assignment, and downgrade
    the rest to CONFLICT status (removing their output files).
    """
    # Group PASS results by the en_chapter they were assigned.
    from collections import defaultdict
    en_to_results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in results:
        if r["status"] == "PASS" and r.get("en_chapter"):
            en_to_results[r["en_chapter"]].append(r)

    # Resolve conflicts: keep the best by final_confidence (fall back to
    # semantic_score when final_confidence is absent).
    def _conf(r: dict[str, Any]) -> float:
        if r.get("final_confidence") is not None:
            return float(r["final_confidence"])
        return float(r.get("semantic_score", 0.0))

    conflict_jp_names: set[str] = set()
    for en_name, group in en_to_results.items():
        if len(group) <= 1:
            continue
        group.sort(key=_conf, reverse=True)
        winner = group[0]
        losers = group[1:]
        for loser in losers:
            conflict_jp_names.add(loser["jp_chapter"])
            print(
                f"[CONFLICT] {en_name} claimed by "
                f"{winner['jp_chapter']} (conf={_conf(winner):.4f}) AND "
                f"{loser['jp_chapter']} (conf={_conf(loser):.4f}) → "
                f"downgrading {loser['jp_chapter']}"
            )

    # Update results in-place and delete orphaned output files.
    for r in results:
        if r["jp_chapter"] in conflict_jp_names:
            old_output = r.get("output_chapter")
            r["status"] = "CONFLICT"
            r["reason"] = (
                f"en_chapter {r['en_chapter']} also matched by a higher-confidence JP chapter"
            )
            r["output_written"] = False
            r["output_chapter"] = None
            if old_output:
                for d in (output_jp, output_en):
                    p = d / old_output
                    if p.exists():
                        p.unlink(missing_ok=True)

    return results


# =============================================================================
# Reporting
# =============================================================================

def summarize_results(results: list[dict[str, Any]]) -> None:
    total = len(results)
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    skipped = [r for r in results if r["status"] == "SKIP"]
    errors = [r for r in results if r["status"] == "ERROR"]
    conflicts = [r for r in results if r["status"] == "CONFLICT"]

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Total    : {total}")
    print(f"PASS     : {len(passed)}")
    print(f"FAIL     : {len(failed)}")
    print(f"SKIP     : {len(skipped)}")
    print(f"ERROR    : {len(errors)}")
    print(f"CONFLICT : {len(conflicts)}")

    pass_confs = [float(r["final_confidence"]) for r in passed if r.get("final_confidence") is not None]
    if pass_confs:
        print(f"PASS avg final confidence   : {statistics.mean(pass_confs):.4f}")
        print(f"PASS median final confidence: {statistics.median(pass_confs):.4f}")

    if failed:
        print("\nTop failures:")
        for r in failed[:20]:
            print(f"  {r['jp_chapter']} -> {r['en_chapter']} | {r['reason']}")

    if conflicts:
        print("\nConflicts (EN chapter claimed by multiple JP chapters):")
        for r in conflicts[:20]:
            print(f"  {r['jp_chapter']} -> {r['en_chapter']} | {r['reason']}")

    if errors:
        print("\nTop errors:")
        for r in errors[:10]:
            print(f"  {r['jp_chapter']} -> {r['en_chapter']} | {r['reason']}")


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accuracy-first JP/EN chapter matching pipeline using embeddings + targeted LLM checks."
    )
    parser.add_argument("root", help="Novel root folder containing JP/EN or JP-Working/EN-Working")
    parser.add_argument("--jp-dir", default="JP-Working", help="JP chapter directory name")
    parser.add_argument("--en-dir", default="EN-Working", help="EN chapter directory name")
    parser.add_argument("--output-jp", default="JP-Output", help="Output JP directory name")
    parser.add_argument("--output-en", default="EN-Output", help="Output EN directory name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K EN candidates per JP chapter")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent validation workers")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--write-full-matrix", action="store_true", help="Write the full similarity matrix JSON")
    parser.add_argument("--audit-mode", choices=["all", "borderline", "random", "none"], default=AUDIT_MODE_DEFAULT)
    parser.add_argument("--audit-random-rate", type=float, default=AUDIT_RANDOM_RATE_DEFAULT)
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", DEFAULT_DEPLOYMENT))
    parser.add_argument("--api-key", default=os.getenv("AZURE_OPENAI_API_KEY", "none"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--clear-output", action="store_true", help="Delete existing output files before writing new ones"
    )
    args = parser.parse_args()

    root = Path(args.root)
    jp_dir = root / args.jp_dir
    en_dir = root / args.en_dir
    output_jp = root / args.output_jp
    output_en = root / args.output_en
    state_path = root / STATE_FILE
    topk_path = root / TOPK_FILE
    matrix_path = root / FULL_MATRIX_FILE

    if not jp_dir.exists():
        if (root / "JP").exists():
            (root / "JP-Working").mkdir(exist_ok=True)
            for f in (root / "JP").glob("*.txt"):
                shutil.copy(f, root / "JP-Working" / f.name)
            jp_dir = root / "JP-Working"
        else:
            raise SystemExit(f"Missing folder: {jp_dir}")
    if not en_dir.exists():
        if (root / "EN").exists():
            (root / "EN-Working").mkdir(exist_ok=True)
            for f in (root / "EN").glob("*.txt"):
                shutil.copy(f, root / "EN-Working" / f.name)
            en_dir = root / "EN-Working"
        else:
            raise SystemExit(f"Missing folder: {en_dir}")

    output_jp.mkdir(parents=True, exist_ok=True)
    output_en.mkdir(parents=True, exist_ok=True)

    if args.clear_output:
        for f in output_jp.glob("*.txt"):
            f.unlink(missing_ok=True)
        for f in output_en.glob("*.txt"):
            f.unlink(missing_ok=True)

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("Loading chapters...")
    jp_files = load_chapter_folder(jp_dir, "JP")
    en_files = load_chapter_folder(en_dir, "EN")

    print(f"JP chapters: {len(jp_files)}")
    print(f"EN chapters: {len(en_files)}")

    print("Embedding JP chapters...")
    jp_embeddings = batched_encode(model, [c.text for c in jp_files], args.batch_size)

    print("Embedding EN chapters...")
    en_embeddings = batched_encode(model, [c.text for c in en_files], args.batch_size)

    print("Ranking candidates...")
    topk_map, full_matrix = build_similarity_maps(
        jp_files=jp_files,
        en_files=en_files,
        jp_embeddings=jp_embeddings,
        en_embeddings=en_embeddings,
        top_k=args.top_k,
        write_full_matrix=args.write_full_matrix,
    )
    reverse_best = build_reverse_best(topk_map)

    save_json_atomic(topk_path, topk_map)
    print(f"Top-K index written -> {topk_path}")

    if full_matrix is not None:
        save_json_atomic(matrix_path, full_matrix)
        print(f"Full matrix written -> {matrix_path}")

    en_map = {c.name: c for c in en_files}
    client = AzureChatClient(args.endpoint, args.deployment, args.api_key)

    results: list[dict[str, Any]] = []
    started = time.time()

    # FIX-8: process_one now delegates all decision logic to validate_candidate.
    def process_one(jp: ChapterFile) -> dict[str, Any]:
        candidates = topk_map.get(jp.name, [])
        if not candidates:
            return make_result(
                jp,
                ChapterFile(name="", path=Path(""), text="", chapter_num=None, language="EN"),
                "SKIP", "no_candidates",
                0.0, 0.0, None, None, False, None,
            )

        best = candidates[0]
        best_en = en_map.get(best["en_name"])
        if best_en is None:
            return make_result(
                jp,
                ChapterFile(name=best["en_name"], path=Path(best["en_name"]), text="", chapter_num=None, language="EN"),
                "SKIP", "missing_en_file",
                float(best["score"]), float(best["margin"]), None, None, False, None,
            )

        # FIX-3: margin is now the drop from this candidate to the next in the
        # top-K list (already computed correctly in build_similarity_maps).
        result = validate_candidate(
            client=client,
            jp=jp,
            en=best_en,
            semantic_score=float(best["score"]),
            margin=float(best["margin"]),
            reciprocal=reciprocal_match(reverse_best, jp.name, best_en.name),
            audit_mode=args.audit_mode,
            audit_random_rate=args.audit_random_rate,
            output_jp=output_jp,
            output_en=output_en,
        )

        if result["status"] == "PASS":
            return result

        # Try fallback candidates when the best one did not pass.
        for fb in pick_fallback_candidates(candidates, en_map, best_en.name, MAX_FALLBACK_TRIES):
            fb_en = en_map.get(fb["en_name"])
            if fb_en is None:
                continue

            # FIX-3: use the fallback candidate's own margin from the top-K list.
            fb_result = validate_candidate(
                client=client,
                jp=jp,
                en=fb_en,
                semantic_score=float(fb["score"]),
                margin=float(fb["margin"]),
                reciprocal=reciprocal_match(reverse_best, jp.name, fb_en.name),
                audit_mode=args.audit_mode,
                audit_random_rate=args.audit_random_rate,
                output_jp=output_jp,
                output_en=output_en,
                fallback_used=True,
                fallback_from=best_en.name,
            )

            result = fb_result
            if fb_result["status"] == "PASS":
                return fb_result

        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, jp): jp.name for jp in jp_files}
        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Validating", unit="chapter"),
            start=1,
        ):
            jp_name = futures[future]
            try:
                res = future.result()
            except Exception as e:
                jp = next((x for x in jp_files if x.name == jp_name), None)
                if jp is None:
                    continue
                res = make_result(
                    jp,
                    ChapterFile(name="", path=Path(""), text="", chapter_num=None, language="EN"),
                    "ERROR", str(e),
                    0.0, 0.0, None, None, False, None,
                )
            results.append(res)
            print(f"[{i}/{len(futures)}] {jp_name}: {res['status']} | {res['reason']}")

    # FIX-9: resolve any EN chapters claimed by multiple JP chapters.
    results = resolve_en_conflicts(results, output_jp, output_en)

    elapsed = time.time() - started

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 3),
        "root": str(root),
        "jp_dir": str(jp_dir),
        "en_dir": str(en_dir),
        "output_jp": str(output_jp),
        "output_en": str(output_en),
        "model_name": args.model_name,
        "top_k": args.top_k,
        "audit_mode": args.audit_mode,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "failed": sum(1 for r in results if r["status"] == "FAIL"),
            "skipped": sum(1 for r in results if r["status"] == "SKIP"),
            "errors": sum(1 for r in results if r["status"] == "ERROR"),
            "conflicts": sum(1 for r in results if r["status"] == "CONFLICT"),
            "fallback_passes": sum(1 for r in results if r.get("fallback_used") and r["status"] == "PASS"),
        },
        "chapters": sorted(
            results,
            key=lambda x: (
                x["jp_chapter_num"] is None,
                x["jp_chapter_num"] if x["jp_chapter_num"] is not None else x["jp_chapter"],
            ),
        ),
    }

    save_json_atomic(state_path, report)
    summarize_results(results)

    print(f"\nState written -> {state_path}")
    print(f"Top-K index   -> {topk_path}")
    if full_matrix is not None:
        print(f"Matrix written -> {matrix_path}")


if __name__ == "__main__":
    main()