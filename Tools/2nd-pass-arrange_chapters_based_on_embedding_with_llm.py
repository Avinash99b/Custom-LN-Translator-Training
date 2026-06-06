"""
2nd-pass arrangement
This script performs the second pass of chapter arrangement by validating the best candidate matches from the similarity matrix using an LLM. It applies a series of pre-filters to skip obviously bad matches, then calls the LLM to verify if the chapters are the same story. It combines the semantic score and LLM confidence to make a final decision. If the best match fails but is close, it tries nearby EN chapters as fallbacks. The results are categorised into PASS/FAIL/SKIP/ERROR and written to output folders and a structured JSON state file.
Usage:
  python 2nd-pass-arrangement-validator.py <matrix_path> <base_path> [--jp-dir JP_DIR] [--en-dir EN_DIR]
Example:
  python 2nd-pass-arrangement-validator.py chapter_similarity_matrix.json /path/to/novel --jp-dir JP --en-dir EN
  
  It also looks +/-2 chapters away in the EN folder for potential matches if the best candidate fails with a story mismatch or low confidence, and accepts the first one that passes the LLM verification.

"""


#!/usr/bin/env python3

import json
import argparse
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

LM_STUDIO_URL = "https://bathu-mpkwvf7h-eastus2.cognitiveservices.azure.com/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"
API_KEY = os.getenv("LM_STUDIO_API_KEY", "[REACTED_GOTCHA]")

WORKERS = 10
SAMPLE_HEAD_LINES = 30

MIN_LENGTH_RATIO = 0.15
MAX_LENGTH_RATIO = 8.0
MIN_SEMANTIC_SCORE = 0.45

SEMANTIC_WEIGHT = 0.70
LLM_WEIGHT = 0.30

CONFIDENCE_THRESHOLD = 0.57
DRIFT_PENALTY = 0.25

CHAPTER_GAP_THRESHOLD = 20
CHAPTER_GAP_SEMANTIC_THRESHOLD = 0.6

FALLBACK_NEIGHBOR_WINDOW = 2

ARRANGEMENT_STATE_FILE = "arrangement_state.json"

OUTPUT_LOCK = Lock()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def sample_text_sections(text: str) -> str:
    """
    Extract the head lines from text and return them as a sample.
    """
    lines = text.strip().splitlines()
    total = len(lines)

    if total == 0:
        return ""

    head_end = min(SAMPLE_HEAD_LINES, total)
    head = lines[:head_end]

    return "---HEAD---\n" + "\n".join(head)


def extract_json_from_response(raw: str) -> dict:
    """
    Robustly extract a JSON object from a model response.
    Handles:
      - clean JSON
      - markdown code fences (```json ... ```)
      - leading/trailing prose
      - multiple JSON objects (first wins)
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from model")

    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "")

    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from model response: {raw[:300]!r}")


def validate_llm_json(data: dict) -> dict:
    """
    Validate and normalise the structured JSON returned by the model.
    """
    same_story = data.get("same_story")
    if not isinstance(same_story, bool):
        if str(same_story).lower() in ("true", "1"):
            same_story = True
        elif str(same_story).lower() in ("false", "0"):
            same_story = False
        else:
            same_story = False

    raw_conf = data.get("confidence", 0.5)
    try:
        confidence = float(raw_conf)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    reason = str(data.get("reason", "no reason provided"))[:500]

    major_drift = data.get("major_drift")
    if not isinstance(major_drift, bool):
        if str(major_drift).lower() in ("true", "1"):
            major_drift = True
        else:
            major_drift = False

    return {
        "same_story": same_story,
        "confidence": confidence,
        "reason": reason,
        "major_drift": major_drift,
    }


def ask_llm(jp_name: str, en_name: str, jp_sample: str, en_sample: str) -> dict:
    """
    Call the LM Studio OpenAI-compatible chat completions endpoint.
    Returns a validated dict with keys:
      same_story, confidence, reason, major_drift
    Raises RuntimeError on unrecoverable failure.
    """
    system_prompt = (
        "You are a bilingual light novel chapter alignment specialist.\n\n"
        "Your task is to determine whether two chapters are intended to be the SAME chapter.\n\n"
        "IMPORTANT:\n"
        "Focus primarily on:\n"
        "- chapter title\n"
        "- chapter number (if present)\n"
        "- opening scene\n"
        "- opening setting\n"
        "- opening characters present\n"
        "- opening dialogue\n"
        "- first major event at the beginning of the chapter\n\n"
        "DO NOT compare the entire chapter.\n"
        "DO NOT reject chapters because later events differ.\n"
        "DO NOT reject chapters because one version contains extra content.\n"
        "DO NOT reject chapters because one version is longer.\n"
        "DO not trust the chapter numbers inside the text, as they may be wrong or missing. Use them as a hint but not a deciding factor.\n"
        "DO NOT reject chapters because translator notes, status screens, afterwords, or bonus text are present.\n"
        "DO NOT reject chapters because scenes are expanded, condensed, reordered slightly, or localized.\n\n"
        "If the chapter titles match or are clear translations of each other, and the opening portion of both chapters starts from the same scene, treat them as the SAME chapter even if later content differs.\n\n"
        "Only mark chapters as different if:\n"
        "- the titles clearly refer to different chapters\n"
        "- the opening scene is clearly different\n"
        "- the opening characters and setting do not match\n"
        "- one chapter obviously begins at a different point in the story\n\n"
        "DO NOT MARK AS DIFFERENT just because the chapter numbers are different or missing. Use the chapter numbers as a weak signal, but rely more on the content of the opening scene and title.\n\n"
        "Return ONLY a JSON object.\n"
        "No markdown.\n"
        "No explanation outside the JSON.\n\n"
        "Example output:\n"
        "{\"same_story\": true, \"confidence\": 0.97, \"reason\": \"matching title and opening scene\", \"major_drift\": false}\n"
        "or\n"
        "{\"same_story\": false, \"confidence\": 0.08, \"reason\": \"different title and opening scene\", \"major_drift\": true}\n\n"
        "Fields:\n"
        "  same_story  : boolean\n"
        "  confidence  : float 0.0-1.0\n"
        "  reason      : short string\n"
        "  major_drift : boolean"
    )

    user_prompt = (
        f"Japanese chapter ({jp_name}):\n\n"
        f"{jp_sample}\n\n"
        f"---\n\n"
        f"English chapter ({en_name}):\n\n"
        f"{en_sample}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 12000,
    }

    try:
        response = requests.post(
            LM_STUDIO_URL,
            json=payload,
            timeout=300,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"LM Studio request failed for {jp_name} -> {en_name}: {e}") from e

    try:
        body = response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse LM Studio HTTP response as JSON: {e}") from e

    raw_text = ""
    try:
        raw_text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected LM Studio response structure for {jp_name} -> {en_name}: {e}\n"
            f"Response body: {json.dumps(body)[:500]}"
        ) from e

    try:
        parsed = extract_json_from_response(raw_text)
        validated = validate_llm_json(parsed)
        validated["raw_response"] = raw_text
        return validated
    except Exception as e:
        raise RuntimeError(
            f"JSON extraction failed for {jp_name} -> {en_name}: {e}\n"
            f"Raw model output: {raw_text!r}"
        ) from e


def compute_final_confidence(
    semantic_score: float,
    llm_confidence: float,
    same_story: bool,
) -> float:
    """
    Combine semantic score and LLM confidence.
    Apply a strong penalty when the model reports story mismatch.
    """
    semantic_score = max(0.0, min(1.0, semantic_score))
    llm_confidence = max(0.0, min(1.0, llm_confidence))

    combined = SEMANTIC_WEIGHT * semantic_score + LLM_WEIGHT * llm_confidence

    if not same_story:
        combined *= DRIFT_PENALTY

    return round(combined, 6)


def extract_chapter_number(chapter_name: str):
    match = re.search(r"(\d+)\.txt$", chapter_name)
    if not match:
        return None
    return int(match.group(1))


def chapter_sort_key(chapter_name: str):
    chapter_num = extract_chapter_number(chapter_name)
    if chapter_num is None:
        return (1, chapter_name)
    return (0, chapter_num, chapter_name)


def build_state_record(
    jp_name: str,
    en_name: str,
    semantic_score: float,
    jp_chapter_num,
    en_chapter_num,
    jp_len: int,
    en_len: int,
    ratio,
    llm_result,
    final_confidence,
    status: str,
    reason: str,
    output_chapter,
    output_written: bool,
    primary_en_chapter: str | None = None,
    fallback_used: bool = False,
    fallback_from_en_chapter: str | None = None,
) -> dict:
    chapter_gap = None
    if jp_chapter_num is not None and en_chapter_num is not None:
        chapter_gap = abs(jp_chapter_num - en_chapter_num)

    record = {
        "jp_chapter": jp_name,
        "en_chapter": en_name,
        "primary_en_chapter": primary_en_chapter if primary_en_chapter is not None else en_name,
        "fallback_used": fallback_used,
        "fallback_from_en_chapter": fallback_from_en_chapter,
        "jp_chapter_num": jp_chapter_num,
        "en_chapter_num": en_chapter_num,
        "chapter_gap": chapter_gap,
        "status": status,
        "reason": reason,
        "semantic_val": round(semantic_score, 6),
        "llm_same_story": None,
        "llm_confidence": None,
        "llm_reason": None,
        "llm_major_drift": None,
        "llm_raw_response": None,
        "final_confidence": None,
        "length_ratio": None,
        "jp_length": jp_len,
        "en_length": en_len,
        "output_chapter": output_chapter,
        "output_written": output_written,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if ratio is not None:
        record["length_ratio"] = round(ratio, 6)

    if llm_result is not None:
        record["llm_same_story"] = llm_result["same_story"]
        record["llm_confidence"] = round(llm_result["confidence"], 6)
        record["llm_reason"] = llm_result["reason"]
        record["llm_major_drift"] = llm_result["major_drift"]
        record["llm_raw_response"] = llm_result.get("raw_response", "")

    if final_confidence is not None:
        record["final_confidence"] = round(final_confidence, 6)

    return record


def should_try_nearby_fallback(result: dict) -> bool:
    if result.get("status") not in {"FAIL", "SKIP"}:
        return False

    reason = str(result.get("reason", "")).lower()
    return any(
        marker in reason
        for marker in (
            "story_mismatch",
            "major_drift_detected",
            "low_confidence",
        )
    )


def get_nearby_en_candidates(
    matrix: dict,
    jp_name: str,
    current_en_name: str,
    en_dir: Path,
    window: int = FALLBACK_NEIGHBOR_WINDOW,
) -> list[tuple[str, float, int]]:
    """
    Return nearby EN chapters within +/- window of the current EN chapter number.
    Output is sorted by nearest chapter gap first, then higher semantic score.
    """
    en_scores = matrix.get(jp_name, {})
    current_num = extract_chapter_number(current_en_name)
    if current_num is None:
        return []

    candidates = []
    for en_name, score in en_scores.items():
        if en_name == current_en_name:
            continue

        en_num = extract_chapter_number(en_name)
        if en_num is None:
            continue

        gap = abs(en_num - current_num)
        if 1 <= gap <= window:
            en_file = en_dir / en_name
            if en_file.exists():
                candidates.append((en_name, score, gap))

    candidates.sort(key=lambda x: (x[2], -x[1], x[0]))
    return candidates


def validate_and_arrange(
    jp_name: str,
    en_name: str,
    semantic_score: float,
    jp_file: Path,
    en_file: Path,
    jp_output_dir: Path,
    en_output_dir: Path,
    matrix: dict,
    en_dir: Path,
    primary_en_chapter: str | None = None,
    fallback_used: bool = False,
    fallback_from_en_chapter: str | None = None,
) -> dict:
    jp_chapter_num = extract_chapter_number(jp_name)
    en_chapter_num = extract_chapter_number(en_name)

    jp_full = read_text(jp_file)
    en_full = read_text(en_file)

    jp_len = len(jp_full)
    en_len = len(en_full)
    ratio = en_len / max(jp_len, 1)

    effective_primary_en = primary_en_chapter if primary_en_chapter is not None else en_name

    def make_record(
        status: str,
        reason: str,
        llm_result=None,
        final_confidence=None,
        output_chapter=None,
        output_written: bool = False,
    ) -> dict:
        return build_state_record(
            jp_name,
            en_name,
            semantic_score,
            jp_chapter_num,
            en_chapter_num,
            jp_len,
            en_len,
            ratio,
            llm_result,
            final_confidence,
            status,
            reason,
            output_chapter,
            output_written,
            primary_en_chapter=effective_primary_en,
            fallback_used=fallback_used,
            fallback_from_en_chapter=fallback_from_en_chapter,
        )

    # --- Pre-filters (no LLM call) ---

    if (
        jp_chapter_num is not None
        and en_chapter_num is not None
        and abs(jp_chapter_num - en_chapter_num) > CHAPTER_GAP_THRESHOLD
        and semantic_score < CHAPTER_GAP_SEMANTIC_THRESHOLD
    ):
        gap = abs(jp_chapter_num - en_chapter_num)
        return make_record(
            "SKIP",
            f"chapter_gap_too_large (gap={gap}, semantic={semantic_score:.2f})",
            None,
            compute_final_confidence(semantic_score, 0.0, False),
            None,
            False,
        )

    if semantic_score < MIN_SEMANTIC_SCORE:
        return make_record(
            "SKIP",
            f"semantic_score_too_low ({semantic_score:.2f})",
            None,
            compute_final_confidence(semantic_score, 0.0, False),
            None,
            False,
        )

    if ratio < MIN_LENGTH_RATIO:
        return make_record(
            "SKIP",
            f"length_ratio_too_small ({ratio:.2f})",
            None,
            compute_final_confidence(semantic_score, 0.0, False),
            None,
            False,
        )

    if ratio > MAX_LENGTH_RATIO:
        return make_record(
            "SKIP",
            f"length_ratio_too_large ({ratio:.2f})",
            None,
            compute_final_confidence(semantic_score, 0.0, False),
            None,
            False,
        )

    # --- Build samples ---
    jp_sample = sample_text_sections(jp_full)
    en_sample = sample_text_sections(en_full)

    # --- LLM verification ---
    try:
        llm_result = ask_llm(jp_name, en_name, jp_sample, en_sample)
    except Exception as e:
        return make_record("ERROR", str(e), None, None, None, False)

    same_story = llm_result["same_story"]
    llm_confidence = llm_result["confidence"]
    major_drift = llm_result["major_drift"]

    final_confidence = compute_final_confidence(
        semantic_score, llm_confidence, same_story
    )

    # --- Drift hard-fail ---
    if major_drift:
        return make_record(
            "FAIL",
            f"major_drift_detected ({llm_result['reason']})",
            llm_result,
            final_confidence,
            None,
            False,
        )

    # --- Confidence gate ---
    if final_confidence < CONFIDENCE_THRESHOLD:
        return make_record(
            "SKIP",
            (
                f"low_confidence "
                f"(semantic={semantic_score:.2f}, "
                f"llm_conf={llm_confidence:.2f}, "
                f"same_story={same_story}, "
                f"final={final_confidence:.2f})"
            ),
            llm_result,
            final_confidence,
            None,
            False,
        )

    # --- Story mismatch ---
    if not same_story:
        return make_record(
            "FAIL",
            f"story_mismatch ({llm_result['reason']})",
            llm_result,
            final_confidence,
            None,
            False,
        )

    # --- Chapter number required for output ---
    if jp_chapter_num is None:
        return make_record(
            "SKIP",
            "could_not_extract_chapter_number",
            llm_result,
            final_confidence,
            None,
            False,
        )

    # --- Write output ---
    output_name = f"{jp_chapter_num}.txt"

    with OUTPUT_LOCK:
        jp_out_path = jp_output_dir / output_name
        en_out_path = en_output_dir / output_name
        jp_out_path.write_text(jp_full, encoding="utf-8")
        en_out_path.write_text(en_full, encoding="utf-8")

    return make_record(
        "PASS",
        "same_story",
        llm_result,
        final_confidence,
        str(jp_chapter_num),
        True,
    )


def validate_and_arrange_with_fallback(
    jp_name: str,
    en_name: str,
    semantic_score: float,
    jp_file: Path,
    en_file: Path,
    jp_output_dir: Path,
    en_output_dir: Path,
    matrix: dict,
    en_dir: Path,
) -> dict:
    """
    First try the best EN match. If LLM says it is not the same chapter,
    retry with nearby EN chapters within +/-2 and accept the first one
    the LLM verifies as same_story.
    """
    primary_result = validate_and_arrange(
        jp_name,
        en_name,
        semantic_score,
        jp_file,
        en_file,
        jp_output_dir,
        en_output_dir,
        matrix,
        en_dir,
        primary_en_chapter=en_name,
        fallback_used=False,
        fallback_from_en_chapter=None,
    )

    if primary_result["status"] == "PASS":
        return primary_result

    if not should_try_nearby_fallback(primary_result):
        return primary_result

    nearby_candidates = get_nearby_en_candidates(matrix, jp_name, en_name, en_dir, window=FALLBACK_NEIGHBOR_WINDOW)

    for candidate_en_name, candidate_score, gap in nearby_candidates:
        candidate_file = en_dir / candidate_en_name
        if not candidate_file.exists():
            continue

        fallback_result = validate_and_arrange(
            jp_name,
            candidate_en_name,
            candidate_score,
            jp_file,
            candidate_file,
            jp_output_dir,
            en_output_dir,
            matrix,
            en_dir,
            primary_en_chapter=en_name,
            fallback_used=True,
            fallback_from_en_chapter=en_name,
        )

        if fallback_result["status"] == "PASS":
            fallback_result["reason"] = (
                f"same_story (fallback accepted from {en_name} to {candidate_en_name}; "
                f"initial_reason={primary_result['reason']})"
            )
            return fallback_result

    return primary_result


def load_similarity_matrix(matrix_path: Path) -> dict:
    try:
        return json.loads(matrix_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load similarity matrix: {e}") from e


def find_best_matches(matrix: dict, jp_dir: Path, en_dir: Path) -> dict:
    matches: dict = {}

    for jp_name, en_scores in matrix.items():
        if not en_scores:
            continue

        best_en, score = max(en_scores.items(), key=lambda x: x[1])

        jp_file = jp_dir / jp_name
        en_file = en_dir / best_en

        if not jp_file.exists() or not en_file.exists():
            continue

        matches[jp_name] = {
            "en_chapter": best_en,
            "score": score,
            "jp_file": jp_file,
            "en_file": en_file,
        }

    return matches


def main(
    matrix_path: str,
    base_path: str,
    jp_dir_name: str = "JP",
    en_dir_name: str = "EN",
) -> None:
    matrix_file = Path(matrix_path)
    if not matrix_file.exists():
        raise RuntimeError(f"Similarity matrix not found: {matrix_path}")

    base = Path(base_path)
    jp_dir = base / jp_dir_name
    en_dir = base / en_dir_name

    if not jp_dir.exists():
        raise RuntimeError(f"Missing folder: {jp_dir}")
    if not en_dir.exists():
        raise RuntimeError(f"Missing folder: {en_dir}")

    jp_output_dir = base / "JP-Output"
    en_output_dir = base / "EN-Output"
    jp_output_dir.mkdir(exist_ok=True)
    en_output_dir.mkdir(exist_ok=True)

    state_path = base / ARRANGEMENT_STATE_FILE

    print("Loading similarity matrix...")
    matrix = load_similarity_matrix(matrix_file)

    print("Finding best matches...")
    matches = find_best_matches(matrix, jp_dir, en_dir)
    print(f"Found {len(matches)} potential matches")
    print()

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_jp = {
            executor.submit(
                validate_and_arrange_with_fallback,
                jp_name,
                match_info["en_chapter"],
                match_info["score"],
                match_info["jp_file"],
                match_info["en_file"],
                jp_output_dir,
                en_output_dir,
                matrix,
                en_dir,
            ): jp_name
            for jp_name, match_info in sorted(
                matches.items(),
                key=lambda item: chapter_sort_key(item[0]),
            )
        }

        total = len(future_to_jp)
        completed = 0

        for future in as_completed(future_to_jp):
            completed += 1
            try:
                result = future.result()
                results.append(result)
                fb = " [fallback]" if result.get("fallback_used") else ""
                print(
                    f"[{completed}/{total}] "
                    f"{result['status']}{fb} "
                    f"{result['jp_chapter']} -> {result['en_chapter']} "
                    f"({result['reason']})"
                )
            except Exception as e:
                print(f"[{completed}/{total}] ERROR: {e}")

    # --- Categorise ---
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    skipped = [r for r in results if r["status"] == "SKIP"]
    errors = [r for r in results if r["status"] == "ERROR"]

    # --- Collision detection ---
    en_to_jp: dict[str, list[str]] = {}
    for result in passed:
        en_to_jp.setdefault(result["en_chapter"], []).append(result["jp_chapter"])

    collisions = [
        {"en_chapter": en_ch, "jp_chapters": jp_chs, "count": len(jp_chs)}
        for en_ch, jp_chs in en_to_jp.items()
        if len(jp_chs) > 1
    ]

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed : {len(passed)}")
    print(f"Failed : {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors : {len(errors)}")
    print()

    if collisions:
        print("⚠️  COLLISIONS DETECTED:")
        print("-" * 60)
        for collision in sorted(collisions, key=lambda x: x["en_chapter"]):
            print(f"  EN: {collision['en_chapter']}")
            for jp_ch in sorted(collision["jp_chapters"]):
                print(f"    <- JP: {jp_ch}")
        print()

    if passed:
        print("✓ Successfully arranged chapters:")
        for result in sorted(passed, key=lambda x: int(x["output_chapter"])):
            print(
                f"  Chapter {result['output_chapter']}: "
                f"{result['jp_chapter']} + {result['en_chapter']}"
            )

    print()
    print(
        f"Output files created in:\n"
        f"  {jp_output_dir}\n"
        f"  {en_output_dir}"
    )

    # --- State file ---
    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_file),
        "base_path": str(base),
        "jp_dir": str(jp_dir),
        "en_dir": str(en_dir),
        "output_dirs": {
            "jp": str(jp_output_dir),
            "en": str(en_output_dir),
        },
        "summary": {
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
            "errors": len(errors),
            "collisions": len(collisions),
        },
        "chapters": {
            result["jp_chapter"]: result
            for result in sorted(
                results,
                key=lambda x: (
                    x["jp_chapter_num"] is None,
                    x["jp_chapter_num"]
                    if x["jp_chapter_num"] is not None
                    else x["jp_chapter"],
                ),
            )
        },
    }

    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\nStructured state written to:\n  {state_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Validate and arrange translated light novel chapters "
            "using similarity matrix and LM Studio JSON verification."
        )
    )
    parser.add_argument(
        "matrix_path",
        help="Path to the similarity matrix JSON file",
    )
    parser.add_argument(
        "base_path",
        help="Path to the novel folder containing JP and EN subfolders",
    )
    parser.add_argument(
        "--jp-dir",
        default="JP",
        help="Name of the Japanese chapters directory (default: JP)",
    )
    parser.add_argument(
        "--en-dir",
        default="EN",
        help="Name of the English chapters directory (default: EN)",
    )

    args = parser.parse_args()

    main(
        args.matrix_path,
        args.base_path,
        args.jp_dir,
        args.en_dir,
    )