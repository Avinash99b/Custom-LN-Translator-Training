#!/usr/bin/env python3

import json
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

WORKERS = 10
LLM_SAMPLE_LINES = 50

MIN_LENGTH_RATIO = 0.15
MAX_LENGTH_RATIO = 8.0
MIN_SEMANTIC_SCORE = 0.45
SEMANTIC_WEIGHT = 0.85
LLM_WEIGHT = 0.15
CONFIDENCE_THRESHOLD = 0.57
CHAPTER_GAP_THRESHOLD = 20  # Max allowed gap in chapter numbers for matching
CHAPTER_GAP_SEMANTIC_THRESHOLD = 0.6  # Minimum semantic score to allow large chapter gaps
ARRANGEMENT_STATE_FILE = "arrangement_state.json"

OUTPUT_LOCK = Lock()


def read_text(path: Path):
    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def sample_text(text: str, max_lines: int):
    lines = text.strip().splitlines()

    if max_lines <= 0:
        return ""

    return "\n".join(lines[:max_lines])


def compute_biased_semantic_value(semantic_score: float):
    semantic_score = max(0.0, min(1.0, semantic_score))
    return SEMANTIC_WEIGHT * semantic_score


def parse_llm_response(raw_response: str):
    normalized = raw_response.strip().upper()

    normalized = re.sub(
        r"^[\s\"'`*_-]+|[\s\"'`*_.!,?:;\-]+$",
        "",
        normalized
    )

    if normalized == "SAME":
        return True

    if normalized == "DIFFER":
        return False

    raise RuntimeError(
        f"Unexpected model output: {raw_response.strip()}"
    )


def compute_confidence(semantic_score: float, llm_match: bool):
    semantic_score = max(0.0, min(1.0, semantic_score))
    llm_score = 1.0 if llm_match else 0.0
    return (
        SEMANTIC_WEIGHT * semantic_score
        + LLM_WEIGHT * llm_score
    )


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


def ask_llm(chapter, jp_text, en_text):

    prompt = f"""
You compare a Japanese light novel chapter and its English translation.

Your goal is to determine whether they tell the SAME STORY.

Focus on:
- major events
- character actions
- character relationships
- important dialogue outcomes
- locations
- chapter progression
- plot developments

IGNORE:
- wording differences
- translation style
- localization choices
- added humor
- rewritten jokes
- naturalization of expressions
- cultural adaptation
- sentence structure
- paragraph structure
- small additions that do not change the story
- small omissions that do not change the story
- differences in tone or narration

Answer SAME if:
- the plot is substantially the same
- the same events occur
- the same characters participate
- the same outcomes happen
- the chapter serves the same purpose in the story

Answer DIFFER only if there is a meaningful story mismatch, such as:
- missing scenes
- extra scenes
- different events
- different character actions
- different character relationships
- different outcomes
- different chapter progression
- merged or split chapters
- major omissions
- major additions
- content from another chapter

Be conservative.

If the story is mostly the same, answer SAME.

Answer with EXACTLY one word:

SAME

or

DIFFER

Japanese:

{jp_text}

English:

{en_text}
"""

    raw_response = ""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 2
                }
            },
            timeout=300
        )

        response.raise_for_status()

        raw_response = (
            response.json()
            .get("response", "")
        )
        raw_response = str(raw_response)

        match = parse_llm_response(raw_response)

        return {
            "accepted": match,
            "raw_response": raw_response,
            "parsed_response": "SAME" if match else "DIFFER"
        }

    except Exception as error:
        raise RuntimeError(
            f"LLM validation failed for {chapter[0]} -> {chapter[1]}: {error}"
        ) from error


def build_state_record(
    jp_name,
    en_name,
    semantic_score,
    jp_chapter_num,
    en_chapter_num,
    jp_len,
    en_len,
    ratio,
    llm_result,
    confidence,
    status,
    reason,
    output_chapter,
    output_written,
    llm_sample_lines
):
    chapter_gap = None

    if jp_chapter_num is not None and en_chapter_num is not None:
        chapter_gap = abs(jp_chapter_num - en_chapter_num)

    record = {
        "jp_chapter": jp_name,
        "en_chapter": en_name,
        "jp_chapter_num": jp_chapter_num,
        "en_chapter_num": en_chapter_num,
        "chapter_gap": chapter_gap,
        "status": status,
        "reason": reason,
        "semantic_val": round(semantic_score, 6),
        "biased_semantic_val": round(compute_biased_semantic_value(semantic_score), 6),
        "llm_accepted": None,
        "llm_val": None,
        "llm_parsed": None,
        "llm_raw_response": None,
        "confidence": None,
        "length_ratio": None,
        "jp_length": jp_len,
        "en_length": en_len,
        "llm_sample_lines": llm_sample_lines,
        "output_chapter": output_chapter,
        "output_written": output_written,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if ratio is not None:
        record["length_ratio"] = round(ratio, 6)

    if llm_result is not None:
        record["llm_accepted"] = llm_result["accepted"]
        record["llm_val"] = 1.0 if llm_result["accepted"] else 0.0
        record["llm_parsed"] = llm_result["parsed_response"]
        record["llm_raw_response"] = llm_result["raw_response"]

    if confidence is not None:
        record["confidence"] = round(confidence, 6)

    return record


def validate_and_arrange(
    jp_name,
    en_name,
    semantic_score,
    jp_file,
    en_file,
    jp_output_dir,
    en_output_dir,
    llm_sample_lines
):
    """Validate a JP-EN pair and arrange output if valid."""

    jp_chapter_num = extract_chapter_number(jp_name)
    en_chapter_num = extract_chapter_number(en_name)
    jp_full = read_text(jp_file)
    en_full = read_text(en_file)

    jp_len = len(jp_full)
    en_len = len(en_full)

    ratio = en_len / max(jp_len, 1)

    if (
        jp_chapter_num is not None
        and en_chapter_num is not None
        and abs(jp_chapter_num - en_chapter_num) > CHAPTER_GAP_THRESHOLD
        and semantic_score < CHAPTER_GAP_SEMANTIC_THRESHOLD
    ):
        return build_state_record(
            jp_name,
            en_name,
            semantic_score,
            jp_chapter_num,
            en_chapter_num,
            jp_len,
            en_len,
            ratio,
            None,
            compute_confidence(semantic_score, False),
            "SKIP",
            (
                f"chapter_gap_too_large "
                f"(gap={abs(jp_chapter_num - en_chapter_num)}, "
                f"semantic={semantic_score:.2f})"
            ),
            None,
            False,
            llm_sample_lines
        )

    if semantic_score < MIN_SEMANTIC_SCORE:
        return build_state_record(
            jp_name,
            en_name,
            semantic_score,
            jp_chapter_num,
            en_chapter_num,
            jp_len,
            en_len,
            ratio,
            None,
            compute_confidence(semantic_score, False),
            "SKIP",
            (
                f"semantic_score_too_low "
                f"({semantic_score:.2f})"
            ),
            None,
            False,
            llm_sample_lines
        )

    if ratio < MIN_LENGTH_RATIO:
        return build_state_record(
            jp_name,
            en_name,
            semantic_score,
            jp_chapter_num,
            en_chapter_num,
            jp_len,
            en_len,
            ratio,
            None,
            compute_confidence(semantic_score, False),
            "SKIP",
            (
                f"length_ratio_too_small "
                f"({ratio:.2f})"
            ),
            None,
            False,
            llm_sample_lines
        )

    if ratio > MAX_LENGTH_RATIO:
        return build_state_record(
            jp_name,
            en_name,
            semantic_score,
            jp_chapter_num,
            en_chapter_num,
            jp_len,
            en_len,
            ratio,
            None,
            compute_confidence(semantic_score, False),
            "SKIP",
            (
                f"length_ratio_too_large "
                f"({ratio:.2f})"
            ),
            None,
            False,
            llm_sample_lines
        )

    jp_sample = sample_text(jp_full, llm_sample_lines)
    en_sample = sample_text(en_full, llm_sample_lines)

    try:
        llm_result = ask_llm(
            (jp_name, en_name),
            jp_sample,
            en_sample
        )
    except Exception as e:
        return build_state_record(
            jp_name,
            en_name,
            semantic_score,
            jp_chapter_num,
            en_chapter_num,
            jp_len,
            en_len,
            ratio,
            None,
            None,
            "ERROR",
            str(e),
            None,
            False,
            llm_sample_lines
        )

    match = llm_result["accepted"]

    confidence = compute_confidence(
        semantic_score,
        match
    )

    if confidence < CONFIDENCE_THRESHOLD:
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
            confidence,
            "SKIP",
            (
                f"low_confidence "
                f"(semantic={semantic_score:.2f}, "
                f"llm={'SAME' if match else 'DIFFER'}, "
                f"confidence={confidence:.2f})"
            ),
            None,
            False,
            llm_sample_lines
        )

    if not match and confidence <= CONFIDENCE_THRESHOLD:
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
            confidence,
            "FAIL",
            "story_drift",
            None,
            False,
            llm_sample_lines
        )

    if jp_chapter_num is None:
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
            confidence,
            "SKIP",
            "could_not_extract_chapter_number",
            None,
            False,
            llm_sample_lines
        )

    chapter_num = str(jp_chapter_num)
    output_name = f"{chapter_num}.txt"

    # Write output files
    with OUTPUT_LOCK:
        jp_out_path = jp_output_dir / output_name
        en_out_path = en_output_dir / output_name

        jp_out_path.write_text(jp_full, encoding="utf-8")
        en_out_path.write_text(en_full, encoding="utf-8")

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
        confidence,
        "PASS",
        "same_story",
        chapter_num,
        True,
        llm_sample_lines
    )


def load_similarity_matrix(matrix_path: Path):
    """Load the similarity matrix JSON."""
    try:
        data = json.loads(
            matrix_path.read_text(
                encoding="utf-8"
            )
        )
        return data
    except Exception as e:
        raise RuntimeError(
            f"Failed to load similarity matrix: {e}"
        )


def find_best_matches(
    matrix,
    jp_dir,
    en_dir
):
    """For each JP chapter, find the best EN match."""

    matches = {}

    for jp_name, en_scores in matrix.items():
        if not en_scores:
            continue

        # Find highest score
        best_en = max(
            en_scores.items(),
            key=lambda x: x[1]
        )

        en_name = best_en[0]
        score = best_en[1]

        jp_file = jp_dir / jp_name
        en_file = en_dir / en_name

        if not jp_file.exists():
            continue

        if not en_file.exists():
            continue

        matches[jp_name] = {
            "en_chapter": en_name,
            "score": score,
            "jp_file": jp_file,
            "en_file": en_file
        }

    return matches


def main(
    matrix_path,
    base_path,
    jp_dir_name="JP",
    en_dir_name="EN",
    llm_sample_lines=LLM_SAMPLE_LINES
):

    matrix_file = Path(matrix_path)
    if not matrix_file.exists():
        raise RuntimeError(
            f"Similarity matrix not found: {matrix_path}"
        )

    base = Path(base_path)
    jp_dir = base / jp_dir_name
    en_dir = base / en_dir_name

    if not jp_dir.exists():
        raise RuntimeError(
            f"Missing folder: {jp_dir}"
        )

    if not en_dir.exists():
        raise RuntimeError(
            f"Missing folder: {en_dir}"
        )

    jp_output_dir = base / "JP-Output"
    en_output_dir = base / "EN-Output"

    jp_output_dir.mkdir(exist_ok=True)
    en_output_dir.mkdir(exist_ok=True)

    state_path = base / ARRANGEMENT_STATE_FILE

    print("Loading similarity matrix...")
    matrix = load_similarity_matrix(matrix_file)

    print("Finding best matches...")
    matches = find_best_matches(
        matrix,
        jp_dir,
        en_dir
    )

    print(f"Found {len(matches)} potential matches")
    print()

    results = []
    futures = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        for jp_name, match_info in sorted(
            matches.items(),
            key=lambda item: chapter_sort_key(item[0])
        ):

            futures.append(
                executor.submit(
                    validate_and_arrange,
                    jp_name,
                    match_info["en_chapter"],
                    match_info["score"],
                    match_info["jp_file"],
                    match_info["en_file"],
                    jp_output_dir,
                    en_output_dir,
                    llm_sample_lines
                )
            )

        total = len(futures)
        completed = 0

        for future in as_completed(futures):

            completed += 1

            try:

                result = future.result()
                results.append(result)

                print(
                    f"[{completed}/{total}] "
                    f"{result['status']} "
                    f"{result['jp_chapter']} -> "
                    f"{result['en_chapter']} "
                    f"({result['reason']})"
                )

            except Exception as e:

                print(
                    f"[{completed}/{total}] "
                    f"ERROR: {e}"
                )

    # Analyze results
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    skipped = [r for r in results if r["status"] == "SKIP"]
    errors = [r for r in results if r["status"] == "ERROR"]

    # Detect collisions (multiple JP chapters -> same EN chapter)
    en_to_jp = {}
    collisions = []

    for result in passed:
        en_ch = result["en_chapter"]
        jp_ch = result["jp_chapter"]

        if en_ch not in en_to_jp:
            en_to_jp[en_ch] = []

        en_to_jp[en_ch].append(jp_ch)

    for en_ch, jp_chapters in en_to_jp.items():
        if len(jp_chapters) > 1:
            collisions.append({
                "en_chapter": en_ch,
                "jp_chapters": jp_chapters,
                "count": len(jp_chapters)
            })

    # Summary
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
        for collision in sorted(
            collisions,
            key=lambda x: x["en_chapter"]
        ):
            print(
                f"  EN: {collision['en_chapter']}"
            )
            for jp_ch in sorted(
                collision['jp_chapters']
            ):
                print(f"    <- JP: {jp_ch}")
        print()

    if passed:
        print("✓ Successfully arranged chapters:")
        for result in sorted(
            passed,
            key=lambda x: int(x['output_chapter'])
        ):
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

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_file),
        "base_path": str(base),
        "jp_dir": str(jp_dir),
        "en_dir": str(en_dir),
        "output_dirs": {
            "jp": str(jp_output_dir),
            "en": str(en_output_dir)
        },
        "summary": {
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
            "errors": len(errors),
            "collisions": len(collisions)
        },
        "chapters": {
            result["jp_chapter"]: result
            for result in sorted(
                results,
                key=lambda x: (
                    x["jp_chapter_num"] is None,
                    x["jp_chapter_num"] if x["jp_chapter_num"] is not None else x["jp_chapter"]
                )
            )
        }
    }

    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\nStructured state written to:\n  {state_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Validate and arrange translated light novel chapters "
            "using similarity matrix and LLM verification."
        )
    )
    parser.add_argument(
        "matrix_path",
        help="Path to the similarity matrix JSON file"
    )
    parser.add_argument(
        "base_path",
        help="Path to the novel folder containing JP and EN subfolders"
    )
    parser.add_argument(
        "--jp-dir",
        default="JP",
        help="Name of the Japanese chapters directory (default: JP)"
    )
    parser.add_argument(
        "--en-dir",
        default="EN",
        help="Name of the English chapters directory (default: EN)"
    )
    parser.add_argument(
        "--llm-sample-lines",
        type=int,
        default=LLM_SAMPLE_LINES,
        help=(
            "Number of lines from the start of each chapter to send to the "
            "LLM (default: 120)"
        )
    )

    args = parser.parse_args()

    main(
        args.matrix_path,
        args.base_path,
        args.jp_dir,
        args.en_dir,
        args.llm_sample_lines
    )
