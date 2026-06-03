#!/usr/bin/env python3

import json
import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

WORKERS = 4
SAMPLE_SIZE = 1000

MIN_LENGTH_RATIO = 0.15
MAX_LENGTH_RATIO = 8.0
LLM_RESPONSE_LOG = "llm_response_log.jsonl"

LOG_LOCK = Lock()


def read_text(path: Path):
    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def sample_text(text: str):
    text = text.strip()

    if len(text) <= SAMPLE_SIZE * 3:
        return text

    middle_start = max(
        0,
        len(text) // 2 - SAMPLE_SIZE // 2
    )

    return (
        text[:SAMPLE_SIZE]
        + "\n\n[MIDDLE]\n\n"
        + text[
            middle_start:
            middle_start + SAMPLE_SIZE
        ]
        + "\n\n[END]\n\n"
        + text[-SAMPLE_SIZE:]
    )


def log_llm_exchange(log_path: Path, entry: dict):
    record = dict(entry)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    line = json.dumps(
        record,
        ensure_ascii=False
    )

    with LOG_LOCK:
        with log_path.open(
            "a",
            encoding="utf-8"
        ) as handle:
            handle.write(line)
            handle.write("\n")


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


def ask_llm(chapter, jp_text, en_text, log_path: Path):

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

        log_llm_exchange(
            log_path,
            {
                "chapter": chapter,
                "status": "ok",
                "raw_response": raw_response,
                "parsed": "SAME" if match else "DIFFER"
            }
        )

        return match

    except Exception as error:
        log_llm_exchange(
            log_path,
            {
                "chapter": chapter,
                "status": "error",
                "raw_response": raw_response,
                "error": str(error)
            }
        )

        raise


def validate_pair(
    chapter,
    jp_file,
    en_file,
    log_path
):

    jp_full = read_text(jp_file)
    en_full = read_text(en_file)

    jp_len = len(jp_full)
    en_len = len(en_full)

    ratio = (
        en_len /
        max(jp_len, 1)
    )

    if ratio < MIN_LENGTH_RATIO:
        return {
            "chapter": chapter,
            "status": "FAIL",
            "reason": (
                f"length_ratio_too_small "
                f"({ratio:.2f})"
            )
        }

    if ratio > MAX_LENGTH_RATIO:
        return {
            "chapter": chapter,
            "status": "FAIL",
            "reason": (
                f"length_ratio_too_large "
                f"({ratio:.2f})"
            )
        }

    jp_sample = sample_text(jp_full)
    en_sample = sample_text(en_full)

    match = ask_llm(
        chapter,
        jp_sample,
        en_sample,
        log_path
    )

    return {
        "chapter": chapter,
        "status":
            "PASS"
            if match
            else "FAIL",
        "reason":
            "same_story"
            if match
            else "story_drift"
    }


def load_existing(report_path):

    if not report_path.exists():
        return {}

    try:
        data = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        return {
            int(item["chapter"]): item
            for item in data
        }

    except Exception:
        return {}


def save_results(
    report_path,
    results
):

    ordered = sorted(
        results.values(),
        key=lambda x:
        x["chapter"]
    )

    report_path.write_text(
        json.dumps(
            ordered,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def find_jp_chapters(jp_dir):

    chapters = {}

    for file in jp_dir.glob("*.txt"):

        match = re.search(
            r"ch-(\d+)",
            file.name
        )

        if match:
            chapter = int(
                match.group(1)
            )

            chapters[chapter] = file

    return chapters


def main(base_path, start_chapter: int):

    base = Path(base_path)

    jp_dir = base / "JP"
    en_dir = base / "EN"

    if not jp_dir.exists():
        raise RuntimeError(
            f"Missing folder: {jp_dir}"
        )

    if not en_dir.exists():
        raise RuntimeError(
            f"Missing folder: {en_dir}"
        )

    report_path = (
        base /
        "validation_report.json"
    )

    llm_log_path = base / LLM_RESPONSE_LOG

    results = load_existing(
        report_path
    )

    if start_chapter > 1:
        results = {
            chapter: result
            for chapter, result in results.items()
            if chapter < start_chapter
        }

    jp_chapters = find_jp_chapters(
        jp_dir
    )

    futures = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        for chapter in sorted(
            jp_chapters.keys()
        ):

            if chapter < start_chapter:
                continue

            if chapter in results:
                continue

            en_file = (
                en_dir /
                f"{chapter}.txt"
            )

            if not en_file.exists():

                results[chapter] = {
                    "chapter": chapter,
                    "status": "SKIP",
                    "reason":
                        "missing_english"
                }

                continue

            futures.append(
                executor.submit(
                    validate_pair,
                    chapter,
                    jp_chapters[
                        chapter
                    ],
                    en_file,
                    llm_log_path
                )
            )

        total = len(futures)
        completed = 0

        for future in as_completed(
            futures
        ):

            completed += 1

            try:

                result = future.result()

                results[
                    result["chapter"]
                ] = result

                save_results(
                    report_path,
                    results
                )

                print(
                    f"[{completed}/{total}] "
                    f"{result['status']} "
                    f"Chapter "
                    f"{result['chapter']} "
                    f"({result['reason']})"
                )

            except Exception as e:

                print(
                    f"[{completed}/{total}] "
                    f"ERROR: {e}"
                )

    save_results(
        report_path,
        results
    )

    passed = sum(
        1
        for r in results.values()
        if r["status"] == "PASS"
    )

    failed = sum(
        1
        for r in results.values()
        if r["status"] == "FAIL"
    )

    skipped = sum(
        1
        for r in results.values()
        if r["status"] == "SKIP"
    )

    print()
    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    print(f"Skipped: {skipped}")
    print()
    print(
        f"Saved report to:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Validate translated light novel chapters with an LLM."
        )
    )
    parser.add_argument(
        "base_path",
        help="Path to the novel folder containing JP and EN subfolders"
    )
    parser.add_argument(
        "-s",
        "--start-chapter",
        type=int,
        default=1,
        help=(
            "Start validating from this chapter number onward"
        )
    )

    args = parser.parse_args()

    if args.start_chapter < 1:
        print(
            "Error: --start-chapter must be >= 1",
            file=sys.stderr
        )
        sys.exit(1)

    main(
        args.base_path,
        args.start_chapter
    )
