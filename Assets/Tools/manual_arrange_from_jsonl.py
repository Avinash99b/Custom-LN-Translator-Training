#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


FAILED_STATUSES = {"FAIL", "DIFFER"}
DEFAULT_JSONL_NAME = "llm_response_log.jsonl"


def extract_chapter_number(name: str):
    match = re.search(r"(\d+)(?=\.txt$)", name)
    if not match:
        return None
    return int(match.group(1))


def read_jsonl(path: Path):
    entries = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue

            try:
                entries.append(json.loads(text))
            except json.JSONDecodeError as error:
                print(
                    f"Skipping invalid JSONL line {line_number} in {path.name}: {error}"
                )

    return entries


def find_jsonl_file(base_path: Path, explicit_jsonl):
    if explicit_jsonl:
        candidate = Path(explicit_jsonl)
        if not candidate.is_absolute():
            candidate = base_path / candidate
        if not candidate.exists():
            raise RuntimeError(f"JSONL file not found: {candidate}")
        return candidate

    preferred = base_path / DEFAULT_JSONL_NAME
    if preferred.exists():
        return preferred

    jsonl_files = sorted(base_path.glob("*.jsonl"))
    if not jsonl_files:
        return None

    if len(jsonl_files) == 1:
        return jsonl_files[0]

    print("Multiple JSONL files found; using the first one:")
    for file in jsonl_files:
        print(f"  {file.name}")

    return jsonl_files[0]


def extract_failed_pairs(entries):
    pairs = []

    for entry in entries:
        jp_chapter = entry.get("jp_chapter")
        en_chapter = entry.get("en_chapter")

        if jp_chapter is None and en_chapter is None and "chapter" in entry:
            chapter = str(entry["chapter"])
            jp_chapter = f"{chapter}.txt"
            en_chapter = f"{chapter}.txt"

        if not jp_chapter or not en_chapter:
            continue

        status = str(entry.get("status", "")).upper()
        parsed = str(entry.get("parsed", "")).upper()

        if status in FAILED_STATUSES or parsed in FAILED_STATUSES:
            pairs.append(
                {
                    "jp_chapter": str(jp_chapter),
                    "en_chapter": str(en_chapter),
                    "source": entry,
                }
            )

    seen = set()
    unique_pairs = []

    for pair in pairs:
        key = (pair["jp_chapter"], pair["en_chapter"])
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)

    return unique_pairs


def ask_yes_no(prompt: str):
    while True:
        answer = input(f"{prompt} [Y/n]: ").strip().lower()

        if answer in {"", "y", "yes"}:
            return True

        if answer in {"", "n", "no"}:
            return False

        print("Please answer y or n.")


def ask_chapter_number(prompt: str):
    while True:
        answer = input(prompt).strip()
        if not answer:
            print("Chapter number cannot be empty.")
            continue

        if not answer.isdigit():
            print("Please enter a numeric chapter number.")
            continue

        return int(answer)


def copy_pair(jp_file: Path, en_file: Path, output_name: str, jp_output_dir: Path, en_output_dir: Path):
    jp_target = jp_output_dir / output_name
    en_target = en_output_dir / output_name

    shutil.copy2(jp_file, jp_target)
    shutil.copy2(en_file, en_target)


def update_jsonl_entries(jsonl_path: Path, jp_name: str, en_name: str):
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    updated_lines = []
    updated_count = 0

    for line in lines:
        text = line.strip()
        if not text:
            updated_lines.append(line)
            continue

        try:
            entry = json.loads(text)
        except json.JSONDecodeError:
            updated_lines.append(line)
            continue

        matches_pair = (
            str(entry.get("jp_chapter", "")) == jp_name
            and str(entry.get("en_chapter", "")) == en_name
        )

        if matches_pair and (
            str(entry.get("status", "")).upper() in FAILED_STATUSES
            or str(entry.get("parsed", "")).upper() in FAILED_STATUSES
        ):
            entry["status"] = "ok"
            entry["parsed"] = "SAME"
            updated_count += 1

        updated_lines.append(
            json.dumps(entry, ensure_ascii=False)
        )

    if updated_count:
        jsonl_path.write_text(
            "\n".join(updated_lines) + "\n",
            encoding="utf-8"
        )

    return updated_count


def validate_and_copy_pair(
    jsonl_path: Path,
    jp_name: str,
    en_name: str,
    jp_dir: Path,
    en_dir: Path,
    jp_output_dir: Path,
    en_output_dir: Path,
    used_output_names: set,
    used_en_sources: set,
    copied: list,
    collisions: list,
    skipped: list,
):
    jp_file = jp_dir / jp_name
    en_file = en_dir / en_name

    if not jp_file.exists() or not en_file.exists():
        skipped.append(
            {
                "jp_chapter": jp_name,
                "en_chapter": en_name,
                "reason": "missing_source_file",
            }
        )
        print(f"SKIP  Chapter JAP: {jp_name}, EN: {en_name} (missing source file)")
        return

    output_number = extract_chapter_number(jp_name)
    if output_number is None:
        skipped.append(
            {
                "jp_chapter": jp_name,
                "en_chapter": en_name,
                "reason": "could_not_extract_output_chapter",
            }
        )
        print(
            f"SKIP  Chapter JAP: {jp_name}, EN: {en_name} (could not extract output chapter number)"
        )
        return

    output_name = f"{output_number}.txt"

    if output_name in used_output_names or (jp_output_dir / output_name).exists() or (en_output_dir / output_name).exists():
        collisions.append(
            {
                "jp_chapter": jp_name,
                "en_chapter": en_name,
                "reason": "output_chapter_collision",
                "output_chapter": output_name,
            }
        )
        print(
            f"COLLISION output chapter {output_name} already exists or was already assigned; skipping."
        )
        return

    if en_name in used_en_sources:
        collisions.append(
            {
                "jp_chapter": jp_name,
                "en_chapter": en_name,
                "reason": "english_source_collision",
                "output_chapter": output_name,
            }
        )
        print(
            f"COLLISION English chapter {en_name} was already assigned to another output chapter; skipping."
        )
        return

    updated_count = update_jsonl_entries(jsonl_path, jp_name, en_name)
    if updated_count:
        print(f"Updated {updated_count} JSONL entr{'y' if updated_count == 1 else 'ies'} to SAME.")

    copy_pair(jp_file, en_file, output_name, jp_output_dir, en_output_dir)

    used_output_names.add(output_name)
    used_en_sources.add(en_name)
    copied.append(
        {
            "jp_chapter": jp_name,
            "en_chapter": en_name,
            "output_chapter": output_name,
        }
    )

    print(
        f"COPIED JAP {jp_name} + EN {en_name} -> output chapter {output_name}"
    )


def main(base_path: str, jsonl_path):
    base = Path(base_path)

    jp_dir = base / "JP"
    en_dir = base / "EN"

    if not jp_dir.exists():
        raise RuntimeError(f"Missing folder: {jp_dir}")

    if not en_dir.exists():
        raise RuntimeError(f"Missing folder: {en_dir}")

    detected_jsonl = find_jsonl_file(base, jsonl_path)
    if detected_jsonl is None:
        print(f"No JSONL file found in {base}")
        return

    print(f"Using JSONL file: {detected_jsonl}")

    entries = read_jsonl(detected_jsonl)
    failed_pairs = extract_failed_pairs(entries)

    if not failed_pairs:
        print("No failed chapter pairs found in the JSONL file.")
        return

    failed_pairs.sort(
        key=lambda item: (
            extract_chapter_number(item["jp_chapter"]) or 10**18,
            item["jp_chapter"],
            item["en_chapter"],
        )
    )

    jp_output_dir = base / "JP-Output"
    en_output_dir = base / "EN-Output"
    jp_output_dir.mkdir(exist_ok=True)
    en_output_dir.mkdir(exist_ok=True)

    used_output_names = set()
    used_en_sources = set()
    copied = []
    skipped = []
    collisions = []

    print()
    print(f"Found {len(failed_pairs)} failed chapter pair(s).")
    print()

    for pair in failed_pairs:
        jp_name = pair["jp_chapter"]
        en_name = pair["en_chapter"]

        jp_file = jp_dir / jp_name
        en_file = en_dir / en_name

        if not jp_file.exists() or not en_file.exists():
            skipped.append(
                {
                    "jp_chapter": jp_name,
                    "en_chapter": en_name,
                    "reason": "missing_source_file",
                }
            )
            print(f"SKIP  Chapter JAP: {jp_name}, EN: {en_name} (missing source file)")
            continue

        output_number = extract_chapter_number(jp_name)
        if output_number is None:
            skipped.append(
                {
                    "jp_chapter": jp_name,
                    "en_chapter": en_name,
                    "reason": "could_not_extract_output_chapter",
                }
            )
            print(
                f"SKIP  Chapter JAP: {jp_name}, EN: {en_name} (could not extract output chapter number)"
            )
            continue

        output_name = f"{output_number}.txt"

        print()
        print(f"Failed pair: JAP {jp_name} <-> EN {en_name}")

        if ask_yes_no(
            f"Have you manually verified Chapter JAP: {jp_name}, EN: {en_name} as valid?"
        ):
            validate_and_copy_pair(
                detected_jsonl,
                jp_name,
                en_name,
                jp_dir,
                en_dir,
                jp_output_dir,
                en_output_dir,
                used_output_names,
                used_en_sources,
                copied,
                collisions,
                skipped,
            )
            continue

        if not ask_yes_no(
            f"Have you found the actual matching chapter pair for JAP {jp_name} and EN {en_name}?"
        ):
            skipped.append(
                {
                    "jp_chapter": jp_name,
                    "en_chapter": en_name,
                    "reason": "user_declined_manual_verification",
                }
            )
            print("Skipped by user.")
            continue

        actual_jp_number = ask_chapter_number(
            "Enter actual JP chapter number: "
        )
        actual_en_number = ask_chapter_number(
            "Enter actual EN chapter number: "
        )

        actual_jp_name = f"{actual_jp_number}.txt"
        actual_en_name = f"{actual_en_number}.txt"

        print(
            f"Using actual pair JAP {actual_jp_name} / EN {actual_en_name}"
        )

        validate_and_copy_pair(
            detected_jsonl,
            actual_jp_name,
            actual_en_name,
            jp_dir,
            en_dir,
            jp_output_dir,
            en_output_dir,
            used_output_names,
            used_en_sources,
            copied,
            collisions,
            skipped,
        )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Copied   : {len(copied)}")
    print(f"Skipped  : {len(skipped)}")
    print(f"Collisions: {len(collisions)}")

    if collisions:
        print()
        print("Collisions:")
        for item in collisions:
            print(
                f"  JAP {item['jp_chapter']} / EN {item['en_chapter']} -> {item['reason']} ({item['output_chapter']})"
            )

    print()
    print(f"JP output: {jp_output_dir}")
    print(f"EN output: {en_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Interactively recover failed chapter pairs from a JSONL log and copy validated pairs into output folders."
        )
    )
    parser.add_argument(
        "base_path",
        help="Path to the novel folder containing JP and EN subfolders"
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optional JSONL file name or path. Defaults to llm_response_log.jsonl if present."
    )

    args = parser.parse_args()

    try:
        main(args.base_path, args.jsonl)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)