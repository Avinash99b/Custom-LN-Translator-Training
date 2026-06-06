#!/usr/bin/env python3
"""
delete_failed_chapters.py

Reads a validation JSON report and deletes chapter pairs that failed validation.

Features
--------
- Verifies JP and EN chapter numbers match before deletion.
- Shows a deletion plan first.
- Requires explicit confirmation ("yup go ahead").
- Deletes both JP and EN files for failed chapters.
- Detects if failed chapters were already deleted.
- Prints:
      "3rd pass deletion already completed"
  when all FAIL chapters are already gone.
- Safe to run repeatedly.

Usage
-----
python delete_failed_chapters.py validation_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIRM_TEXT = "yup go ahead"


def extract_chapter_number(path_str: str) -> str:
    """
    Extract chapter number from:
        .../1.txt   -> "1"
        .../23.txt  -> "23"
    """
    return Path(path_str).stem


def load_report(report_path: Path) -> dict:
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_deletion_plan(report: dict):
    chapters = report.get("chapters", [])

    failures = []
    errors = []

    for ch in chapters:
        status = str(ch.get("status", "")).upper()

        if status != "FAIL":
            continue

        jp_file = ch.get("jp_file")
        en_file = ch.get("en_file")

        if not jp_file or not en_file:
            errors.append(
                f"Missing jp_file/en_file for chapter_id={ch.get('chapter_id')}"
            )
            continue

        jp_num = extract_chapter_number(jp_file)
        en_num = extract_chapter_number(en_file)

        # Mandatory chapter-number consistency check
        if jp_num != en_num:
            errors.append(
                f"\nChapter number mismatch detected\n"
                f"JP file : {jp_file}\n"
                f"EN file : {en_file}\n"
                f"JP num  : {jp_num}\n"
                f"EN num  : {en_num}\n"
            )
            continue

        failures.append(
            {
                "chapter": jp_num,
                "jp_file": Path(jp_file),
                "en_file": Path(en_file),
                "reason": ch.get("reason", ""),
            }
        )

    return failures, errors


def print_plan(failures):
    print("\n================ DELETION PLAN ================\n")

    for item in failures:
        print(f"Chapter {item['chapter']}")
        print(f"  JP: {item['jp_file']}")
        print(f"  EN: {item['en_file']}")

        if item["reason"]:
            print(f"  Reason: {item['reason']}")

        print()

    print(f"Total failed chapter pairs to delete: {len(failures)}")
    print("\n===============================================\n")


def delete_files(failures):
    deleted_pairs = 0

    for item in failures:
        print(f"\nProcessing Chapter {item['chapter']}")

        for file_path in (item["jp_file"], item["en_file"]):
            try:
                if file_path.exists():
                    file_path.unlink()
                    print(f"Deleted: {file_path}")
                else:
                    print(f"Already missing: {file_path}")

            except Exception as e:
                print(f"ERROR deleting {file_path}: {e}")

        deleted_pairs += 1

    return deleted_pairs


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "    python delete_failed_chapters.py validation_report.json"
        )
        sys.exit(1)

    report_path = Path(sys.argv[1])

    if not report_path.exists():
        print(f"ERROR: JSON file not found:\n{report_path}")
        sys.exit(1)

    try:
        report = load_report(report_path)
    except Exception as e:
        print(f"Failed to load JSON:\n{e}")
        sys.exit(1)

    failures, errors = build_deletion_plan(report)

    # --------------------------------------------------
    # Mandatory consistency checks
    # --------------------------------------------------
    if errors:
        print("\nERRORS DETECTED:\n")

        for err in errors:
            print(err)

        print(
            "\nAborting because mandatory chapter-number "
            "consistency checks failed."
        )
        sys.exit(1)

    if not failures:
        print("No failed chapters found in validation report.")
        return

    # --------------------------------------------------
    # Check if chapters were already deleted
    # --------------------------------------------------
    existing_failures = []
    already_deleted = []

    for item in failures:
        jp_exists = item["jp_file"].exists()
        en_exists = item["en_file"].exists()

        if not jp_exists and not en_exists:
            already_deleted.append(item)
        else:
            existing_failures.append(item)

    # Everything already removed
    if len(already_deleted) == len(failures):
        print(
            "\n3rd pass deletion already completed.\n"
            f"All {len(failures)} failed chapter pairs "
            f"have already been deleted.\n"
        )
        return

    if already_deleted:
        print(
            f"\n{len(already_deleted)} failed chapter pair(s) "
            f"were already deleted in a previous run.\n"
        )

    failures = existing_failures

    if not failures:
        print(
            "\n3rd pass deletion already completed.\n"
            "Nothing left to delete.\n"
        )
        return

    # --------------------------------------------------
    # Show deletion plan
    # --------------------------------------------------
    print_plan(failures)

    print(
        "Type exactly the following text to continue:\n\n"
        f"    {CONFIRM_TEXT}\n"
    )

    user_input = input("Confirmation: ").strip()

    if user_input != CONFIRM_TEXT:
        print("\nDeletion cancelled.")
        return

    print("\nStarting deletion...\n")

    deleted_pairs = delete_files(failures)

    print("\n===============================================")
    print("Deletion completed.")
    print(f"Chapter pairs processed : {deleted_pairs}")
    print(f"Files expected removed  : {deleted_pairs * 2}")
    print("===============================================\n")


if __name__ == "__main__":
    main()