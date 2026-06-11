#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

JP_PATTERN = re.compile(
    r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]'
)

MIN_JP_CHARS = 20  # Adjust if needed


def count_japanese_chars(path: Path) -> int:
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
        return len(JP_PATTERN.findall(text))
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Find EN files containing Japanese text and optionally delete paired files."
    )

    parser.add_argument(
        "root",
        help="Novel folder containing EN-Output and JP-Output"
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag, runs in dry-run mode."
    )

    args = parser.parse_args()

    root = Path(args.root)

    en_dir = root / "EN-Output"
    jp_dir = root / "JP-Output"

    if not en_dir.exists():
        raise FileNotFoundError(f"Missing: {en_dir}")

    if not jp_dir.exists():
        raise FileNotFoundError(f"Missing: {jp_dir}")

    print(f"Mode: {'DELETE' if args.delete else 'DRY RUN'}")
    print(f"Novel: {root.name}\n")

    matches = 0

    for en_file in sorted(en_dir.rglob("*.txt")):
        jp_count = count_japanese_chars(en_file)

        if jp_count < MIN_JP_CHARS:
            continue

        jp_file = jp_dir / en_file.relative_to(en_dir)

        matches += 1

        print(f"[MATCH] {en_file.name} ({jp_count} JP chars)")
        print(f"  EN: {en_file}")
        print(f"  JP: {jp_file if jp_file.exists() else 'MISSING'}")

        if args.delete:
            en_file.unlink(missing_ok=True)
            if jp_file.exists():
                jp_file.unlink()

            print("  Deleted pair")

        print()

    print(f"Found {matches} suspicious pair(s).")

    if not args.delete:
        print("Dry run complete. Re-run with --delete to remove files.")


if __name__ == "__main__":
    main()