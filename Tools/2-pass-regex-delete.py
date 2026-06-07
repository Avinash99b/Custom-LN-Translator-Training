#!/usr/bin/env python3
"""
remove_note_chapters.py

Scan translated chapter files for common translator note / author note patterns.
If a file matches, delete it and its paired JP-Aligned file with the same chapter name.

Example structure:
Assets/Novels/NovelName/EN-Aligned/242.txt
Assets/Novels/NovelName/JP-Aligned/242.txt

Usage:
    python remove_note_chapters.py Assets/Novels
    python remove_note_chapters.py Assets/Novels --delete
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable


# Common patterns seen in light novel aligned files.
# Add/remove patterns here as needed.
PATTERNS: list[re.Pattern[str]] = [
    # =========================
    # Author / Translator Notes
    # =========================
    re.compile(r"\bAuthor\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bAuthor'?s?\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bTranslator\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bTranslator'?s?\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bTranslation\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bEditor'?s?\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bNote\s*to\s*Readers\s*:", re.IGNORECASE),
    re.compile(r"\bFootnote\s*:", re.IGNORECASE),

    # =========================
    # TL / TN Variants
    # =========================
    re.compile(r"\bTL\s*:", re.IGNORECASE),
    re.compile(r"\bTLN\s*:", re.IGNORECASE),
    re.compile(r"\bTN\s*:", re.IGNORECASE),
    re.compile(r"\bT\/N\s*:", re.IGNORECASE),

    re.compile(r"\bTL\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bTL\s*Notes?\s*:", re.IGNORECASE),
    re.compile(r"\bTN\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bTN\s*Notes?\s*:", re.IGNORECASE),

    re.compile(r"\bTranslator'?s?\s*Comments?\s*:", re.IGNORECASE),

    # =========================
    # Embedded Parenthetical Notes
    # =========================
    re.compile(r"\(\s*TL\s*:", re.IGNORECASE),
    re.compile(r"\(\s*TLN\s*:", re.IGNORECASE),
    re.compile(r"\(\s*TN\s*:", re.IGNORECASE),
    re.compile(r"\(\s*T\/N\s*:", re.IGNORECASE),

    re.compile(r"\(\s*TL\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\(\s*TN\s*Note\s*:", re.IGNORECASE),

    re.compile(r"\[\s*TL\s*:", re.IGNORECASE),
    re.compile(r"\[\s*TLN\s*:", re.IGNORECASE),
    re.compile(r"\[\s*TN\s*:", re.IGNORECASE),

    re.compile(r"【\s*TL\s*:", re.IGNORECASE),
    re.compile(r"【\s*TLN\s*:", re.IGNORECASE),
    re.compile(r"【\s*TN\s*:", re.IGNORECASE),

    # =========================
    # Common LN Fan-TL Artifacts
    # =========================
    re.compile(r"\bA\/N\s*:", re.IGNORECASE),
    re.compile(r"^\s*AN\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bAfterword\b", re.IGNORECASE),
    re.compile(r"\bTranslator\s*Corner\b", re.IGNORECASE),
    re.compile(r"\bTranslator\s*Comment\b", re.IGNORECASE),
    re.compile(r"\bTranslator\s*Comments\b", re.IGNORECASE),

    # =========================
    # Release Group Credits
    # =========================
    re.compile(r"\btranslated\s+by\b", re.IGNORECASE),
    re.compile(r"\bedited\s+by\b", re.IGNORECASE),
    re.compile(r"\bproofread\s+by\b", re.IGNORECASE),
    re.compile(r"\braw\s+provider\b", re.IGNORECASE),
    re.compile(r"\btranslation\s+group\b", re.IGNORECASE),

    # =========================
    # Common LN Sites
    # =========================
    re.compile(r"baka[- ]?tsuki", re.IGNORECASE),
    re.compile(r"novelupdates", re.IGNORECASE),
    re.compile(r"syosetu", re.IGNORECASE),
    re.compile(r"wuxiaworld", re.IGNORECASE),
    re.compile(r"royalroad", re.IGNORECASE),

    # =========================
    # Donation / Patreon Stuff
    # =========================
    re.compile(r"patreon", re.IGNORECASE),
    re.compile(r"buy\s+me\s+a\s+coffee", re.IGNORECASE),
    re.compile(r"paypal", re.IGNORECASE),
    re.compile(r"support\s+the\s+translator", re.IGNORECASE),
    re.compile(r"\bdonation\b", re.IGNORECASE),
    re.compile(r"\bdonate\b", re.IGNORECASE),

    # =========================
    # Japanese Explanation Notes
    # =========================
    re.compile(r"left\s+it\s+in\s+romaji", re.IGNORECASE),
    re.compile(r"left\s+in\s+romaji", re.IGNORECASE),
    re.compile(r"\bromanized\b", re.IGNORECASE),
    re.compile(r"\bhonorific\b", re.IGNORECASE),
    re.compile(r"\bkeikaku\s+means\s+plan\b", re.IGNORECASE),

    # =========================
    # Misc Fan-TL Commentary
    # =========================
    re.compile(r"\bTL\s+here\b", re.IGNORECASE),
    re.compile(r"\btranslator\s+here\b", re.IGNORECASE),
    re.compile(r"\bnote\s+from\s+translator\b", re.IGNORECASE),
    re.compile(r"\btranslator'?s?\s+rambling\b", re.IGNORECASE),
    re.compile(r"\bchapter\s+summary\b", re.IGNORECASE),
]

# Optional: lines that are often harmless, but still worth flagging if they appear as standalone notes.
# You can expand this if your dataset uses more variants.
EXTRA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bTranslator's\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bTranslation\s*Note\s*:", re.IGNORECASE),
    re.compile(r"\bNote\s*to\s*Readers\s*:", re.IGNORECASE),
]

ALL_PATTERNS = PATTERNS + EXTRA_PATTERNS


def file_contains_any_pattern(text: str) -> tuple[bool, list[str]]:
    """
    Returns (matched?, matched_pattern_names)
    """
    hits: list[str] = []
    for pat in ALL_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return (len(hits) > 0), hits


def paired_jp_file(en_file: Path) -> Path | None:
    """
    Convert .../EN-Aligned/242.txt -> .../JP-Aligned/242.txt
    """
    parts = list(en_file.parts)
    try:
        idx = parts.index("EN-Aligned")
    except ValueError:
        return None

    parts[idx] = "JP-Aligned"
    return Path(*parts)


def is_aligned_text_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".txt" and any(
        part.endswith("-Aligned") for part in path.parts
    )


def iter_en_aligned_txt_files(root: Path) -> Iterable[Path]:
    """
    Recursively find all .txt files under root that are in an EN-Aligned folder.
    """
    for path in root.rglob("*.txt"):
        if is_aligned_text_file(path) and "EN-Aligned" in path.parts:
            yield path


def delete_file(path: Path, *, enabled: bool) -> None:
    if not path.exists():
        return
    if enabled:
        path.unlink()
    else:
        # dry-run mode: do nothing
        return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and remove chapter files containing translator/author notes."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder containing novel, e.g. Assets/Novels/NovelName",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag, the script only prints matches.",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root folder does not exist or is not a directory: {root}")

    matched_count = 0
    deleted_pairs = 0
    missing_jp = 0

    for en_file in iter_en_aligned_txt_files(root):
        try:
            text = en_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[SKIP] Could not read: {en_file} ({e})")
            continue

        matched, hit_patterns = file_contains_any_pattern(text)
        if not matched:
            continue

        matched_count += 1
        jp_file = paired_jp_file(en_file)

        print(f"[MATCH] {en_file}")
        for pat in hit_patterns:
            print(f"        pattern: {pat}")

        if jp_file is None:
            print("        paired JP file: could not resolve")
            continue

        if jp_file.exists():
            print(f"        paired JP file: {jp_file}")
        else:
            print(f"        paired JP file: MISSING -> {jp_file}")
            missing_jp += 1

        if args.delete:
            try:
                delete_file(en_file, enabled=True)
                if jp_file.exists():
                    delete_file(jp_file, enabled=True)
                deleted_pairs += 1
                print("        deleted: yes")
            except Exception as e:
                print(f"        delete failed: {e}")
        else:
            print("        deleted: no (dry-run)")

    print("\nSummary")
    print(f"  matched EN files : {matched_count}")
    print(f"  deleted pairs     : {deleted_pairs if args.delete else 0} (dry-run)" if not args.delete else f"  deleted pairs     : {deleted_pairs}")
    print(f"  missing JP files  : {missing_jp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())