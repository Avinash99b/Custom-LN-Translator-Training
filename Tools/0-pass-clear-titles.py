#!/usr/bin/env python3
"""
clean_chapter_titles.py

Removes chapter numbering headers from the first 4 non-empty lines of every
.txt file under NovelName/EN/ and NovelName/JP/.

Handles:
    Chapter 8 : Hero's Return      -> Hero's Return
    Volume 22 Chapter 7: A Second  -> A Second
    Chapter 8                      -> (line removed)
    第8話 勇者の帰還                -> 勇者の帰還
    第八話                         -> (line removed)

Usage:
    python clean_chapter_titles.py Assets/Novels/MushokuTensei
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

MAX_PREVIEW_FILES = 20

# Lines that have a chapter number AND a title -> keep only the title
TITLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*chapter\s+\d+(?:\.\d+)?\s*[:：\-–—]\s+(.+?)\s*$", re.IGNORECASE), "Chapter title normalized"),
    (re.compile(r"^\s*(?:volume|vol\.?)\s*\d+\s+(?:chapter|ch\.?)\s*\d+(?:\.\d+)?(?:\s*[:：\-–—]\s*|\s+)(.+?)\s*$", re.IGNORECASE), "Volume+chapter title normalized"),
    (re.compile(r"^\s*(?:volume|vol\.?)\s*\d+\s+\d+(?:\.\d+)?(?:\s*[:：\-–—]\s*|\s+)(.+?)\s*$", re.IGNORECASE), "Volume numeric title normalized"),
    (re.compile(r"^\s*第\s*[0-9０-９]+\s*[話章節巻部]\s*[:：\-–—]?\s*(.+?)\s*$"), "JP chapter title normalized"),
    (re.compile(r"^\s*第\s*[一二三四五六七八九十百千万]+\s*[話章節巻部]\s*[:：\-–—]?\s*(.+?)\s*$"), "JP kanji chapter title normalized"),
    (re.compile(r"^\s*[【\[]\s*第\s*[0-9０-９一二三四五六七八九十百千万]+\s*[話章節巻部]\s*[】\]]\s*(.+?)\s*$"), "JP bracketed chapter title normalized"),
]

# Lines that are ONLY a chapter number -> remove entirely
REMOVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*chapter\s+\d+(?:\.\d+)?\s*[:：]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:volume|vol\.?)\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*第\s*[0-9０-９]+\s*[話章節巻部]\s*$"),
    re.compile(r"^\s*第\s*[一二三四五六七八九十百千万]+\s*[話章節巻部]\s*$"),
    re.compile(r"^\s*[【\[]\s*第\s*[0-9０-９一二三四五六七八九十百千万]+\s*[話章節巻部]\s*[】\]]\s*$"),
    re.compile(r"^\s*[0-9０-９]+\s*[\.．]\s*$"),
]


def process_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    """Apply patterns to the first 4 non-empty lines. Returns (new_lines, ops)."""
    ops: list[str] = []
    non_empty_seen = 0

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        non_empty_seen += 1
        if non_empty_seen > 4:
            break

        # Try title normalization first
        for pattern, label in TITLE_PATTERNS:
            m = pattern.match(line)
            if m:
                lines[i] = m.group(1).strip()
                ops.append(label)
                break
        else:
            # Try full removal
            for pattern in REMOVE_PATTERNS:
                if pattern.match(line):
                    lines[i] = ""
                    ops.append("Standalone header removed")
                    break

    return lines, ops


def process_file(path: Path) -> dict | None:
    try:
        original = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    lines = original.splitlines()
    lines, ops = process_lines(lines)

    if not ops:
        return None

    new_text = "\n".join(lines)
    if original.endswith("\n"):
        new_text += "\n"

    if new_text == original:
        return None

    return {"path": path, "old": original, "new": new_text, "ops": ops}


def preview(text: str, n: int = 6) -> str:
    return "\n".join(text.splitlines()[:n])


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python clean_chapter_titles.py Assets/Novels/MushokuTensei")
        return 1

    novel_path = Path(sys.argv[1])
    if not novel_path.is_dir():
        print(f"Not a folder: {novel_path}")
        return 1

    matches: list[dict] = []
    print(f"\nScanning: {novel_path}\n")

    for txt_file in novel_path.rglob("*.txt"):
        if not any(p in {"EN", "JP"} for p in txt_file.parts):
            continue
        result = process_file(txt_file)
        if result:
            matches.append(result)

    if not matches:
        print("No chapter headers found.")
        return 0

    # Summary table
    counter: Counter = Counter()
    for m in matches:
        counter.update(m["ops"])

    rows = [*counter.items(), ("Total edits", sum(counter.values())), ("Files affected", len(matches))]
    w = max(len(r[0]) for r in rows)
    print(f"\n{'Action':<{w}}  Count")
    print("-" * (w + 8))
    for label, count in rows:
        print(f"{label:<{w}}  {count}")

    # Dry-run preview
    print("\n" + "=" * 100)
    print("DRY RUN PREVIEW")
    print("=" * 100)

    for item in matches[:MAX_PREVIEW_FILES]:
        print(f"\nFILE: {item['path']}")
        print("--- BEFORE ---")
        print(preview(item["old"]))
        print("--- AFTER  ---")
        print(preview(item["new"]))
        print("-" * 100)

    if len(matches) > MAX_PREVIEW_FILES:
        print(f"\n... {len(matches) - MAX_PREVIEW_FILES} more files not shown.")

    confirm = input("\nApply changes? Type 'y' to continue: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return 0

    for item in matches:
        item["path"].write_text(item["new"], encoding="utf-8")

    print(f"\nDone. Modified {len(matches)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())