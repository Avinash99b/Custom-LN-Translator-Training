#!/usr/bin/env python3
"""
clean_chapter_titles.py

Scans all .txt files under a single novel folder and cleans chapter headers
from the FIRST 4 NON-EMPTY LINES ONLY.

Expected input:
    Assets/Novels/NovelName/

It will only work inside:
    NovelName/EN/
    NovelName/JP/

What it does:
-------------
1. Removes explicit chapter numbering while preserving titles.

    Chapter 8 : Hero's Return
    -> Hero's Return

    Volume 22 Chapter 7: A Second
    -> A Second

    Volume 22  7 — A Second
    -> A Second

    第8話 勇者の帰還
    -> 勇者の帰還

2. Removes standalone chapter-number lines only.

    Chapter 8
    -> removed

    第八話
    -> removed

3. Does NOT remove perspective headers, decorative lines,
   side-story labels, or other descriptive header text.

4. Removes repeated top titles even when punctuation differs.

    Volume 5 Side Story — Return of Roxy
    Volume 5 Side Story - Return of Roxy

    -> keep one, remove the duplicate

5. Processes only the first 4 non-empty lines.

6. Dry-run first, then asks for explicit 'y' confirmation.

Usage:
    python clean_chapter_titles.py Assets/Novels/MushokuTensei
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

MAX_PREVIEW_FILES = 20
MAX_DEDUP_SCAN_LINES = 12


TITLE_PATTERNS = [
    (
        re.compile(
            r"^\s*chapter\s+\d+(?:\.\d+)?\s*[:：\-–—]?\s+(.+?)\s*$",
            re.IGNORECASE,
        ),
        lambda m: m.group(1).strip(),
        "Chapter title normalized",
    ),
    (
        re.compile(
            r"^\s*(?:volume|vol\.?)\s*\d+\s+(?:chapter|ch(?:apter)?\.?)\s*\d+(?:\.\d+)?(?:\s*[:：\-–—]\s*|\s+)(.+?)\s*$",
            re.IGNORECASE,
        ),
        lambda m: m.group(1).strip(),
        "Volume chapter title normalized",
    ),
    (
        re.compile(
            r"^\s*(?:volume|vol\.?)\s*\d+\s+\d+(?:\.\d+)?(?:\s*[:：\-–—]\s*|\s+)(.+?)\s*$",
            re.IGNORECASE,
        ),
        lambda m: m.group(1).strip(),
        "Volume numeric title normalized",
    ),
    (
        re.compile(
            r"^\s*第\s*[0-9０-９]+\s*[話章節巻部]\s*[:：\-–—]?\s*(.+?)\s*$"
        ),
        lambda m: m.group(1).strip(),
        "Japanese chapter title normalized",
    ),
    (
        re.compile(
            r"^\s*第\s*[一二三四五六七八九十百千万]+\s*[話章節巻部]\s*[:：\-–—]?\s*(.+?)\s*$"
        ),
        lambda m: m.group(1).strip(),
        "Japanese kanji chapter title normalized",
    ),
    (
        re.compile(
            r"^\s*[【\[]\s*第\s*[0-9０-９一二三四五六七八九十百千万]+\s*[話章節巻部]\s*[】\]]\s*(.+?)\s*$"
        ),
        lambda m: m.group(1).strip(),
        "Bracketed Japanese chapter title normalized",
    ),
]

REMOVE_LINE_PATTERNS = [
    re.compile(
        r"^\s*chapter\s+\d+(?:\.\d+)?\s*[:：]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*第\s*[0-9０-９]+\s*[話章節巻部]\s*$"
    ),
    re.compile(
        r"^\s*第\s*[一二三四五六七八九十百千万]+\s*[話章節巻部]\s*$"
    ),
    re.compile(
        r"^\s*[【\[]\s*第\s*[0-9０-９一二三四五六七八九十百千万]+\s*[話章節巻部]\s*[】\]]\s*$"
    ),
    re.compile(
        r"^\s*[0-9０-９]+\s*[\.．]\s*$"
    ),
]


def normalize_for_dedupe(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).strip()
    s = s.replace("’", "'").replace("`", "'")
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n-:：|/\\[]【】()（）「」『』\"'")
    return s.lower()


def looks_title_like(text: str) -> bool:
    s = unicodedata.normalize("NFKC", text).strip()

    if not s:
        return False

    if len(s) > 120:
        return False

    if re.search(
        r"\b(?:volume|vol\.?|chapter|ch(?:apter)?\.?|part)\b|第|話|章|節|巻|部|side story|prologue|epilogue",
        s,
        re.IGNORECASE,
    ):
        return True

    if not re.search(r"[。．.!?！？]$", s):
        return True

    return False


def clean_first_four_non_empty_lines(lines: list[str]) -> tuple[bool, list[str]]:
    changed = False
    operations: list[str] = []
    non_empty_seen = 0

    for idx in range(len(lines)):
        if not lines[idx].strip():
            continue

        non_empty_seen += 1

        if non_empty_seen > 4:
            break

        original = lines[idx]
        matched = False

        for pattern, replacer, label in TITLE_PATTERNS:
            match = pattern.match(original)
            if match:
                new_line = replacer(match)
                if new_line != original:
                    lines[idx] = new_line
                    changed = True
                    operations.append(label)
                matched = True
                break

        if matched:
            continue

        for pattern in REMOVE_LINE_PATTERNS:
            if pattern.match(original):
                lines[idx] = ""
                changed = True
                operations.append("Standalone header removed")
                break

    return changed, operations


def deduplicate_top_titles(lines: list[str]) -> tuple[bool, int]:
    changed = False
    removed_count = 0

    limit = min(MAX_DEDUP_SCAN_LINES, len(lines))
    seen: set[str] = set()

    for i in range(limit):
        current = lines[i].strip()

        if not current:
            continue

        if not looks_title_like(current):
            continue

        key = normalize_for_dedupe(current)

        if key in seen:
            lines[i] = ""
            changed = True
            removed_count += 1
        else:
            seen.add(key)

    return changed, removed_count


def process_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    lines = text.splitlines()
    original_text = text

    changed = False
    operations: list[str] = []

    title_changed, title_ops = clean_first_four_non_empty_lines(lines)
    if title_changed:
        changed = True
        operations.extend(title_ops)

    dedup_changed, dedup_removed = deduplicate_top_titles(lines)
    if dedup_changed:
        changed = True
        operations.extend(["Duplicate top title removed"] * dedup_removed)

    if not changed:
        return None

    new_text = "\n".join(lines)

    return {
        "path": path,
        "old_text": original_text,
        "new_text": new_text,
        "operations": operations,
    }


def preview_first_lines(text: str, n: int = 6) -> str:
    return "\n".join(text.splitlines()[:n])


def print_summary_table(matches: list[dict]) -> None:
    counter = Counter()
    for item in matches:
        counter.update(item["operations"])

    total_edits = sum(counter.values())

    rows = [
        ("Chapter title normalized", counter["Chapter title normalized"], "Removed numbering, kept title"),
        ("Volume chapter title normalized", counter["Volume chapter title normalized"], "Volume + chapter header"),
        ("Volume numeric title normalized", counter["Volume numeric title normalized"], "Volume + numeric header"),
        ("Japanese chapter title normalized", counter["Japanese chapter title normalized"], "Japanese chapter header"),
        ("Japanese kanji chapter title normalized", counter["Japanese kanji chapter title normalized"], "Japanese kanji chapter header"),
        ("Bracketed Japanese chapter title normalized", counter["Bracketed Japanese chapter title normalized"], "Bracketed chapter header"),
        ("Standalone header removed", counter["Standalone header removed"], "Header was only a number"),
        ("Duplicate top title removed", counter["Duplicate top title removed"], "Collapsed repeated title line"),
        ("Total edit actions", total_edits, "All proposed edits combined"),
        ("Files to modify", len(matches), "Unique files affected"),
    ]

    col1 = max(len("Action"), max(len(r[0]) for r in rows))
    col2 = max(len("Count"), max(len(str(r[1])) for r in rows))
    col3 = max(len("Notes"), max(len(r[2]) for r in rows))

    print("\nProposed changes")
    print("=" * (col1 + col2 + col3 + 10))
    print(f"{'Action':<{col1}} | {'Count':>{col2}} | {'Notes':<{col3}}")
    print(f"{'-' * col1}-+-{'-' * col2}-+-{'-' * col3}")

    for action, count, notes in rows:
        print(f"{action:<{col1}} | {count:>{col2}} | {notes:<{col3}}")

    print("=" * (col1 + col2 + col3 + 10))


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "    python clean_chapter_titles.py Assets/Novels/MushokuTensei"
        )
        return 1

    novel_path = Path(sys.argv[1])

    if not novel_path.exists():
        print(f"Novel folder not found: {novel_path}")
        return 1

    if not novel_path.is_dir():
        print(f"Not a folder: {novel_path}")
        return 1

    matches = []

    print(f"\nScanning novel folder: {novel_path}\n")

    for txt_file in novel_path.rglob("*.txt"):
        if not any(part in {"EN", "JP"} for part in txt_file.parts):
            continue

        result = process_file(txt_file)
        if result:
            matches.append(result)

    if not matches:
        print("No chapter headers found.")
        return 0

    print_summary_table(matches)

    print("\n" + "=" * 120)
    print("DRY RUN PREVIEW")
    print("=" * 120)

    sample = matches[:MAX_PREVIEW_FILES]

    for item in sample:
        print(f"\nFILE: {item['path']}")

        print("\n--- BEFORE ---")
        print(preview_first_lines(item["old_text"]))

        print("\n--- AFTER ----")
        print(preview_first_lines(item["new_text"]))

        print("-" * 120)

    if len(matches) > MAX_PREVIEW_FILES:
        print(
            f"\n... {len(matches) - MAX_PREVIEW_FILES} more files omitted from preview "
            f"to keep the summary readable."
        )

    confirm = input("\nApply changes? Type 'y' to continue: ").strip().lower()

    if confirm != "y":
        print("\nAborted.")
        return 0

    modified = 0
    for item in matches:
        Path(item["path"]).write_text(item["new_text"], encoding="utf-8")
        modified += 1

    print(f"\nDone. Modified {modified} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())