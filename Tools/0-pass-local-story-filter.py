#!/usr/bin/env python3
"""
clean_notes.py

Scans JP-Working and EN-Working folders for translator notes, author notes,
advertisements, afterwords, and any other non-story content in light novel
chapter files.

For every file:
  1. Create working copies (JP-Working / EN-Working) from originals if absent.
  2. Operate ONLY on the Working folders — originals are never touched.
  3. Send numbered lines to LM Studio and ask it to identify ranges to remove.
  4. Parse the JSON response, validate ranges, remove lines.
  5. Save the cleaned file in-place inside the Working folder.
  6. Write a progress JSON after every file so interrupted runs can be resumed.

Usage:
    python clean_notes.py <novel_folder>
    python clean_notes.py <novel_folder> --found-only
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

LM_STUDIO_URL   = "http://localhost:1234/v1/chat/completions"
MODEL           = "qwen3.5-4b-claude-4.6-opus-reasoning-distilled"   # LM Studio ignores this but it must be present
LLM_TIMEOUT     = 300             # seconds per request
MAX_TOKENS      = 12000            # enough for a full removal list

# How many lines from the head / tail to include in the prompt.
# For most chapter files the non-story content is near the top or bottom.
# We send the FULL file — line numbering makes the model's job accurate.
# (Set to None to always send the full file.)
MAX_LINES_TO_SEND = None          # None = send everything


# ============================================================
# SYSTEM / USER PROMPT
# ============================================================

SYSTEM_PROMPT = """\
You are a professional light novel editor specialising in cleaning up
fan-translated and officially translated chapter files.

Your ONLY job is to identify lines that must be removed from the file.
You must NEVER rewrite, summarise, or output the chapter text.

Lines that MUST be removed (if present):
  - Translator notes       [TN:], [TL:], TL Note, 訳注
  - Editor notes           [EN:], [ED:], Editor Note
  - Author afterwords      後書き, 感想, "Thank you for reading" messages
  - Author greeting notes  "Hello everyone, it's the author"
  - Patreon promotions     Any mention of Patreon, patron, donate
  - Discord advertisements Any mention of Discord servers, invite links
  - Website advertisements Any mention of novel-update sites, reading sites
  - Purchase / store links Any "buy the volume", Amazon, bookstore links
  - Blu-ray advertisements Any Blu-ray, DVD, BD advertisement
  - Release announcements  "Volume X on sale", "coming soon"
  - Shout-outs / credits   "Special thanks to our editors/patrons …"
  - Activity reports       活動報告, お知らせ (notices unrelated to story)
  - Copyright notices      © lines, ISBN, publisher boilerplate
  - Disclaimer blocks      "This is a work of fiction …"
  - Ko-fi / donation links Any ko-fi, buymeacoffee, tip-jar links

Lines that must NEVER be removed:
  - Chapter titles, volume titles, prologue/epilogue headers
  - Section dividers (---, ***, ◆, etc.)
  - Status lines like "Chapter X", "Volume Y", "Prologue", "Epilogue"
  - Any line that looks like it could be part of the story text
  - DO NOT REMOVE STATUS PANELS OF ANY KIND LIKE LEVEL, HP, STATS, SKILL LISTS, etc.
  - Story prose, dialogue, action, description — even a single sentence
  - Character name labels that introduce dialogue (「…」 lines)

When in doubt about any line: KEEP IT (do not include it in the output).

You will receive the chapter text with 1-based line numbers in the format:
  1: <line content>
  2: <line content>
  …

Output ONLY valid JSON, no markdown fences, no extra commentary:

{
  "remove_ranges": [
    { "start": <int>, "end": <int>, "reason": "<short label>" }
  ]
}

If nothing should be removed output:
{
  "remove_ranges": []
}
"""

USER_TEMPLATE = """\
Here is the chapter text. Identify all non-story lines to remove.

{numbered_text}
"""


# ============================================================
# LM STUDIO API
# ============================================================

def call_lm_studio(numbered_text: str) -> str:
    """
    Send a chat-completion request to LM Studio and return the raw
    assistant response string.
    """
    payload = {
        "model":       MODEL,
        "temperature": 0,
        "max_tokens":  MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(
                numbered_text=numbered_text)},
        ],
    }
    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ============================================================
# NUMBERED TEXT
# ============================================================

def build_numbered_text(lines: list) -> str:
    """Return the lines with 1-based line numbers prepended."""
    parts = []
    for i, line in enumerate(lines, start=1):
        parts.append(f"{i}: {line}")
    return "\n".join(parts)


# ============================================================
# JSON PARSING & RANGE VALIDATION
# ============================================================

def parse_removal_response(raw: str, total_lines: int) -> dict:
    """
    Parse the LLM response JSON and return a normalised dict:
      {
        "ok":             bool,
        "remove_ranges":  [ {"start":int,"end":int,"reason":str}, ... ],
        "error":          str | None,
      }

    Ranges are validated:
      - start and end must be integers
      - start <= end
      - both must be within [1, total_lines]
    Malformed ranges are silently dropped.
    """
    # Strip accidental markdown fences if the model ignores the instruction
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # remove first and last fence lines
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {"ok": False, "remove_ranges": [],
                "error": f"JSON decode error: {exc}  |  raw={raw[:200]!r}"}

    if not isinstance(obj, dict):
        return {"ok": False, "remove_ranges": [],
                "error": f"Expected dict, got {type(obj).__name__}"}

    raw_ranges = obj.get("remove_ranges", [])
    if not isinstance(raw_ranges, list):
        return {"ok": False, "remove_ranges": [],
                "error": f"remove_ranges is not a list: {raw_ranges!r}"}

    valid_ranges = []
    for entry in raw_ranges:
        if not isinstance(entry, dict):
            continue
        try:
            start  = int(entry["start"])
            end    = int(entry["end"])
            reason = str(entry.get("reason", "unspecified"))
        except (KeyError, ValueError, TypeError):
            continue

        # Clamp and validate
        if start > end:
            start, end = end, start          # tolerate reversed ranges
        if start < 1 or end > total_lines:
            continue                          # out-of-bounds — skip
        if start > total_lines or end < 1:
            continue

        valid_ranges.append({"start": start, "end": end, "reason": reason})

    return {"ok": True, "remove_ranges": valid_ranges, "error": None}


# ============================================================
# LINE REMOVAL
# ============================================================

def apply_removals(lines: list, remove_ranges: list) -> tuple:
    """
    Remove all lines covered by the validated ranges (1-based, inclusive).
    Returns (cleaned_lines, total_removed_count).
    """
    if not remove_ranges:
        return lines, 0

    # Build a set of 0-based indices to remove
    to_remove: set = set()
    for rng in remove_ranges:
        for idx in range(rng["start"] - 1, rng["end"]):   # convert to 0-based
            if 0 <= idx < len(lines):
                to_remove.add(idx)

    cleaned = [l for i, l in enumerate(lines) if i not in to_remove]

    # Collapse 3+ consecutive blank lines to 2
    collapsed, blanks = [], 0
    for line in cleaned:
        if line.strip() == "":
            blanks += 1
            if blanks <= 2:
                collapsed.append(line)
        else:
            blanks = 0
            collapsed.append(line)

    return collapsed, len(to_remove)


# ============================================================
# FILE HELPERS
# ============================================================

def read_lines(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def write_lines(path: Path, lines: list):
    path.write_text("\n".join(lines), encoding="utf-8")


def natural_sort_key(name: str):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", name)]


def collect_files(root: Path) -> list:
    """
    Return a list of (filename, working_path, language) tuples for every .txt
    file found in JP-Working and EN-Working, sorted naturally by filename.
    """
    result = []
    for lang, subdir in (("JP", "JP-Working"), ("EN", "EN-Working")):
        folder = root / subdir
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.txt"),
                        key=lambda x: natural_sort_key(x.name)):
            result.append((f.name, f, lang))
    return result


# ============================================================
# WORKING FOLDERS
# ============================================================

def ensure_working_folders(root: Path):
    """
    Create JP-Working and EN-Working by copying from JP / EN.
    Only copies files that do not yet exist in the working folder (so that a
    partially completed run is not reset).
    """
    for subdir in ("JP", "EN"):
        src_dir  = root / subdir
        dst_name = subdir + "-Working"
        dst_dir  = root / dst_name

        if not src_dir.exists():
            print(f"  Warning: source folder not found, skipping: {src_dir}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        src_files = list(src_dir.glob("*.txt"))
        copied    = 0
        skipped   = 0
        for src_file in src_files:
            dst_file = dst_dir / src_file.name
            if dst_file.exists():
                skipped += 1
            else:
                shutil.copy2(src_file, dst_file)
                copied += 1

        print(f"  {dst_name}: {copied} file(s) copied, {skipped} already present.")


# ============================================================
# PROGRESS  (compatible structure with reference script)
# ============================================================

def load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {e["key"]: e for e in data if "key" in e}
    except Exception:
        return {}


def save_progress(path: Path, ordered_keys: list, by_key: dict):
    ordered = [by_key[k] for k in ordered_keys if k in by_key]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


# ============================================================
# RESUME PROMPT
# ============================================================

def ask_resume(prog_file: Path, completed: dict, all_keys: list) -> str:
    done_set   = set(completed.keys())
    error_keys = {k for k, v in completed.items() if v.get("status") == "ERROR"}
    missing    = set(all_keys) - done_set

    print(f"\n{'─'*60}")
    print(f"Progress file  : {prog_file.name}")
    print(f"  On disk      : {len(all_keys)}")
    print(f"  Done         : {len(done_set)}")
    print(f"  Errors       : {len(error_keys)}")
    print(f"  Missing      : {len(missing)}")

    if missing:
        print("\n  New / missing:")
        for k in sorted(missing, key=lambda x: natural_sort_key(x.split("/", 1)[-1])):
            print(f"    + {k}")
    if error_keys:
        print("\n  Errored:")
        for k in sorted(error_keys, key=lambda x: natural_sort_key(x.split("/", 1)[-1])):
            print(f"    ! {k}")

    print(f"\n{'─'*60}")
    print("  [s] Skip completed  — only process new/missing (errors kept as-is)")
    print("  [e] Retry errors    — re-run errored + new/missing files")
    print("  [r] Restart         — wipe progress and process everything")
    print(f"{'─'*60}")

    while True:
        ans = input("Choice [s/e/r]: ").strip().lower()
        if ans in ("s", "skip"):
            return "skip"
        if ans in ("e", "errors", "error", "retry"):
            return "errors"
        if ans in ("r", "restart"):
            prog_file.unlink(missing_ok=True)
            print("Progress wiped.\n")
            return "restart"
        print("Please enter s, e, or r.")


# ============================================================
# PER-FILE PROCESSING
# ============================================================

def process_file(chapter_name: str, file_path: Path, language: str) -> dict:
    key = f"{language}/{chapter_name}"

    # ── Read file ─────────────────────────────────────────────────────────────
    lines = read_lines(file_path)
    if not lines:
        return _build_result(key, chapter_name, language, "CLEAN", [], 0)

    # Optionally truncate for very large files (disabled by default)
    send_lines = lines if MAX_LINES_TO_SEND is None else lines[:MAX_LINES_TO_SEND]
    numbered   = build_numbered_text(send_lines)

    # ── Call LM Studio ────────────────────────────────────────────────────────
    try:
        raw_response = call_lm_studio(numbered)
    except requests.exceptions.ConnectionError:
        err = "Cannot connect to LM Studio. Is it running on localhost:1234?"
        return _build_result(key, chapter_name, language, "ERROR", [], 0,
                             error=err)
    except requests.exceptions.Timeout:
        err = f"LM Studio request timed out after {LLM_TIMEOUT}s."
        return _build_result(key, chapter_name, language, "ERROR", [], 0,
                             error=err)
    except Exception as exc:
        return _build_result(key, chapter_name, language, "ERROR", [], 0,
                             error=str(exc))

    # ── Parse response ────────────────────────────────────────────────────────
    parsed = parse_removal_response(raw_response, len(send_lines))

    if not parsed["ok"]:
        return _build_result(key, chapter_name, language, "ERROR", [], 0,
                             error=parsed["error"])

    remove_ranges = parsed["remove_ranges"]

    if not remove_ranges:
        return _build_result(key, chapter_name, language, "CLEAN", [], 0)

    # ── Apply removals ────────────────────────────────────────────────────────
    cleaned_lines, n_removed = apply_removals(lines, remove_ranges)

    if n_removed == 0:
        # All ranges were filtered out during apply (shouldn't happen after
        # validation, but be safe)
        return _build_result(key, chapter_name, language, "CLEAN", [], 0)

    # ── Save cleaned file ─────────────────────────────────────────────────────
    write_lines(file_path, cleaned_lines)

    return _build_result(key, chapter_name, language, "CLEANED",
                         remove_ranges, n_removed)


def _build_result(key, chapter, language, status,
                  removed_ranges, removed_line_count, error=None):
    r = {
        "key":               key,
        "chapter":           chapter,
        "language":          language,
        "status":            status,
        "removed_ranges":    removed_ranges,
        "removed_line_count": removed_line_count,
    }
    if error:
        r["error"] = error
    return r


# ============================================================
# SUMMARY
# ============================================================

def print_summary(by_key: dict, found_only: bool = False):
    all_r   = list(by_key.values())
    total   = len(all_r)
    clean   = sum(1 for r in all_r if r.get("status") == "CLEAN")
    cleaned = sum(1 for r in all_r if r.get("status") == "CLEANED")
    errors  = sum(1 for r in all_r if r.get("status") == "ERROR")

    print(f"\n{'='*60}")
    print(f"SUMMARY  —  {total} files")
    print(f"  CLEAN   (no extra content)      : {clean}")
    print(f"  CLEANED (extra content removed) : {cleaned}")
    print(f"  ERROR                           : {errors}")
    print(f"{'='*60}")

    modified = [r for r in all_r if r.get("status") == "CLEANED"]
    if modified:
        print("\nFiles with content removed:")
        for r in modified:
            total_removed = r.get("removed_line_count", 0)
            ranges        = r.get("removed_ranges", [])
            print(f"  [{r['language']}] {r['chapter']}  "
                  f"— {total_removed} line(s) removed "
                  f"({len(ranges)} range(s))")
            if not found_only:
                for rng in ranges:
                    print(f"      lines {rng['start']}–{rng['end']} "
                          f"({rng.get('reason','?')})")

    error_items = [r for r in all_r if r.get("status") == "ERROR"]
    if error_items:
        print("\nFiles with errors:")
        for r in error_items:
            print(f"  [{r['language']}] {r['chapter']}  — {r.get('error','?')}")

    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove translator notes, author notes, advertisements, and other "
            "non-story content from light novel chapter files using LM Studio."
        )
    )
    parser.add_argument(
        "folder",
        help="Root novel folder (contains JP / EN sub-folders)",
    )
    parser.add_argument(
        "--found-only",
        action="store_true",
        help="Suppress per-range details in the final summary",
    )
    args = parser.parse_args()

    root      = Path(args.folder)
    prog_file = root / "clean_notes_progress.json"

    if not root.exists():
        print(f"Error: folder not found: {root}")
        sys.exit(1)

    # ── 1. Ensure working folders ─────────────────────────────────────────────
    print("\nPreparing working folders …")
    ensure_working_folders(root)

    # ── 2. Collect working files ──────────────────────────────────────────────
    all_files = collect_files(root)
    if not all_files:
        print("No .txt files found in JP-Working / EN-Working.")
        sys.exit(0)

    ordered_keys = [f"{lang}/{name}" for name, _, lang in all_files]
    file_map     = {f"{lang}/{name}": (name, path, lang)
                    for name, path, lang in all_files}

    # ── 3. Resume logic ───────────────────────────────────────────────────────
    results_by_key: dict = {}

    if prog_file.exists():
        completed = load_progress(prog_file)
        mode      = ask_resume(prog_file, completed, ordered_keys)

        if mode == "restart":
            results_by_key = {}
            pending_keys   = list(ordered_keys)
            # Refresh working folders from originals on full restart
            print("\nRefreshing working folders from originals …")
            for subdir in ("JP-Working", "EN-Working"):
                working = root / subdir
                if working.exists():
                    shutil.rmtree(working)
            ensure_working_folders(root)

        elif mode == "skip":
            results_by_key = dict(completed)
            pending_keys   = [k for k in ordered_keys if k not in completed]
            print(f"Skipping {len(completed)} already-done file(s).\n")

        elif mode == "errors":
            results_by_key = {k: v for k, v in completed.items()
                               if v.get("status") != "ERROR"}
            err_keys        = {k for k, v in completed.items()
                                if v.get("status") == "ERROR"}
            missing_keys    = set(ordered_keys) - set(completed.keys())
            pending_keys    = [k for k in ordered_keys
                                if k in err_keys or k in missing_keys]
            print(f"Re-running {len(err_keys)} error(s) + "
                  f"{len(missing_keys)} missing. "
                  f"Keeping {len(results_by_key)} clean result(s).\n")

        else:
            results_by_key = {}
            pending_keys   = list(ordered_keys)
    else:
        results_by_key = {}
        pending_keys   = list(ordered_keys)

    if not pending_keys:
        print("Nothing left to process.")
        save_progress(prog_file, ordered_keys, results_by_key)
        print_summary(results_by_key, args.found_only)
        sys.exit(0)

    print(f"\nFiles to process : {len(pending_keys)}")
    print(f"Already done     : {len(results_by_key)}")
    print(f"LM Studio URL    : {LM_STUDIO_URL}")
    print(f"Model hint       : {MODEL}\n")

    # ── 4. Sequential processing with progress bar ────────────────────────────
    with tqdm(total=len(pending_keys), unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
              ) as pbar:
        for key in pending_keys:
            if key not in file_map:
                pbar.update(1)
                continue

            name, path, lang = file_map[key]
            pbar.set_description(f"[{lang}] {name}")

            try:
                result = process_file(name, path, lang)
            except Exception as exc:
                lang_p, name_p = key.split("/", 1)
                result = _build_result(key, name_p, lang_p, "ERROR",
                                       [], 0, error=str(exc))

            results_by_key[key] = result
            save_progress(prog_file, ordered_keys, results_by_key)

            status = result.get("status", "ERROR")
            removed = result.get("removed_line_count", 0)
            suffix = ""
            if status == "CLEANED":
                suffix = f" (-{removed} lines)"
            elif status == "ERROR":
                suffix = f" ERR: {result.get('error','?')[:60]}"

            tqdm.write(f"  {'✓' if status != 'ERROR' else '✗'} "
                       f"[{lang}] {name}  →  {status}{suffix}")
            pbar.update(1)

    # ── 5. Final save + summary ───────────────────────────────────────────────
    save_progress(prog_file, ordered_keys, results_by_key)
    print(f"\nProgress saved → {prog_file}")
    print_summary(results_by_key, args.found_only)


if __name__ == "__main__":
    main()