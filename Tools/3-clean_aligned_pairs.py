#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import sys
from pathlib import Path
import time
import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

AZURE_OPENAI_URL = "https://bathu-mpkwvf7h-eastus2.cognitiveservices.azure.com/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "none")  # keep as requested
LLM_TIMEOUT = 300
MAX_TOKENS = 16000

SOURCE_JP = "JP-Aligned"
SOURCE_EN = "EN-Aligned"
OUTPUT_JP = "JP-Output"
OUTPUT_EN = "EN-Output"

MAX_CONSECUTIVE_DIVERGENCES = 2
MAX_WORKERS = 10  # Added to easily control concurrency

MAX_BACKOFF_RETRIES = 5

# Confidence thresholds
CONFIDENCE_ACCEPT  = 80   # >= this → same story, proceed to cleaning
CONFIDENCE_UNSURE  = 50   # >= this but < ACCEPT → LOW_CONFIDENCE (flagged, not cleaned)
                          # <  this → DIVERGED (definitely different)


# ============================================================
# SYSTEM / USER PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a professional light novel editor.

You will receive TWO aligned chapter files:

* a Japanese chapter
* an English chapter

==================================================
STAGE 1 — STORY MATCH CHECK
===========================

Determine whether the JP and EN chapters represent the same chapter/story segment.

Base your judgment primarily on:

* major story events
* event ordering
* character identities
* scene progression
* narrative structure

IGNORE OR HEAVILY DISCOUNT:

* chapter title wording
* chapter numbering differences
* translator notes
* formatting differences
* paragraph splitting/merging
* minor translation omissions
* localization differences

Before making a decision, assign a confidence score:

0   = completely different stories
50  = uncertain
100 = definitely same chapter

Return the confidence as an integer from 0–100.

Use these rules:

confidence >= 80
=> same_story = true

confidence < 80
=> same_story = false

If confidence < 80 return exactly:

{
  "same_story": false,
  "confidence": 63,
  "reason": "Short reason under 10 words"
}

Do NOT perform cleaning when same_story is false.

==================================================
STAGE 2 — CLEANING
==================

Only perform this stage when confidence >= 80.

Clean non-story content from BOTH chapters separately.

Remove lines such as:

* translator notes
* editor notes
* author greetings
* advertisements
* Patreon / Ko-fi links
* Discord links
* website/store promotions
* release notices
* copyright boilerplate
* disclaimers unrelated to story

Never remove:

* story prose
* dialogue
* narration
* chapter titles
* prologues
* epilogues
* volume headings
* status screens
* stat blocks
* skill lists
* inventory screens
* system messages
* anything that may be part of the story

When uncertain, keep the content.

For each side, extract the chapter title and remove any chapter-number prefix.

If the title line contains only a chapter number or marker,
return an empty string "".

If no title is identifiable:
chapter_title = null
chapter_title_line = -1

Output ONLY valid JSON.

If confidence >= 80:

{
  "same_story": true,
  "confidence": 92,
  "jp": {
    "remove_ranges": [
      {
        "start": 1,
        "end": 2,
        "reason": "translator_note"
      }
    ],
    "chapter_title": "Title",
    "chapter_title_line": 3
  },
  "en": {
    "remove_ranges": [
      {
        "start": 1,
        "end": 2,
        "reason": "translator_note"
      }
    ],
    "chapter_title": "Title",
    "chapter_title_line": 3
  }
}

If nothing should be removed AND no title exists:

{
  "remove_ranges": [],
  "chapter_title": null,
  "chapter_title_line": -1
}
"""


USER_TEMPLATE = """\
Here are the aligned chapter pairs. First check whether they are the same story.
If they are the same, clean both sides separately.

JP chapter:
{jp_numbered_text}

EN chapter:
{en_numbered_text}
"""


# ============================================================
# API CALL
# ============================================================

def call_lm_studio(jp_numbered_text: str, en_numbered_text: str) -> str:
    if not API_KEY or API_KEY == "hidden":
        raise RuntimeError(
            "API key not set. Please export AZURE_OPENAI_API_KEY with your Azure OpenAI key."
        )

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    jp_numbered_text=jp_numbered_text,
                    en_numbered_text=en_numbered_text,
                ),
            },
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    while True:
        resp = requests.post(
            AZURE_OPENAI_URL,
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT,
        )

        if not resp.ok and resp.status_code in (429, 503) and MAX_BACKOFF_RETRIES > 0:
            MAX_BACKOFF_RETRIES -= 1
            print(f"Rate limited or service unavailable. Retrying in {2 ** (5 - MAX_BACKOFF_RETRIES)} seconds... (Retries left: {MAX_BACKOFF_RETRIES})")
            time.sleep(2 ** (5 - MAX_BACKOFF_RETRIES))  # Exponential backoff
            continue

        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} Client Error: {resp.reason} for url: {AZURE_OPENAI_URL} | body={resp.text[:1000]!r}",
                response=resp,
            )

        data = resp.json()
        return data["choices"][0]["message"]["content"]
    return ""  # Unreachable, but satisfies function signature


# ============================================================
# NUMBERED TEXT
# ============================================================

def build_numbered_text(lines: list[str]) -> str:
    parts = []
    for i, line in enumerate(lines, start=1):
        parts.append(f"{i}: {line}")
    return "\n".join(parts)


# ============================================================
# JSON PARSING & RANGE VALIDATION
# ============================================================

def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_confidence(obj: dict) -> int | None:
    """
    Extract and validate the confidence integer from a parsed JSON object.
    Returns None if absent or unparseable.
    """
    raw = obj.get("confidence")
    if raw is None:
        return None
    try:
        val = int(raw)
        return max(0, min(100, val))   # clamp to [0, 100]
    except (TypeError, ValueError):
        return None


def _validate_side(obj: dict, side: str, total_lines: int) -> dict:
    side_obj = obj.get(side)
    if not isinstance(side_obj, dict):
        return {
            "ok": False,
            "error": f"Missing or invalid '{side}' section",
            "remove_ranges": [],
            "chapter_title": None,
            "chapter_title_line": -1,
        }

    raw_ranges = side_obj.get("remove_ranges", [])
    if not isinstance(raw_ranges, list):
        return {
            "ok": False,
            "error": f"{side}.remove_ranges is not a list",
            "remove_ranges": [],
            "chapter_title": None,
            "chapter_title_line": -1,
        }

    valid_ranges = []
    for entry in raw_ranges:
        if not isinstance(entry, dict):
            continue
        try:
            start = int(entry["start"])
            end = int(entry["end"])
            reason = str(entry.get("reason", "unspecified"))
        except (KeyError, ValueError, TypeError):
            continue

        if start > end:
            start, end = end, start
        if start < 1 or end > total_lines:
            continue

        valid_ranges.append({"start": start, "end": end, "reason": reason})

    chapter_title = side_obj.get("chapter_title", None)
    if chapter_title is not None and not isinstance(chapter_title, str):
        chapter_title = str(chapter_title)

    raw_ctl = side_obj.get("chapter_title_line", -1)
    try:
        chapter_title_line = int(raw_ctl)
    except (TypeError, ValueError):
        chapter_title_line = -1

    if chapter_title_line != -1 and not (1 <= chapter_title_line <= total_lines):
        chapter_title_line = -1
        chapter_title = None

    return {
        "ok": True,
        "remove_ranges": valid_ranges,
        "chapter_title": chapter_title,
        "chapter_title_line": chapter_title_line,
        "error": None,
    }


def parse_pair_response(raw: str, jp_total_lines: int, en_total_lines: int) -> dict:
    cleaned = _strip_code_fences(raw)

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "same_story": False,
            "confidence": None,
            "diverged": False,
            "low_conf": False,
            "reason": None,
            "jp": None,
            "en": None,
            "error": f"JSON decode error: {exc} | raw={raw[:250]!r}",
        }

    if not isinstance(obj, dict):
        return {
            "ok": False,
            "same_story": False,
            "confidence": None,
            "diverged": False,
            "low_conf": False,
            "reason": None,
            "jp": None,
            "en": None,
            "error": f"Expected dict, got {type(obj).__name__}",
        }

    confidence = _parse_confidence(obj)
    reason     = obj.get("reason") if isinstance(obj.get("reason"), str) else None

    # ── Legacy divergence sentinel (kept for backward compat) ────────────────
    if obj.get("ERROR") == "Stories are not same":
        conf = confidence if confidence is not None else 0
        diverged = conf < CONFIDENCE_UNSURE
        low_conf = not diverged   # UNSURE ≤ conf < ACCEPT, but model said false
        return {
            "ok": True,
            "same_story": False,
            "confidence": conf,
            "diverged": diverged,
            "low_conf": low_conf,
            "reason": reason,
            "jp": None,
            "en": None,
            "error": None,
        }

    # ── Model returned same_story = false ────────────────────────────────────
    if obj.get("same_story") is False or obj.get("error") == "Stories are not same":
        conf = confidence if confidence is not None else 0

        if conf < CONFIDENCE_UNSURE:
            # Definitely different stories
            diverged, low_conf = True, False
            label = f"DIVERGED (conf={conf})"
        else:
            # Model is unsure (CONFIDENCE_UNSURE ≤ conf < CONFIDENCE_ACCEPT)
            diverged, low_conf = False, True
            label = f"LOW_CONFIDENCE (conf={conf})"

        return {
            "ok": True,
            "same_story": False,
            "confidence": conf,
            "diverged": diverged,
            "low_conf": low_conf,
            "reason": reason,
            "jp": None,
            "en": None,
            "error": None,
        }

    # ── Model returned same_story = true ─────────────────────────────────────
    if obj.get("same_story") is not True:
        return {
            "ok": False,
            "same_story": False,
            "confidence": confidence,
            "diverged": False,
            "low_conf": False,
            "reason": reason,
            "jp": None,
            "en": None,
            "error": "Missing same_story field (expected true or false)",
        }

    jp = _validate_side(obj, "jp", jp_total_lines)
    en = _validate_side(obj, "en", en_total_lines)

    if not jp["ok"]:
        return {
            "ok": False,
            "same_story": True,
            "confidence": confidence,
            "diverged": False,
            "low_conf": False,
            "reason": reason,
            "jp": None,
            "en": None,
            "error": jp["error"],
        }
    if not en["ok"]:
        return {
            "ok": False,
            "same_story": True,
            "confidence": confidence,
            "diverged": False,
            "low_conf": False,
            "reason": reason,
            "jp": None,
            "en": None,
            "error": en["error"],
        }

    return {
        "ok": True,
        "same_story": True,
        "confidence": confidence,
        "diverged": False,
        "low_conf": False,
        "reason": reason,
        "jp": jp,
        "en": en,
        "error": None,
    }


# ============================================================
# LINE REMOVAL
# ============================================================

def apply_removals(lines: list[str], remove_ranges: list[dict]) -> tuple[list[str], int]:
    if not remove_ranges:
        return lines, 0

    to_remove: set[int] = set()
    for rng in remove_ranges:
        for idx in range(rng["start"] - 1, rng["end"]):
            if 0 <= idx < len(lines):
                to_remove.add(idx)

    cleaned = [l for i, l in enumerate(lines) if i not in to_remove]

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
# CHAPTER TITLE REWRITE
# ============================================================

def apply_title_rewrite(lines: list[str], chapter_title: str | None, chapter_title_line: int) -> tuple[list[str], bool, str]:
    if chapter_title is None or chapter_title_line == -1:
        return lines, False, ""

    idx = chapter_title_line - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, ""

    original = lines[idx]
    cleaned_title = chapter_title.strip()

    if original.strip() == cleaned_title:
        return lines, False, ""

    updated = list(lines)
    updated[idx] = cleaned_title
    return updated, True, original


# ============================================================
# FILE HELPERS
# ============================================================

def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def write_lines(path: Path, lines: list[str]):
    path.write_text("\n".join(lines), encoding="utf-8")


def natural_sort_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def collect_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    jp_dir = root / SOURCE_JP
    en_dir = root / SOURCE_EN

    if not jp_dir.exists() or not en_dir.exists():
        return []

    jp_files = {f.name: f for f in jp_dir.glob("*.txt")}
    en_files = {f.name: f for f in en_dir.glob("*.txt")}

    common = sorted(set(jp_files) & set(en_files), key=natural_sort_key)
    return [(name, jp_files[name], en_files[name]) for name in common]


def collect_missing_pairs(root: Path) -> list[str]:
    jp_dir = root / SOURCE_JP
    en_dir = root / SOURCE_EN

    if not jp_dir.exists() and not en_dir.exists():
        return []

    jp_names = {f.name for f in jp_dir.glob("*.txt")} if jp_dir.exists() else set()
    en_names = {f.name for f in en_dir.glob("*.txt")} if en_dir.exists() else set()

    missing = sorted((jp_names ^ en_names), key=natural_sort_key)
    return missing


# ============================================================
# WORKING / OUTPUT FOLDERS
# ============================================================

def ensure_output_folders(root: Path):
    """
    Create the cleaned output folders by copying from the aligned source folders.
    Originals are never touched.
    """
    for src_name, dst_name in ((SOURCE_JP, OUTPUT_JP), (SOURCE_EN, OUTPUT_EN)):
        src_dir = root / src_name
        dst_dir = root / dst_name

        if not src_dir.exists():
            print(f"  Warning: source folder not found, skipping: {src_dir}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        skipped = 0
        for src_file in src_dir.glob("*.txt"):
            dst_file = dst_dir / src_file.name
            if dst_file.exists():
                skipped += 1
            else:
                shutil.copy2(src_file, dst_file)
                copied += 1

        print(f"  {dst_name}: {copied} file(s) copied, {skipped} already present.")


# ============================================================
# PROGRESS
# ============================================================

def load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {e["key"]: e for e in data if "key" in e}
    except Exception:
        return {}


def save_progress(path: Path, ordered_keys: list[str], by_key: dict):
    ordered = [by_key[k] for k in ordered_keys if k in by_key]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ask_resume(prog_file: Path, completed: dict, all_keys: list[str]) -> str:
    done_set     = set(completed.keys())
    divergent    = {k for k, v in completed.items() if v.get("status") == "DIVERGED"}
    low_conf     = {k for k, v in completed.items() if v.get("status") == "LOW_CONFIDENCE"}
    errored      = {k for k, v in completed.items() if v.get("status") == "ERROR"}
    missing      = set(all_keys) - done_set

    print(f"\n{'─'*60}")
    print(f"Progress file  : {prog_file.name}")
    print(f"  On disk      : {len(all_keys)}")
    print(f"  Done         : {len(done_set)}")
    print(f"  Diverged     : {len(divergent)}")
    print(f"  Low-conf     : {len(low_conf)}")
    print(f"  Errors       : {len(errored)}")
    print(f"  Missing      : {len(missing)}")

    if missing:
        print("\n  New / missing:")
        for k in sorted(missing, key=natural_sort_key):
            print(f"    + {k}")
    if divergent:
        print("\n  Diverged (conf < 50):")
        for k in sorted(divergent, key=natural_sort_key):
            conf = completed[k].get("confidence")
            conf_str = f"conf={conf}" if conf is not None else "conf=?"
            print(f"    ! {k}  [{conf_str}]")
    if low_conf:
        print("\n  Low-confidence (50 ≤ conf < 80):")
        for k in sorted(low_conf, key=natural_sort_key):
            conf = completed[k].get("confidence")
            conf_str = f"conf={conf}" if conf is not None else "conf=?"
            print(f"    ? {k}  [{conf_str}]")
    if errored:
        print("\n  Errored:")
        for k in sorted(errored, key=natural_sort_key):
            print(f"    ! {k}")

    print(f"\n{'─'*60}")
    print("  [s] Skip completed  — only process new/missing (errors kept as-is)")
    print("  [e] Retry errors    — re-run errored + diverged + low-conf + new/missing files")
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
# PAIR PROCESSING
# ============================================================

def _build_result(
    key,
    chapter,
    status,
    confidence=None,
    reason=None,
    error=None,
    jp_removed=0,
    en_removed=0,
    jp_changes=None,
    en_changes=None,
):
    r = {
        "key": key,
        "chapter": chapter,
        "status": status,
        "confidence": confidence,
        "jp_removed_line_count": jp_removed,
        "en_removed_line_count": en_removed,
    }
    if reason is not None:
        r["reason"] = reason
    if error:
        r["error"] = error
    if jp_changes is not None:
        r["jp_changes"] = jp_changes
    if en_changes is not None:
        r["en_changes"] = en_changes
    return r


def process_pair(chapter_name: str, jp_src: Path, en_src: Path, jp_dst: Path, en_dst: Path) -> dict:
    key = chapter_name

    jp_lines = read_lines(jp_src)
    en_lines = read_lines(en_src)

    if not jp_lines and not en_lines:
        return _build_result(key, chapter_name, "CLEAN", confidence=None)

    jp_numbered = build_numbered_text(jp_lines)
    en_numbered = build_numbered_text(en_lines)

    try:
        raw_response = call_lm_studio(jp_numbered, en_numbered)
    except requests.exceptions.Timeout:
        err = f"Azure request timed out after {LLM_TIMEOUT}s."
        return _build_result(key, chapter_name, "ERROR", error=err)
    except requests.exceptions.ConnectionError:
        err = "Cannot connect to Azure endpoint."
        return _build_result(key, chapter_name, "ERROR", error=err)
    except Exception as exc:
        return _build_result(key, chapter_name, "ERROR", error=str(exc))

    parsed = parse_pair_response(raw_response, len(jp_lines), len(en_lines))

    if not parsed["ok"]:
        return _build_result(
            key, chapter_name, "ERROR",
            confidence=parsed.get("confidence"),
            error=parsed["error"],
        )

    # ── Definitely different stories (conf < CONFIDENCE_UNSURE) ──────────────
    if parsed["diverged"]:
        return _build_result(
            key, chapter_name, "DIVERGED",
            confidence=parsed.get("confidence"),
            reason=parsed.get("reason"),
        )

    # ── Uncertain match (CONFIDENCE_UNSURE ≤ conf < CONFIDENCE_ACCEPT) ───────
    if parsed["low_conf"]:
        return _build_result(
            key, chapter_name, "LOW_CONFIDENCE",
            confidence=parsed.get("confidence"),
            reason=parsed.get("reason"),
        )

    # ── Confirmed same story — proceed with cleaning ──────────────────────────
    jp_info = parsed["jp"]
    en_info = parsed["en"]

    jp_working, jp_removed = apply_removals(jp_lines, jp_info["remove_ranges"])
    en_working, en_removed = apply_removals(en_lines, en_info["remove_ranges"])

    jp_title_changed = False
    en_title_changed = False
    jp_original_title = ""
    en_original_title = ""
    jp_clean_title = ""
    en_clean_title = ""

    if jp_info["chapter_title"] is not None and jp_info["chapter_title_line"] != -1:
        lines_before = sum(
            1 for rng in jp_info["remove_ranges"]
            for idx in range(rng["start"] - 1, rng["end"])
            if idx < jp_info["chapter_title_line"] - 1
        )
        adjusted_line = jp_info["chapter_title_line"] - lines_before
        jp_working, jp_title_changed, jp_original_title = apply_title_rewrite(
            jp_working, jp_info["chapter_title"], adjusted_line
        )
        if jp_title_changed:
            jp_clean_title = jp_info["chapter_title"]

    if en_info["chapter_title"] is not None and en_info["chapter_title_line"] != -1:
        lines_before = sum(
            1 for rng in en_info["remove_ranges"]
            for idx in range(rng["start"] - 1, rng["end"])
            if idx < en_info["chapter_title_line"] - 1
        )
        adjusted_line = en_info["chapter_title_line"] - lines_before
        en_working, en_title_changed, en_original_title = apply_title_rewrite(
            en_working, en_info["chapter_title"], adjusted_line
        )
        if en_title_changed:
            en_clean_title = en_info["chapter_title"]

    any_change = any((jp_removed > 0, en_removed > 0, jp_title_changed, en_title_changed))

    if any_change:
        write_lines(jp_dst, jp_working)
        write_lines(en_dst, en_working)

    removals_changed = (jp_removed > 0) or (en_removed > 0)
    title_changed = jp_title_changed or en_title_changed

    if not any_change:
        status = "CLEAN"
    elif removals_changed and title_changed:
        status = "CLEANED+TITLE"
    elif removals_changed:
        status = "CLEANED"
    else:
        status = "TITLE_FIXED"

    return _build_result(
        key,
        chapter_name,
        status,
        confidence=parsed.get("confidence"),
        jp_removed=jp_removed,
        en_removed=en_removed,
        jp_changes={
            "chapter_title": jp_clean_title if jp_title_changed else None,
            "original_title": jp_original_title if jp_title_changed else None,
            "remove_ranges": jp_info["remove_ranges"],
        },
        en_changes={
            "chapter_title": en_clean_title if en_title_changed else None,
            "original_title": en_original_title if en_title_changed else None,
            "remove_ranges": en_info["remove_ranges"],
        },
    )


# ============================================================
# SUMMARY
# ============================================================

def _conf_str(conf) -> str:
    """Format a confidence value for display."""
    return f"conf={conf}" if conf is not None else "conf=?"


def print_summary(by_key: dict, found_only: bool = False):
    all_r = list(by_key.values())
    total = len(all_r)

    clean      = sum(1 for r in all_r if r.get("status") == "CLEAN")
    cleaned    = sum(1 for r in all_r if r.get("status") == "CLEANED")
    title_fixed = sum(1 for r in all_r if r.get("status") == "TITLE_FIXED")
    both       = sum(1 for r in all_r if r.get("status") == "CLEANED+TITLE")
    low_conf   = sum(1 for r in all_r if r.get("status") == "LOW_CONFIDENCE")
    diverged   = sum(1 for r in all_r if r.get("status") == "DIVERGED")
    errors     = sum(1 for r in all_r if r.get("status") == "ERROR")

    print(f"\n{'='*60}")
    print(f"SUMMARY  —  {total} pairs")
    print(f"  CLEAN           (no changes)              : {clean}")
    print(f"  CLEANED         (extra content removed)   : {cleaned}")
    print(f"  TITLE_FIXED     (chapter number stripped) : {title_fixed}")
    print(f"  CLEANED+TITLE   (both)                    : {both}")
    print(f"  LOW_CONFIDENCE  (50 ≤ conf < 80, skipped) : {low_conf}")
    print(f"  DIVERGED        (conf < 50)               : {diverged}")
    print(f"  ERROR                                     : {errors}")
    print(f"{'='*60}")

    modified = [r for r in all_r if r.get("status") in ("CLEANED", "CLEANED+TITLE", "TITLE_FIXED")]
    if modified:
        print("\nPairs with changes:")
        for r in modified:
            jp_removed = r.get("jp_removed_line_count", 0)
            en_removed = r.get("en_removed_line_count", 0)
            jp_changes = r.get("jp_changes") or {}
            en_changes = r.get("en_changes") or {}
            conf       = r.get("confidence")

            notes = []
            if jp_removed > 0:
                notes.append(f"JP -{jp_removed} lines")
            if en_removed > 0:
                notes.append(f"EN -{en_removed} lines")

            title_note = []
            if jp_changes.get("chapter_title") is not None:
                title_note.append(
                    f'JP title: "{jp_changes.get("original_title","")}" → "{jp_changes.get("chapter_title","")}"'
                )
            if en_changes.get("chapter_title") is not None:
                title_note.append(
                    f'EN title: "{en_changes.get("original_title","")}" → "{en_changes.get("chapter_title","")}"'
                )

            print(
                f"  {r['chapter']}  [{_conf_str(conf)}]  —  "
                f"{', '.join(notes) if notes else 'title only'}"
            )
            if title_note:
                print("      " + " | ".join(title_note))
            if not found_only:
                if jp_changes.get("remove_ranges"):
                    for rng in jp_changes["remove_ranges"]:
                        print(f"      JP lines {rng['start']}–{rng['end']} ({rng.get('reason','?')})")
                if en_changes.get("remove_ranges"):
                    for rng in en_changes["remove_ranges"]:
                        print(f"      EN lines {rng['start']}–{rng['end']} ({rng.get('reason','?')})")

    low_conf_items = [r for r in all_r if r.get("status") == "LOW_CONFIDENCE"]
    if low_conf_items:
        print("\nPairs skipped (low confidence — 50 ≤ conf < 80):")
        for r in low_conf_items:
            conf   = r.get("confidence")
            reason = r.get("reason", "")
            reason_str = f"  — {reason}" if reason else ""
            print(f"  ? {r['chapter']}  [{_conf_str(conf)}]{reason_str}")

    diverged_items = [r for r in all_r if r.get("status") == "DIVERGED"]
    if diverged_items:
        print("\nPairs marked as diverged (conf < 50):")
        for r in diverged_items:
            conf   = r.get("confidence")
            reason = r.get("reason", "")
            reason_str = f"  — {reason}" if reason else ""
            print(f"  ✗ {r['chapter']}  [{_conf_str(conf)}]{reason_str}")

    error_items = [r for r in all_r if r.get("status") == "ERROR"]
    if error_items:
        print("\nPairs with errors:")
        for r in error_items:
            print(f"  ! {r['chapter']}  —  {r.get('error','?')}")

    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean aligned JP/EN chapter pairs using Azure GPT-4o-mini. "
            "The model first checks whether the stories match, then cleans both sides."
        )
    )
    parser.add_argument(
        "folder",
        help="Root novel folder (contains JP-Aligned / EN-Aligned sub-folders)",
    )
    parser.add_argument(
        "--found-only",
        action="store_true",
        help="Suppress per-range details in the final summary",
    )
    args = parser.parse_args()

    root = Path(args.folder)
    prog_file = root / "aligned_clean_progress.json"

    if not root.exists():
        print(f"Error: folder not found: {root}")
        sys.exit(1)

    print(f"\nConfidence thresholds:")
    print(f"  ≥ {CONFIDENCE_ACCEPT}  → same story, proceed to cleaning")
    print(f"  {CONFIDENCE_UNSURE}–{CONFIDENCE_ACCEPT - 1} → LOW_CONFIDENCE (skipped, logged)")
    print(f"  < {CONFIDENCE_UNSURE}   → DIVERGED (definitely different)\n")

    print("Preparing output folders …")
    ensure_output_folders(root)

    pairs = collect_pairs(root)
    if not pairs:
        print("No aligned .txt pairs found in JP-Aligned / EN-Aligned.")
        missing = collect_missing_pairs(root)
        if missing:
            print("Mismatched files:")
            for name in missing:
                print(f"  {name}")
        sys.exit(0)

    ordered_keys = [name for name, _, _ in pairs]
    pair_map = {
        name: (jp_path, en_path,
               root / OUTPUT_JP / name,
               root / OUTPUT_EN / name)
        for name, jp_path, en_path in pairs
    }

    results_by_key: dict = {}
    consecutive_divergences = 0

    if prog_file.exists():
        completed = load_progress(prog_file)
        mode = ask_resume(prog_file, completed, ordered_keys)

        if mode == "restart":
            results_by_key = {}
            pending_keys = list(ordered_keys)
            consecutive_divergences = 0
            print("\nRefreshing output folders from originals …")
            for subdir in (OUTPUT_JP, OUTPUT_EN):
                working = root / subdir
                if working.exists():
                    shutil.rmtree(working)
            ensure_output_folders(root)

        elif mode == "skip":
            results_by_key = dict(completed)
            pending_keys = [k for k in ordered_keys if k not in completed]
            print(f"Skipping {len(completed)} already-done pair(s).\n")

        elif mode == "errors":
            skip_statuses = {"ERROR", "DIVERGED", "LOW_CONFIDENCE"}
            results_by_key = {k: v for k, v in completed.items() if v.get("status") not in skip_statuses}
            retry_keys     = {k for k, v in completed.items() if v.get("status") in skip_statuses}
            missing_keys   = set(ordered_keys) - set(completed.keys())
            pending_keys   = [k for k in ordered_keys if k in retry_keys or k in missing_keys]
            print(
                f"Re-running {len(retry_keys)} error/diverged/low-conf pair(s) + "
                f"{len(missing_keys)} missing. "
                f"Keeping {len(results_by_key)} clean result(s).\n"
            )

        else:
            results_by_key = {}
            pending_keys = list(ordered_keys)
    else:
        pending_keys = list(ordered_keys)

    if not pending_keys:
        print("Nothing left to process.")
        save_progress(prog_file, ordered_keys, results_by_key)
        print_summary(results_by_key, args.found_only)
        sys.exit(0)

    print(f"\nPairs to process : {len(pending_keys)}")
    print(f"Already done     : {len(results_by_key)}")
    print(f"Azure URL        : {AZURE_OPENAI_URL}")
    print(f"Model hint       : {MODEL}")
    print(f"Workers          : {MAX_WORKERS}")
    print(f"Consecutive divergence limit: {MAX_CONSECUTIVE_DIVERGENCES + 1}th mismatch stops the run\n")

    # ============================================================
    # PARALLEL EXECUTION LOOP
    # ============================================================
    with tqdm(
        total=len(pending_keys),
        unit="pair",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_key = {}
            for key in pending_keys:
                if key not in pair_map:
                    continue
                jp_src, en_src, jp_dst, en_dst = pair_map[key]
                future = executor.submit(process_pair, key, jp_src, en_src, jp_dst, en_dst)
                future_to_key[future] = key

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                
                try:
                    result = future.result()
                except Exception as exc:
                    result = _build_result(key, key, "ERROR", error=f"Thread crash: {exc}")

                # Because execution order isn't guaranteed, we log a divergence if it happens
                # during the order of thread completion.
                if result.get("status") == "DIVERGED":
                    consecutive_divergences += 1
                else:
                    consecutive_divergences = 0

                results_by_key[key] = result
                save_progress(prog_file, ordered_keys, results_by_key)

                status = result.get("status", "ERROR")
                conf   = result.get("confidence")
                conf_tag = f" [{_conf_str(conf)}]"
                suffix = ""

                if status in ("CLEANED", "CLEANED+TITLE"):
                    suffix = f" (JP -{result.get('jp_removed_line_count', 0)} / EN -{result.get('en_removed_line_count', 0)} lines)"
                if status in ("TITLE_FIXED", "CLEANED+TITLE"):
                    jp_changes = result.get("jp_changes") or {}
                    en_changes = result.get("en_changes") or {}
                    title_bits = []
                    if jp_changes.get("chapter_title") is not None:
                        title_bits.append(f'JP: "{jp_changes.get("original_title","")}" → "{jp_changes.get("chapter_title","")}"')
                    if en_changes.get("chapter_title") is not None:
                        title_bits.append(f'EN: "{en_changes.get("original_title","")}" → "{en_changes.get("chapter_title","")}"')
                    if title_bits:
                        suffix += "  title: " + " | ".join(title_bits)
                elif status == "LOW_CONFIDENCE":
                    reason = result.get("reason", "")
                    suffix = f" skipped — {reason}" if reason else " skipped (uncertain match)"
                elif status == "DIVERGED":
                    reason = result.get("reason", "")
                    suffix = f" — {reason}" if reason else " Stories are not same"
                elif status == "ERROR":
                    suffix = f" ERR: {result.get('error', '?')[:80]}"

                icon = "✓" if status not in ("ERROR", "DIVERGED", "LOW_CONFIDENCE") else ("?" if status == "LOW_CONFIDENCE" else "✗")
                tqdm.write(f"  {icon} [{key}]{conf_tag}  →  {status}{suffix}")
                pbar.update(1)

                if consecutive_divergences > MAX_CONSECUTIVE_DIVERGENCES:
                    tqdm.write(
                        f"\nStopping early: stories diverged {consecutive_divergences} times in a row."
                    )
                    # Cancel any tasks that haven't started yet to clean up threads gracefully
                    for f in future_to_key:
                        f.cancel()
                    break

    save_progress(prog_file, ordered_keys, results_by_key)
    print(f"\nProgress saved → {prog_file}")
    print_summary(results_by_key, args.found_only)


if __name__ == "__main__":
    main()