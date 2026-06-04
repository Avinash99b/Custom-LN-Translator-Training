#!/usr/bin/env python3
"""
kaggle_vllm_novel_cleaner.py

Kaggle-friendly novel cleaner for JP-Output / EN-Output folders.

What it does:
  1) Extracts /kaggle/working/Novel.zip into /kaggle/working/Novel (if needed)
  2) Backs up JP-Output / EN-Output to .BKP copies once
  3) Loads a local vLLM model on Kaggle's dual T4s
  4) Scans files in batches for translator notes / editor notes / ads / afterwords
  5) Removes only clearly non-story lines using deterministic regex + excerpt matching
  6) Re-checks cleaned files for a few passes
  7) Saves progress so runs can be resumed

Notes:
  - This script preserves story text as aggressively as possible.
  - It does NOT ask the model to rewrite the chapter.
  - It only asks the model to identify clear non-story content.

Usage examples:
    python kaggle_vllm_novel_cleaner.py
    python kaggle_vllm_novel_cleaner.py --root /kaggle/working/Novel
    python kaggle_vllm_novel_cleaner.py --zip-path /kaggle/working/Novel.zip
    python kaggle_vllm_novel_cleaner.py --batch-size 48 --max-clean-passes 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Kaggle + tokenizers behave better with this off.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from vllm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover
    print("Failed to import vLLM. Make sure it is installed in Kaggle.")
    raise


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_ZIP_PATH = Path("/kaggle/working/Novel.zip")
DEFAULT_ROOT = Path("/kaggle/working/Novel")
DEFAULT_PROGRESS_FILE = "clean_extra_content.json"
DEFAULT_OUTPUT_ZIP = Path("/kaggle/working/Novel_Cleaned.zip")

# Best-first candidate order.
# Qwen3 is the latest generation in the Qwen series, and the collection
# includes 14B-AWQ and 8B-AWQ checkpoints. vLLM supports AWQ and multi-GPU
# tensor parallelism, so this fits Kaggle's dual T4 setup well.
MODEL_CANDIDATES = [
    {"name": "Qwen/Qwen3-8B-AWQ", "quantization": "awq"},
    {"name": "Qwen/Qwen2.5-14B-Instruct-AWQ", "quantization": "awq"},
]

# Kaggle dual T4: conservative default batch size, can be raised.
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_MAX_TOKENS = 192
DEFAULT_GPU_MEM_UTIL = 0.95
DEFAULT_TENSOR_PARALLEL_SIZE = 1

SCAN_HEAD_LINES = 30
SCAN_TAIL_LINES = 20
SCAN_TAIL_LINES_RECHECK = 10
MAX_CLEAN_PASSES = 3


# ============================================================
# PROMPT
# ============================================================

DETECTION_PROMPT = """\
You are an expert at analysing light novel / web novel text files.

Decide whether the given snippet contains CLEAR, CERTAIN non-story content.

Extra-content types to detect:
  editor_note       — [E/N:], [Editor:], ED: blocks
  translator_note   — [TN:], [TL:], 訳注 blocks
  afterword         — author thank-you messages, 後書き, 感想, ありがとう
  table_of_contents — 目次, TOC
  copyright_notice  — © lines, ISBN, publisher boilerplate
  disclaimer        — "This is a work of fiction" blocks
  advertisement     — "Check out our other titles", Patreon links
  release_announcement — sponsored-by, ko-fi, shout-outs
  chapter_index     — numbered sub-chapter list within the file
  stat_box          — character stat blocks: レベル, HP, 天職, skills list, ステータス
  other_non_novel   — anything else that is clearly NOT story prose

DO NOT flag:
- Chapter or volume titles (プロローグ, エピローグ, 第一章, Chapter 1, Prologue, etc.)
- Section dividers alone (---, ***, 「---」)
- Story prose describing events, dialogue, character actions — even at the tail
- Any line you are not fully certain is extra content

If you removed extra content already and the remaining text looks like story prose, answer CLEAN.
When in doubt, answer CLEAN.

Output format — plain text only:

STATUS: CLEAN
or
STATUS: FOUND
FINDING: type=<type> | lang=JP|EN|BOTH | loc=head|tail|middle | excerpt=<verbatim, max 60 chars, no newlines>

Rules:
- First line is always STATUS: CLEAN or STATUS: FOUND
- Each FINDING line has exactly those 4 pipe-separated fields
- No markdown, no JSON, no extra commentary
- If CLEAN, output only the STATUS line
"""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class FileJob:
    key: str
    chapter_name: str
    language: str
    path: Path


# ============================================================
# FILE HELPERS
# ============================================================

def natural_sort_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def write_lines(path: Path, lines: Sequence[str]):
    path.write_text("\n".join(lines), encoding="utf-8")


def collapse_blank_lines(lines: Sequence[str]) -> List[str]:
    collapsed: List[str] = []
    blanks = 0
    for line in lines:
        if line.strip() == "":
            blanks += 1
            if blanks <= 2:
                collapsed.append(line)
        else:
            blanks = 0
            collapsed.append(line)
    return collapsed


def build_snippet(lines: Sequence[str], tail_lines: int = SCAN_TAIL_LINES) -> str:
    head = list(lines[:SCAN_HEAD_LINES])
    tail = list(lines[-tail_lines:]) if len(lines) > SCAN_HEAD_LINES + tail_lines else []
    if tail:
        return "\n".join(head) + "\n\n[… middle omitted …]\n\n" + "\n".join(tail)
    return "\n".join(head)


def collect_files(root: Path) -> List[FileJob]:
    result: List[FileJob] = []
    for lang, subdir in (("JP", "JP-Output.BKP"), ("EN", "EN-Output.BKP")):
        folder = root / subdir
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.txt"), key=lambda x: natural_sort_key(x.name)):
            key = f"{lang}/{f.name}"
            result.append(FileJob(key=key, chapter_name=f.name, language=lang, path=f))
    return result


# ============================================================
# BACKUP / ZIP
# ============================================================

def ensure_extracted(zip_path: Path, root: Path):
    if root.exists() and any((root / d).exists() for d in ("JP-Output", "EN-Output", "JP-Output.BKP", "EN-Output.BKP")):
        return
    if not zip_path.exists():
        return

    print(f"Extracting {zip_path.name} → {root} ...")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    print("Extraction done.")


def ensure_backups(root: Path):
    for subdir in ("JP-Output", "EN-Output"):
        src = root / subdir
        dst = root / f"{subdir}.BKP"
        if dst.exists():
            print(f"  Backup already exists, skipping: {dst.name}")
            continue
        if not src.exists():
            print(f"  Warning: source not found, skipping: {src}")
            continue
        print(f"  Backing up {src.name} → {dst.name} ...")
        shutil.copytree(src, dst)
        print("  Done.")


def make_output_zip(root: Path, output_zip: Path):
    base = output_zip.with_suffix("")
    if output_zip.exists():
        output_zip.unlink()
    shutil.make_archive(str(base), "zip", root)


# ============================================================
# PROGRESS
# ============================================================

def load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {e["key"]: e for e in data if isinstance(e, dict) and "key" in e}
    except Exception:
        return {}


def save_progress(path: Path, ordered_keys: Sequence[str], results_by_key: dict):
    ordered = [results_by_key[k] for k in ordered_keys if k in results_by_key]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ask_resume(prog_file: Path, completed: dict, all_keys: Sequence[str]) -> str:
    done_set = set(completed.keys())
    error_keys = {k for k, v in completed.items() if v.get("status") == "ERROR"}
    missing = set(all_keys) - done_set

    print(f"\n{'─'*60}")
    print(f"Progress file  : {prog_file.name}")
    print(f"  On disk      : {len(all_keys)}")
    print(f"  Done         : {len(done_set)}")
    print(f"  Errors       : {len(error_keys)}")
    print(f"  Missing      : {len(missing)}")

    if missing:
        print("\n  New / missing:")
        for k in sorted(missing, key=lambda x: natural_sort_key(x.split('/', 1)[-1])):
            print(f"    + {k}")
    if error_keys:
        print("\n  Errored:")
        for k in sorted(error_keys, key=lambda x: natural_sort_key(x.split('/', 1)[-1])):
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
# DETECTION RESPONSE PARSING
# ============================================================

def _parse_response(raw: str) -> dict:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return {"status": "ERROR", "findings": [], "error": "empty LLM response"}

    first = lines[0]
    if not first.upper().startswith("STATUS:"):
        return {"status": "ERROR", "findings": [], "error": f"unexpected first line: {first!r}"}

    status = first.split(":", 1)[1].strip().upper()
    if status not in ("CLEAN", "FOUND"):
        return {"status": "ERROR", "findings": [], "error": f"unknown status: {status!r}"}

    if status == "CLEAN":
        return {"status": "CLEAN", "findings": []}

    findings = []
    for line in lines[1:]:
        if not line.upper().startswith("FINDING:"):
            continue
        body = line.split(":", 1)[1].strip()
        fields = {}
        for part in body.split("|"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.strip().lower()] = v.strip()
        findings.append(
            {
                "type": fields.get("type", "other_non_novel"),
                "language": fields.get("lang", "?"),
                "location": fields.get("loc", "?"),
                "excerpt": fields.get("excerpt", ""),
            }
        )

    return {"status": "FOUND", "findings": findings}


# ============================================================
# STRIP LOGIC
# ============================================================

_STRIP_PATTERNS = [
    # JP author afterwords / thanks
    re.compile(r"(いつも|何時も)読んで(下さり|くださり)", re.IGNORECASE),
    re.compile(r"毎度.*(読んで|お読み)", re.IGNORECASE),
    re.compile(r"感想(・|·|、|も).*(有難う|ありがとう)", re.IGNORECASE),
    re.compile(r"(誤字脱字|脱字|誤字).*(報告|有難う|ありがとう)", re.IGNORECASE),
    re.compile(r"(有難う|ありがとう)ございます。?(感想|意見|報告)", re.IGNORECASE),
    re.compile(r"よろしくお願いします！?$", re.IGNORECASE),
    # TL / Editor notes (EN)
    re.compile(r"^\s*\[?(T[LN]|TL\s*Note|Editor|ED|E/N)\s*[:\[]", re.IGNORECASE),
    re.compile(r"^\s*\[?(Translator|Translation)\s*(Note|s)?[:\[]", re.IGNORECASE),
    # Release / Patreon / ads
    re.compile(r"(Patreon|patreon\.com|ko-fi\.com)", re.IGNORECASE),
    re.compile(r"(sponsored by|brought to you by)", re.IGNORECASE),
    re.compile(r"(無料配信|是非、見に行って|見に行ってみて)", re.IGNORECASE),
    # Copyright / disclaimer
    re.compile(r"©|Copyright\s+\d{4}", re.IGNORECASE),
    re.compile(r"(work of fiction|any resemblance)", re.IGNORECASE),
    # Content warnings / boilerplate
    re.compile(r"(グロ注意|R\s*18|成人向け|暴力表現|閲覧注意)", re.IGNORECASE),
    re.compile(r"タグ(も|を).*(変更|修正)", re.IGNORECASE),
    re.compile(r"活動報告", re.IGNORECASE),
]

_STAT_BLOCK_START = re.compile(
    r"(レベル\s*[:：]\s*\d|天職\s*[:：]|ステータス|【ステ|"
    r"技能\s*[:：]|筋力\s*[:：]\s*\d|HP\s*[:：]\s*\d|"
    r"魔力\s*[:：]\s*\d|耐性\s*[:：]\s*\d|体力\s*[:：]\s*\d|"
    r"\d+歳\s+[男女]\s+レベル)",
    re.IGNORECASE,
)

_TITLE_PATTERNS = re.compile(
    r"^(プロローグ|エピローグ|第[一二三四五六七八九十百\d]+[章節話幕]|"
    r"Chapter\s+\d|Prologue|Epilogue|Interlude|Side\s+Story|"
    r"volume\s+\d|Vol\s*\.|番外編|閑話)",
    re.IGNORECASE,
)


def _is_title_line(line: str) -> bool:
    return bool(_TITLE_PATTERNS.match(line.strip()))


def _line_matches_strip_pattern(line: str) -> bool:
    return any(pat.search(line) for pat in _STRIP_PATTERNS)


def _excerpt_matches_line(line: str, excerpt: str) -> bool:
    needle = excerpt[:40].strip()
    return bool(needle) and needle in line


def _finding_is_locatable(lines: Sequence[str], finding: dict) -> bool:
    excerpt = finding.get("excerpt", "")
    needle = excerpt[:30].strip()
    if not needle:
        return False
    return any(needle in line for line in lines)


def _remove_stat_block(lines: Sequence[str], anchor_idx: int) -> set:
    to_remove = set()
    i = anchor_idx
    while i < len(lines):
        line = lines[i]
        if line.strip() == "" and i > anchor_idx:
            break
        to_remove.add(i)
        i += 1
    return to_remove


def remove_extra_lines(lines: Sequence[str], findings: Sequence[dict]) -> Tuple[List[str], int]:
    """
    Remove lines identified by the LLM findings.

    Rules:
      1) stat_box findings remove the whole stat block starting at the match.
      2) Other findings remove lines matching known strip patterns or the excerpt.
      3) Collapse excessive blank lines after removal.
    """
    to_remove: set = set()
    n = len(lines)

    for finding in findings:
        ftype = finding.get("type", "")
        loc = finding.get("location", "middle")
        excerpt = finding.get("excerpt", "")

        if loc == "head":
            region = range(min(SCAN_HEAD_LINES + 5, n))
        elif loc == "tail":
            region = range(max(0, n - SCAN_TAIL_LINES - 5), n)
        else:
            region = range(n)

        if ftype == "stat_box":
            matched_any = False
            for i in region:
                if _STAT_BLOCK_START.search(lines[i]):
                    to_remove |= _remove_stat_block(lines, i)
                    matched_any = True
                    break
            if not matched_any and excerpt:
                for i in region:
                    if _excerpt_matches_line(lines[i], excerpt):
                        to_remove |= _remove_stat_block(lines, i)
                        matched_any = True
                        break
        else:
            matched = False
            for i in region:
                line = lines[i]
                if _is_title_line(line):
                    continue
                if _line_matches_strip_pattern(line) or _excerpt_matches_line(line, excerpt):
                    to_remove.add(i)
                    matched = True

            if not matched and excerpt:
                needle = excerpt[:30].strip()
                for i in range(n):
                    if needle and needle in lines[i] and not _is_title_line(lines[i]):
                        to_remove.add(i)

    if not to_remove:
        return list(lines), 0

    cleaned = [l for i, l in enumerate(lines) if i not in to_remove]
    cleaned = collapse_blank_lines(cleaned)
    return cleaned, len(to_remove)


# ============================================================
# vLLM LOADING / BATCH DETECTION
# ============================================================

def _candidate_to_kwargs(candidate: dict) -> dict:
    kwargs = {
        "model": candidate["name"],
        "tensor_parallel_size": DEFAULT_TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": DEFAULT_GPU_MEM_UTIL,
        "max_model_len": DEFAULT_MAX_MODEL_LEN,
        "trust_remote_code": True,
        "dtype": "auto",
    }
    if candidate.get("quantization"):
        kwargs["quantization"] = candidate["quantization"]
    return kwargs


def load_llm(model_override: Optional[str] = None) -> Tuple[LLM, str]:
    attempts = []
    candidates = ([{"name": model_override, "quantization": "awq"}] if model_override else MODEL_CANDIDATES)

    last_error = None
    for cand in candidates:
        try:
            print(f"Loading model: {cand['name']}")
            llm = LLM(**_candidate_to_kwargs(cand))
            print(f"Loaded model: {cand['name']}")
            return llm, cand["name"]
        except Exception as exc:
            last_error = exc
            attempts.append((cand["name"], str(exc)))
            print(f"  Failed: {cand['name']}")
            print(f"  Reason: {exc}")

    msg = ["Could not load any candidate model."]
    for name, err in attempts:
        msg.append(f"- {name}: {err}")
    raise RuntimeError("\n".join(msg)) from last_error


def build_prompt(snippet: str) -> str:
    return DETECTION_PROMPT + "\n\nText to analyse:\n\n" + snippet


def detect_batch(llm: LLM, prompts: Sequence[str], max_tokens: int) -> List[dict]:
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(list(prompts), sampling_params)
    return [_parse_response(o.outputs[0].text.strip()) for o in outputs]


# ============================================================
# RESULT BUILDING
# ============================================================

def _build_result(
    key: str,
    chapter: str,
    language: str,
    status: str,
    findings: Sequence[dict],
    lines_removed: int,
    passes: int,
    history: Sequence[dict],
    error: Optional[str] = None,
):
    r = {
        "key": key,
        "chapter": chapter,
        "language": language,
        "status": status,
        "findings": list(findings),
        "finding_types": sorted({f.get("type", "?") for f in findings}),
        "lines_removed": lines_removed,
        "passes": passes,
        "clean_history": list(history),
    }
    if error:
        r["error"] = error
    return r


def print_summary(by_key: dict, found_only: bool = False):
    all_r = list(by_key.values())
    total = len(all_r)
    clean = sum(1 for r in all_r if r.get("status") == "CLEAN")
    cleaned = sum(1 for r in all_r if r.get("status") == "CLEANED")
    found = sum(1 for r in all_r if r.get("status") == "FOUND")
    errors = sum(1 for r in all_r if r.get("status") == "ERROR")

    print(f"\n{'='*60}")
    print(f"SUMMARY  —  {total} files")
    print(f"  CLEAN   (no extra content)      : {clean}")
    print(f"  CLEANED (extra content removed) : {cleaned}")
    print(f"  FOUND   (could not fully clean) : {found}")
    print(f"  ERROR                           : {errors}")
    print(f"{'='*60}")

    dirty = [r for r in all_r if r.get("status") in ("CLEANED", "FOUND")]
    if dirty:
        print("\nFiles modified or still with issues:")
        for r in dirty:
            tag = "✓ cleaned" if r["status"] == "CLEANED" else "⚠ partially cleaned"
            print(
                f"  [{r['language']}] {r['chapter']}  {tag}  "
                f"(removed {r.get('lines_removed', 0)} line(s), {r.get('passes', 0)} pass(es))"
            )
            if not found_only:
                for f in r.get("findings", []):
                    print(f"    • {f['type']} ({f['language']}, {f['location']}): {f['excerpt'][:70]!r}")
    print()


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_all(root: Path, llm: LLM, model_name: str, batch_size: int, max_clean_passes: int, max_tokens: int, found_only: bool):
    prog_file = root / DEFAULT_PROGRESS_FILE

    # Ensure backups and collect files.
    print("\nChecking backups ...")
    ensure_backups(root)

    all_files = collect_files(root)
    if not all_files:
        print("No .txt files found in JP-Output.BKP / EN-Output.BKP.")
        return {}

    ordered_keys = [f.key for f in all_files]
    file_map = {f.key: f for f in all_files}

    results_by_key: Dict[str, dict] = {}
    pending_keys: List[str]

    if prog_file.exists():
        completed = load_progress(prog_file)
        mode = ask_resume(prog_file, completed, ordered_keys)

        if mode == "restart":
            results_by_key = {}
            pending_keys = list(ordered_keys)
        elif mode == "skip":
            results_by_key = dict(completed)
            pending_keys = [k for k in ordered_keys if k not in completed]
            print(f"Skipping {len(completed)} already-done file(s).\n")
        elif mode == "errors":
            results_by_key = {k: v for k, v in completed.items() if v.get("status") != "ERROR"}
            err_keys = {k for k, v in completed.items() if v.get("status") == "ERROR"}
            missing_keys = set(ordered_keys) - set(completed.keys())
            pending_keys = [k for k in ordered_keys if k in err_keys or k in missing_keys]
            print(f"Re-running {len(err_keys)} error(s) + {len(missing_keys)} missing. Keeping {len(results_by_key)} clean result(s).\n")
        else:
            results_by_key = {}
            pending_keys = list(ordered_keys)
    else:
        results_by_key = {}
        pending_keys = list(ordered_keys)

    if not pending_keys:
        print("Nothing to process.")
        save_progress(prog_file, ordered_keys, results_by_key)
        print_summary(results_by_key, found_only)
        return results_by_key

    print(f"Files to process : {len(pending_keys)}")
    print(f"Already done     : {len(results_by_key)}")
    print(f"Model            : {model_name}")
    print(f"Batch size       : {batch_size}")
    print(f"Max passes       : {max_clean_passes}")
    print(f"Max tokens       : {max_tokens}\n")

    # Round-based processing: all still-dirty files are rescanned together.
    had_any_removals: Dict[str, bool] = {k: bool(results_by_key.get(k, {}).get("lines_removed", 0)) for k in ordered_keys}

    for round_num in range(1, max_clean_passes + 1):
        round_keys = [k for k in pending_keys if k in file_map]
        if not round_keys:
            break

        print(f"\n===== ROUND {round_num}/{max_clean_passes} — scanning {len(round_keys)} file(s) =====")
        next_pending: List[str] = []

        for start in range(0, len(round_keys), batch_size):
            batch_keys = round_keys[start : start + batch_size]
            jobs = []
            prompts = []

            for key in batch_keys:
                job = file_map[key]
                lines = read_lines(job.path)
                tail_lines = SCAN_TAIL_LINES if round_num == 1 else SCAN_TAIL_LINES_RECHECK
                snippet = build_snippet(lines, tail_lines=tail_lines)
                jobs.append((job, lines))
                prompts.append(build_prompt(snippet))

            responses = detect_batch(llm, prompts, max_tokens=max_tokens)

            for (job, lines), result in zip(jobs, responses):
                key = job.key
                current_history = results_by_key.get(key, {}).get("clean_history", [])
                current_findings = results_by_key.get(key, {}).get("findings", [])
                current_removed = results_by_key.get(key, {}).get("lines_removed", 0)
                passes_done = results_by_key.get(key, {}).get("passes", 0)

                if result["status"] == "ERROR":
                    print(f"  [ERROR] {key}: {result.get('error')}")
                    results_by_key[key] = _build_result(
                        key, job.chapter_name, job.language, "ERROR", current_findings, current_removed, passes_done, current_history, result.get("error")
                    )
                    continue

                if result["status"] == "CLEAN":
                    status = "CLEANED" if had_any_removals.get(key, False) else "CLEAN"
                    print(f"  [CLEAN] {key} -> {status}")
                    results_by_key[key] = _build_result(
                        key,
                        job.chapter_name,
                        job.language,
                        status,
                        current_findings,
                        current_removed,
                        passes_done,
                        current_history + [{"round": round_num, "action": "clean"}],
                    )
                    continue

                findings = result.get("findings", [])
                locatable = [f for f in findings if _finding_is_locatable(lines, f)]
                ghost = [f for f in findings if not _finding_is_locatable(lines, f)]

                if ghost:
                    print(f"  [GHOST] {key}: skipping {len(ghost)} unlocatable finding(s)")
                    for f in ghost[:3]:
                        print(f"    ✗ {f['type']} / {f['location']}: {f['excerpt'][:60]!r}")

                if not locatable:
                    status = "CLEANED" if had_any_removals.get(key, False) else "CLEAN"
                    print(f"  [GUARD] {key}: no locatable findings; treating as {status}")
                    results_by_key[key] = _build_result(
                        key,
                        job.chapter_name,
                        job.language,
                        status,
                        current_findings,
                        current_removed,
                        passes_done,
                        current_history + [{"round": round_num, "action": "hallucination_guard"}],
                    )
                    continue

                cleaned_lines, n_removed = remove_extra_lines(lines, locatable)
                if n_removed > 0:
                    had_any_removals[key] = True
                    current_removed += n_removed
                    passes_done += 1
                    current_findings.extend(locatable)
                    current_history.append(
                        {
                            "round": round_num,
                            "action": "removed",
                            "removed": n_removed,
                            "types": [f.get("type", "?") for f in locatable],
                        }
                    )
                    write_lines(job.path, cleaned_lines)
                    next_pending.append(key)
                    print(f"  [FIX ] {key}: removed {n_removed} line(s); will recheck")
                    results_by_key[key] = _build_result(
                        key,
                        job.chapter_name,
                        job.language,
                        "CLEANED",
                        current_findings,
                        current_removed,
                        passes_done,
                        current_history,
                    )
                else:
                    # Findings were locatable, but no lines got removed.
                    # Keep the file for another pass only if we haven't exhausted passes.
                    current_history.append(
                        {
                            "round": round_num,
                            "action": "no_removal",
                            "types": [f.get("type", "?") for f in locatable],
                        }
                    )
                    status = "CLEANED" if had_any_removals.get(key, False) else "FOUND"
                    print(f"  [WARN] {key}: could not remove anything; status={status}")
                    results_by_key[key] = _build_result(
                        key,
                        job.chapter_name,
                        job.language,
                        status,
                        current_findings,
                        current_removed,
                        passes_done,
                        current_history,
                    )
                    if round_num < max_clean_passes:
                        next_pending.append(key)

        pending_keys = list(dict.fromkeys(next_pending))
        save_progress(prog_file, ordered_keys, results_by_key)
        print(f"Round {round_num} done. Pending for next round: {len(pending_keys)}")

        if not pending_keys:
            break

    # Finalize any still-pending files as FOUND/CLEANED based on what happened.
    for key in pending_keys:
        if key not in file_map:
            continue
        job = file_map[key]
        prev = results_by_key.get(key, {})
        status = "CLEANED" if had_any_removals.get(key, False) else prev.get("status", "FOUND")
        results_by_key[key] = _build_result(
            key,
            job.chapter_name,
            job.language,
            status,
            prev.get("findings", []),
            prev.get("lines_removed", 0),
            prev.get("passes", 0),
            prev.get("clean_history", []),
            prev.get("error"),
        )

    save_progress(prog_file, ordered_keys, results_by_key)
    print(f"\nProgress saved → {prog_file}")
    print_summary(results_by_key, found_only)
    return results_by_key


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description="Find and remove editor/translator notes and other non-novel content from JP-Output and EN-Output using Kaggle vLLM.",
    )
    p.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH, help="Path to Novel.zip (default: /kaggle/working/Novel.zip)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Extraction root / novel folder (default: /kaggle/working/Novel)")
    p.add_argument("--output-zip", type=Path, default=DEFAULT_OUTPUT_ZIP, help="Output zip path for cleaned tree")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Prompts per vLLM batch")
    p.add_argument("--max-clean-passes", type=int, default=MAX_CLEAN_PASSES, help="Maximum re-check rounds")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for model output")
    p.add_argument("--found-only", action="store_true", help="Suppress per-finding details in summary")
    p.add_argument("--model", type=str, default=None, help="Force a specific model instead of auto fallback")
    return p.parse_args(argv)


def main():
    args = parse_args()

    ensure_extracted(args.zip_path, args.root)
    if not args.root.exists():
        print(f"Root folder not found: {args.root}")
        sys.exit(1)

    llm, model_name = load_llm(args.model)
    process_all(
        root=args.root,
        llm=llm,
        model_name=model_name,
        batch_size=max(1, args.batch_size),
        max_clean_passes=max(1, args.max_clean_passes),
        max_tokens=max(1, args.max_tokens),
        found_only=args.found_only,
    )

    print("Creating output zip ...")
    make_output_zip(args.root, args.output_zip)
    print(f"Saved cleaned archive → {args.output_zip}")


if __name__ == "__main__":
    main()
