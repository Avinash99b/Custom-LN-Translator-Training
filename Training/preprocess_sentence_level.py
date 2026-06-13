#!/usr/bin/env python3
"""
Build Qwen3-4B SFT data from aligned_sentences.jsonl files.

Directory layout:

root/
  NovelA/
    aligned_sentences.jsonl
  NovelB/
    aligned_sentences.jsonl

Each line in aligned_sentences.jsonl:
  {"chapter":"6.txt","pair_index":0,"jp":"...","en":"...","similarity":0.68,...}

Output JSONL (chat format, Qwen3 no-think mode):
  {
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Qwen3 no-think system prompt
# Appending /no_think to the system prompt signals Qwen3 to skip <think> blocks.
# Keep it short — every token here is repeated across all training examples.
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a literary translation model. "
    "Translate faithfully, preserving tone, names, and paragraph structure."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return rows


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def norm(text: Any) -> str:
    return "" if text is None else str(text).strip()


# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------

def build_side_text(window: List[Dict[str, Any]], side: str, separator: str) -> str:
    parts = [norm(item.get(side, "")) for item in window]
    return separator.join(p for p in parts if p).strip()


def make_example(
    window: List[Dict[str, Any]],
    direction: str,
    separator: str,
    system_prompt: str,
) -> Optional[Dict[str, Any]]:
    src, tgt = ("jp", "en") if direction == "jp2en" else ("en", "jp")
    source = build_side_text(window, src, separator)
    target = build_side_text(window, tgt, separator)
    if not source or not target:
        return None

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": source},
            {"role": "assistant", "content": target},
        ]
    }


def chapter_examples(
    rows: List[Dict[str, Any]],
    window_size: int,
    stride: int,
    direction: str,
    separator: str,
    system_prompt: str,
    min_similarity: float,
) -> List[Dict[str, Any]]:
    # Filter low-quality pairs
    if min_similarity > 0.0:
        rows = [r for r in rows if float(r.get("similarity", 1.0)) >= min_similarity]

    if len(rows) < window_size:
        return []

    rows = sorted(rows, key=lambda x: safe_int(x.get("pair_index", 0)))

    examples: List[Dict[str, Any]] = []
    for start in range(0, len(rows) - window_size + 1, stride):
        ex = make_example(
            window=rows[start : start + window_size],
            direction=direction,
            separator=separator,
            system_prompt=system_prompt,
        )
        if ex is not None:
            examples.append(ex)
    return examples


def build_dataset(
    root_dir: Path,
    window_size: int,
    direction: str,
    separator: str,
    system_prompt: str,
    stride: int,
    min_similarity: float,
    deduplicate: bool,
) -> List[Dict[str, Any]]:
    all_examples: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for novel_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        aligned_path = novel_dir / "aligned_sentences.jsonl"
        if not aligned_path.exists():
            continue

        rows = load_jsonl(aligned_path)

        chapters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            ch = norm(row.get("chapter", ""))
            if ch:
                chapters[ch].append(row)

        for ch_name in sorted(chapters):
            for ex in chapter_examples(
                rows=chapters[ch_name],
                window_size=window_size,
                stride=stride,
                direction=direction,
                separator=separator,
                system_prompt=system_prompt,
                min_similarity=min_similarity,
            ):
                if deduplicate:
                    # Hash on (user_content, assistant_content) to catch exact duplicates
                    msgs = ex["messages"]
                    key = hashlib.md5(
                        (msgs[1]["content"] + "\x00" + msgs[2]["content"]).encode()
                    ).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                all_examples.append(ex)

    return all_examples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3-4B SFT JSONL from aligned_sentences.jsonl files."
    )
    parser.add_argument("--root",         type=Path, required=True,  help="Root folder containing novel sub-folders.")
    parser.add_argument("--out",          type=Path, required=True,  help="Output JSONL path.")
    parser.add_argument("--window_size",  type=int,  default=3,      help="Aligned pairs per training example (default: 3).")
    parser.add_argument("--stride",       type=int,  default=1,      help="Window stride; 1 = fully overlapping (default: 1).")
    parser.add_argument("--direction",    type=str,  default="jp2en", choices=["jp2en", "en2jp"],
                        help="Translation direction (default: jp2en).")
    parser.add_argument("--separator",    type=str,  default="\n",   help="Segment separator inside a window (default: newline).")
    parser.add_argument("--min_sim",      type=float, default=0.5,   help="Drop pairs below this similarity score (default: 0.5).")
    parser.add_argument("--no_dedup",     action="store_true",       help="Disable exact-duplicate removal.")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System message for every example.")

    args = parser.parse_args()

    if args.window_size < 1:
        raise ValueError("--window_size must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not args.root.exists():
        raise FileNotFoundError(f"Root directory not found: {args.root}")

    examples = build_dataset(
        root_dir=args.root,
        window_size=args.window_size,
        direction=args.direction,
        separator=args.separator,
        system_prompt=args.system_prompt,
        stride=args.stride,
        min_similarity=args.min_sim,
        deduplicate=not args.no_dedup,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"root            : {args.root}")
    print(f"output          : {args.out}")
    print(f"direction       : {args.direction}")
    print(f"window_size     : {args.window_size}")
    print(f"stride          : {args.stride}")
    print(f"min_similarity  : {args.min_sim}")
    print(f"deduplication   : {not args.no_dedup}")
    print(f"examples_written: {len(examples)}")


if __name__ == "__main__":
    main()