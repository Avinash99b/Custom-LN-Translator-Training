#!/usr/bin/env python3
"""
Build Qwen-style SFT data from aligned_sentences.jsonl files.

Expected directory layout:

root/
  NovelA/
    aligned_sentences.jsonl
  NovelB/
    aligned_sentences.jsonl
  ...

Each line in aligned_sentences.jsonl should look like:
{"chapter":"6.txt","pair_index":0,"jp":"...","en":"...","similarity":0.68,...}

This script:
- groups rows by chapter
- sorts by pair_index
- creates overlapping windows of aligned pairs
- writes a JSONL file in chat format:
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    ...
  }

The output is suitable for chat-template-based SFT pipelines.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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
                raise ValueError(f"Invalid JSON at {path} line {line_no}: {e}") from e
    return rows


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def build_window_text(
    window: List[Dict[str, Any]],
    side: str,
    separator: str,
    include_indices: bool = False,
) -> str:
    """
    side: "jp" or "en"
    """
    parts: List[str] = []
    for item in window:
        text = normalize_text(item.get(side, ""))
        if not text:
            continue

        if include_indices:
            idx = safe_int(item.get("pair_index", -1), -1)
            parts.append(f"[{idx}] {text}")
        else:
            parts.append(text)

    return separator.join(parts).strip()


def make_chat_example(
    novel_name: str,
    chapter_name: str,
    window: List[Dict[str, Any]],
    direction: str,
    separator: str,
    system_prompt: str,
    include_metadata: bool = True,
    include_indices_in_prompt: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Build one supervised chat example from a window of aligned pairs.

    direction:
      - jp2en: Japanese source -> English target
      - en2jp: English source -> Japanese target
    """
    if direction not in {"jp2en", "en2jp"}:
        raise ValueError("direction must be one of: jp2en, en2jp")

    source_side = "jp" if direction == "jp2en" else "en"
    target_side = "en" if direction == "jp2en" else "jp"

    source_text = build_window_text(
        window=window,
        side=source_side,
        separator=separator,
        include_indices=include_indices_in_prompt,
    )
    target_text = build_window_text(
        window=window,
        side=target_side,
        separator=separator,
        include_indices=False,
    )

    if not source_text or not target_text:
        return None

    pair_indices = [safe_int(x.get("pair_index", -1), -1) for x in window]

    if direction == "jp2en":
        task_line = (
            "Translate the Japanese text into natural English. "
            "Preserve paragraph breaks and meaning."
        )
        user_content = source_text
    else:
        task_line = (
            "Translate the English text into natural Japanese. "
            "Preserve paragraph breaks and meaning."
        )
        user_content = source_text

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        {
            "role": "user",
            "content": user_content.strip(),
        },
        {
            "role": "assistant",
            "content": target_text.strip(),
        },
    ]

    example: Dict[str, Any] = {
        "messages": messages,
    }

    if include_metadata:
        example.update(
            {
                "novel": novel_name,
                "chapter": chapter_name,
                "direction": direction,
                "window_size": len(window),
                "window_pair_indices": pair_indices,
                "task": task_line,
                "source_side": source_side,
                "target_side": target_side,
            }
        )

    return example


def build_examples_for_chapter(
    novel_name: str,
    chapter_name: str,
    rows: List[Dict[str, Any]],
    window_size: int,
    direction: str,
    separator: str,
    system_prompt: str,
    stride: int = 1,
    include_metadata: bool = True,
    include_indices_in_prompt: bool = False,
) -> List[Dict[str, Any]]:
    if len(rows) < window_size:
        return []

    rows = sorted(rows, key=lambda x: safe_int(x.get("pair_index", 0), 0))

    examples: List[Dict[str, Any]] = []
    for start in range(0, len(rows) - window_size + 1, stride):
        window = rows[start : start + window_size]
        ex = make_chat_example(
            novel_name=novel_name,
            chapter_name=chapter_name,
            window=window,
            direction=direction,
            separator=separator,
            system_prompt=system_prompt,
            include_metadata=include_metadata,
            include_indices_in_prompt=include_indices_in_prompt,
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
    include_metadata: bool,
    include_indices_in_prompt: bool,
) -> List[Dict[str, Any]]:
    all_examples: List[Dict[str, Any]] = []

    novel_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    for novel_dir in novel_dirs:
        aligned_path = novel_dir / "aligned_sentences.jsonl"
        if not aligned_path.exists():
            continue

        rows = load_jsonl(aligned_path)

        chapters: dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            chapter = normalize_text(row.get("chapter", ""))
            if not chapter:
                continue
            chapters[chapter].append(row)

        for chapter_name in sorted(chapters.keys()):
            chapter_rows = chapters[chapter_name]
            examples = build_examples_for_chapter(
                novel_name=novel_dir.name,
                chapter_name=chapter_name,
                rows=chapter_rows,
                window_size=window_size,
                direction=direction,
                separator=separator,
                system_prompt=system_prompt,
                stride=stride,
                include_metadata=include_metadata,
                include_indices_in_prompt=include_indices_in_prompt,
            )
            all_examples.extend(examples)

    return all_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen-style SFT JSONL from aligned_sentences.jsonl files."
    )
    parser.add_argument("--root", type=Path, required=True, help="Root folder containing novel folders.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="How many aligned sentence pairs per training example.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="jp2en",
        choices=["jp2en", "en2jp"],
        help="Translation direction.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n",
        help="Separator between pairs inside each window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Window stride. 1 makes fully overlapping windows.",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Do not include extra metadata fields in each row.",
    )
    parser.add_argument(
        "--include_indices_in_prompt",
        action="store_true",
        help="Prefix each source segment with [pair_index] in the prompt.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a careful literary translation model. "
            "Translate faithfully, preserve tone, punctuation, names, and paragraph structure."
        ),
        help="System message used for every example.",
    )
    args = parser.parse_args()

    if args.window_size < 1:
        raise ValueError("--window_size must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not args.root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {args.root}")

    examples = build_dataset(
        root_dir=args.root,
        window_size=args.window_size,
        direction=args.direction,
        separator=args.separator,
        system_prompt=args.system_prompt,
        stride=args.stride,
        include_metadata=not args.no_metadata,
        include_indices_in_prompt=args.include_indices_in_prompt,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"novel_root      : {args.root}")
    print(f"output_file     : {args.out}")
    print(f"window_size     : {args.window_size}")
    print(f"direction       : {args.direction}")
    print(f"stride          : {args.stride}")
    print(f"examples_written: {len(examples)}")


if __name__ == "__main__":
    main()