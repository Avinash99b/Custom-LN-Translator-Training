
#!/usr/bin/env python3
"""
Local preprocessing pipeline for Japanese -> English Light Novel translation fine-tuning.

Outputs:
  tokenized_dataset/
    full_chapter/train/
    full_chapter/validation/
    chunked/train/
    chunked/validation/
    metadata/

Design goals:
- Scan novels recursively.
- Match JP/EN chapter pairs by filename.
- Validate and clean text.
- Generate two complementary datasets:
    1) full_chapter: contiguous chapter-preserving samples
    2) chunked: overlapping chunk samples with paragraph-aware boundaries
- Tokenize locally with the Qwen tokenizer.
- Save tokenized, Arrow-backed Hugging Face datasets for fast Kaggle loading.
- Be resumable: each chapter pair is materialized as a shard file; existing shards are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset, DatasetDict, Features, Value, Sequence as HFSequence, load_dataset, load_from_disk
from transformers import AutoTokenizer

# -----------------------------
# Text normalization / cleaning
# -----------------------------

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
SPACE_RE = re.compile(r"[ \t]+")
TRAILING_WS_RE = re.compile(r"[ \t]+\n")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def normalize_text(text: str) -> str:
    """Lightweight normalization that preserves prose structure."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ZERO_WIDTH_RE.sub("", text)
    text = CTRL_RE.sub("", text)
    # Normalize full-width spaces and collapse horizontal whitespace.
    text = text.replace("\u3000", " ")
    text = SPACE_RE.sub(" ", text)
    text = TRAILING_WS_RE.sub("\n", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    """
    Split on blank lines, preserving paragraph order.
    Falls back to line-based segmentation for texts without blank paragraphs.
    """
    text = normalize_text(text)
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paras) <= 1:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) > 1:
            paras = lines
        else:
            paras = [text]
    return paras

def join_paragraphs(paras: Sequence[str]) -> str:
    return normalize_text("\n\n".join(p.strip() for p in paras if p and p.strip()))

def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)

def slugify(name: str) -> str:
    name = re.sub(r"[^\w\-\.]+", "_", name.strip(), flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    return name.strip("._") or "unknown"

# -----------------------------
# Pair scanning / validation
# -----------------------------

@dataclass(frozen=True)
class ChapterPair:
    novel: str
    chapter: str
    jp_path: str
    en_path: str
    split: str  # train / validation

def find_novel_dirs(root: Path) -> List[Path]:
    """Find directories that contain both EN-Output and JP-Output.

    Supports both:
      1) root/NovelA/{JP-Output,EN-Output}
      2) root/{JP-Output,EN-Output}
    """
    result = []

    # The root itself may be a novel folder.
    if (root / "JP-Output").is_dir() and (root / "EN-Output").is_dir():
        result.append(root)

    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if (path / "JP-Output").is_dir() and (path / "EN-Output").is_dir():
            result.append(path)

    # Deduplicate nested matches by keeping the shallowest unique paths.
    result = sorted(set(result), key=lambda p: (len(p.parts), str(p)))
    pruned = []
    for p in result:
        if not any(str(p).startswith(str(existing) + os.sep) or p == existing for existing in pruned):
            pruned.append(p)
    return pruned

def collect_pairs(root: Path, validation_novel_frac: float = 0.05) -> List[ChapterPair]:
    pairs: List[ChapterPair] = []
    novel_dirs = find_novel_dirs(root)

    if not novel_dirs:
        return []

    single_novel_mode = len(novel_dirs) == 1

    print(f"[scan] discovered {len(novel_dirs)} novel(s) single_novel_mode={single_novel_mode}")

    for novel_dir in novel_dirs:
        novel_name = novel_dir.name
        jp_dir = novel_dir / "JP-Output"
        en_dir = novel_dir / "EN-Output"

        jp_files = {p.stem: p for p in jp_dir.glob("*.txt")}
        en_files = {p.stem: p for p in en_dir.glob("*.txt")}
        common = sorted(jp_files.keys() & en_files.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

        total_chapters = len(common)

        for idx, ch in enumerate(common):
            if single_novel_mode:
                # For a single novel, split by chapter so validation is guaranteed to exist.
                if total_chapters < 20:
                    split = "validation" if idx == total_chapters - 1 else "train"
                else:
                    split = "validation" if (idx % 20) == 0 else "train"
            else:
                # For multi-novel corpora, keep the original novel-level split.
                split = "validation" if (stable_hash(novel_name) % 10_000) < int(validation_novel_frac * 10_000) else "train"

            pairs.append(
                ChapterPair(
                    novel=novel_name,
                    chapter=ch,
                    jp_path=str(jp_files[ch]),
                    en_path=str(en_files[ch]),
                    split=split,
                )
            )
    return pairs

def validate_pair(jp_text: str, en_text: str, min_chars: int = 20) -> Tuple[bool, str]:
    jp = normalize_text(jp_text)
    en = normalize_text(en_text)
    if len(jp) < min_chars or len(en) < min_chars:
        return False, "too_short"
    jp_chars = len(jp)
    en_chars = len(en)
    ratio = max(jp_chars, en_chars) / max(1, min(jp_chars, en_chars))
    if ratio > 12.0:
        return False, f"length_ratio_extreme:{ratio:.2f}"
    # Avoid pathological files that are mostly one line / one token.
    if len(split_paragraphs(jp)) == 1 and len(jp) > 50_000:
        return False, "suspicious_jp_single_block"
    if len(split_paragraphs(en)) == 1 and len(en) > 60_000:
        return False, "suspicious_en_single_block"
    return True, "ok"

# -----------------------------
# Alignment / chunking helpers
# -----------------------------

PROMPT_TEMPLATE = """Translate the following Japanese light novel passage into natural English light novel prose.

Preserve meaning, tone, dialogue, names, honorific intent, and paragraph breaks.
Output only the English translation.

Japanese:
{jp}

English:
"""

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)

def pair_paragraphs_by_ratio(jp_paras: Sequence[str], en_paras: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Approximate paragraph alignment by proportional slicing.
    Each JP paragraph gets a contiguous EN slice covering the same relative span.
    This is robust enough for chapter-level translation corpora that are already aligned by filename.
    """
    jp_paras = [p for p in jp_paras if p.strip()]
    en_paras = [p for p in en_paras if p.strip()]
    if not jp_paras:
        return []
    if not en_paras:
        return [(p, "") for p in jp_paras]

    n = len(jp_paras)
    m = len(en_paras)
    units: List[Tuple[str, str]] = []
    for i, jp in enumerate(jp_paras):
        start_ratio = i / n
        end_ratio = (i + 1) / n
        e_start = int(math.floor(start_ratio * m))
        e_end = int(math.floor(end_ratio * m))
        if i == n - 1:
            e_end = m
        e_start = max(0, min(e_start, m - 1))
        e_end = max(e_start + 1, min(e_end, m))
        en_seg = join_paragraphs(en_paras[e_start:e_end])
        units.append((normalize_text(jp), en_seg))
    return units

def build_windows_by_token_budget(
    units: Sequence[Tuple[str, str]],
    tokenizer,
    budget_tokens: int,
    overlap_ratio: float,
    min_keep_ratio: float,
) -> List[Tuple[str, str, int]]:
    """
    Build overlapping windows at paragraph boundaries while respecting a token budget.
    Returns list of (jp_chunk, en_chunk, chunk_token_len).
    """
    if not units:
        return []

    # Token length per aligned unit, measured on the full prompt+target formatting.
    unit_lens = []
    for jp_u, en_u in units:
        prompt = PROMPT_TEMPLATE.format(jp=jp_u.strip())
        ids = tokenizer(prompt + en_u.strip(), add_special_tokens=False).input_ids
        unit_lens.append(max(1, len(ids)))

    windows: List[Tuple[str, str, int]] = []
    start = 0
    total_units = len(units)

    while start < total_units:
        end = start
        total = 0
        while end < total_units and total < budget_tokens:
            total += unit_lens[end]
            end += 1

        # Tail handling: keep a final shorter chunk if it is not tiny.
        if total < int(budget_tokens * min_keep_ratio) and end < total_units:
            break

        jp_chunk = join_paragraphs([u[0] for u in units[start:end]])
        en_chunk = join_paragraphs([u[1] for u in units[start:end]])
        windows.append((jp_chunk, en_chunk, total))

        if end >= total_units:
            break

        # Convert desired overlap ratio into a token advance.
        # e.g. overlap_ratio=0.30 -> advance by 70% of current window.
        target_advance = max(1, int(total * (1.0 - overlap_ratio)))
        advanced = 0
        new_start = start
        while new_start < end and advanced < target_advance:
            advanced += unit_lens[new_start]
            new_start += 1
        if new_start <= start:
            new_start = start + 1
        start = new_start

    return windows

def tokenize_pair(tokenizer, jp_text: str, en_text: str) -> Dict[str, List[int]]:
    """
    Build a causal-LM training example:
      prompt -> Japanese source
      response -> English target
    We mask prompt tokens with -100 in labels.
    """
    prompt = PROMPT_TEMPLATE.format(jp=jp_text.strip())
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(en_text.strip(), add_special_tokens=False).input_ids

    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(eos_id)

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def segment_full_chapter(
    tokenizer,
    jp_text: str,
    en_text: str,
    max_tokens: int,
    min_keep_ratio: float = 0.60,
) -> List[Tuple[str, str, int]]:
    """
    Create full-chapter-preserving contiguous segments at paragraph boundaries.
    If a chapter exceeds max_tokens, it is split into sequential pieces without overlap.
    """
    jp_paras = split_paragraphs(jp_text)
    en_paras = split_paragraphs(en_text)
    units = pair_paragraphs_by_ratio(jp_paras, en_paras)

    if not units:
        return []

    unit_lens = []
    for jp_u, en_u in units:
        prompt = PROMPT_TEMPLATE.format(jp=jp_u.strip())
        ids = tokenizer(prompt + en_u.strip(), add_special_tokens=False).input_ids
        unit_lens.append(max(1, len(ids)))

    windows: List[Tuple[str, str, int]] = []
    start = 0
    total_units = len(units)
    while start < total_units:
        end = start
        total = 0
        while end < total_units and total < max_tokens:
            total += unit_lens[end]
            end += 1

        if total < int(max_tokens * min_keep_ratio) and end < total_units:
            # Avoid too-small fragments except for the final tail.
            break

        jp_chunk = join_paragraphs([u[0] for u in units[start:end]])
        en_chunk = join_paragraphs([u[1] for u in units[start:end]])
        windows.append((jp_chunk, en_chunk, total))

        if end >= total_units:
            break
        start = end

    return windows

# -----------------------------
# Worker setup
# -----------------------------

_WORKER_TOKENIZER = None

def worker_init(tokenizer_name: str):
    global _WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if _WORKER_TOKENIZER.pad_token is None:
        _WORKER_TOKENIZER.pad_token = _WORKER_TOKENIZER.eos_token

def process_one_pair(
    pair: ChapterPair,
    out_root: Path,
    tokenizer_name: str,
    full_max_tokens: int,
    chunk_tokens: int,
    chunk_overlap: float,
    min_keep_ratio: float,
    force_rebuild: bool = False,
) -> Dict:
    """
    Process one chapter pair into a shard file containing full and chunked samples.
    This is the resumable unit of work.
    """
    global _WORKER_TOKENIZER
    shard_dir = out_root / "staging" / pair.split / slugify(pair.novel)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{slugify(pair.chapter)}.jsonl"

    if shard_path.exists() and not force_rebuild:
        return {"shard": str(shard_path), "skipped": True}

    with open(pair.jp_path, "r", encoding="utf-8", errors="ignore") as f:
        jp_text = f.read()
    with open(pair.en_path, "r", encoding="utf-8", errors="ignore") as f:
        en_text = f.read()

    jp_text = normalize_text(jp_text)
    en_text = normalize_text(en_text)
    ok, reason = validate_pair(jp_text, en_text)
    if not ok:
        payload = {
            "status": "skipped",
            "reason": reason,
            "novel": pair.novel,
            "chapter": pair.chapter,
            "split": pair.split,
        }
        with open(shard_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {"shard": str(shard_path), "skipped": True, "reason": reason}

    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Full chapter segments: contiguous, no overlap.
    full_segments = segment_full_chapter(tokenizer, jp_text, en_text, max_tokens=full_max_tokens, min_keep_ratio=min_keep_ratio)

    # Chunked windows: shorter, overlapping, paragraph-aware.
    jp_paras = split_paragraphs(jp_text)
    en_paras = split_paragraphs(en_text)
    units = pair_paragraphs_by_ratio(jp_paras, en_paras)
    chunk_segments = build_windows_by_token_budget(
        units=units,
        tokenizer=tokenizer,
        budget_tokens=chunk_tokens,
        overlap_ratio=chunk_overlap,
        min_keep_ratio=min_keep_ratio,
    )

    records = []
    # Full chapter samples
    for idx, (jp_chunk, en_chunk, tok_len) in enumerate(full_segments):
        tok = tokenize_pair(tokenizer, jp_chunk, en_chunk)
        records.append({
            "id": f"{slugify(pair.novel)}::{pair.chapter}::full::{idx}",
            "novel": pair.novel,
            "chapter": pair.chapter,
            "split": pair.split,
            "dataset_type": "full_chapter",
            "segment_index": idx,
            "segment_count": len(full_segments),
            "chunk_size_hint": full_max_tokens,
            "chunk_overlap_ratio": 0.0,
            "source_text": jp_chunk,
            "target_text": en_chunk,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": tok["labels"],
            "source_chars": len(jp_chunk),
            "target_chars": len(en_chunk),
            "tokens": len(tok["input_ids"]),
        })

    # Chunked samples
    for idx, (jp_chunk, en_chunk, tok_len) in enumerate(chunk_segments):
        tok = tokenize_pair(tokenizer, jp_chunk, en_chunk)
        # Discard tiny partials that are unlikely to help training.
        if len(tok["input_ids"]) < int(chunk_tokens * min_keep_ratio):
            continue
        records.append({
            "id": f"{slugify(pair.novel)}::{pair.chapter}::chunk::{idx}",
            "novel": pair.novel,
            "chapter": pair.chapter,
            "split": pair.split,
            "dataset_type": "chunked",
            "segment_index": idx,
            "segment_count": len(chunk_segments),
            "chunk_size_hint": chunk_tokens,
            "chunk_overlap_ratio": chunk_overlap,
            "source_text": jp_chunk,
            "target_text": en_chunk,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": tok["labels"],
            "source_chars": len(jp_chunk),
            "target_chars": len(en_chunk),
            "tokens": len(tok["input_ids"]),
        })

    shard_payload = {
        "status": "ok",
        "novel": pair.novel,
        "chapter": pair.chapter,
        "split": pair.split,
        "records": records,
        "stats": {
            "full_segments": len(full_segments),
            "chunk_segments": len(chunk_segments),
            "full_tokens": sum(r["tokens"] for r in records if r["dataset_type"] == "full_chapter"),
            "chunk_tokens": sum(r["tokens"] for r in records if r["dataset_type"] == "chunked"),
        },
    }
    with open(shard_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(shard_payload, ensure_ascii=False) + "\n")
    return {"shard": str(shard_path), "skipped": False, "records": len(records)}

# -----------------------------
# Consolidation
# -----------------------------

def load_jsonl_payloads(paths: List[Path]) -> List[Dict]:
    payloads = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.loads(f.readline())
            if obj.get("status") == "ok":
                payloads.extend(obj.get("records", []))
        except Exception:
            continue
    return payloads

def build_dataset_from_shards(shard_paths: List[Path]) -> Dataset:
    """
    Build an Arrow-backed dataset by reading the JSONL shard files via the datasets library.
    This keeps final Kaggle loading fast while allowing resumable preprocessing.
    """
    if not shard_paths:
        raise ValueError("No shard paths found.")

    # The JSONL shard files have one JSON object per file.
    # We flatten them into an in-memory list only at consolidation time.
    records = load_jsonl_payloads(shard_paths)
    if not records:
        raise ValueError("No valid records found in shards.")

    ds = Dataset.from_list(records)
    ds = ds.sort("id")
    return ds

def summarize_dataset(ds: Dataset, name: str) -> Dict:
    total_tokens = sum(int(x) for x in ds["tokens"]) if len(ds) else 0
    longest = max((int(x) for x in ds["tokens"]), default=0)
    avg = (total_tokens / len(ds)) if len(ds) else 0.0
    novels = len(set(ds["novel"])) if len(ds) else 0
    chapters = len(set((n, c) for n, c in zip(ds["novel"], ds["chapter"]))) if len(ds) else 0
    return {
        "name": name,
        "samples": len(ds),
        "novels": novels,
        "chapters": chapters,
        "total_tokens": int(total_tokens),
        "average_tokens": float(avg),
        "longest_sample_tokens": int(longest),
    }

def save_dataset_split(ds: Dataset, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory containing novel folders, or a single novel folder.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for tokenized dataset.")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-4B", help="Tokenizer/model name.")
    parser.add_argument("--full_max_tokens", type=int, default=3072, help="Max tokens per contiguous full-chapter segment.")
    parser.add_argument("--chunk_tokens", type=int, default=3072, help="Target tokens per chunked window.")
    parser.add_argument("--chunk_overlap", type=float, default=0.30, help="Overlap ratio between chunk windows.")
    parser.add_argument("--min_keep_ratio", type=float, default=0.35, help="Discard tiny tail fragments smaller than this fraction of budget.")
    parser.add_argument("--validation_novel_frac", type=float, default=0.05, help="Novel-level validation split fraction.")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1), help="Worker processes.")
    parser.add_argument("--force_rebuild", action="store_true", help="Rebuild shard files even if they already exist.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"[scan] root={root}")
    pairs = collect_pairs(root, validation_novel_frac=args.validation_novel_frac)
    if not pairs:
        raise SystemExit("No chapter pairs found. Check the folder structure.")

    print(f"[scan] pairs={len(pairs)}")

    # Persist a manifest for resume/debugging.
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "chapter_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pairs], f, ensure_ascii=False, indent=2)

    # Parallel shard generation.
    results = []
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=worker_init,
        initargs=(args.tokenizer,),
    ) as pool:
        futures = [
            pool.submit(
                process_one_pair,
                pair,
                out_dir,
                args.tokenizer,
                args.full_max_tokens,
                args.chunk_tokens,
                args.chunk_overlap,
                args.min_keep_ratio,
                args.force_rebuild,
            )
            for pair in pairs
        ]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if len(results) % 20 == 0 or len(results) == len(futures):
                done = sum(1 for r in results if not r.get("skipped", False))
                skipped = sum(1 for r in results if r.get("skipped", False))
                print(f"[progress] done={done}/{len(pairs)} skipped={skipped}")

    # Collect shard files by split.
    shard_root = out_dir / "staging"
    train_shards = sorted(shard_root.glob("train/**/*.jsonl"))
    val_shards = sorted(shard_root.glob("validation/**/*.jsonl"))

    print(f"[merge] train_shards={len(train_shards)} validation_shards={len(val_shards)}")

    # Build datasets from shards and filter by dataset_type.
    train_ds = build_dataset_from_shards(train_shards)

    if val_shards:
        val_ds = build_dataset_from_shards(val_shards)
    else:
        print("[warning] No validation shards found; creating an empty validation split.")
        val_ds = train_ds.select([])

    full_train = train_ds.filter(lambda x: x["dataset_type"] == "full_chapter")
    chunk_train = train_ds.filter(lambda x: x["dataset_type"] == "chunked")

    if len(val_ds):
        full_val = val_ds.filter(lambda x: x["dataset_type"] == "full_chapter")
        chunk_val = val_ds.filter(lambda x: x["dataset_type"] == "chunked")
    else:
        full_val = val_ds
        chunk_val = val_ds

    # Save Arrow-backed datasets for fast Kaggle loading.
    save_dataset_split(full_train, out_dir / "full_chapter" / "train")
    save_dataset_split(full_val, out_dir / "full_chapter" / "validation")
    save_dataset_split(chunk_train, out_dir / "chunked" / "train")
    save_dataset_split(chunk_val, out_dir / "chunked" / "validation")

    # Statistics
    stats = {
        "full_chapter_train": summarize_dataset(full_train, "full_chapter_train"),
        "full_chapter_validation": summarize_dataset(full_val, "full_chapter_validation"),
        "chunked_train": summarize_dataset(chunk_train, "chunked_train"),
        "chunked_validation": summarize_dataset(chunk_val, "chunked_validation"),
    }

    full_tokens = stats["full_chapter_train"]["total_tokens"] + stats["full_chapter_validation"]["total_tokens"]
    chunk_tokens = stats["chunked_train"]["total_tokens"] + stats["chunked_validation"]["total_tokens"]

    stats["corpus"] = {
        "novels": len({p.novel for p in pairs}),
        "chapters": len(pairs),
        "total_full_tokens": int(full_tokens),
        "total_chunk_tokens": int(chunk_tokens),
        "dataset_expansion_factor_from_chunking": (chunk_tokens / max(1, full_tokens)) if full_tokens else 0.0,
        "estimated_training_tokens_mixture_50_50": int((full_tokens + chunk_tokens) / 2),
        "estimated_training_tokens_all": int(full_tokens + chunk_tokens),
        "average_chapter_tokens": float(full_tokens / max(1, stats["full_chapter_train"]["samples"] + stats["full_chapter_validation"]["samples"])),
        "average_chunk_tokens": float(chunk_tokens / max(1, stats["chunked_train"]["samples"] + stats["chunked_validation"]["samples"])),
        "longest_full_sample_tokens": max(stats["full_chapter_train"]["longest_sample_tokens"], stats["full_chapter_validation"]["longest_sample_tokens"]),
        "longest_chunk_tokens": max(stats["chunked_train"]["longest_sample_tokens"], stats["chunked_validation"]["longest_sample_tokens"]),
    }

    with open(metadata_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] datasets saved to: {out_dir}")

if __name__ == "__main__":
    main()
