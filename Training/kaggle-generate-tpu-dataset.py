#!/usr/bin/env python3
"""
dataset_preprocessor_hf.py

Local JP→EN light novel preprocessing for Hugging Face / Qwen chat training.

What it does:
- Reads aligned chapter files from:
    input_dir/<Novel>/JP-Output/*.txt
    input_dir/<Novel>/EN-Output/*.txt
- Cleans and optionally chunks chapters locally
- Builds `messages` examples
- Renders Qwen-style chat text locally using a Hugging Face tokenizer
- Optionally exports tokenized fields too

Outputs:
- train.jsonl
- val.jsonl
- test.jsonl   (optional)
- stats.json

Optional per-example fields:
- text
- input_ids
- attention_mask
"""

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English light novel translator. "
    "Your translations are accurate, fluent, and faithful to the original tone. "
    "You preserve character names in their romanised form, maintain the author's "
    "stylistic voice, and handle honorifics naturally within the English prose. "
    "Translate the following Japanese passage into English. "
    "Output only the translated English text."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
NL = "\n"


@dataclass
class ChunkConfig:
    enabled: bool = False
    chunk_size: int = 400
    overlap: int = 80
    min_chunk_chars: int = 60
    split_on_paragraphs: bool = True


@dataclass
class HFConfig:
    tokenizer_id_or_path: str = "Qwen/Qwen3-4B"
    cache_dir: Optional[Path] = None
    local_files_only: bool = True
    max_length: int = 4096
    save_text: bool = True
    save_tokenized: bool = False
    add_eos: bool = True


@dataclass
class PreprocessConfig:
    input_dir: Path = Path("./novels")
    output_dir: Path = Path("./dataset")
    split_ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    hf: HFConfig = field(default_factory=HFConfig)
    seed: int = 42
    jp_folder: str = "JP-Output"
    en_folder: str = "EN-Output"
    encoding: str = "utf-8"
    system_prompt: str = SYSTEM_PROMPT


def clean_text(text: str) -> str:
    lines = [l.rstrip() for l in text.splitlines()]
    cleaned: List[str] = []
    blanks = 0
    for line in lines:
        if line.strip() == "":
            blanks += 1
            if blanks <= 2:
                cleaned.append("")
        else:
            blanks = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_by_chars(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chars: int,
    split_on_paragraphs: bool,
) -> List[str]:
    if not text:
        return []

    if split_on_paragraphs:
        paragraphs = split_into_paragraphs(text)
        if not paragraphs:
            return [text] if len(text) >= min_chars else []

        chunks: List[str] = []
        current_paras: List[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_paras and current_len + para_len > chunk_size:
                chunk_text = "\n\n".join(current_paras)
                if len(chunk_text) >= min_chars:
                    chunks.append(chunk_text)

                overlap_paras: List[str] = []
                overlap_len = 0
                for p in reversed(current_paras):
                    if overlap_len + len(p) <= overlap:
                        overlap_paras.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break

                current_paras = overlap_paras + [para]
                current_len = sum(len(p) for p in current_paras)
            else:
                current_paras.append(para)
                current_len += para_len

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            if len(chunk_text) >= min_chars:
                chunks.append(chunk_text)

        return chunks

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def align_chunks(jp_chunks: List[str], en_chunks: List[str]) -> List[Tuple[str, str]]:
    if len(jp_chunks) != len(en_chunks):
        log.warning(
            "JP chunks (%d) != EN chunks (%d); truncating to shorter.",
            len(jp_chunks),
            len(en_chunks),
        )
    n = min(len(jp_chunks), len(en_chunks))
    return list(zip(jp_chunks[:n], en_chunks[:n]))


def load_tokenizer(hf: HFConfig):
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": hf.local_files_only,
    }
    if hf.cache_dir is not None:
        kwargs["cache_dir"] = str(hf.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(hf.tokenizer_id_or_path, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    return tokenizer


def render_chat_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    text = ""
    for msg in messages:
        text += f"{IM_START}{msg['role']}{NL}{msg['content']}{IM_END}{NL}"
    return text


def tokenize_text(tokenizer, text: str, max_length: int, add_eos: bool) -> Dict[str, Any]:
    if add_eos and tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=False,
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


def load_novel(
    novel_dir: Path,
    jp_folder: str,
    en_folder: str,
    encoding: str,
) -> List[Tuple[int, str, str]]:
    jp_dir = novel_dir / jp_folder
    en_dir = novel_dir / en_folder

    if not jp_dir.is_dir():
        log.warning("JP folder missing: %s — skipping novel.", jp_dir)
        return []
    if not en_dir.is_dir():
        log.warning("EN folder missing: %s — skipping novel.", en_dir)
        return []

    def chapter_files(directory: Path) -> Dict[int, Path]:
        result = {}
        for fp in directory.iterdir():
            if fp.suffix.lower() == ".txt":
                m = re.match(r"^(\d+)", fp.stem)
                if m:
                    result[int(m.group(1))] = fp
        return result

    jp_files = chapter_files(jp_dir)
    en_files = chapter_files(en_dir)
    common = sorted(set(jp_files) & set(en_files))

    if not common:
        log.warning("No matched chapters in %s", novel_dir.name)
        return []

    chapters = []
    for ch in common:
        try:
            jp_text = clean_text(jp_files[ch].read_text(encoding=encoding))
            en_text = clean_text(en_files[ch].read_text(encoding=encoding))
            if jp_text and en_text:
                chapters.append((ch, jp_text, en_text))
            else:
                log.warning("Chapter %d in %s is empty — skipping.", ch, novel_dir.name)
        except Exception as exc:
            log.error("Failed reading chapter %d in %s: %s", ch, novel_dir.name, exc)
    return chapters


def build_examples(
    novel_name: str,
    chapter: int,
    jp_text: str,
    en_text: str,
    cfg: PreprocessConfig,
    tokenizer=None,
) -> List[dict]:
    if cfg.chunk.enabled:
        jp_chunks = chunk_text_by_chars(
            jp_text,
            cfg.chunk.chunk_size,
            cfg.chunk.overlap,
            cfg.chunk.min_chunk_chars,
            cfg.chunk.split_on_paragraphs,
        )
        en_chunks = chunk_text_by_chars(
            en_text,
            int(cfg.chunk.chunk_size * 1.5),
            int(cfg.chunk.overlap * 1.5),
            cfg.chunk.min_chunk_chars,
            cfg.chunk.split_on_paragraphs,
        )
        pairs = align_chunks(jp_chunks, en_chunks)
        if not pairs:
            log.warning(
                "No valid chunks for %s ch%d — falling back to full chapter.",
                novel_name,
                chapter,
            )
            pairs = [(jp_text, en_text)]
    else:
        pairs = [(jp_text, en_text)]

    total = len(pairs)
    examples = []

    for idx, (jp, en) in enumerate(pairs):
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": jp.strip()},
            {"role": "assistant", "content": en.strip()},
        ]

        ex = {
            "messages": messages,
            "novel": novel_name,
            "chapter": chapter,
            "chunk_idx": idx,
            "total_chunks": total,
        }

        if tokenizer is not None and cfg.hf.save_text:
            text = render_chat_text(tokenizer, messages)
            ex["text"] = text

            if cfg.hf.save_tokenized:
                ex.update(tokenize_text(tokenizer, text, cfg.hf.max_length, cfg.hf.add_eos))

        examples.append(ex)

    return examples


def preprocess(cfg: PreprocessConfig) -> None:
    random.seed(cfg.seed)

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    tokenizer = load_tokenizer(cfg.hf)

    all_examples: List[dict] = []
    novel_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not novel_dirs:
        log.error("No subdirectories found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d novel folder(s) in %s", len(novel_dirs), input_dir)
    log.info("Tokenizer: %s", cfg.hf.tokenizer_id_or_path)

    for novel_dir in novel_dirs:
        novel_name = novel_dir.name
        chapters = load_novel(novel_dir, cfg.jp_folder, cfg.en_folder, cfg.encoding)
        if not chapters:
            continue

        novel_examples = []
        for chapter, jp_text, en_text in chapters:
            exs = build_examples(novel_name, chapter, jp_text, en_text, cfg, tokenizer=tokenizer)
            novel_examples.extend(exs)

        log.info(
            "  %-30s  chapters=%d  examples=%d",
            novel_name,
            len(chapters),
            len(novel_examples),
        )
        all_examples.extend(novel_examples)

    if not all_examples:
        log.error("No examples were generated. Check your folder structure.")
        sys.exit(1)

    log.info("Total examples before split: %d", len(all_examples))

    random.shuffle(all_examples)

    train_r, val_r, test_r = cfg.split_ratios
    n = len(all_examples)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_set = all_examples[:n_train]
    val_set = all_examples[n_train : n_train + n_val]
    test_set = all_examples[n_train + n_val :]

    def write_jsonl(examples: List[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("Wrote %d examples → %s", len(examples), path)

    write_jsonl(train_set, output_dir / "train.jsonl")
    write_jsonl(val_set, output_dir / "val.jsonl")
    if test_set:
        write_jsonl(test_set, output_dir / "test.jsonl")

    stats = {
        "total_examples": n,
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "chunking_enabled": cfg.chunk.enabled,
        "chunk_config": asdict(cfg.chunk) if cfg.chunk.enabled else None,
        "novels": len(novel_dirs),
        "seed": cfg.seed,
        "tokenizer_id_or_path": cfg.hf.tokenizer_id_or_path,
        "saved_text": cfg.hf.save_text,
        "saved_tokenized": cfg.hf.save_tokenized,
        "max_length": cfg.hf.max_length if cfg.hf.save_tokenized else None,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Done. Train=%d  Val=%d  Test=%d", len(train_set), len(val_set), len(test_set))


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(
        description="Preprocess JP→EN light novel data for Hugging Face fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", default="./novels")
    parser.add_argument("--output_dir", default="./dataset")
    parser.add_argument("--jp_folder", default="JP-Output")
    parser.add_argument("--en_folder", default="EN-Output")
    parser.add_argument("--split", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"), default=[0.90, 0.05, 0.05])
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tokenizer_id_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--online_tokenizer", action="store_true", help="Allow downloading tokenizer files if needed.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--no_save_text", action="store_true")
    parser.add_argument("--save_tokenized", action="store_true")
    parser.add_argument("--no_eos", action="store_true")

    chunk_group = parser.add_argument_group("Overlap-based Chunking")
    chunk_group.add_argument("--enable_chunking", action="store_true")
    chunk_group.add_argument("--chunk_size", type=int, default=400)
    chunk_group.add_argument("--overlap", type=int, default=80)
    chunk_group.add_argument("--min_chunk_chars", type=int, default=60)
    chunk_group.add_argument("--no_paragraph_split", action="store_true")

    args = parser.parse_args()

    total = sum(args.split)
    if abs(total - 1.0) > 1e-6:
        parser.error(f"--split ratios must sum to 1.0 (got {total:.4f})")

    hf_cfg = HFConfig(
        tokenizer_id_or_path=args.tokenizer_id_or_path,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        local_files_only=not args.online_tokenizer,
        max_length=args.max_length,
        save_text=not args.no_save_text,
        save_tokenized=args.save_tokenized,
        add_eos=not args.no_eos,
    )

    return PreprocessConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        split_ratios=tuple(args.split),
        jp_folder=args.jp_folder,
        en_folder=args.en_folder,
        encoding=args.encoding,
        seed=args.seed,
        hf=hf_cfg,
        chunk=ChunkConfig(
            enabled=args.enable_chunking,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_chars=args.min_chunk_chars,
            split_on_paragraphs=not args.no_paragraph_split,
        ),
    )


if __name__ == "__main__":
    cfg = parse_args()

    log.info("=" * 60)
    log.info("Light Novel JP→EN Dataset Preprocessor (Hugging Face)")
    log.info("=" * 60)
    log.info("Input:    %s", cfg.input_dir)
    log.info("Output:   %s", cfg.output_dir)
    log.info(
        "Split:    train=%.0f%%  val=%.0f%%  test=%.0f%%",
        cfg.split_ratios[0] * 100,
        cfg.split_ratios[1] * 100,
        cfg.split_ratios[2] * 100,
    )
    log.info("Chunking: %s", "ENABLED" if cfg.chunk.enabled else "DISABLED")
    log.info("Tokenizer: %s", cfg.hf.tokenizer_id_or_path)
    log.info("Offline:   %s", cfg.hf.local_files_only)
    log.info("Save text: %s", cfg.hf.save_text)
    log.info("Tokenized: %s", cfg.hf.save_tokenized)
    log.info("=" * 60)

    preprocess(cfg)