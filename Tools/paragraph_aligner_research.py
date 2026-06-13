"""
paragraph_aligner.py
====================
Aligns JP/EN sentences within pre-aligned chapter pairs and outputs a JSONL
dataset. Mirrors the offset-aware diagonal approach from embedding_research.py
but operates at sentence level within each chapter.

Key design decisions
--------------------
* Accuracy-first: trusts diagonal segments, discards inter-segment gaps.
* JP sentences split on Japanese punctuation (。！？); EN on standard sentence
  boundaries.
* Supports 1-to-1, 1-to-2, and 2-to-1 pairings via a merge-candidate pass
  after the primary diagonal alignment.
* BGE-M3 embeddings (same model as the chapter aligner) for cross-lingual
  sentence similarity.
* Incremental writes: each chapter's pairs are flushed to the JSONL immediately
  after processing — a crash mid-run loses at most one chapter's work.

Usage
-----
    python paragraph_aligner.py novel_root [--output pairs.jsonl- default: aligned_sentences.jsonl]

Expected structure
------------------
    jp_dir/   1.txt, 2.txt, ...  (JP chapters, aligned)
    en_dir/   1.txt, 2.txt, ...  (EN chapters, matching filenames)

Output JSONL schema (one JSON object per line)
----------------------------------------------
    {
      "chapter":    "3.txt",
      "pair_index": 7,
      "jp":         "彼女は微笑んだ。",
      "en":         "She smiled.",
      "similarity": 0.8421,
      "jp_indices": [4],          # source sentence indices (0-based)
      "en_indices": [6],
      "pairing":    "1:1"         # "1:1" | "1:2" | "2:1"
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL              = "BAAI/bge-m3"
BATCH_SIZE         = 16          # increase if VRAM allows
MIN_SEGMENT_LEN    = 2           # min consecutive sentences sharing same offset
MERGE_SIM_BOOST    = 0.04        # merge must beat base sim by this delta
MIN_SENTENCE_CHARS = 4           # discard fragments shorter than this


# ── Sentence splitting ─────────────────────────────────────────────────────────

_JP_SPLIT = re.compile(r"(?<=[。！？\!?])\s*")
_EN_SPLIT  = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def split_jp(text: str) -> list[str]:
    raw = _JP_SPLIT.split(text)
    return [s.strip() for s in raw if len(s.strip()) >= MIN_SENTENCE_CHARS]


def split_en(text: str) -> list[str]:
    text = re.sub(r"\n{2,}", "\n\n", text)
    sentences: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for p in _EN_SPLIT.split(para):
            p = p.strip()
            if len(p) >= MIN_SENTENCE_CHARS:
                sentences.append(p)
    return sentences


# ── Diagonal alignment ────────────────────────────────────────────────────────

@dataclass
class Segment:
    jp_start:  int
    jp_end:    int      # exclusive
    offset:    int      # en_idx = jp_idx + offset
    avg_score: float


def find_segments(sim: np.ndarray, argmax: list[int]) -> list[Segment]:
    n       = len(argmax)
    m       = sim.shape[1]
    offsets = [argmax[i] - i for i in range(n)]

    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        off, start = offsets[i], i
        while i < n and offsets[i] == off:
            i += 1
        runs.append((start, i, off))

    confirmed: list[Segment] = []
    for start, end, off in runs:
        if end - start < MIN_SEGMENT_LEN:
            continue
        scores = [float(sim[ji, ji + off]) for ji in range(start, end) if 0 <= ji + off < m]
        avg    = float(np.mean(scores)) if scores else 0.0
        confirmed.append(Segment(start, end, off, avg))
    return confirmed


def assign_statuses(n: int, segments: list[Segment]) -> list[str]:
    status = ["UNASSIGNED"] * n
    for seg in segments:
        for i in range(seg.jp_start, seg.jp_end):
            status[i] = "MATCH"

    i = 0
    while i < n:
        if status[i] == "UNASSIGNED":
            gap_start = i
            while i < n and status[i] == "UNASSIGNED":
                i += 1
            label = "DISCARD" if (i - gap_start) == 1 else "DRIFT"
            for j in range(gap_start, i):
                status[j] = label
        else:
            i += 1
    return status


# ── Merge-candidate pass ──────────────────────────────────────────────────────

def try_merge(
    sim: np.ndarray, jp_idx: int, en_idx: int, base_sim: float,
) -> tuple[list[int], list[int], float, str] | None:
    n_jp, n_en = sim.shape
    best_sim, best_result = base_sim, None

    # 1:2 — one JP, two EN
    if en_idx + 1 < n_en:
        combined = (base_sim + float(sim[jp_idx, en_idx + 1])) / 2
        if combined >= best_sim + MERGE_SIM_BOOST:
            best_sim, best_result = combined, ([jp_idx], [en_idx, en_idx + 1], combined, "1:2")

    # 2:1 — two JP, one EN
    if jp_idx + 1 < n_jp:
        combined = (base_sim + float(sim[jp_idx + 1, en_idx])) / 2
        if combined >= best_sim + MERGE_SIM_BOOST:
            best_sim, best_result = combined, ([jp_idx, jp_idx + 1], [en_idx], combined, "2:1")

    return best_result


# ── Per-chapter alignment ─────────────────────────────────────────────────────

@dataclass
class AlignedPair:
    chapter:    str
    pair_index: int
    jp:         str
    en:         str
    similarity: float
    jp_indices: list[int]
    en_indices: list[int]
    pairing:    str


def align_chapter(
    chapter_name: str,
    jp_text: str,
    en_text: str,
    model: SentenceTransformer,
) -> list[AlignedPair]:

    jp_sents = split_jp(jp_text)
    en_sents = split_en(en_text)

    if not jp_sents or not en_sents:
        print(f"    [WARN] {chapter_name}: empty after splitting "
              f"(JP={len(jp_sents)}, EN={len(en_sents)}), skipping.")
        return []

    jp_emb = model.encode(jp_sents, batch_size=BATCH_SIZE,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
    en_emb = model.encode(en_sents, batch_size=BATCH_SIZE,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)

    sim    = np.matmul(jp_emb, en_emb.T)
    n_jp   = sim.shape[0]

    argmax   = [int(np.argmax(sim[i])) for i in range(n_jp)]
    segments = find_segments(sim, argmax)
    statuses = assign_statuses(n_jp, segments)

    jp_to_seg: dict[int, Segment] = {}
    for seg in segments:
        for i in range(seg.jp_start, seg.jp_end):
            jp_to_seg[i] = seg

    pairs: list[AlignedPair] = []
    pair_idx = 0
    used_en: set[int] = set()

    i = 0
    while i < n_jp:
        if statuses[i] != "MATCH":
            i += 1
            continue

        seg    = jp_to_seg[i]
        en_idx = i + seg.offset

        if en_idx < 0 or en_idx >= sim.shape[1] or en_idx in used_en:
            i += 1
            continue

        base_sim_val = float(sim[i, en_idx])
        merge        = try_merge(sim, i, en_idx, base_sim_val)

        if merge and any(e in used_en for e in merge[1]):
            merge = None

        if merge:
            jp_idxs, en_idxs, merged_sim, label = merge
            used_en.update(en_idxs)
            pairs.append(AlignedPair(
                chapter    = chapter_name,
                pair_index = pair_idx,
                jp         = " ".join(jp_sents[j] for j in jp_idxs),
                en         = " ".join(en_sents[e] for e in en_idxs),
                similarity = round(merged_sim, 6),
                jp_indices = jp_idxs,
                en_indices = en_idxs,
                pairing    = label,
            ))
            pair_idx += 1
            i += len(jp_idxs)
        else:
            used_en.add(en_idx)
            pairs.append(AlignedPair(
                chapter    = chapter_name,
                pair_index = pair_idx,
                jp         = jp_sents[i],
                en         = en_sents[en_idx],
                similarity = round(base_sim_val, 6),
                jp_indices = [i],
                en_indices = [en_idx],
                pairing    = "1:1",
            ))
            pair_idx += 1
            i += 1

    return pairs


# ── Flush helpers ─────────────────────────────────────────────────────────────

def flush_pairs(pairs: list[AlignedPair], fh) -> None:
    """Write pairs to an already-open file handle, one JSON line each."""
    for p in pairs:
        record = {
            "chapter":    p.chapter,
            "pair_index": p.pair_index,
            "jp":         p.jp,
            "en":         p.en,
            "similarity": p.similarity,
            "jp_indices": p.jp_indices,
            "en_indices": p.en_indices,
            "pairing":    p.pairing,
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def write_stats(stats_path: Path, chapter_stats: list[dict], all_pairs: list[AlignedPair]) -> None:
    total_11    = sum(c["pairing_counts"]["1:1"] for c in chapter_stats)
    total_12    = sum(c["pairing_counts"]["1:2"] for c in chapter_stats)
    total_21    = sum(c["pairing_counts"]["2:1"] for c in chapter_stats)
    overall_sim = float(np.mean([p.similarity for p in all_pairs])) if all_pairs else 0.0
    stats = {
        "total_pairs":        len(all_pairs),
        "overall_avg_sim":    round(overall_sim, 4),
        "pairing_breakdown":  {"1:1": total_11, "1:2": total_12, "2:1": total_21},
        "chapters":           chapter_stats,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ── File helpers ──────────────────────────────────────────────────────────────

def natural_key(p: Path) -> tuple:
    m = re.search(r"\d+", p.stem)
    return (0, int(m.group()), p.stem.lower()) if m else (1, 0, p.stem.lower())


def load_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main(jp_dir: Path, en_dir: Path, output_path: Path) -> None:
    jp_files = {f.name: f for f in sorted(jp_dir.glob("*.txt"), key=natural_key)}
    en_files = {f.name: f for f in sorted(en_dir.glob("*.txt"), key=natural_key)}

    common  = sorted(jp_files.keys() & en_files.keys(), key=lambda n: natural_key(Path(n)))
    jp_only = jp_files.keys() - en_files.keys()
    en_only = en_files.keys() - jp_files.keys()

    if jp_only:
        print(f"[WARN] {len(jp_only)} JP files with no EN match: {sorted(jp_only)[:5]}")
    if en_only:
        print(f"[WARN] {len(en_only)} EN files with no JP match: {sorted(en_only)[:5]}")
    if not common:
        print("[ERROR] No matching filenames between JP and EN directories.")
        sys.exit(1)

    print(f"Found {len(common)} paired chapter(s) to align.\n")
    print(f"Loading model: {MODEL}")
    model = SentenceTransformer(MODEL, trust_remote_code=True)
    print()

    stats_path    = output_path.with_suffix(".stats.json")
    all_pairs:     list[AlignedPair] = []
    chapter_stats: list[dict]        = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Open JSONL once; flush per chapter ───────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as jsonl_fh:

        for ch_num, name in enumerate(common, 1):
            t0      = time.perf_counter()
            jp_text = load_file(jp_files[name])
            en_text = load_file(en_files[name])

            if not jp_text or not en_text:
                print(f"  [{ch_num:>3}/{len(common)}] SKIP  {name}  (empty file)")
                continue

            pairs = align_chapter(name, jp_text, en_text, model)
            elapsed = time.perf_counter() - t0

            # ── Incremental write ─────────────────────────────────────────────
            flush_pairs(pairs, jsonl_fh)

            # ── Accumulate stats ──────────────────────────────────────────────
            pairing_counts = {"1:1": 0, "1:2": 0, "2:1": 0}
            for p in pairs:
                pairing_counts[p.pairing] = pairing_counts.get(p.pairing, 0) + 1
            avg_sim = float(np.mean([p.similarity for p in pairs])) if pairs else 0.0

            chapter_stats.append({
                "chapter":        name,
                "pairs":          len(pairs),
                "pairing_counts": pairing_counts,
                "avg_similarity": round(avg_sim, 4),
            })
            all_pairs.extend(pairs)

            # ── Per-chapter stats line ────────────────────────────────────────
            running_total = len(all_pairs)
            print(
                f"  [{ch_num:>3}/{len(common)}] {name:<20} "
                f"pairs={len(pairs):>4}  "
                f"1:1={pairing_counts['1:1']:>3} "
                f"1:2={pairing_counts['1:2']:>3} "
                f"2:1={pairing_counts['2:1']:>3}  "
                f"avg_sim={avg_sim:.4f}  "
                f"({elapsed:.1f}s)  "
                f"[total so far: {running_total}]"
            )

            # ── Incrementally overwrite stats sidecar ─────────────────────────
            write_stats(stats_path, chapter_stats, all_pairs)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_11    = sum(c["pairing_counts"]["1:1"] for c in chapter_stats)
    total_12    = sum(c["pairing_counts"]["1:2"] for c in chapter_stats)
    total_21    = sum(c["pairing_counts"]["2:1"] for c in chapter_stats)
    overall_sim = float(np.mean([p.similarity for p in all_pairs])) if all_pairs else 0.0

    print(f"\n{'═'*60}")
    print(f"  Chapters processed   : {len(chapter_stats)}")
    print(f"  Total aligned pairs  : {len(all_pairs)}")
    print(f"  Overall avg sim      : {overall_sim:.4f}")
    print(f"  Pairing breakdown    : 1:1={total_11}  1:2={total_12}  2:1={total_21}")
    print(f"  Output JSONL         : {output_path}")
    print(f"  Stats sidecar        : {stats_path}")
    print(f"{'═'*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align JP/EN sentences within pre-aligned chapter pairs."
    )
    parser.add_argument("novel_root",  type=Path, help="Root directory of the novel with JP and EN chapters")
    args = parser.parse_args()
    jp_dir = args.novel_root / "JP-Output"
    en_dir = args.novel_root / "EN-Output"

    for d, label in [[args.novel_root, "JP-Output"], [args.novel_root, "EN-Output"]]:
        if not d.is_dir():
            print(f"[ERROR] Directory not found: {d}")
            sys.exit(1)

    main(jp_dir, en_dir, Path(args.novel_root) / "aligned_sentences.jsonl")