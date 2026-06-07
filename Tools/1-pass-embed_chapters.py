"""
embedding_research.py
=====================
Embeds JP/EN chapters, detects offset-aware diagonal segments,
and copies only confirmed MATCH pairs into JP-Aligned / EN-Aligned.

Accuracy-first: no absolute score threshold.
Trust the diagonal pattern, discard everything else.

Usage:
  python embedding_research.py <root_dir>

Expected structure:
  root_dir/
    JP/   1.txt, 2.txt, ...
    EN/   1.txt, 2.txt, ...

Output:
  root_dir/
    JP-Aligned/          renumbered from 1.txt
    EN-Aligned/          renumbered from 1.txt
    aligned_pairs_manifest.json
    chapter_similarity_matrix.json
"""

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL          = "BAAI/bge-m3"
FIRST_N_LINES  = 100
BATCH_SIZE     = 2

# Minimum consecutive chapters that must share the same offset to be treated
# as a real diagonal segment. Raise to be more conservative.
MIN_SEGMENT_LENGTH = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def natural_key(path: Path):
    stem = path.stem
    m = re.search(r"\d+", stem)
    return (0, int(m.group()), stem.lower()) if m else (1, stem.lower())


def load_text(path: Path, n_lines: int) -> str:
    try:
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(lines) >= n_lines:
                    break
                line = line.strip()
                if line:
                    lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        print(f"  [WARN] Failed reading {path}: {e}")
        return ""


def clear_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


# ── Offset-aware diagonal detection ───────────────────────────────────────────

@dataclass
class Segment:
    jp_start:  int      # inclusive index into jp_names
    jp_end:    int      # exclusive
    offset:    int      # en_index = jp_index + offset
    avg_score: float


def find_segments(sim_matrix: np.ndarray, argmax: list[int]) -> list[Segment]:
    """
    Group consecutive JP rows that share the same (argmax - i) offset.
    Accept groups of MIN_SEGMENT_LENGTH or more as confirmed segments.
    """
    n = len(argmax)
    offsets = [argmax[i] - i for i in range(n)]

    raw: list[tuple[int, int, int]] = []   # (start, end, offset)
    i = 0
    while i < n:
        off = offsets[i]
        start = i
        while i < n and offsets[i] == off:
            i += 1
        raw.append((start, i, off))

    confirmed: list[Segment] = []
    m = sim_matrix.shape[1]
    for start, end, off in raw:
        if end - start < MIN_SEGMENT_LENGTH:
            continue
        scores = [
            float(sim_matrix[ji, ji + off])
            for ji in range(start, end)
            if 0 <= ji + off < m
        ]
        avg = float(np.mean(scores)) if scores else 0.0
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main(root_dir_str: str):
    root   = Path(root_dir_str)
    jp_dir = root / "JP"
    en_dir = root / "EN"

    for d in (jp_dir, en_dir):
        if not d.exists():
            print(f"[ERROR] Missing folder: {d}")
            sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────

    jp_files = sorted(jp_dir.glob("*.txt"), key=natural_key)
    en_files = sorted(en_dir.glob("*.txt"), key=natural_key)

    print(f"JP chapters : {len(jp_files)}")
    print(f"EN chapters : {len(en_files)}")

    jp_names, jp_texts = [], []
    for f in jp_files:
        t = load_text(f, FIRST_N_LINES)
        if t:
            jp_names.append(f.name)
            jp_texts.append(t)
        else:
            print(f"  [WARN] Skipping empty: {f.name}")

    en_names, en_texts = [], []
    for f in en_files:
        t = load_text(f, FIRST_N_LINES)
        if t:
            en_names.append(f.name)
            en_texts.append(t)
        else:
            print(f"  [WARN] Skipping empty: {f.name}")

    if not jp_names or not en_names:
        print("[ERROR] No usable chapters found.")
        sys.exit(1)

    # ── Embed ─────────────────────────────────────────────────────────────────

    print(f"\nLoading model: {MODEL}")
    model = SentenceTransformer(MODEL, trust_remote_code=True)

    print("\nEmbedding JP chapters...")
    jp_emb = model.encode(
        jp_texts, batch_size=BATCH_SIZE,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    )

    print("\nEmbedding EN chapters...")
    en_emb = model.encode(
        en_texts, batch_size=BATCH_SIZE,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    )

    # ── Similarity matrix ─────────────────────────────────────────────────────

    print("\nComputing similarity matrix...")
    sim = np.matmul(jp_emb, en_emb.T)   # (N, M)
    n, m = sim.shape

    # Save full matrix
    matrix_path = root / "chapter_similarity_matrix.json"
    matrix_out  = {
        jp_names[i]: {en_names[j]: round(float(sim[i, j]), 6) for j in range(m)}
        for i in range(n)
    }
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix_out, f, ensure_ascii=False, indent=2)
    print(f"Saved matrix → {matrix_path}")

    # ── Diagonal analysis ─────────────────────────────────────────────────────

    print("\nRunning offset-aware diagonal analysis...")
    argmax    = [int(np.argmax(sim[i])) for i in range(n)]
    segments  = find_segments(sim, argmax)
    statuses  = assign_statuses(n, segments)

    # Build jp_index → segment lookup
    jp_to_seg: dict[int, Segment] = {}
    for seg in segments:
        for i in range(seg.jp_start, seg.jp_end):
            jp_to_seg[i] = seg

    print(f"\nDetected {len(segments)} diagonal segment(s):")
    for idx, seg in enumerate(segments):
        print(f"  Segment {idx+1}: JP[{seg.jp_start+1}..{seg.jp_end}] "
              f"→ EN offset {seg.offset:+d}  "
              f"({seg.jp_end - seg.jp_start} chapters, avg score {seg.avg_score:.4f})")

    matched   = statuses.count("MATCH")
    discarded = statuses.count("DISCARD")
    drifted   = statuses.count("DRIFT")
    print(f"\n  ✓ MATCH   : {matched}")
    print(f"  ~ DISCARD : {discarded}  (isolated single gaps)")
    print(f"  ! DRIFT   : {drifted}  (multi-chapter gaps, discarded)")

    # ── Copy confirmed pairs ───────────────────────────────────────────────────

    jp_aligned = root / "JP-Aligned"
    en_aligned = root / "EN-Aligned"
    clear_dir(jp_aligned)
    clear_dir(en_aligned)

    kept       = []
    pair_index = 1

    for i, jp_name in enumerate(jp_names):
        if statuses[i] != "MATCH":
            continue

        seg    = jp_to_seg[i]
        en_idx = i + seg.offset

        if en_idx < 0 or en_idx >= m:
            print(f"  [WARN] Segment offset puts EN index out of range for {jp_name}, skipping.")
            continue

        en_name = en_names[en_idx]
        score   = round(float(sim[i, en_idx]), 6)

        dst_name = f"{pair_index}.txt"
        shutil.copy2(jp_dir / jp_name, jp_aligned / dst_name)
        shutil.copy2(en_dir / en_name, en_aligned / dst_name)

        kept.append({
            "aligned_number": pair_index,
            "jp_source":      jp_name,
            "en_source":      en_name,
            "similarity":     score,
            "segment_offset": seg.offset,
        })
        pair_index += 1

    manifest_path = root / "aligned_pairs_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*50}")
    print(f"  Confirmed pairs copied : {len(kept)}")
    print(f"  Discarded (gaps)       : {n - len(kept)}")
    print(f"  JP-Aligned → {jp_aligned}")
    print(f"  EN-Aligned → {en_aligned}")
    print(f"  Manifest   → {manifest_path}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embedding_research.py <root_dir>")
        sys.exit(1)
    main(sys.argv[1].strip())