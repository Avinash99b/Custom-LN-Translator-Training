"""
preprocessing_metrics_sentence_pairs.py
────────────────────────────────────────
Analyses a training.jsonl file of JP→EN sentence-pair fine-tuning data,
reports health metrics, and optionally:
  • removes genuinely bad examples (OOM-risk, empty, think-contaminated)
  • caps over-represented novels to --cap-novel N samples
  • writes a per-example sample-weight file for weighted DataLoader training
"""

import json
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Dataset health check + optional cleaning")
    p.add_argument("--jsonl",        default="training.jsonl",  help="Path to training JSONL")
    p.add_argument("--model",        default="Qwen/Qwen3-4B",   help="HF model id for tokenizer")
    p.add_argument("--max-len",      type=int, default=1024,      help="Hard token ceiling (OOM guard)")
    p.add_argument("--min-len",      type=int, default=20,       help="Minimum total tokens")
    p.add_argument("--min-assistant",type=int, default=5,        help="Minimum assistant tokens")
    p.add_argument("--batch-size",   type=int, default=20,        help="Batch size for padding-waste estimate")
    p.add_argument("--tokenize-bs",  type=int, default=64,       help="Tokenizer batch size (RAM tuning)")
    p.add_argument("--cap-novel",    type=int, default=None,
                   help="Cap each novel to at most N examples (removes excess from tail of file)")
    p.add_argument("--weights-out",  default=None,
                   help="If set, write per-example inverse-frequency weights to this .npy file")
    p.add_argument("--yes",          action="store_true",
                   help="Auto-confirm all destructive operations (no prompts)")
    return p.parse_args()

# ── Load tokenizer ────────────────────────────────────────────────────────────
def load_tokenizer(model_id: str):
    print(f"Loading tokenizer from '{model_id}'...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tok

# ── Load JSONL ────────────────────────────────────────────────────────────────
def load_records(path: str):
    print(f"Loading data from '{path}'...")
    records = []
    malformed = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i} is malformed JSON: {e}")
                malformed += 1
    print(f"Loaded {len(records)} records  ({malformed} malformed lines skipped)\n")
    return records

# ── Template rendering ────────────────────────────────────────────────────────
def render_text(tokenizer, rec: dict) -> str:
    return tokenizer.apply_chat_template(
        rec["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

def has_think_content(rec: dict) -> bool:
    content = next(
        (m["content"] for m in rec["messages"] if m["role"] == "assistant"), ""
    )
    return bool(re.search(r"<think>\s*\S", content))

# ── Batched tokenisation ──────────────────────────────────────────────────────
def batch_token_lengths(tokenizer, text_list, add_special_tokens: bool, batch_size: int):
    out = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
            truncation=False,
            padding=False,
            return_length=True,
        )
        out.extend(enc["length"])
    return out

# ── Statistics printer ────────────────────────────────────────────────────────
def print_stats(arr: np.ndarray, label: str):
    print(f"\n── {label} ──")
    print(f"  count   : {len(arr)}")
    print(f"  mean    : {np.mean(arr):.2f}")
    print(f"  median  : {np.median(arr):.2f}")
    print(f"  std     : {np.std(arr):.2f}")
    print(f"  min     : {np.min(arr)}")
    print(f"  max     : {np.max(arr)}")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  p{p:<5} : {np.percentile(arr, p):.1f}")

# ── Section header ────────────────────────────────────────────────────────────
def header(n: int, title: str):
    print(f"\n{'═'*60}")
    print(f"{n}. {title}")
    print("═"*60)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    tokenizer = load_tokenizer(args.model)
    records   = load_records(args.jsonl)

    if not records:
        print("No records found. Exiting.")
        sys.exit(1)

    # ── Extract fields and render templates ──
    print("Rendering chat templates...")
    line_indices  = []
    texts         = []
    novels        = []
    chapters      = []
    directions    = []
    think_flags   = []
    assistant_texts = []
    user_texts      = []

    for line_idx, rec in records:
        line_indices.append(line_idx)
        texts.append(render_text(tokenizer, rec))
        novels.append(rec.get("novel", "unknown"))
        chapters.append(rec.get("chapter", "unknown"))
        directions.append(rec.get("direction", "unknown"))
        think_flags.append(has_think_content(rec))
        for msg in rec["messages"]:
            if msg["role"] == "assistant":
                assistant_texts.append(msg["content"])
            elif msg["role"] == "user":
                user_texts.append(msg["content"])

    # ── Tokenise ──
    print("Tokenizing in batches...")
    bs = args.tokenize_bs
    token_lengths     = np.array(batch_token_lengths(tokenizer, texts,          True,  bs))
    assistant_lengths = np.array(batch_token_lengths(tokenizer, assistant_texts, False, bs))
    user_lengths      = np.array(batch_token_lengths(tokenizer, user_texts,      False, bs))

    N = len(records)

    # ════════════════════════════════════════════════════════════
    # 1. Token length distribution
    # ════════════════════════════════════════════════════════════
    header(1, "TOKEN LENGTH DISTRIBUTION (full prompt, add_special_tokens=True)")
    print_stats(token_lengths,     "Total tokens per example")
    print_stats(assistant_lengths, "Assistant (target) tokens per example")
    print_stats(user_lengths,      "User (source) tokens per example")

    over_max = int((token_lengths > args.max_len).sum())
    print(f"\n  Examples over MAX_LEN ({args.max_len}): {over_max}  ({100*over_max/N:.2f}%)")

    # ════════════════════════════════════════════════════════════
    # 2. Dataset composition
    # ════════════════════════════════════════════════════════════
    header(2, "DATASET COMPOSITION")
    novel_counts = Counter(novels)
    print(f"\n  Novels ({len(novel_counts)}):")
    for nov, cnt in novel_counts.most_common():
        pct = 100 * cnt / N
        bar = "█" * int(pct / 2)
        print(f"    {nov:<42} {cnt:>7}  ({pct:5.1f}%)  {bar}")

    direction_counts = Counter(directions)
    print(f"\n  Directions: {dict(direction_counts)}")
    print(f"  Unique chapters: {len(set(chapters))}")

    # ════════════════════════════════════════════════════════════
    # 3. Class imbalance
    # ════════════════════════════════════════════════════════════
    header(3, "CLASS IMBALANCE")
    max_cnt   = novel_counts.most_common(1)[0][1]
    min_cnt   = novel_counts.most_common()[-1][1]
    ratio     = max_cnt / max(min_cnt, 1)
    flag      = "  !! Severe imbalance" if ratio > 10 else ""
    print(f"  Max/min novel ratio: {ratio:.1f}x{flag}")

    print("\n  Avg / max tokens per novel:")
    novel_tokens = defaultdict(list)
    for nov, tl in zip(novels, token_lengths):
        novel_tokens[nov].append(tl)
    for nov, tls in sorted(novel_tokens.items()):
        arr = np.array(tls)
        print(f"    {nov:<42} mean={np.mean(arr):.0f}  p99={np.percentile(arr,99):.0f}  max={np.max(arr)}")

    if args.cap_novel:
        cap = args.cap_novel
        print(f"\n  Cap setting: --cap-novel {cap}")
        for nov, cnt in novel_counts.most_common():
            if cnt > cap:
                excess = cnt - cap
                print(f"    {nov}: {cnt} → {cap}  (removes {excess})")
            else:
                print(f"    {nov}: {cnt}  (under cap, unchanged)")

    # ════════════════════════════════════════════════════════════
    # 4. Outlier detection  ← FIXED: no 3-sigma / IQR noise
    # ════════════════════════════════════════════════════════════
    header(4, "OUTLIER DETECTION")

    outlier_reasons: dict[int, tuple] = {}   # line_idx → (tokens, [reasons], preview)

    for i, (line_idx, tl) in enumerate(zip(line_indices, token_lengths)):
        reasons = []

        # Hard OOM guard
        if tl > args.max_len:
            reasons.append(f"over_max_len({tl}>{args.max_len})")

        # Suspiciously short — likely a scraping artifact
        if tl < args.min_len:
            reasons.append(f"too_short({tl}<{args.min_len})")

        # Empty / near-empty translation output
        if assistant_lengths[i] < args.min_assistant:
            reasons.append(f"empty_assistant({assistant_lengths[i]}<{args.min_assistant})")

        # <think> contamination in labels
        if think_flags[i]:
            reasons.append("think_content_in_labels")

        if reasons:
            outlier_reasons[line_idx] = (tl, reasons, texts[i][:120])

    print(f"  Total outliers detected: {len(outlier_reasons)}")
    print(f"  Criteria applied:")
    print(f"    over_max_len    > {args.max_len}  (OOM during backward pass)")
    print(f"    too_short       < {args.min_len}   (scraping artifact)")
    print(f"    empty_assistant < {args.min_assistant}   (blank translation)")
    print(f"    think_in_labels : <think> with real content in assistant turn")
    print(f"\n  NOTE: 3-sigma / IQR flags removed — translation token lengths are")
    print(f"  naturally right-skewed; long examples are valid, not errors.")

    think_contaminated = sum(think_flags)
    print(f"\n  <think> contaminated examples: {think_contaminated}")
    if think_contaminated == 0:
        print("  ✓ No think contamination — enable_thinking=False is working correctly")

    if outlier_reasons:
        print(f"\n  Outlier breakdown by reason:")
        reason_counts: Counter = Counter()
        for _, (_, reasons, _) in outlier_reasons.items():
            for r in reasons:
                key = r.split("(")[0]
                reason_counts[key] += 1
        for reason, cnt in reason_counts.most_common():
            print(f"    {reason:<30} {cnt}")

        print(f"\n  Sample outliers (first 20):")
        for line_idx, (tl, reasons, preview) in list(outlier_reasons.items())[:20]:
            print(f"    line={line_idx:>6}  tokens={tl:>4}  reasons={reasons}")
            print(f"           preview: {repr(preview)}")

    # ════════════════════════════════════════════════════════════
    # 5. Duplication check
    # ════════════════════════════════════════════════════════════
    header(5, "DUPLICATION")
    seen: set = set()
    dupes = 0
    for _, rec in records:
        key = (rec.get("novel"), rec.get("chapter"), str(rec.get("window_pair_indices")))
        if key in seen:
            dupes += 1
        seen.add(key)
    print(f"  Exact duplicate (novel+chapter+window) pairs: {dupes}")
    if dupes == 0:
        print("  ✓ No duplicates found")

    # ════════════════════════════════════════════════════════════
    # 6. Padding waste estimate
    # ════════════════════════════════════════════════════════════
    header(6, f"PADDING WASTE ESTIMATE (batch_size={args.batch_size})")
    sorted_lens = np.sort(token_lengths)
    batches = [
        sorted_lens[i : i + args.batch_size]
        for i in range(0, len(sorted_lens), args.batch_size)
    ]
    pad_waste = [
        ((b.max() * len(b)) - b.sum()) / (b.max() * len(b)) * 100
        for b in batches if len(b) > 0
    ]
    avg_waste = np.mean(pad_waste)
    max_waste = np.max(pad_waste)
    print(f"  Avg padding waste per batch : {avg_waste:.1f}%")
    print(f"  Max padding waste per batch : {max_waste:.1f}%")
    if avg_waste > 30:
        print("  !! High padding waste — consider packing=True in your trainer")
    else:
        print("  ✓ Padding waste is acceptable (packing=True optional)")

    # ════════════════════════════════════════════════════════════
    # 7. Health summary
    # ════════════════════════════════════════════════════════════
    header(7, "HEALTH SUMMARY")
    issues = []
    if over_max > 0:
        issues.append(f"  !! {over_max} examples exceed MAX_LEN={args.max_len} — will OOM during backward")
    if ratio > 10:
        issues.append(
            f"  !! Novel imbalance ratio {ratio:.1f}x — model may overfit dominant novel\n"
            f"     → Use --cap-novel to reduce, or --weights-out for weighted sampling"
        )
    if dupes > 0:
        issues.append(f"  !! {dupes} duplicate examples found")
    if think_contaminated > 0:
        issues.append(f"  !! {think_contaminated} examples have <think> content in labels")
    if avg_waste > 30:
        issues.append(f"  !! High padding waste {avg_waste:.1f}% — enable packing=True")

    if not issues:
        print("  ✓ No critical issues found")
    else:
        for iss in issues:
            print(iss)

    # ════════════════════════════════════════════════════════════
    # 8. Outlier removal
    # ════════════════════════════════════════════════════════════
    if outlier_reasons:
        header(8, "OUTLIER REMOVAL")
        print(f"  {len(outlier_reasons)} genuinely bad lines identified (OOM / empty / think-contaminated).")

        if args.yes:
            ans = "y"
            print("  Auto-confirming removal (--yes flag set).")
        else:
            ans = input("  Remove these from JSONL? This will overwrite the file. [y/N]: ").strip().lower()

        if ans == "y":
            bad_lines = set(outlier_reasons.keys())
            lines_out = []
            removed   = 0
            with open(args.jsonl, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i in bad_lines:
                        removed += 1
                    else:
                        lines_out.append(line)
            with open(args.jsonl, "w", encoding="utf-8") as f:
                f.writelines(lines_out)
            kept = len(lines_out)
            print(f"  Done. Kept {kept}, removed {removed}. File overwritten.")
            # Update in-memory state so cap / weights steps work on clean data
            records   = [(i, r) for (i, r) in records if i not in bad_lines]
            novels    = [nov for nov, li in zip(novels, line_indices) if li not in bad_lines]
            chapters  = [ch  for ch,  li in zip(chapters, line_indices) if li not in bad_lines]
        else:
            print("  Skipped. No changes made.")

    # ════════════════════════════════════════════════════════════
    # 9. Novel capping
    # ════════════════════════════════════════════════════════════
    if args.cap_novel:
        cap = args.cap_novel
        header(9, f"NOVEL CAPPING (--cap-novel {cap})")

        # Identify lines to keep: take first `cap` occurrences per novel
        novel_seen: Counter = Counter()
        keep_flags  = []
        for _, rec in records:
            nov = rec.get("novel", "unknown")
            if novel_seen[nov] < cap:
                keep_flags.append(True)
                novel_seen[nov] += 1
            else:
                keep_flags.append(False)

        kept_recs    = sum(keep_flags)
        removed_recs = len(records) - kept_recs
        print(f"  Keeping {kept_recs} records, removing {removed_recs} excess.")

        if args.yes:
            ans = "y"
            print("  Auto-confirming cap (--yes flag set).")
        else:
            ans = input("  Apply cap and overwrite JSONL? [y/N]: ").strip().lower()

        if ans == "y":
            lines_out = []
            with open(args.jsonl, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            # records[i][0] is the original line index in the file
            keep_line_indices = {
                records[i][0] for i, keep in enumerate(keep_flags) if keep
            }
            removed = 0
            for li, line in enumerate(all_lines):
                if li in keep_line_indices or line.strip() == "":
                    lines_out.append(line)
                else:
                    removed += 1

            with open(args.jsonl, "w", encoding="utf-8") as f:
                f.writelines(lines_out)
            print(f"  Done. Wrote {len(lines_out)} lines (removed {removed}).")
        else:
            print("  Skipped. No changes made.")

    # ════════════════════════════════════════════════════════════
    # 10. Weighted sampler output
    # ════════════════════════════════════════════════════════════
    if args.weights_out:
        header(10, "WEIGHTED SAMPLER OUTPUT")

        # Inverse-frequency weights: weight_i = 1 / count(novel_i)
        # Normalised so weights sum to len(records)
        novel_freq = Counter(novels)
        weights    = np.array([1.0 / novel_freq[nov] for nov in novels], dtype=np.float32)
        weights   *= len(weights) / weights.sum()   # normalise

        out_path = Path(args.weights_out)
        np.save(out_path, weights)
        print(f"  Saved {len(weights)} weights → '{out_path}'")
        print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print()
        print("  To use in training (HuggingFace Trainer):")
        print("    from torch.utils.data import WeightedRandomSampler")
        print("    import numpy as np")
        print(f"    weights = np.load('{out_path}')")
        print("    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)")
        print("    # Pass to a custom Trainer subclass that overrides get_train_dataloader()")

    print("\nDone.\n")


if __name__ == "__main__":
    main()