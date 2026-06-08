from pathlib import Path
import numpy as np
from datasets import load_from_disk

TOKENIZED_ROOT = Path("/home/avinash/Projects/Custom-LN-Translator-Training/PreProcessingOutputs/")

splits = {
    "full_train": load_from_disk(str(TOKENIZED_ROOT / "full_chapter" / "train")),
    "full_val": load_from_disk(str(TOKENIZED_ROOT / "full_chapter" / "validation")),
    "chunk_train": load_from_disk(str(TOKENIZED_ROOT / "chunked" / "train")),
    "chunk_val": load_from_disk(str(TOKENIZED_ROOT / "chunked" / "validation")),
}


def analyze_dataset(ds, name):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    lengths = np.array([len(x["input_ids"]) for x in ds], dtype=np.int32)

    total_tokens = lengths.sum()

    print(f"Samples                : {len(ds):,}")
    print(f"Total Tokens           : {total_tokens:,}")
    print(f"Avg Tokens / Sample    : {lengths.mean():.2f}")
    print(f"Median Tokens          : {np.median(lengths):.2f}")
    print(f"Min Tokens             : {lengths.min():,}")
    print(f"Max Tokens             : {lengths.max():,}")

    print("\nPercentiles")
    print(f"P50                    : {np.percentile(lengths,50):.0f}")
    print(f"P75                    : {np.percentile(lengths,75):.0f}")
    print(f"P90                    : {np.percentile(lengths,90):.0f}")
    print(f"P95                    : {np.percentile(lengths,95):.0f}")
    print(f"P99                    : {np.percentile(lengths,99):.0f}")

    print("\nEstimated packed blocks")

    for seq_len in [1024, 2048, 3072]:
        blocks = int(np.ceil(total_tokens / seq_len))
        print(f"{seq_len:4d} tokens -> {blocks:,} packed sequences")

    print("\nLength buckets")

    buckets = [
        (0,512),
        (512,1024),
        (1024,2048),
        (2048,3072),
        (3072,4096),
        (4096,8192),
        (8192,999999)
    ]

    for low, high in buckets:
        count = ((lengths >= low) & (lengths < high)).sum()
        pct = count / len(lengths) * 100
        print(f"{low:5d}-{high:<5d}: {count:6d} ({pct:6.2f}%)")

    return {
        "samples": len(ds),
        "tokens": int(total_tokens),
        "avg_len": float(lengths.mean()),
    }


stats = {}

for name, ds in splits.items():
    stats[name] = analyze_dataset(ds, name)

print("\n\n" + "=" * 80)
print("GLOBAL SUMMARY")
print("=" * 80)

total_train_tokens = (
    stats["full_train"]["tokens"] +
    stats["chunk_train"]["tokens"]
)

total_val_tokens = (
    stats["full_val"]["tokens"] +
    stats["chunk_val"]["tokens"]
)

print(f"Train Tokens : {total_train_tokens:,}")
print(f"Val Tokens   : {total_val_tokens:,}")
print(f"All Tokens   : {total_train_tokens + total_val_tokens:,}")

print("\nApprox packed sequences")

for seq_len in [1024, 2048, 3072]:
    packed = int(np.ceil(total_train_tokens / seq_len))
    print(f"{seq_len:4d} -> {packed:,} sequences")