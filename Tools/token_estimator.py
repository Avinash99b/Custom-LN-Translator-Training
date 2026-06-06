#!/usr/bin/env python3

from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("pip install tiktoken")
    raise SystemExit(1)


def token_count(text: str, encoder):
    return len(encoder.encode(text))


def analyze_novel(novel_dir: Path, encoder):
    en_dir = novel_dir / "EN-Output"
    jp_dir = novel_dir / "JP-Output"

    if not en_dir.exists() or not jp_dir.exists():
        return None

    en_files = {
        f.name: f
        for f in en_dir.glob("*.txt")
    }

    jp_files = {
        f.name: f
        for f in jp_dir.glob("*.txt")
    }

    paired_names = sorted(
        set(en_files.keys()) &
        set(jp_files.keys())
    )

    orphan_en = sorted(
        set(en_files.keys()) -
        set(jp_files.keys())
    )

    orphan_jp = sorted(
        set(jp_files.keys()) -
        set(en_files.keys())
    )

    en_tokens = 0
    jp_tokens = 0

    en_chars = 0
    jp_chars = 0

    pair_count = 0

    for filename in paired_names:
        try:
            en_text = en_files[filename].read_text(
                encoding="utf-8",
                errors="ignore"
            )

            jp_text = jp_files[filename].read_text(
                encoding="utf-8",
                errors="ignore"
            )

        except Exception:
            continue

        pair_count += 1

        en_chars += len(en_text)
        jp_chars += len(jp_text)

        en_tokens += token_count(
            en_text,
            encoder
        )

        jp_tokens += token_count(
            jp_text,
            encoder
        )

    return {
        "novel": novel_dir.name,
        "pairs": pair_count,
        "en_tokens": en_tokens,
        "jp_tokens": jp_tokens,
        "en_chars": en_chars,
        "jp_chars": jp_chars,
        "orphan_en": len(orphan_en),
        "orphan_jp": len(orphan_jp),
    }


def main():
    root_input = "/home/avinash/Projects/Custom-LN-Translator-Training/Assets/Novels".strip()

    root = Path(root_input)

    if not root.exists():
        print("Folder not found.")
        return

    print("\nLoading tokenizer...")
    encoder = tiktoken.get_encoding(
        "o200k_base"
    )

    total_pairs = 0

    total_en_tokens = 0
    total_jp_tokens = 0

    total_en_chars = 0
    total_jp_chars = 0

    total_orphan_en = 0
    total_orphan_jp = 0

    novels_processed = 0

    print("\nPER NOVEL")
    print("=" * 100)

    for novel_dir in sorted(root.iterdir()):
        if not novel_dir.is_dir():
            continue

        result = analyze_novel(
            novel_dir,
            encoder
        )

        if result is None:
            continue

        novels_processed += 1

        total_pairs += result["pairs"]

        total_en_tokens += result["en_tokens"]
        total_jp_tokens += result["jp_tokens"]

        total_en_chars += result["en_chars"]
        total_jp_chars += result["jp_chars"]

        total_orphan_en += result["orphan_en"]
        total_orphan_jp += result["orphan_jp"]

        combined_tokens = (
            result["en_tokens"] +
            result["jp_tokens"]
        )

        print(
            f"{result['novel'][:45]:45} | "
            f"Pairs={result['pairs']:6,} | "
            f"Tokens={combined_tokens:10,} | "
            f"Orphan EN={result['orphan_en']:4,} | "
            f"Orphan JP={result['orphan_jp']:4,}"
        )

    combined_tokens = (
        total_en_tokens +
        total_jp_tokens
    )

    combined_chars = (
        total_en_chars +
        total_jp_chars
    )

    print("\n")
    print("=" * 100)
    print("FINAL TRAINING CORPUS STATISTICS")
    print("=" * 100)

    print(
        f"Novels Processed       : "
        f"{novels_processed:,}"
    )

    print(
        f"Aligned Chapter Pairs  : "
        f"{total_pairs:,}"
    )

    print()

    print(
        f"EN Tokens              : "
        f"{total_en_tokens:,}"
    )

    print(
        f"JP Tokens              : "
        f"{total_jp_tokens:,}"
    )

    print(
        f"Combined Tokens        : "
        f"{combined_tokens:,}"
    )

    print()

    print(
        f"EN Dataset Size        : "
        f"{total_en_tokens/1_000_000:.3f}M"
    )

    print(
        f"JP Dataset Size        : "
        f"{total_jp_tokens/1_000_000:.3f}M"
    )

    print(
        f"Combined Dataset Size  : "
        f"{combined_tokens/1_000_000:.3f}M"
    )

    print()

    print(
        f"Characters             : "
        f"{combined_chars:,}"
    )

    print(
        f"Orphan EN Files        : "
        f"{total_orphan_en:,}"
    )

    print(
        f"Orphan JP Files        : "
        f"{total_orphan_jp:,}"
    )

    if total_pairs:
        print()

        print(
            f"Avg EN Tokens/Pair     : "
            f"{total_en_tokens/total_pairs:,.1f}"
        )

        print(
            f"Avg JP Tokens/Pair     : "
            f"{total_jp_tokens/total_pairs:,.1f}"
        )

        print(
            f"Avg Total Tokens/Pair  : "
            f"{combined_tokens/total_pairs:,.1f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()