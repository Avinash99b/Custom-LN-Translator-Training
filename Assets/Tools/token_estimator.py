#!/usr/bin/env python3

from pathlib import Path
import statistics

try:
    import tiktoken
except ImportError:
    print("Install first:")
    print("pip install tiktoken")
    raise SystemExit(1)


def analyze_folder(folder: Path, encoder):
    files = sorted(folder.glob("*.txt"))

    token_counts = []
    char_counts = []

    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        tokens = len(encoder.encode(text))

        token_counts.append(tokens)
        char_counts.append(len(text))

    if not token_counts:
        return {
            "files": 0,
            "tokens": 0,
            "chars": 0,
            "avg_tokens": 0,
            "median_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
        }

    return {
        "files": len(token_counts),
        "tokens": sum(token_counts),
        "chars": sum(char_counts),
        "avg_tokens": statistics.mean(token_counts),
        "median_tokens": statistics.median(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }


def print_stats(name, stats):
    print(f"\n{name}")
    print("=" * 60)
    print(f"Files               : {stats['files']:,}")
    print(f"Characters          : {stats['chars']:,}")
    print(f"Tokens              : {stats['tokens']:,}")
    print(f"Avg Tokens/File     : {stats['avg_tokens']:,.1f}")
    print(f"Median Tokens/File  : {stats['median_tokens']:,.1f}")
    print(f"Min Tokens/File     : {stats['min_tokens']:,}")
    print(f"Max Tokens/File     : {stats['max_tokens']:,}")


def main():
    root_input = input(
        "\nEnter root folder path (containing EN and JP folders):\n> "
    ).strip()

    root = Path(root_input)

    if not root.exists():
        print("Folder does not exist.")
        return

    print("\nLoading GPT-4o-mini tokenizer...")
    encoder = tiktoken.get_encoding("o200k_base")

    en_stats = analyze_folder(root / "EN", encoder)
    jp_stats = analyze_folder(root / "JP", encoder)

    total_files = en_stats["files"] + jp_stats["files"]
    total_tokens = en_stats["tokens"] + jp_stats["tokens"]
    total_chars = en_stats["chars"] + jp_stats["chars"]

    print("\n" + "=" * 60)
    print("GPT-4o-mini TOKEN ANALYSIS")
    print("=" * 60)

    print_stats("EN", en_stats)
    print_stats("JP", jp_stats)

    print("\nCOMBINED")
    print("=" * 60)
    print(f"Files               : {total_files:,}")
    print(f"Characters          : {total_chars:,}")
    print(f"Tokens              : {total_tokens:,}")

    if total_files:
        print(
            f"Avg Tokens/File     : {total_tokens / total_files:,.1f}"
        )

    print("\nDATASET SIZE")
    print("=" * 60)
    print(f"EN Tokens           : {en_stats['tokens'] / 1_000_000:.3f}M")
    print(f"JP Tokens           : {jp_stats['tokens'] / 1_000_000:.3f}M")
    print(f"Total Tokens        : {total_tokens / 1_000_000:.3f}M")

    if total_tokens:
        print(
            f"EN Share            : {en_stats['tokens'] / total_tokens * 100:.2f}%"
        )
        print(
            f"JP Share            : {jp_stats['tokens'] / total_tokens * 100:.2f}%"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()