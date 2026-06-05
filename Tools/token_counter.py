#!/usr/bin/env python3

from pathlib import Path
import re
import statistics

# ==================================================
# CONFIG
# ==================================================

NOVELS_ROOT = Path("/home/avinash/Projects/Custom-LN-Translator-Training/Assets/Novels")

PARAGRAPHS_PER_CHUNK = 4
STRIDE = 2

# rough estimates
JP_CHARS_PER_TOKEN = 1.8
EN_CHARS_PER_TOKEN = 4.0


# ==================================================
# HELPERS
# ==================================================

def split_paragraphs(text: str):
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    return paragraphs


def estimate_tokens_jp(text: str):
    return len(text) / JP_CHARS_PER_TOKEN


def estimate_tokens_en(text: str):
    return len(text) / EN_CHARS_PER_TOKEN


def chunk_count(paragraphs):

    if len(paragraphs) < PARAGRAPHS_PER_CHUNK:
        return 1

    count = 0

    for start in range(
        0,
        len(paragraphs) - PARAGRAPHS_PER_CHUNK + 1,
        STRIDE
    ):
        count += 1

    return count


# ==================================================
# MAIN
# ==================================================

def main():

    total_chapters = 0

    total_jp_tokens = 0
    total_en_tokens = 0

    total_examples = 0

    chapter_token_sizes = []

    print()
    print("=" * 80)
    print("NOVEL DATASET ANALYSIS")
    print("=" * 80)

    for novel_dir in sorted(NOVELS_ROOT.iterdir()):

        if not novel_dir.is_dir():
            continue

        jp_dir = novel_dir / "JP-Output"
        en_dir = novel_dir / "EN-Output"

        if not jp_dir.exists():
            continue

        if not en_dir.exists():
            continue

        novel_examples = 0
        novel_tokens = 0
        novel_chapters = 0

        for jp_file in sorted(jp_dir.glob("*.txt")):

            en_file = en_dir / jp_file.name

            if not en_file.exists():
                continue

            jp_text = jp_file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            en_text = en_file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            jp_tokens = estimate_tokens_jp(jp_text)
            en_tokens = estimate_tokens_en(en_text)

            total_jp_tokens += jp_tokens
            total_en_tokens += en_tokens

            chapter_tokens = jp_tokens + en_tokens

            chapter_token_sizes.append(
                chapter_tokens
            )

            jp_paragraphs = split_paragraphs(
                jp_text
            )

            chunks = chunk_count(
                jp_paragraphs
            )

            total_examples += chunks

            novel_examples += chunks
            novel_tokens += chapter_tokens
            novel_chapters += 1

            total_chapters += 1

        print(
            f"{novel_dir.name:40s}"
            f" chapters={novel_chapters:5d}"
            f" chunks={novel_examples:7d}"
            f" tokens={int(novel_tokens):10,d}"
        )

    print()
    print("=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    total_tokens = (
        total_jp_tokens
        + total_en_tokens
    )

    print(
        f"Total Chapters           : "
        f"{total_chapters:,}"
    )

    print(
        f"Estimated JP Tokens      : "
        f"{int(total_jp_tokens):,}"
    )

    print(
        f"Estimated EN Tokens      : "
        f"{int(total_en_tokens):,}"
    )

    print(
        f"Estimated Total Tokens   : "
        f"{int(total_tokens):,}"
    )

    print()

    print(
        f"Training Examples        : "
        f"{total_examples:,}"
    )

    print(
        f"Avg Tokens / Chapter     : "
        f"{statistics.mean(chapter_token_sizes):.0f}"
    )

    print(
        f"Median Tokens / Chapter  : "
        f"{statistics.median(chapter_token_sizes):.0f}"
    )

    print()

    effective_tokens = (
        statistics.mean(chapter_token_sizes)
        * total_examples
    )

    print(
        f"Effective Chunked Tokens : "
        f"{int(effective_tokens):,}"
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()