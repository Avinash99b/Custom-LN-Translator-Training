import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"

ROOT_DIR = input("Enter the root directory containing JP and EN folders: ").strip()
JP_DIR = Path(ROOT_DIR) / "JP"
EN_DIR = Path(ROOT_DIR) / "EN"

OUTPUT_JSON = Path(ROOT_DIR) / "chapter_similarity_matrix.json"

# Number of lines to read from each file
FIRST_N_LINES = 100


def load_text(path: Path, n_lines: int):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = []

            for i, line in enumerate(f):
                if i >= n_lines:
                    break

                line = line.strip()

                if line:
                    lines.append(line)

            return "\n".join(lines)

    except Exception as e:
        print(f"Failed reading {path}: {e}")
        return ""


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def print_progress(prefix: str, current: int, total: int):
    bar_width = 30
    filled_width = int(bar_width * current / total) if total else bar_width
    empty_width = bar_width - filled_width
    bar = "#" * filled_width + "-" * empty_width
    end_char = "\n" if current >= total else "\r"
    print(f"{prefix} [{bar}] {current}/{total}", end=end_char, flush=True)


print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

jp_files = sorted(JP_DIR.glob("*.txt"))
en_files = sorted(EN_DIR.glob("*.txt"))

print(f"JP chapters: {len(jp_files)}")
print(f"EN chapters: {len(en_files)}")

# --------------------------------------------------
# Read text
# --------------------------------------------------

jp_texts = {}
en_texts = {}

for file in jp_files:
    jp_texts[file.name] = load_text(file, FIRST_N_LINES)

for file in en_files:
    en_texts[file.name] = load_text(file, FIRST_N_LINES)

# --------------------------------------------------
# Generate embeddings once
# --------------------------------------------------

print("Embedding JP chapters...")

jp_items = [(filename, text) for filename, text in jp_texts.items() if text]

jp_embeddings = {}

total_jp = len(jp_items)

for current_jp, (filename, text) in enumerate(jp_items, start=1):
    print_progress("JP embeddings", current_jp, total_jp)

    jp_embeddings[filename] = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )

if total_jp == 0:
    print("JP embeddings [------------------------------] 0/0")

print("Embedding EN chapters...")

en_items = [(filename, text) for filename, text in en_texts.items() if text]

en_embeddings = {}

total_en = len(en_items)

for current_en, (filename, text) in enumerate(en_items, start=1):
    print_progress("EN embeddings", current_en, total_en)

    en_embeddings[filename] = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )

if total_en == 0:
    print("EN embeddings [------------------------------] 0/0")

# --------------------------------------------------
# Similarity matrix
# --------------------------------------------------

print("Computing similarity matrix...")

results = {}

total = len(jp_embeddings)
current = 0

for jp_name, jp_emb in jp_embeddings.items():

    current += 1

    print(f"[{current}/{total}] {jp_name}")

    results[jp_name] = {}

    for en_name, en_emb in en_embeddings.items():

        score = cosine_similarity(
            jp_emb,
            en_emb
        )

        results[jp_name][en_name] = round(score, 6)

# --------------------------------------------------
# Save
# --------------------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print(f"Saved -> {OUTPUT_JSON}")

