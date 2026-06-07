import json
from pathlib import Path
import sys
import numpy as np
from sentence_transformers import SentenceTransformer




FIRST_N_LINES = 300
BATCH_SIZE = 32

MODELS = [
    "sentence-transformers/LaBSE",
]
def load_text(path: Path, n_lines: int):
    try:
        lines = []

        with open(path, "r", encoding="utf-8") as f:
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
def main(ROOT_DIR: str):
    JP_DIR = Path(ROOT_DIR) / "JP"
    EN_DIR = Path(ROOT_DIR) / "EN"

    OUTPUT_JSON = Path(ROOT_DIR) / "chapter_similarity_matrix.json"
    for model_name in MODELS:
    
        print(f"Loading model: {model_name}...")
        model = SentenceTransformer(model_name, trust_remote_code=True)

        jp_files = sorted(JP_DIR.glob("*.txt"))
        en_files = sorted(EN_DIR.glob("*.txt"))

        print(f"JP chapters: {len(jp_files)}")
        print(f"EN chapters: {len(en_files)}")

        # --------------------------------------------------
        # Load texts
        # --------------------------------------------------

        jp_names = []
        jp_texts = []

        for file in jp_files:
            text = load_text(file, FIRST_N_LINES)

            if text:
                jp_names.append(file.name)

                # E5 expects prefixes
                jp_texts.append(f"passage: {text}")

        en_names = []
        en_texts = []

        for file in en_files:
            text = load_text(file, FIRST_N_LINES)

            if text:
                en_names.append(file.name)

                # E5 expects prefixes
                en_texts.append(f"passage: {text}")

        # --------------------------------------------------
        # Embeddings
        # --------------------------------------------------

        print("Generating JP embeddings...")

        jp_embeddings = model.encode(
            jp_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print("Generating EN embeddings...")

        en_embeddings = model.encode(
            en_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # --------------------------------------------------
        # Similarity Matrix
        # --------------------------------------------------

        print("Computing similarity matrix...")

        similarity_matrix = np.matmul(
            jp_embeddings,
            en_embeddings.T
        )

        # --------------------------------------------------
        # Convert to JSON
        # --------------------------------------------------

        results = {}

        for jp_idx, jp_name in enumerate(jp_names):

            row = {}

            for en_idx, en_name in enumerate(en_names):

                row[en_name] = round(
                    float(similarity_matrix[jp_idx, en_idx]),
                    6
                )

            results[jp_name] = row

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        with open(
            Path(OUTPUT_JSON).parent / f"{Path(OUTPUT_JSON).stem}.{model_name.replace('/', '_')}.json",
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

if __name__ == "__main__":
    #load root dir from cli args
    if(len(sys.argv) < 2):
        print("Usage: python embeding_research.py <root_dir>")
        sys.exit(1)

    ROOT_DIR = sys.argv[1].strip()
    main(ROOT_DIR)