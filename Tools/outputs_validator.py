import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

ENDPOINT = "https://bathu-mpkwvf7h-eastus2.services.ai.azure.com/openai/v1/"
DEPLOYMENT_NAME = "gpt-oss-120b"

API_KEY_ENV = "NOVEL_VERIFICATION_LLM_KEY"

EMBEDDING_MODEL = "BAAI/bge-m3"

SUMMARY_MODEL_MAX_TOKENS = 120
COMPARISON_MODEL_MAX_TOKENS = 300

LLM_WEIGHT = 0.90
EMBEDDING_WEIGHT = 0.10

THRESHOLD = 0.80

MAX_WORKERS = 5  # Parallel threads for chapter processing


# ============================================================
# PROMPTS
# ============================================================

SUMMARY_PROMPT = """
You are extracting the story content from a light novel chapter.

Summarize ONLY:

- major events
- character actions
- character relationships
- important dialogue outcomes
- locations
- plot progression

Ignore:
- writing style
- narration style
- localization
- wording

Rules:
- Maximum 5 short sentences.
- Plain text only.
- No markdown.
"""


COMPARE_PROMPT = """
You compare two chapter summaries.

Determine whether they describe the same story chapter.

Return ONLY valid format like this SAME|DIFFER, confidence: 0.0 - 1.0.

Schema:

SAME|DIFFER, confidence: 0.0 - 1.0 

DO NOT RETURN ANYTHING OTHER THAN THE FORMAT LIKE ABOVE. NO EXPLANATIONS, NO MARKDOWN, NO EXTRA FIELDS.

Rules:

SAME if:
- same plot
- same events
- same progression
- same outcomes

DIFFER if:
- major scenes differ
- major events differ
- content comes from another chapter
- important outcomes differ
- chapters appear merged/split

Confidence:
0.0 - 1.0

If uncertain:
return SAME with lower confidence.

No explanations.
No extra fields.
"""


# ============================================================
# OPENAI
# ============================================================

def create_client():
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing environment variable: {API_KEY_ENV}")
    return OpenAI(base_url=ENDPOINT, api_key=api_key)


# ============================================================
# FILE HELPERS
# ============================================================

def read_text(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore")


def natural_sort_key(filename: str):
    """
    Sort filenames naturally so 1.txt < 2.txt < 10.txt instead of
    lexicographic order where '10' < '2'.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


def collect_pairs(root_folder: Path):
    jp_dir = root_folder / "JP-Output"
    en_dir = root_folder / "EN-Output"

    if not jp_dir.exists():
        raise RuntimeError(f"Missing folder: {jp_dir}")
    if not en_dir.exists():
        raise RuntimeError(f"Missing folder: {en_dir}")

    jp_files = {f.name: f for f in jp_dir.glob("*.txt")}
    en_files = {f.name: f for f in en_dir.glob("*.txt")}

    common = sorted(
        set(jp_files.keys()) & set(en_files.keys()),
        key=natural_sort_key,
    )

    return [(name, jp_files[name], en_files[name]) for name in common]


# ============================================================
# RESUME HELPERS
# ============================================================

def load_progress(output_file: Path) -> dict:
    """Return a dict of chapter_name -> result for already-completed chapters."""
    if not output_file.exists():
        return {}
    try:
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        return {entry["chapter"]: entry for entry in data if "chapter" in entry}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_progress(output_file: Path, results: list):
    """Atomically write the current results list to disk."""
    tmp = output_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(output_file)


def ask_resume(output_file: Path) -> bool:
    """
    Ask the user whether to continue from existing progress or start over.
    Returns True  → continue (skip already-done chapters).
    Returns False → start over (delete existing file).
    """
    print(f"\nFound existing progress file: {output_file}")

    try:
        with open(output_file, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  {len(existing)} chapter(s) already processed.")
    except Exception:
        print("  (Could not read progress file details.)")

    while True:
        answer = input("\nContinue from where you left off? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            output_file.unlink(missing_ok=True)
            print("Starting over — existing progress deleted.\n")
            return False
        print("Please enter 'y' or 'n'.")


# ============================================================
# LLM
# ============================================================

def generate_summary(client, chapter_text: str) -> str:
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        temperature=0,
        max_tokens=SUMMARY_MODEL_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user",   "content": chapter_text},
        ],
    )
    return response.choices[0].message.content.strip()


def compare_summaries(client, jp_summary: str, en_summary: str) -> dict:
    prompt = f"Summary A:\n\n{jp_summary}\n\nSummary B:\n\n{en_summary}"

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        temperature=0,
        max_tokens=COMPARISON_MODEL_MAX_TOKENS,
        response_format={"type": "text"},
        messages=[
            {"role": "system", "content": COMPARE_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    print(f"  LLM raw output: {content}")

    split = content.split(", confidence:")
    if len(split) != 2:
        raise ValueError(f"Invalid response format: {content}")

    result_part = split[0].strip()
    confidence_part = split[1].strip()

    if not result_part.startswith("SAME") and not result_part.startswith("DIFFER"):
        raise ValueError(f"Invalid result value: {result_part}")

    try:
        confidence_value = float(confidence_part)
    except ValueError:
        raise ValueError(f"Invalid confidence value: {confidence_part}")

    return {"result": result_part, "confidence": confidence_value}


# ============================================================
# EMBEDDINGS
# ============================================================

class SummaryEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(self, text: str):
        return self.model.encode(f"passage: {text}", normalize_embeddings=True)

    def similarity(self, text_a: str, text_b: str) -> float:
        return float(np.dot(self.encode(text_a), self.encode(text_b)))


# ============================================================
# SCORING
# ============================================================

def calculate_final_score(llm_confidence: float, embedding_similarity: float) -> float:
    return llm_confidence * LLM_WEIGHT + embedding_similarity * EMBEDDING_WEIGHT


# ============================================================
# PER-CHAPTER WORKER
# ============================================================

def process_chapter(
    chapter_name: str,
    jp_file: Path,
    en_file: Path,
    client: OpenAI,
    embedder: SummaryEmbedder,
) -> dict:
    """Process one chapter end-to-end and return its result dict."""
    print(f"\n[START] {chapter_name}")

    jp_text = read_text(jp_file)
    en_text = read_text(en_file)

    jp_summary = generate_summary(client, jp_text)
    en_summary = generate_summary(client, en_text)

    comparison        = compare_summaries(client, jp_summary, en_summary)
    embedding_sim     = embedder.similarity(jp_summary, en_summary)
    llm_confidence    = float(comparison["confidence"])
    final_score       = calculate_final_score(llm_confidence, embedding_sim)
    final_result      = "SAME" if final_score >= THRESHOLD else "DIFFER"

    result = {
        "chapter":             chapter_name,
        "llm_result":          comparison["result"],
        "llm_confidence":      round(llm_confidence,  4),
        "embedding_similarity": round(embedding_sim,   4),
        "final_score":         round(final_score,      4),
        "final_result":        final_result,
    }

    print(f"[DONE]  {chapter_name} → {final_result} (score={result['final_score']})")
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify JP↔EN chapter alignment in a novel folder."
    )
    parser.add_argument("folder", help="Root folder of the novel")
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Parallel worker threads (default: {MAX_WORKERS})",
    )
    args = parser.parse_args()

    root_folder = Path(args.folder)
    output_file = root_folder / "validation_2nd_pass.json"

    # ── Resume / start-over prompt ──────────────────────────────────────────
    completed: dict = {}
    if output_file.exists():
        resume = ask_resume(output_file)
        if resume:
            completed = load_progress(output_file)
            print(f"Resuming — {len(completed)} chapter(s) already done, skipping them.\n")
        # else: file was deleted, completed stays {}

    # ── Collect all chapter pairs (naturally sorted) ────────────────────────
    all_pairs = collect_pairs(root_folder)
    if not all_pairs:
        print("No matching .txt files found in JP-Output / EN-Output.")
        sys.exit(0)

    # ── Seed results list in canonical order ────────────────────────────────
    # Pre-fill already-completed entries so ordering is preserved at save time.
    ordered_names  = [name for name, _, _ in all_pairs]
    results_by_name: dict = dict(completed)  # mutable working copy

    # Chapters that still need processing
    pending = [
        (name, jp, en)
        for name, jp, en in all_pairs
        if name not in completed
    ]

    if not pending:
        print("All chapters already processed. Nothing to do.")
        _write_ordered(output_file, ordered_names, results_by_name)
        sys.exit(0)

    print(f"Chapters to process : {len(pending)}")
    print(f"Already done        : {len(completed)}")
    print(f"Worker threads      : {args.workers}\n")

    # ── Initialise shared resources ─────────────────────────────────────────
    client   = create_client()
    embedder = SummaryEmbedder()

    # ── Parallel execution ──────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_name = {
            executor.submit(process_chapter, name, jp, en, client, embedder): name
            for name, jp, en in pending
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\n[ERROR] {name}: {exc}")
                result = {
                    "chapter":              name,
                    "llm_result":           "ERROR",
                    "llm_confidence":       0.0,
                    "embedding_similarity": 0.0,
                    "final_score":          0.0,
                    "final_result":         "ERROR",
                    "error":                str(exc),
                }

            results_by_name[name] = result
            # Save after every completed chapter (crash-safe)
            _write_ordered(output_file, ordered_names, results_by_name)

    # ── Final ordered save ──────────────────────────────────────────────────
    _write_ordered(output_file, ordered_names, results_by_name)
    print(f"\nSaved: {output_file}")

    # ── Summary ─────────────────────────────────────────────────────────────
    total  = len(results_by_name)
    same   = sum(1 for r in results_by_name.values() if r.get("final_result") == "SAME")
    differ = sum(1 for r in results_by_name.values() if r.get("final_result") == "DIFFER")
    errors = sum(1 for r in results_by_name.values() if r.get("final_result") == "ERROR")
    print(f"\nSummary: {total} chapters — {same} SAME | {differ} DIFFER | {errors} ERROR")


def _write_ordered(
    output_file: Path,
    ordered_names: list,
    results_by_name: dict,
):
    """Write results in the canonical chapter order (1.txt, 2.txt, …)."""
    ordered = [results_by_name[n] for n in ordered_names if n in results_by_name]
    save_progress(output_file, ordered)


if __name__ == "__main__":
    main()