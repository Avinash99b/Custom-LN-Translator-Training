#!/usr/bin/env python3
"""
Verify that JP-Output/n.txt and EN-Output/n.txt are the same chapter
using a normal Azure OpenAI-compatible inference endpoint.

Usage:
  export AZURE_OPENAI_API_KEY="..."
  export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com/"
  export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"

  python verify_outputs.py /path/to/base_folder

Expected folder layout:
  base_folder/
    JP-Output/
      1.txt
      2.txt
      ...
    EN-Output/
      1.txt
      2.txt
      ...

What it does:
- Matches files strictly by number stem: n.txt with n.txt
- Compares the first N lines from each file (default: 50)
- Calls the normal chat/completions endpoint
- Writes a JSON report with PASS / FAIL / MISSING entries
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_SAMPLE_LINES = 50
DEFAULT_WORKERS = 8
REQUEST_TIMEOUT = 300


SYSTEM_PROMPT = """You are a bilingual light novel chapter alignment checker.

Your task is to determine whether the Japanese and English text fragments are the SAME chapter.

Focus primarily on:
- chapter title
- chapter number if present
- opening scene
- opening setting
- opening characters
- opening dialogue
- first major event at the beginning

Rules:
- Compare ONLY the provided opening text.
- Do NOT compare the full chapter.
- Do NOT reject because later content differs.
- Do NOT reject because one version is longer.
- Do NOT reject because chapter numbers differ or are missing.
- Do NOT reject because of translator notes, afterwords, or localized wording.
- Treat them as the same if they clearly begin from the same chapter/scene.

Return ONLY valid JSON:
{
  "same_chapter": true,
  "confidence": 0.0,
  "reason": "short explanation",
  "major_drift": false
}
"""


@dataclass
class ChapterPair:
    chapter_id: str
    jp_file: Path
    en_file: Path
    jp_sample: str
    en_sample: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def first_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:n]).strip()


def chapter_num_key(name: str):
    m = re.fullmatch(r"(\d+)\.txt", name)
    if m:
        return (0, int(m.group(1)))
    return (1, name)


def extract_json_from_response(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        raise ValueError("Empty response from model")

    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError(f"Could not extract JSON from response: {raw[:300]!r}")


def normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    same_chapter = data.get("same_chapter")
    if not isinstance(same_chapter, bool):
        same_chapter = str(same_chapter).strip().lower() in {"true", "1", "yes"}

    major_drift = data.get("major_drift")
    if not isinstance(major_drift, bool):
        major_drift = str(major_drift).strip().lower() in {"true", "1", "yes"}

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "no reason provided")).strip()
    if not reason:
        reason = "no reason provided"

    return {
        "same_chapter": same_chapter,
        "confidence": confidence,
        "reason": reason[:500],
        "major_drift": major_drift,
    }


def build_pairs(base_path: Path, sample_lines: int) -> tuple[list[ChapterPair], list[str], list[str]]:
    jp_dir = base_path / "JP-Output"
    en_dir = base_path / "EN-Output"

    if not jp_dir.exists():
        raise FileNotFoundError(f"Missing folder: {jp_dir}")
    if not en_dir.exists():
        raise FileNotFoundError(f"Missing folder: {en_dir}")

    jp_files = {p.name: p for p in jp_dir.glob("*.txt")}
    en_files = {p.name: p for p in en_dir.glob("*.txt")}

    jp_names = sorted(jp_files.keys(), key=chapter_num_key)
    en_names = sorted(en_files.keys(), key=chapter_num_key)

    common_names = sorted(set(jp_files) & set(en_files), key=chapter_num_key)

    pairs: list[ChapterPair] = []
    for name in common_names:
        jp_file = jp_files[name]
        en_file = en_files[name]
        jp_sample = first_n_lines(read_text(jp_file), sample_lines)
        en_sample = first_n_lines(read_text(en_file), sample_lines)

        if not jp_sample or not en_sample:
            continue

        pairs.append(
            ChapterPair(
                chapter_id=Path(name).stem,
                jp_file=jp_file,
                en_file=en_file,
                jp_sample=jp_sample,
                en_sample=en_sample,
            )
        )

    return pairs, jp_names, en_names


def ask_llm(
    endpoint: str,
    deployment: str,
    api_key: str,
    pair: ChapterPair,
    sample_lines: int,
) -> dict[str, Any]:
    user_prompt = (
        f"Chapter number: {pair.chapter_id}\n\n"
        f"Japanese first {sample_lines} lines:\n"
        f"{pair.jp_sample}\n\n"
        f"---\n\n"
        f"English first {sample_lines} lines:\n"
        f"{pair.en_sample}\n"
    )

    payload = {
        "model": deployment,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    url = endpoint.rstrip("/") + "/openai/v1/chat/completions"

    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    body = response.json()
    try:
        raw = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected API response structure: {e}\nBody: {json.dumps(body)[:500]}") from e

    parsed = extract_json_from_response(raw)
    result = normalize_result(parsed)
    result["raw_response"] = raw
    return result


def verify_pair(
    endpoint: str,
    deployment: str,
    api_key: str,
    pair: ChapterPair,
    sample_lines: int,
) -> dict[str, Any]:
    try:
        llm_result = ask_llm(endpoint, deployment, api_key, pair, sample_lines)
        status = "PASS" if llm_result["same_chapter"] and not llm_result["major_drift"] else "FAIL"
        return {
            "chapter_id": pair.chapter_id,
            "jp_file": str(pair.jp_file),
            "en_file": str(pair.en_file),
            "status": status,
            "confidence": llm_result["confidence"],
            "reason": llm_result["reason"],
            "major_drift": llm_result["major_drift"],
            "same_chapter": llm_result["same_chapter"],
            "raw_response": llm_result["raw_response"],
        }
    except Exception as e:
        return {
            "chapter_id": pair.chapter_id,
            "jp_file": str(pair.jp_file),
            "en_file": str(pair.en_file),
            "status": "ERROR",
            "confidence": 0.0,
            "reason": str(e),
            "major_drift": True,
            "same_chapter": False,
            "raw_response": "",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify JP-Output/n.txt against EN-Output/n.txt using normal Azure OpenAI inference."
    )
    parser.add_argument("base_path", help="Folder containing JP-Output/ and EN-Output/")
    parser.add_argument("--sample-lines", type=int, default=DEFAULT_SAMPLE_LINES, help="First N lines to compare")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent requests")
    parser.add_argument(
        "--report",
        default="chapter_verification_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_OPENAI_ENDPOINT", "https://bathu-mpkwvf7h-eastus2.cognitiveservices.azure.com/"),
        help="Azure OpenAI endpoint, e.g. https://resource.openai.azure.com/",
    )
    parser.add_argument(
        "--deployment",
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        help="Azure deployment name",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AZURE_OPENAI_API_KEY", "[REACTED_GOTCHA]"),
        help="Azure API key",
    )
    args = parser.parse_args()

    if not args.endpoint:
        raise SystemExit("Missing endpoint. Set AZURE_OPENAI_ENDPOINT or pass --endpoint.")
    if not args.deployment:
        raise SystemExit("Missing deployment name. Set AZURE_OPENAI_DEPLOYMENT or pass --deployment.")
    if not args.api_key:
        raise SystemExit("Missing API key. Set AZURE_OPENAI_API_KEY or pass --api-key.")

    base_path = Path(args.base_path)
    pairs, jp_names, en_names = build_pairs(base_path, args.sample_lines)

    jp_set = set(jp_names)
    en_set = set(en_names)
    missing_in_en = sorted(jp_set - en_set, key=chapter_num_key)
    missing_in_jp = sorted(en_set - jp_set, key=chapter_num_key)

    print(f"Found {len(pairs)} matched chapter pairs")
    if missing_in_en:
        print(f"Missing in EN-Output: {', '.join(missing_in_en[:10])}" + (" ..." if len(missing_in_en) > 10 else ""))
    if missing_in_jp:
        print(f"Missing in JP-Output: {', '.join(missing_in_jp[:10])}" + (" ..." if len(missing_in_jp) > 10 else ""))

    results: list[dict[str, Any]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                verify_pair,
                args.endpoint,
                args.deployment,
                args.api_key,
                pair,
                args.sample_lines,
            )
            for pair in pairs
        ]

        total = len(futures)
        completed = 0

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{total}] {result['chapter_id']}: "
                f"{result['status']} "
                f"(confidence={result['confidence']:.2f}) "
                f"{result['reason']}"
            )

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    errors = [r for r in results if r["status"] == "ERROR"]

    report = {
        "generated_at": time.time(),
        "base_path": str(base_path),
        "jp_dir": "JP-Output",
        "en_dir": "EN-Output",
        "sample_lines": args.sample_lines,
        "deployment": args.deployment,
        "summary": {
            "total_pairs": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "errors": len(errors),
            "missing_in_en": missing_in_en,
            "missing_in_jp": missing_in_jp,
            "elapsed_sec": round(time.time() - started, 3),
        },
        "chapters": sorted(results, key=lambda x: chapter_num_key(f"{x['chapter_id']}.txt")),
    }

    report_path = Path(f"{base_path}/output_dir_pass_verification_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to: {report_path.resolve()}")
    print(f"PASS={len(passed)} FAIL={len(failed)} ERROR={len(errors)}")


if __name__ == "__main__":
    main()