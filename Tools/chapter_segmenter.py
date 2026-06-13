#!/usr/bin/env python3
"""
Chapter Segmenter → Training JSONL
====================================
Reads JP-Output / EN-Output aligned chapter pairs from a root folder,
uses an LLM to:
  1. Segment the story lines into semantically coherent training chunks
     — noise/exclusion detection is done LOCALLY via regex (no LLM tokens spent)
  2. Assemble and emit JSONL training samples ready for SFT fine-tuning

The LLM never rewrites or paraphrases the source text.
All output text is extracted verbatim from the original files using the
exclusion lists (computed locally) and segment boundaries the model returns.

Folder structure expected:
  root/
    NovelName/
      JP-Output/   1.txt, 2.txt, ...
      EN-Output/   1.txt, 2.txt, ...  (always aligned with JP-Output)

Output:
  root/training_data.jsonl   — all novels combined
  root/training_data_stats.json — per-novel / per-chapter stats

Usage:
  python chapter_segmenter.py --root /path/to/novels [options]

Options:
  --root          Root folder containing novel subdirectories (required)
  --model         Azure OpenAI deployment name (default: deepseek-v4-pro)
  --api-key       Azure OpenAI API key (or set DIGITAL_OCEAN_API_KEY / AZURE_API_KEY / AZURE_OPENAI_API_KEY)
  --endpoint      Azure OpenAI endpoint URL (or set OPENAI_API_URL)
  --max-tokens    Max tokens per training chunk JP+EN combined (default: 3800)
  --concurrency   Parallel API calls (default: 3)
  --resume        Skip chapters already in output JSONL
  --dry-run       Print what would be processed, no API calls
  --novels        Comma-separated list of novel names to process (default: all)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_URL = "https://inference.do-ai.run/v1/chat/completions"

DEFAULT_MODEL = "deepseek-v4-pro"

API_KEY = os.getenv("DIGITAL_OCEAN_API_KEY") or os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")

LLM_MAX_TOKENS = 16000

MAX_RETRIES = 5
RETRY_BACKOFF = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local noise detection (replaces LLM exclusion task)
# ---------------------------------------------------------------------------

# --- Japanese noise patterns ---
_JP_NOISE_PATTERNS: list[re.Pattern] = [
    # Volume / chapter / part headers
    re.compile(r"^第\s*[０-９0-9一二三四五六七八九十百千]+\s*[章話節部巻編幕]", re.UNICODE),
    re.compile(r"^[Ｖｖ][Ｏｏ][Ｌｌ]\s*[０-９0-9]+", re.IGNORECASE | re.UNICODE),
    re.compile(r"^プロローグ|^エピローグ|^幕間|^막간", re.UNICODE),
    # Pure horizontal dividers: *, ─, ＊, ◆, ●, ・, 〇 repeated ≥ 2 (spaces allowed between)
    re.compile(r"^[\s*＊─━◆●・〇×＞＜◇○■□▼▽△▲\-=~～※＝]{2,}\s*$", re.UNICODE),
    # Translator / editor tags embedded in JP text
    re.compile(r"^\s*[\(\[（【]?(TL|訳注|注)[:\s：]", re.IGNORECASE | re.UNICODE),
    # Page numbers: lone digit(s)
    re.compile(r"^\s*[０-９0-9]{1,4}\s*$"),
    # Copyright / publisher lines
    re.compile(r"©|Copyright|\bAll rights reserved\b", re.IGNORECASE),
]

# --- English noise patterns ---
_EN_NOISE_PATTERNS: list[re.Pattern] = [
    # Chapter / volume / part headers
    re.compile(r"^\s*(Chapter|Volume|Part|Vol\.?|Prologue|Epilogue|Interlude)\b", re.IGNORECASE),
    # Translator / editor tags
    re.compile(r"^\s*[\[\(]?\s*(TL|T/N|TN|ED|PR|MTL|Note|Translator[''s]*\s*(Note|Comment))\s*[:\])]", re.IGNORECASE),
    re.compile(r"^\s*\*\s*(TL|T/N|TN|ED|PR|Note)\b", re.IGNORECASE),
    # Pure dividers: ***, ---, ===, ~~~, etc.  (must be the entire line)
    re.compile(r"^\s*([*\-=~_#^])\1{1,}\s*$"),
    # Page numbers: lone integer
    re.compile(r"^\s*\d{1,4}\s*$"),
    # Copyright / publisher lines
    re.compile(r"©|Copyright|\bAll rights reserved\b", re.IGNORECASE),
    # Table-of-contents entries: "Chapter 3 ......... 42"
    re.compile(r"^.{0,60}[.…]{3,}\s*\d+\s*$"),
]


def detect_noise_lines(text: str, lang: str) -> set[int]:
    """
    Return a set of 1-indexed line numbers that are noise (to be excluded).
    `lang` is "jp" or "en". Blank/whitespace-only lines are always excluded.
    """
    patterns = _JP_NOISE_PATTERNS if lang == "jp" else _EN_NOISE_PATTERNS
    excluded: set[int] = set()
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            excluded.add(i)
            continue
        for pat in patterns:
            if pat.search(line):
                excluded.add(i)
                break
    return excluded


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChapterPair:
    novel: str
    chapter_id: str
    jp_text: str
    en_text: str
    # Populated before the LLM call so the prompt only shows clean story lines
    jp_excluded: set[int] = field(default_factory=set)
    en_excluded: set[int] = field(default_factory=set)


@dataclass
class TrainingSample:
    novel: str
    chapter_id: str
    chunk_index: int
    jp_chunk: str
    en_chunk: str
    semantic_unit: str       # "narration" | "dialogue" | "inner_monologue" | "action" | "mixed"
    jp_lines: tuple[int, int]
    en_lines: tuple[int, int]


@dataclass
class ProcessingStats:
    total_chapters: int = 0
    processed_chapters: int = 0
    skipped_chapters: int = 0
    failed_chapters: int = 0
    total_samples: int = 0
    total_jp_tokens_est: int = 0
    total_en_tokens_est: int = 0
    per_novel: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a bilingual segmentation assistant for Japanese light novel translation data.

You will receive a JP chapter and its EN translation. Noise/header lines have already been
removed — only story lines are shown. Each language uses 1-indexed line numbers that reflect
the ORIGINAL file (some numbers may be absent because they were pre-filtered).

─── TASK  ·  Segment the story lines ──────────────────────────────────────────
Partition ALL visible JP lines and ALL visible EN lines into semantically coherent
training segments. Each segment pairs a JP span with its corresponding EN span. Rules:
• Each segment must be a complete meaning unit: a scene, a dialogue exchange, a paragraph
  cluster, or a continuous inner monologue.
• Estimated combined token count per segment ≤ {max_tokens}
  (estimate: 1 token ≈ 1.5 JP chars, 4 EN chars).
• jp_begin/jp_end and en_begin/en_end are line numbers from the ORIGINAL file
  (use the numbers shown in the input, not re-indexed positions).
• Asymmetric boundaries are expected — JP and EN rarely split at the same lines.
• Cover every visible line. No gaps, no overlaps within each side.
• Prefer grouping short lines into a meaningful unit over splitting mid-thought.
• Dialogue exchanges should be one segment.

─── OUTPUT FORMAT ──────────────────────────────────────────────────────────────
Return ONLY a JSON object (no markdown, no prose):
{{
  "segments": [
    {{
      "jp_begin": 5,
      "jp_end": 10,
      "en_begin": 3,
      "en_end": 9,
      "semantic_unit": "narration"
    }}
  ]
}}

• semantic_unit: one of "narration", "dialogue", "inner_monologue", "action", "mixed".
"""

USER_PROMPT_TEMPLATE = """Novel: {novel}
Chapter: {chapter_id}
Max tokens per segment: {max_tokens}

=== JP TEXT (story lines only) ===
{jp_numbered}

=== EN TEXT (story lines only) ===
{en_numbered}

Return only the JSON object."""


def number_lines(text: str, excluded: set[int]) -> str:
    """
    Prefix each non-excluded, non-blank line with its 1-based ORIGINAL index.
    The LLM sees original line numbers so its span references are stable.
    """
    lines = text.splitlines()
    parts = []
    for i, line in enumerate(lines, 1):
        if i not in excluded:
            parts.append(f"{i}: {line}")
    return "\n".join(parts)


def build_prompt(pair: ChapterPair, max_tokens: int) -> tuple[str, str]:
    system = SYSTEM_PROMPT.format(max_tokens=max_tokens)
    user = USER_PROMPT_TEMPLATE.format(
        novel=pair.novel,
        chapter_id=pair.chapter_id,
        max_tokens=max_tokens,
        jp_numbered=number_lines(pair.jp_text, pair.jp_excluded),
        en_numbered=number_lines(pair.en_text, pair.en_excluded),
    )
    return system, user


# ---------------------------------------------------------------------------
# Azure / DO OpenAI async client
# ---------------------------------------------------------------------------

async def call_llm(
    session: aiohttp.ClientSession,
    api_key: str,
    endpoint: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 1.0,
) -> str:
    if not api_key:
        raise RuntimeError(
            "API key not set. "
            "Pass --api-key or export DIGITAL_OCEAN_API_KEY / AZURE_API_KEY / AZURE_OPENAI_API_KEY."
        )

    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": LLM_MAX_TOKENS,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error_body = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                text_body = await resp.text()

                if resp.status in (429, 503):
                    wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                    print(f"Response status: {resp.status}, body: {text_body}, HTTP headers: {resp.headers}")
                    log.warning(
                        f"Rate limited / service unavailable (HTTP {resp.status}). "
                        f"Retry {attempt}/{MAX_RETRIES} in {wait:.0f}s..."
                    )
                    last_error_body = text_body
                    await asyncio.sleep(wait)
                    continue

                if resp.status >= 500:
                    wait = RETRY_BACKOFF * attempt
                    log.warning(
                        f"Server error {resp.status}. "
                        f"Retry {attempt}/{MAX_RETRIES} in {wait:.0f}s..."
                    )
                    last_error_body = text_body
                    await asyncio.sleep(wait)
                    continue

                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {text_body}")

                data = json.loads(text_body)
                return data["choices"][0]["message"]["content"]

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError) as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} retries. "
                    f"Last error: {e}. "
                    f"Last response body: {last_error_body}"
                ) from e

            wait = RETRY_BACKOFF * attempt
            log.warning(f"Request failed: {e}. Retry {attempt}/{MAX_RETRIES} in {wait:.0f}s...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"All {MAX_RETRIES} retries exhausted for LLM call")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_llm_response(raw: str, pair: ChapterPair) -> Optional[dict]:
    """
    Parse LLM JSON response. Handles common failure modes:
    - Markdown code fences
    - Leading/trailing prose
    - Truncated JSON (partial output)
    """
    text = re.sub(r"```(?:json)?", "", raw).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        log.error(f"[{pair.novel} ch{pair.chapter_id}] No JSON object found in response")
        return None

    json_str = text[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        log.error(f"[{pair.novel} ch{pair.chapter_id}] JSON parse error: {e}")
        try:
            last_complete = json_str.rfind("},\n")
            if last_complete > 0:
                repaired = json_str[:last_complete] + "}]\n}"
                data = json.loads(repaired)
                log.warning(f"[{pair.novel} ch{pair.chapter_id}] Used repaired JSON (truncated output)")
            else:
                return None
        except json.JSONDecodeError:
            return None

    if "segments" not in data:
        log.error(f"[{pair.novel} ch{pair.chapter_id}] Missing 'segments' field in response")
        return None

    if not data["segments"]:
        log.error(f"[{pair.novel} ch{pair.chapter_id}] Empty segments list")
        return None

    # Inject the locally-computed exclusion sets so downstream code has one
    # authoritative source of truth regardless of which stage produced them.
    data["jp_excluded"] = pair.jp_excluded
    data["en_excluded"] = pair.en_excluded

    return data


def validate_segments(data: dict, pair: ChapterPair) -> list[dict]:
    """
    Validate segment line ranges against the ORIGINAL texts.
    Excluded lines inside a span are silently skipped during text assembly.
    Returns valid segments; logs warnings for out-of-bounds or overlapping spans.
    """
    jp_total = len(pair.jp_text.splitlines())
    en_total = len(pair.en_text.splitlines())

    jp_excluded: set[int] = data["jp_excluded"]
    en_excluded: set[int] = data["en_excluded"]

    segments = data["segments"]
    valid: list[dict] = []
    seen_jp: set[int] = set()
    seen_en: set[int] = set()

    for i, seg in enumerate(segments):
        jp_b = seg.get("jp_begin", 0)
        jp_e = seg.get("jp_end", 0)
        en_b = seg.get("en_begin", 0)
        en_e = seg.get("en_end", 0)

        if not (1 <= jp_b <= jp_e <= jp_total):
            log.warning(
                f"[{pair.novel} ch{pair.chapter_id}] Seg {i}: "
                f"JP range {jp_b}-{jp_e} out of bounds (total={jp_total}). Skipping."
            )
            continue
        if not (1 <= en_b <= en_e <= en_total):
            log.warning(
                f"[{pair.novel} ch{pair.chapter_id}] Seg {i}: "
                f"EN range {en_b}-{en_e} out of bounds (total={en_total}). Skipping."
            )
            continue

        jp_story = set(range(jp_b, jp_e + 1)) - jp_excluded
        en_story = set(range(en_b, en_e + 1)) - en_excluded

        if jp_story & seen_jp:
            log.warning(
                f"[{pair.novel} ch{pair.chapter_id}] Seg {i}: "
                f"JP story lines overlap with a previous segment. Skipping."
            )
            continue
        if en_story & seen_en:
            log.warning(
                f"[{pair.novel} ch{pair.chapter_id}] Seg {i}: "
                f"EN story lines overlap with a previous segment. Skipping."
            )
            continue

        seen_jp |= jp_story
        seen_en |= en_story
        valid.append(seg)

    all_jp_story = set(range(1, jp_total + 1)) - jp_excluded
    all_en_story = set(range(1, en_total + 1)) - en_excluded
    missing_jp = all_jp_story - seen_jp
    missing_en = all_en_story - seen_en

    if missing_jp:
        log.warning(
            f"[{pair.novel} ch{pair.chapter_id}] "
            f"{len(missing_jp)} JP story lines not covered: {sorted(missing_jp)}"
        )
    if missing_en:
        log.warning(
            f"[{pair.novel} ch{pair.chapter_id}] "
            f"{len(missing_en)} EN story lines not covered: {sorted(missing_en)}"
        )

    return valid


# ---------------------------------------------------------------------------
# Training sample construction
# ---------------------------------------------------------------------------

def _extract_span(
    all_lines: list[str],
    begin: int,
    end: int,
    excluded: set[int],
) -> str:
    return "\n".join(
        all_lines[i - 1]
        for i in range(begin, end + 1)
        if i not in excluded and all_lines[i - 1].strip()
    )


def build_training_samples(
    data: dict,
    segments: list[dict],
    pair: ChapterPair,
) -> list[TrainingSample]:
    jp_lines = pair.jp_text.splitlines()
    en_lines = pair.en_text.splitlines()
    jp_excluded: set[int] = data["jp_excluded"]
    en_excluded: set[int] = data["en_excluded"]
    samples = []

    for i, seg in enumerate(segments):
        jp_chunk = _extract_span(jp_lines, seg["jp_begin"], seg["jp_end"], jp_excluded)
        en_chunk = _extract_span(en_lines, seg["en_begin"], seg["en_end"], en_excluded)

        if not jp_chunk.strip() or not en_chunk.strip():
            log.debug(
                f"[{pair.novel} ch{pair.chapter_id}] Seg {i}: "
                f"empty after exclusion — skipped"
            )
            continue

        samples.append(
            TrainingSample(
                novel=pair.novel,
                chapter_id=pair.chapter_id,
                chunk_index=i,
                jp_chunk=jp_chunk,
                en_chunk=en_chunk,
                semantic_unit=seg.get("semantic_unit", "mixed"),
                jp_lines=(seg["jp_begin"], seg["jp_end"]),
                en_lines=(seg["en_begin"], seg["en_end"]),
            )
        )

    return samples


def sample_to_jsonl(
    sample: TrainingSample,
    jp_excluded_count: int = 0,
    en_excluded_count: int = 0,
) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional Japanese to English light novel translator. "
                    "Translate the Japanese text accurately, preserving the narrative style, "
                    "character voice, honorifics context, and literary tone of the original."
                ),
            },
            {
                "role": "user",
                "content": f"Translate the following Japanese text to English:\n\n{sample.jp_chunk}",
            },
            {
                "role": "assistant",
                "content": sample.en_chunk,
            },
        ],
        "metadata": {
            "novel": sample.novel,
            "chapter": sample.chapter_id,
            "chunk_index": sample.chunk_index,
            "semantic_unit": sample.semantic_unit,
            "jp_lines": list(sample.jp_lines),
            "en_lines": list(sample.en_lines),
            "jp_chars": len(sample.jp_chunk),
            "en_chars": len(sample.en_chunk),
            "jp_excluded_lines": jp_excluded_count,
            "en_excluded_lines": en_excluded_count,
        },
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def discover_novels(root: Path, filter_novels: Optional[list[str]] = None) -> list[Path]:
    novels = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if filter_novels and d.name not in filter_novels:
            continue
        jp_out = d / "JP-Output"
        en_out = d / "EN-Output"
        if jp_out.exists() and en_out.exists():
            novels.append(d)
        else:
            log.debug(f"Skipping {d.name}: missing JP-Output or EN-Output")
    return novels


def discover_chapter_pairs(novel_dir: Path) -> list[ChapterPair]:
    jp_dir = novel_dir / "JP-Output"
    en_dir = novel_dir / "EN-Output"

    jp_files = {
        f.stem: f
        for f in sorted(jp_dir.glob("*.txt"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    }
    en_files = {
        f.stem: f
        for f in sorted(en_dir.glob("*.txt"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    }

    common = sorted(set(jp_files) & set(en_files), key=lambda x: int(x) if x.isdigit() else x)
    only_jp = set(jp_files) - set(en_files)
    only_en = set(en_files) - set(jp_files)

    if only_jp:
        log.warning(f"[{novel_dir.name}] JP-only chapters (no EN match): {sorted(only_jp)}")
    if only_en:
        log.warning(f"[{novel_dir.name}] EN-only chapters (no JP match): {sorted(only_en)}")

    pairs = []
    for cid in common:
        try:
            jp_text = jp_files[cid].read_text(encoding="utf-8", errors="replace").strip()
            en_text = en_files[cid].read_text(encoding="utf-8", errors="replace").strip()
            if not jp_text or not en_text:
                log.warning(f"[{novel_dir.name}] Chapter {cid}: empty file, skipping")
                continue

            # --- Detect noise lines locally before constructing the pair ---
            jp_excluded = detect_noise_lines(jp_text, "jp")
            en_excluded = detect_noise_lines(en_text, "en")

            pairs.append(
                ChapterPair(
                    novel=novel_dir.name,
                    chapter_id=cid,
                    jp_text=jp_text,
                    en_text=en_text,
                    jp_excluded=jp_excluded,
                    en_excluded=en_excluded,
                )
            )
        except Exception as e:
            log.error(f"[{novel_dir.name}] Chapter {cid}: read error: {e}")

    return pairs


def load_existing_keys(output_path: Path) -> set[str]:
    keys = set()
    if not output_path.exists():
        return keys
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                meta = obj.get("metadata", {})
                if meta.get("novel") and meta.get("chapter"):
                    keys.add(f"{meta['novel']}|{meta['chapter']}")
            except json.JSONDecodeError:
                pass
    log.info(f"Resume: found {len(keys)} already-processed chapters in {output_path.name}")
    return keys


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

async def process_chapter(
    pair: ChapterPair,
    session: aiohttp.ClientSession,
    api_key: str,
    endpoint: str,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[list[TrainingSample], dict]:
    async with semaphore:
        log.info(
            f"[{pair.novel}] Processing chapter {pair.chapter_id} "
            f"(pre-filtered {len(pair.jp_excluded)} JP / {len(pair.en_excluded)} EN noise lines) ..."
        )
        system, user = build_prompt(pair, max_tokens)

        try:
            raw = await call_llm(session, api_key, endpoint, model, system, user)
        except Exception as e:
            log.error(f"[{pair.novel}] Ch{pair.chapter_id}: API call failed: {e}")
            return [], {}

        data = parse_llm_response(raw, pair)
        if data is None:
            return [], {}

        segments = validate_segments(data, pair)
        if not segments:
            log.error(f"[{pair.novel}] Ch{pair.chapter_id}: No valid segments after validation")
            return [], {}

        samples = build_training_samples(data, segments, pair)
        log.info(
            f"[{pair.novel}] Ch{pair.chapter_id}: {len(samples)} training samples "
            f"(pre-filtered {len(pair.jp_excluded)} JP / {len(pair.en_excluded)} EN noise lines)"
        )
        return samples, data


async def run(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        log.error(f"Root path does not exist: {root}")
        sys.exit(1)

    output_path = root / "training_data.jsonl"
    stats_path = root / "training_data_stats.json"

    api_key = args.api_key or API_KEY
    endpoint = args.endpoint or OPENAI_API_URL

    if not api_key and not args.dry_run:
        log.error(
            "No API key found. "
            "Pass --api-key or export DIGITAL_OCEAN_API_KEY / AZURE_API_KEY / AZURE_OPENAI_API_KEY."
        )
        sys.exit(1)

    filter_novels = [n.strip() for n in args.novels.split(",")] if args.novels else None
    novels = discover_novels(root, filter_novels)

    if not novels:
        log.error(f"No valid novel directories found in {root}")
        sys.exit(1)

    log.info(f"Found {len(novels)} novel(s): {[n.name for n in novels]}")
    log.info(f"Endpoint : {endpoint}")
    log.info(f"Model    : {args.model}")

    all_pairs: list[ChapterPair] = []
    for novel_dir in novels:
        pairs = discover_chapter_pairs(novel_dir)
        log.info(f"  {novel_dir.name}: {len(pairs)} chapter pairs")
        all_pairs.extend(pairs)

    log.info(f"Total chapter pairs to process: {len(all_pairs)}")

    if args.dry_run:
        log.info("DRY RUN — no API calls will be made.")
        for p in all_pairs:
            jp_story_lines = len(p.jp_text.splitlines()) - len(p.jp_excluded)
            en_story_lines = len(p.en_text.splitlines()) - len(p.en_excluded)
            est_tokens = jp_story_lines * 3 + en_story_lines  # rough chars→tokens
            log.info(
                f"  [{p.novel}] ch{p.chapter_id}: "
                f"~{int(est_tokens):,} input tokens after pre-filtering "
                f"({len(p.jp_excluded)} JP / {len(p.en_excluded)} EN noise lines removed)"
            )
        return

    existing_keys: set[str] = set()
    if args.resume:
        existing_keys = load_existing_keys(output_path)

    pending = [
        p for p in all_pairs
        if f"{p.novel}|{p.chapter_id}" not in existing_keys
    ]
    skipped = len(all_pairs) - len(pending)
    if skipped:
        log.info(f"Resuming: skipping {skipped} already-processed chapters")

    if not pending:
        log.info("All chapters already processed. Nothing to do.")
        return

    stats = ProcessingStats(
        total_chapters=len(pending),
        skipped_chapters=skipped,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)

    mode = "a" if args.resume else "w"

    async with aiohttp.ClientSession(connector=connector) as session:
        with open(output_path, mode, encoding="utf-8") as out_f:

            async def process_and_write(pair: ChapterPair) -> None:
                samples, data = await process_chapter(
                    pair, session, api_key, endpoint,
                    args.model, args.max_tokens, semaphore
                )

                if samples:
                    stats.processed_chapters += 1
                    stats.total_samples += len(samples)

                    novel_stats = stats.per_novel.setdefault(
                        pair.novel,
                        {"chapters": 0, "samples": 0, "failed": 0},
                    )
                    novel_stats["chapters"] += 1
                    novel_stats["samples"] += len(samples)

                    for sample in samples:
                        record = sample_to_jsonl(
                            sample,
                            jp_excluded_count=len(pair.jp_excluded),
                            en_excluded_count=len(pair.en_excluded),
                        )
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats.total_jp_tokens_est += len(sample.jp_chunk) // 2
                        stats.total_en_tokens_est += len(sample.en_chunk) // 4
                else:
                    stats.failed_chapters += 1
                    novel_stats = stats.per_novel.setdefault(
                        pair.novel,
                        {"chapters": 0, "samples": 0, "failed": 0},
                    )
                    novel_stats["failed"] += 1

                done = stats.processed_chapters + stats.failed_chapters
                log.info(
                    f"Progress: {done}/{stats.total_chapters} chapters | "
                    f"{stats.total_samples} samples | "
                    f"{stats.failed_chapters} failed"
                )

            await asyncio.gather(*[process_and_write(p) for p in pending])

    stats_data = {
        "total_chapters": stats.total_chapters,
        "processed_chapters": stats.processed_chapters,
        "skipped_chapters": stats.skipped_chapters,
        "failed_chapters": stats.failed_chapters,
        "total_training_samples": stats.total_samples,
        "est_jp_tokens_in_output": stats.total_jp_tokens_est,
        "est_en_tokens_in_output": stats.total_en_tokens_est,
        "per_novel": stats.per_novel,
        "output_file": str(output_path),
        "model_used": args.model,
        "endpoint": endpoint,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  Chapters processed : {stats.processed_chapters}")
    log.info(f"  Chapters failed    : {stats.failed_chapters}")
    log.info(f"  Training samples   : {stats.total_samples}")
    log.info(f"  Output JSONL       : {output_path}")
    log.info(f"  Stats              : {stats_path}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment aligned JP/EN novel chapters into LLM training JSONL"
    )
    parser.add_argument("--root", required=True, help="Root folder containing novel directories")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=None, help="API key (or set DIGITAL_OCEAN_API_KEY / AZURE_API_KEY)")
    parser.add_argument("--endpoint", default=None, help="API endpoint URL (or set OPENAI_API_URL)")
    parser.add_argument("--max-tokens", type=int, default=3800, help="Max combined JP+EN tokens per chunk (default: 3800)")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel API requests (default: 1)")
    parser.add_argument("--resume", action="store_true", help="Skip chapters already in output JSONL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without API calls")
    parser.add_argument("--novels", default="", help="Comma-separated novel names to process (default: all)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()