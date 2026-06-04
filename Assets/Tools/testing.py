import asyncio
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        Progress,
        SpinnerColumn,
    )
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None  # type: ignore


# =========================
# CONFIG
# =========================

ROOT = Path("/home/avinash/Projects/Custom-LN-Translator-Training/Assets/Novels/Arifureta")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHAPTERS = 50
CONCURRENCY = 20
OPENING_CHARS = 2500
RETRIES = 3
RETRY_BASE_SLEEP = 1.5
TIMEOUT_SECONDS = 120

# Your working Azure OpenAI-compatible endpoint.
BASE_URL = "https://bathu-mpkwvf7h-eastus2.cognitiveservices.azure.com/openai/v1/"

GPT4O_KEY = "[REACTED_GOTCHA]"
OSS_KEY = "[REACTED_GOTCHA]"
GPT4O_DEPLOYMENT = "gpt-4o-mini"
OSS_DEPLOYMENT = "gpt-oss-120b"

GPT4O_OUTPUT = RESULTS_DIR / "gpt-4o-mini.jsonl"
OSS_OUTPUT = RESULTS_DIR / "gpt-oss-120b.jsonl"
SUMMARY_OUTPUT = RESULTS_DIR / "benchmark_summary.json"

SYSTEM_PROMPT = (
    "You are a bilingual light novel chapter alignment specialist.\n\n"
    "Your task is to determine whether two chapters are intended to be the SAME chapter.\n\n"
    "IMPORTANT:\n"
    "Focus primarily on:\n"
    "- chapter title\n"
    "- chapter number (if present)\n"
    "- opening scene\n"
    "- opening setting\n"
    "- opening characters present\n"
    "- opening dialogue\n"
    "- first major event at the beginning of the chapter\n\n"
    "DO NOT compare the entire chapter.\n"
    "DO NOT reject chapters because later events differ.\n"
    "DO NOT reject chapters because one version contains extra content.\n"
    "DO NOT reject chapters because one version is longer.\n"
    "DO not trust the chapter numbers inside the text, as they may be wrong or missing. Use them as a hint but not a deciding factor.\n"
    "DO NOT reject chapters because translator notes, status screens, afterwords, or bonus text are present.\n"
    "DO NOT reject chapters because scenes are expanded, condensed, reordered slightly, or localized.\n\n"
    "If the chapter titles match or are clear translations of each other, and the opening portion of both chapters starts from the same scene, treat them as the SAME chapter even if later content differs.\n\n"
    "Only mark chapters as different if:\n"
    "- the titles clearly refer to different chapters\n"
    "- the opening scene is clearly different\n"
    "- the opening characters and setting do not match\n"
    "- one chapter obviously begins at a different point in the story\n\n"
    "DO NOT MARK AS DIFFERENT just because the chapter numbers are different or missing. Use the chapter numbers as a weak signal, but rely more on the content of the opening scene and title.\n\n"
    "Return ONLY a JSON object.\n"
    "No markdown.\n"
    "No explanation outside the JSON.\n\n"
    "Example output:\n"
    '{"same_story": true, "confidence": 0.97, "reason": "matching title and opening scene", "major_drift": false}\n'
    "or\n"
    '{"same_story": false, "confidence": 0.08, "reason": "different title and opening scene", "major_drift": true}\n\n'
    "Fields:\n"
    "  same_story  : boolean\n"
    "  confidence  : float 0.0-1.0\n"
    "  reason      : short string\n"
    "  major_drift : boolean"
)


# =========================
# MODELS
# =========================

@dataclass
class ModelResult:
    chapter: str
    jp_file: str
    en_file: str
    same_story: Optional[bool]
    confidence: Optional[float]
    reason: str
    major_drift: Optional[bool]
    raw_response: str
    parsed_json: Optional[Dict[str, Any]]
    ok: bool
    error: Optional[str]
    latency_sec: float
    input_chars: int
    output_chars: int
    attempt: int
    model: str


# =========================
# CLIENTS
# =========================

# Both deployments are reachable through the same Azure OpenAI-compatible base URL.
gpt4_client = OpenAI(api_key=GPT4O_KEY, base_url=BASE_URL)
oss_client = OpenAI(api_key=OSS_KEY, base_url=BASE_URL)


# =========================
# UTILITIES
# =========================

console = Console() if Console else None
sem = asyncio.Semaphore(CONCURRENCY)
write_lock = asyncio.Lock()
start_time = time.time()
completed = 0
failures = {GPT4O_DEPLOYMENT: 0, OSS_DEPLOYMENT: 0}


def numeric_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def load_chapters() -> List[Tuple[Path, Path]]:
    jp_files = sorted((ROOT / "JP").glob("*.txt"), key=numeric_sort_key)
    en_files = sorted((ROOT / "EN").glob("*.txt"), key=numeric_sort_key)

    en_map = {p.name: p for p in en_files}
    pairs: List[Tuple[Path, Path]] = []

    for jp in jp_files:
        en = en_map.get(jp.name)
        if en is not None:
            pairs.append((jp, en))

    return pairs[:MAX_CHAPTERS]


def extract_opening(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:OPENING_CHARS]


def ensure_json_text(s: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from a model response with a bit of repair."""
    if not s:
        return None

    s = s.strip()

    # Fast path.
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Try to extract the first JSON object from the text.
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None

    candidate = m.group(0)
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        return None

    return None


def normalize_result(model_name: str, chapter: str, jp_file: Path, en_file: Path, raw: str, latency: float, attempt: int, error: Optional[str] = None) -> ModelResult:
    parsed = ensure_json_text(raw)

    same_story = None
    confidence = None
    reason = ""
    major_drift = None
    ok = False

    if parsed is not None:
        same_story = parsed.get("same_story")
        confidence = parsed.get("confidence")
        reason = str(parsed.get("reason", "")).strip()
        major_drift = parsed.get("major_drift")
        ok = isinstance(same_story, bool)

    return ModelResult(
        chapter=chapter,
        jp_file=str(jp_file),
        en_file=str(en_file),
        same_story=same_story,
        confidence=confidence,
        reason=reason,
        major_drift=major_drift,
        raw_response=raw,
        parsed_json=parsed,
        ok=ok,
        error=error,
        latency_sec=latency,
        input_chars=0,
        output_chars=len(raw or ""),
        attempt=attempt,
        model=model_name,
    )


async def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    async with write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_messages(jp_text: str, en_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "JP CHAPTER:\n\n"
                f"{jp_text}\n\n"
                "EN CHAPTER:\n\n"
                f"{en_text}"
            ),
        },
    ]


async def call_model(client: OpenAI, deployment: str, model_name: str, jp_text: str, en_text: str) -> ModelResult:
    chapter = ""
    jp_file = Path(".")
    en_file = Path(".")

    input_chars = len(jp_text) + len(en_text)

    last_error: Optional[str] = None
    for attempt in range(1, RETRIES + 1):
        try:
            start = time.time()

            def _sync_call():
                return client.chat.completions.create(
                    model=deployment,
                    temperature=0,
                    max_tokens=300,
                    messages=build_messages(jp_text, en_text),
                )

            response = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=TIMEOUT_SECONDS)
            latency = time.time() - start
            raw = response.choices[0].message.content or ""
            result = normalize_result(model_name, chapter, jp_file, en_file, raw, latency, attempt)
            result.input_chars = input_chars
            return result

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < RETRIES:
                await asyncio.sleep(RETRY_BASE_SLEEP * attempt)
                continue

            return ModelResult(
                chapter=chapter,
                jp_file=str(jp_file),
                en_file=str(en_file),
                same_story=None,
                confidence=None,
                reason="",
                major_drift=None,
                raw_response="",
                parsed_json=None,
                ok=False,
                error=last_error,
                latency_sec=0.0,
                input_chars=input_chars,
                output_chars=0,
                attempt=attempt,
                model=model_name,
            )

    return ModelResult(
        chapter=chapter,
        jp_file=str(jp_file),
        en_file=str(en_file),
        same_story=None,
        confidence=None,
        reason="",
        major_drift=None,
        raw_response="",
        parsed_json=None,
        ok=False,
        error=last_error,
        latency_sec=0.0,
        input_chars=input_chars,
        output_chars=0,
        attempt=RETRIES,
        model=model_name,
    )


async def call_gpt4(jp_text: str, en_text: str) -> ModelResult:
    return await call_model(gpt4_client, GPT4O_DEPLOYMENT, GPT4O_DEPLOYMENT, jp_text, en_text)


async def call_oss(jp_text: str, en_text: str) -> ModelResult:
    return await call_model(oss_client, OSS_DEPLOYMENT, OSS_DEPLOYMENT, jp_text, en_text)


def score_result(result: ModelResult) -> str:
    if result.ok and isinstance(result.same_story, bool) and isinstance(result.confidence, (int, float)):
        emoji = "✅" if result.same_story else "❌"
        return f"{emoji} {result.same_story}  conf={result.confidence:.2f}"
    if result.error:
        return f"⚠️  ERROR: {result.error}"
    return "⚠️  invalid JSON"


def build_live_table(stats: Dict[str, Any]) -> Table:
    table = Table(title="Light Novel Chapter Alignment Benchmark", expand=True)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("Progress", f"{stats['done']}/{stats['total']}")
    table.add_row("Elapsed", f"{stats['elapsed']:.1f}s")
    table.add_row("Rate", f"{stats['rate']:.2f} chapters/s")
    table.add_row("ETA", stats['eta'])
    table.add_row("GPT-4o-mini OK", str(stats['gpt_ok']))
    table.add_row("GPT-OSS-120B OK", str(stats['oss_ok']))
    table.add_row("GPT-4o-mini Fail", str(stats['gpt_fail']))
    table.add_row("GPT-OSS-120B Fail", str(stats['oss_fail']))
    table.add_row("Disagreements", str(stats['disagreements']))
    return table


async def process_pair(idx: int, jp_path: Path, en_path: Path, progress: Optional[Progress] = None, task_id: Optional[int] = None) -> Dict[str, Any]:
    global completed

    async with sem:
        chapter_id = jp_path.stem
        jp_text = extract_opening(jp_path)
        en_text = extract_opening(en_path)

        if console:
            console.log(f"[bold]Chapter {chapter_id}[/bold] started")

        gpt_task = asyncio.create_task(call_gpt4(jp_text, en_text))
        oss_task = asyncio.create_task(call_oss(jp_text, en_text))

        gpt_result, oss_result = await asyncio.gather(gpt_task, oss_task)

        gpt_result.chapter = chapter_id
        gpt_result.jp_file = str(jp_path)
        gpt_result.en_file = str(en_path)
        gpt_result.input_chars = len(jp_text) + len(en_text)

        oss_result.chapter = chapter_id
        oss_result.jp_file = str(jp_path)
        oss_result.en_file = str(en_path)
        oss_result.input_chars = len(jp_text) + len(en_text)

        await append_jsonl(GPT4O_OUTPUT, asdict(gpt_result))
        await append_jsonl(OSS_OUTPUT, asdict(oss_result))

        completed += 1
        if progress is not None and task_id is not None:
            progress.advance(task_id)

        if console:
            console.log(
                f"[green]Chapter {chapter_id} done[/green] | "
                f"GPT: {score_result(gpt_result)} | "
                f"OSS: {score_result(oss_result)}"
            )

        return {
            "chapter": chapter_id,
            "jp_file": str(jp_path),
            "en_file": str(en_path),
            "gpt4o_mini": asdict(gpt_result),
            "gpt_oss_120b": asdict(oss_result),
            "disagree": bool(
                isinstance(gpt_result.same_story, bool)
                and isinstance(oss_result.same_story, bool)
                and gpt_result.same_story != oss_result.same_story
            ),
        }


async def main() -> None:
    pairs = load_chapters()
    total = len(pairs)

    # Clear previous outputs for a clean benchmark run.
    GPT4O_OUTPUT.write_text("", encoding="utf-8")
    OSS_OUTPUT.write_text("", encoding="utf-8")

    if console:
        console.rule("[bold cyan]Benchmark Setup")
        console.print(f"Root: {ROOT}")
        console.print(f"Chapters: {total}")
        console.print(f"Concurrency: {CONCURRENCY}")
        console.print(f"Opening chars per side: {OPENING_CHARS}")
        console.print(f"GPT model: {GPT4O_DEPLOYMENT}")
        console.print(f"OSS model: {OSS_DEPLOYMENT}")
        console.print(f"Output: {RESULTS_DIR.resolve()}")
        console.rule()

    results: List[Dict[str, Any]] = []
    progress = None
    task_id = None

    if console:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )

    async def run_all() -> None:
        nonlocal task_id, results
        if progress is not None:
            progress.start()
            task_id = progress.add_task("Processing chapters", total=total)

        tasks = [
            asyncio.create_task(process_pair(i + 1, jp, en, progress, task_id))
            for i, (jp, en) in enumerate(pairs)
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                item = await coro
                results.append(item)
            except Exception as e:
                if console:
                    console.log(f"[red]Unhandled task failure:[/red] {e}")

        if progress is not None:
            progress.stop()

    await run_all()

    # Summary stats.
    gpt_ok = sum(1 for r in results if r["gpt4o_mini"]["ok"])
    oss_ok = sum(1 for r in results if r["gpt_oss_120b"]["ok"])
    gpt_fail = total - gpt_ok
    oss_fail = total - oss_ok
    disagreements = sum(1 for r in results if r["disagree"])
    gpt_lat = [r["gpt4o_mini"]["latency_sec"] for r in results if r["gpt4o_mini"]["latency_sec"] > 0]
    oss_lat = [r["gpt_oss_120b"]["latency_sec"] for r in results if r["gpt_oss_120b"]["latency_sec"] > 0]

    elapsed = time.time() - start_time
    summary = {
        "root": str(ROOT),
        "total_chapters": total,
        "completed": len(results),
        "elapsed_sec": elapsed,
        "chapters_per_sec": (len(results) / elapsed) if elapsed > 0 else None,
        "concurrency": CONCURRENCY,
        "opening_chars": OPENING_CHARS,
        "models": [GPT4O_DEPLOYMENT, OSS_DEPLOYMENT],
        "gpt4o_mini": {
            "ok": gpt_ok,
            "fail": gpt_fail,
            "avg_latency_sec": (sum(gpt_lat) / len(gpt_lat)) if gpt_lat else None,
            "median_latency_sec": sorted(gpt_lat)[len(gpt_lat) // 2] if gpt_lat else None,
        },
        "gpt_oss_120b": {
            "ok": oss_ok,
            "fail": oss_fail,
            "avg_latency_sec": (sum(oss_lat) / len(oss_lat)) if oss_lat else None,
            "median_latency_sec": sorted(oss_lat)[len(oss_lat) // 2] if oss_lat else None,
        },
        "disagreements": disagreements,
        "output_files": {
            "gpt4o_mini": str(GPT4O_OUTPUT),
            "gpt_oss_120b": str(OSS_OUTPUT),
        },
    }

    SUMMARY_OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if console:
        stats = {
            "done": len(results),
            "total": total,
            "elapsed": elapsed,
            "rate": (len(results) / elapsed) if elapsed > 0 else 0.0,
            "eta": "0s",
            "gpt_ok": gpt_ok,
            "oss_ok": oss_ok,
            "gpt_fail": gpt_fail,
            "oss_fail": oss_fail,
            "disagreements": disagreements,
        }

        console.rule("[bold green]Benchmark Summary")
        console.print(Panel.fit(build_live_table(stats), border_style="green"))
        console.print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
