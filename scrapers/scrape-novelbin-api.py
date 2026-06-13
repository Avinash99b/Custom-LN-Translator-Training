#!/usr/bin/env python3

import sys
import json
import re
import time
import random
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from html.parser import HTMLParser

# ── CLI args ──────────────────────────────────────────────────────────────────

if len(sys.argv) < 3:
    print(
        "Usage: python scrape.py <novel-id> <output-dir> [start] [end]\n"
        "Examples:\n"
        "  python scrape.py im-a-spider-so-what ./output\n"
        "  python scrape.py im-a-spider-so-what ./output 1 100"
    )
    sys.exit(1)

novel_id   = sys.argv[1]
output_dir = Path(sys.argv[2])
start_arg  = int(sys.argv[3]) if len(sys.argv) > 3 else None
end_arg    = int(sys.argv[4]) if len(sys.argv) > 4 else None

CONCURRENCY = 5
MAX_RETRIES = 5

HEADERS_BASE = {
    "accept": "*/*",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def html_to_text(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>",   "", html, flags=re.IGNORECASE)
    html = re.sub(r"<br\s*/?>",   "\n",   html, flags=re.IGNORECASE)
    html = re.sub(r"</p>",        "\n\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</h[1-6]>",   "\n\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>",     "",     html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&") \
               .replace("&lt;", "<").replace("&gt;", ">")
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def retry_delay(attempt: int) -> float:
    return 2 ** attempt + random.random()

# ── Metadata ──────────────────────────────────────────────────────────────────

metadata: dict = {"novel_id": novel_id, "novel_name": None, "chapters": {}}
metadata_path: Path | None = None

def save_metadata():
    if metadata_path is None:
        return
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

# ── Network ───────────────────────────────────────────────────────────────────

async def fetch_chapter_index(session: aiohttp.ClientSession) -> list[str]:
    url = f"https://novelbin.com/ajax/chapter-archive?novelId={novel_id}"
    print(f"Fetching chapter index from {url} ...")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(
                url,
                headers={**HEADERS_BASE, "referer": f"https://novelbin.com/b/{novel_id}",
                         "x-requested-with": "XMLHttpRequest"},
            ) as resp:
                resp.raise_for_status()
                html = await resp.text()

            slugs = re.findall(
                rf'href="https://novelbin\.com/b/{re.escape(novel_id)}/([^"]+)"',
                html,
            )
            if not slugs:
                raise ValueError("No chapter links found in archive response")

            print(f"Found {len(slugs)} chapters in index.")
            return slugs

        except Exception as e:
            last_error = e
            delay = retry_delay(attempt)
            print(f"Index fetch retry {attempt}/{MAX_RETRIES} in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

    raise last_error


async def fetch_chapter_content(
    session: aiohttp.ClientSession, chapter_slug: str, label: str
) -> dict:
    url = (
        f"https://novelbin.com/ajax/chapter-fragment"
        f"?novel_id={novel_id}&chapter_id={chapter_slug}"
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(
                url,
                headers={**HEADERS_BASE,
                         "accept": "application/json",
                         "referer": f"https://novelbin.com/b/{novel_id}/{chapter_slug}"},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            if not data.get("success"):
                raise ValueError("API returned success=false")
            return data

        except Exception as e:
            last_error = e
            delay = retry_delay(attempt)
            print(f"[{label}] Retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
            await asyncio.sleep(delay)

    raise last_error

# ── Chapter download ──────────────────────────────────────────────────────────

async def download_chapter(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    idx: int,
    chapter_slug: str,
):
    label     = f"{idx} ({chapter_slug})"
    file_path = output_dir / f"{idx}.txt"

    async with sem:
        if file_path.exists():
            print(f"[{label}] Skipped")
            return

        try:
            data    = await fetch_chapter_content(session, chapter_slug, label)
            chapter = data["chapter"]
            text    = html_to_text(chapter["content_html"])

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(text + "\n")

            metadata["chapters"][str(idx)] = {
                "chapter_id":   chapter["chapter_id"],
                "chapter_name": chapter["chapter_name"],
                "slug":         chapter_slug,
                "url":          chapter["url"],
            }
            save_metadata()
            print(f"[{label}] Saved")

        except Exception as e:
            print(f"[{label}] Failed: {e}", file=sys.stderr)

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    global metadata, metadata_path

    async with aiohttp.ClientSession() as session:
        all_slugs = await fetch_chapter_index(session)

    start = start_arg or 1
    end   = end_arg   or len(all_slugs)

    if not (1 <= start <= end <= len(all_slugs)):
        print(
            f"Range {start}-{end} is out of bounds "
            f"(index has {len(all_slugs)} chapters)",
            file=sys.stderr,
        )
        sys.exit(1)

    slug_slice = all_slugs[start - 1 : end]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    metadata["novel_id"] = novel_id
    save_metadata()

    print(f"Downloading chapters {start}–{end} into {output_dir} ...")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_chapter(session, sem, start + i, slug)
            for i, slug in enumerate(slug_slice)
        ]
        await asyncio.gather(*tasks)

    save_metadata()
    print("Finished.")


if __name__ == "__main__":
    asyncio.run(main())