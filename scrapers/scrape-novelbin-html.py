#!/usr/bin/env python3

import asyncio
import json
import os
import sys
from pathlib import Path

import aiofiles
import aiohttp
from bs4 import BeautifulSoup

CONCURRENCY = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
}


def ask_number(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            value = int(raw)
            if min_val <= value <= max_val:
                return value
        print(f"Please enter a number between {min_val} and {max_val}")


async def get_chapter_list(session: aiohttp.ClientSession, novel_slug: str) -> list[dict]:
    url = f"https://novelbin.com/ajax/chapter-archive?novelId={novel_slug}"

    async with session.get(
        url,
        headers={
            **HEADERS,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://novelbin.com/b/{novel_slug}",
        },
        timeout=aiohttp.ClientTimeout(total=30),
    ) as response:
        html = await response.text()

    soup = BeautifulSoup(html, "html.parser")
    chapters = []
    seen = set()

    for el in soup.select("li[data-chapter-item] a"):
        href = el.get("href")
        if not href or href in seen:
            continue
        seen.add(href)
        title = (el.get("title") or el.get_text()).strip()
        chapters.append({"title": title, "url": href})

    if not chapters:
        raise RuntimeError("No chapters found in archive.")

    return chapters


async def fetch_chapter(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(
        url,
        headers=HEADERS,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as response:
        html = await response.text()

    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("#chr-content")

    if not container:
        raise RuntimeError("Could not find #chr-content")

    for tag in container.select("script, style, .js-ad-slot"):
        tag.decompose()

    paragraphs = [p.get_text().strip() for p in container.find_all("p")]
    content = "\n\n".join(p for p in paragraphs if p).strip()

    if not content:
        raise RuntimeError("Chapter content empty")

    return content


async def download_chapter(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    chapters: list[dict],
    index: int,
    output_dir: Path,
    total: int,
    counter: list[int],
) -> None:
    chapter = chapters[index]
    chapter_no = index + 1
    file_path = output_dir / f"{chapter_no}.txt"

    async with semaphore:
        if file_path.exists():
            counter[0] += 1
            print(f"[{counter[0]}/{total}] Skip {chapter_no}")
            return

        try:
            content = await fetch_chapter(session, chapter["url"])
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)
            counter[0] += 1
            print(f"[{counter[0]}/{total}] ✓ {chapter_no}")
        except Exception as err:
            counter[0] += 1
            print(f"[{counter[0]}/{total}] ✗ {chapter_no} :: {err}", file=sys.stderr)


async def main() -> None:
    print("\n==============================")
    print("NovelBin Downloader")
    print("==============================\n")

    novel_slug = input("Novel slug: ").strip()
    output_dir = Path(input("Output folder path: ").strip())

    print("\nFetching chapter archive...")

    async with aiohttp.ClientSession() as session:
        chapters = await get_chapter_list(session, novel_slug)

        print(f"Found {len(chapters)} chapters.\n")
        print(f"1. {chapters[0]['title']}")
        print(f"{len(chapters)}. {chapters[-1]['title']}\n")

        start = ask_number(f"Start chapter [1-{len(chapters)}]: ", 1, len(chapters))
        end = ask_number(f"End chapter [{start}-{len(chapters)}]: ", start, len(chapters))

        output_dir.mkdir(parents=True, exist_ok=True)

        chapters_json = output_dir / "chapters.json"
        async with aiofiles.open(chapters_json, "w", encoding="utf-8") as f:
            await f.write(json.dumps(chapters, indent=2, ensure_ascii=False))

        print(f"\nDownloading chapters {start}-{end}")

        semaphore = asyncio.Semaphore(CONCURRENCY)
        total = end - start + 1
        counter = [0]  # mutable reference for shared counter

        tasks = [
            download_chapter(session, semaphore, chapters, i, output_dir, total, counter)
            for i in range(start - 1, end)
        ]

        await asyncio.gather(*tasks)

    print("\nFinished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)