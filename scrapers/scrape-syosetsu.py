#!/usr/bin/env python3
"""
Syosetu Light Novel Chapter Scraper — Python port of scraper.js

Features:
  - Interactive CLI prompts
  - Parallel downloads with configurable concurrency
  - Retry support
  - Clean TXT output
  - Resume friendly (skips existing files)

Usage:
  python scrape-syosetu.py

Install:
  pip install httpx beautifulsoup4
"""

import asyncio
import os
import re
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://ncode.syosetu.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome Safari",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    text = text.replace("\r", "")
    text = text.replace("\u3000", "  ")   # full-width space → two regular spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prompt(message: str, default: str = "") -> str:
    display = f"{message} [{default}]: " if default else f"{message}: "
    try:
        value = input(display).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return value or default


def prompt_int(message: str, default: int, min_val: int = 1, max_val: int = 9999) -> int:
    while True:
        raw = prompt(message, str(default))
        try:
            value = int(raw)
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a valid integer.")


# ── Network ───────────────────────────────────────────────────────────────────

async def fetch_chapter(client: httpx.AsyncClient, url: str, retries: int) -> str:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url, headers=HEADERS, follow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_err = exc
            print(f"❌ Fetch failed ({attempt}/{retries}) -> {url}")
            if attempt < retries:
                await asyncio.sleep(1.5)
    raise last_err  # type: ignore[misc]


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_chapter(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.select_one(".p-novel__title") or soup.select_one("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    subtitle_tag = soup.select_one(".p-novel__subtitle") or soup.select_one("h1")
    subtitle = subtitle_tag.get_text(strip=True) if subtitle_tag else ""

    lines = []
    for p in soup.select(".js-novel-text p"):
        text = p.get_text(strip=True)
        lines.append(text if text else "")

    body = sanitize_text("\n".join(lines))

    return {"title": title, "subtitle": subtitle, "body": body}


# ── Download task ─────────────────────────────────────────────────────────────

async def save_chapter(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    novel_code: str,
    chapter_index: int,
    output_dir: Path,
    retries: int,
):
    async with sem:
        file_path = output_dir / f"{chapter_index}.txt"

        if file_path.exists():
            print(f"⏭  Skipped chapter {chapter_index} (already exists)")
            return

        url = f"{BASE_URL}/{novel_code}/{chapter_index}/"
        print(f"📥 Fetching chapter {chapter_index}")

        try:
            html = await fetch_chapter(client, url, retries)
        except Exception:
            print(f"💀 Failed chapter {chapter_index}")
            return

        chapter = extract_chapter(html)
        file_path.write_text(chapter["body"], encoding="utf-8")
        print(f"✅ Saved {file_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print("\n📚 Syosetu Chapter Scraper\n")

    novel_code  = prompt("Novel code (example: n5864cn)", "n5864cn")
    start_index = prompt_int("Start chapter index", 1, min_val=1)
    end_index   = prompt_int("End chapter index", 81, min_val=1)
    parallel    = prompt_int("Parallel requests", 5, min_val=1, max_val=50)
    retries     = prompt_int("Retries per chapter", 3, min_val=0, max_val=20)
    output_dir  = Path(prompt("Output directory", "./chapters"))

    output_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(parallel)

    async with httpx.AsyncClient(timeout=20) as client:
        tasks = [
            asyncio.create_task(
                save_chapter(client, sem, novel_code, i, output_dir, retries)
            )
            for i in range(start_index, end_index + 1)
        ]
        await asyncio.gather(*tasks)

    print("\n🎉 All done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")