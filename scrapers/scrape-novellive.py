#!/usr/bin/env python3
"""
Novel Live scraper — Python port of scrape-novellive.js
Usage: python scrape-novellive.py <novel-slug> <start> <end>
Example: python scrape-novellive.py mushoku-tensei-novel 1 500
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
CONCURRENCY = 5
MAX_RETRIES = 5
BASE_URL = "https://novellive.app"

HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.8",
    "priority": "u=0, i",
    "referer": "https://novellive.app/book/mushoku-tensei-novel/chapter-2",
    "sec-ch-ua": '"Chromium";v="148", "Brave";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-arch": '"x86"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-full-version-list": '"Chromium";v="148.0.0.0", "Brave";v="148.0.0.0", "Not/A)Brand";v="99.0.0.0"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": '"Linux"',
    "sec-ch-ua-platform-version": '""',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "sec-gpc": "1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Cookie": "_csrf=LOlngVT8UX-z1-UdXdbQBjz5; cf_clearance=VTBiZOTOIG_HlqD38GB43p0NftV1IzyjEQFbjzS.V6w-1780891923-1.2.1.1-L3zPJ2uML6qgCTXc7rhQorDZaKF62PIGzjdcgmwApyPnnnlRcSf1g9Bd6MP9ahxN.vAjnNW2j9GXfkaX5QHuIoNZNHwiRTxN4q.KBK0m956vIF8I42bWWRazZauVAOt4a9eY14OLMusvs.DcjWMqz75PFcmMdCwyyWw42X6v6HnwnBD_DUTZ3gTlBUMXVRSS.vrBuNiJWwKLKvJWNBHYsJuTvsLPxVbn4cIqY0y.oOO0vh_Xo_qrbp9Mlw1RZImNLNgiSfT8.wMMY5OPlMhpzATWYXLxKY_XIJX.I2.7x_YBWUEpA9FC2YA0M0PYGcoDRAZybkfiRdeuSTpQKH85xW6Y16z8snGk0j7ZKg2OP_JMPNPKyzNLW4eG8jv7xHajP1bJd5UgC.uOd_TI7Zf4FJfjoGhrjolrvjnEfDs3H2k",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", s).strip()


def html_to_text(html: str) -> str:
    """Mirrors the JS htmlToText function."""
    text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_chapter_data(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    novel_title_tag = soup.select_one("h1.tit a")
    novel_title = novel_title_tag.get_text(strip=True) if novel_title_tag else ""

    chapter_title_tag = soup.select_one("span.chapter")
    chapter_title = chapter_title_tag.get_text(strip=True) if chapter_title_tag else ""

    content_div = soup.select_one(".txt")
    content_html = str(content_div) if content_div else ""

    return {
        "novel_name": novel_title,
        "chapter_name": chapter_title,
        "content_html": content_html,
    }


# ── State (shared across coroutines) ─────────────────────────────────────────

class Scraper:
    def __init__(self, novel_slug: str, start: int, end: int):
        self.novel_slug = novel_slug
        self.start = start
        self.end = end

        self.novel_name: str | None = None
        self.output_dir: Path | None = None
        self.metadata_path: Path | None = None
        self.metadata: dict = {
            "novel_id": novel_slug,
            "novel_name": None,
            "chapters": {},
        }
        self._metadata_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(CONCURRENCY)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def save_metadata(self):
        if not self.metadata_path:
            return
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Network ──────────────────────────────────────────────────────────────

    async def fetch_with_retry(self, client: httpx.AsyncClient, chapter_num: int) -> str:
        url = f"{BASE_URL}/book/{self.novel_slug}/chapter-{chapter_num}"
        last_err: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.get(url, headers=HEADERS, follow_redirects=True)
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_err = exc
                delay = (2 ** attempt) + (os.urandom(1)[0] / 255)  # ~exponential + jitter
                print(f"[{chapter_num}] Retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
                await asyncio.sleep(delay)

        raise last_err  # type: ignore[misc]

    # ── Core logic ───────────────────────────────────────────────────────────

    async def initialize(self, client: httpx.AsyncClient):
        html = await self.fetch_with_retry(client, self.start)
        data = extract_chapter_data(html)

        self.novel_name = sanitize(data["novel_name"] or self.novel_slug)
        self.output_dir = Path.cwd() / self.novel_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.output_dir / "metadata.json"
        if self.metadata_path.exists():
            try:
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        self.metadata["novel_id"] = self.novel_slug
        self.metadata["novel_name"] = data["novel_name"] or self.novel_slug
        self.save_metadata()

        print(f"Output directory: {self.output_dir}")

    async def download_chapter(self, client: httpx.AsyncClient, chapter_num: int):
        async with self._sem:
            try:
                file_path = self.output_dir / f"{chapter_num}.txt"  # type: ignore[operator]

                if file_path.exists():
                    print(f"[{chapter_num}] Skipped (already exists)")
                    return

                html = await self.fetch_with_retry(client, chapter_num)
                data = extract_chapter_data(html)

                text = html_to_text(data["content_html"]).strip()
                file_path.write_text(text, encoding="utf-8")

                async with self._metadata_lock:
                    self.metadata.setdefault("chapters", {})[str(chapter_num)] = {
                        "chapter_name": data["chapter_name"],
                        "url": f"{BASE_URL}/book/{self.novel_slug}/chapter-{chapter_num}",
                    }
                    self.save_metadata()

                print(f"[{chapter_num}] Saved: {data['chapter_name']}")

            except Exception as exc:
                print(f"[{chapter_num}] Failed: {exc}")

    async def run(self):
        async with httpx.AsyncClient(timeout=30) as client:
            await self.initialize(client)

            tasks = [
                asyncio.create_task(self.download_chapter(client, i))
                for i in range(self.start, self.end + 1)
            ]
            await asyncio.gather(*tasks)

        self.save_metadata()
        print("✅ Finished scraping!")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python scrape-novellive.py <novel-slug> <start> <end>\n"
            "Example:\n"
            "  python scrape-novellive.py mushoku-tensei-novel 1 500"
        )
        sys.exit(1)

    _, novel_slug, start_arg, end_arg = sys.argv
    start = int(start_arg)
    end = int(end_arg)

    scraper = Scraper(novel_slug, start, end)
    asyncio.run(scraper.run())


if __name__ == "__main__":
    main()