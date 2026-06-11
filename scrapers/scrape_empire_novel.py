#!/usr/bin/env python3
"""
Scraper for empirenovel.com — bypasses Cloudflare by connecting to your
real installed browser (Brave/Chrome) via CDP remote debugging.

How it works:
  1. Script launches your real Brave/Chrome with --remote-debugging-port=9222
  2. Playwright connects to it via CDP (no fake Chromium, no bot fingerprint)
  3. CF sees a genuine browser, challenge clears normally
  4. Chapters scraped in parallel tabs, saved as <output_dir>/<n>.txt

Usage:
    python scrape_empirenovel.py <slug> <output-dir> <start> <end> [options]

Example:
    python scrape_empirenovel.py tondemo-skill-de-isekai-hourou-meshi ./chapters 1 250

Options:
    --workers N       Parallel tabs (default: 3)
    --delay N         Seconds between requests per tab (default: 1.5)
    --browser PATH    Explicit path to Brave/Chrome binary (auto-detected if omitted)
    --profile PATH    Browser profile dir to use (default: a fresh temp profile)
"""

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.async_api import async_playwright, Page, BrowserContext

BASE_URL = "https://www.empirenovel.com/novel"
CDP_PORT  = 9222

# Dotted-censorship fix: j.a.pan. -> japan
_DOTTED_RE = re.compile(r'\b([a-zA-Z])(\.[a-zA-Z])+\.?')

def fix_dotted(text: str) -> str:
    return _DOTTED_RE.sub(lambda m: m.group(0).replace(".", ""), text)


# ---------------------------------------------------------------------------
# Browser detection
# ---------------------------------------------------------------------------
_BROWSER_CANDIDATES = [
    # Brave (common install locations on Linux)
    "/usr/bin/brave-browser",
    "/usr/bin/brave",
    "/snap/bin/brave",
    "/opt/brave.com/brave/brave",
    # Chrome
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/opt/google/chrome/chrome",
    # Chromium
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/snap/bin/chromium",
]

def find_browser(hint: str | None = None) -> str:
    if hint:
        if os.path.isfile(hint) and os.access(hint, os.X_OK):
            return hint
        raise FileNotFoundError(f"Specified browser not found or not executable: {hint}")

    # Try PATH first (handles snap shims etc.)
    for name in ("brave-browser", "brave", "google-chrome", "google-chrome-stable",
                 "chromium-browser", "chromium"):
        found = shutil.which(name)
        if found:
            return found

    for path in _BROWSER_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "Could not find Brave or Chrome. Install one or pass --browser /path/to/binary"
    )


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------
async def wait_for_page(page: Page, url: str, cf_timeout: float = 60.0):
    """Navigate and wait until CF challenge clears (title stops being 'Just a moment...')."""
    await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    deadline = time.monotonic() + cf_timeout
    while time.monotonic() < deadline:
        title = await page.title()
        if "just a moment" not in title.lower():
            return
        await asyncio.sleep(1.5)
    raise TimeoutError(
        f"Cloudflare did not clear within {cf_timeout}s.\n"
        "The browser window should be visible — try clicking the checkbox if shown, "
        "or wait for it to auto-solve."
    )


async def extract_chapter(page: Page) -> tuple[str, str]:
    div = page.locator("#read-novel")
    await div.wait_for(state="visible", timeout=20_000)

    title = await page.evaluate("""() => {
        const div = document.getElementById('read-novel');
        if (!div) return '';
        for (const node of div.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                const t = node.textContent.trim();
                if (t) return t;
            }
            if (node.nodeName === 'P') break;
        }
        return '';
    }""")

    paragraphs = await page.evaluate("""() => {
        return Array.from(document.querySelectorAll('#read-novel p'))
            .map(p => p.innerText.trim())
            .filter(t => t.length > 0);
    }""")

    return fix_dotted(title.strip()), fix_dotted("\n\n".join(paragraphs))


# ---------------------------------------------------------------------------
# Per-chapter worker
# ---------------------------------------------------------------------------
async def scrape_chapter(
    context: BrowserContext,
    slug: str,
    ch: int,
    end: int,
    out_path: Path,
    delay: float,
    sem: asyncio.Semaphore,
) -> str:
    if out_path.exists():
        print(f"[{ch}/{end}] SKIP  {out_path.name}")
        return "skip"

    url = f"{BASE_URL}/{slug}/{ch}"

    async with sem:
        page = await context.new_page()
        try:
            await wait_for_page(page, url, cf_timeout=60.0)
            title, body = await extract_chapter(page)
        except Exception as e:
            print(f"[{ch}/{end}] ERROR  {e}", file=sys.stderr)
            await page.close()
            await asyncio.sleep(delay)
            return "error"

        await asyncio.sleep(delay)
        await page.close()

    content = f"{title}\n\n{body}\n" if title else f"{body}\n"
    out_path.write_text(content, encoding="utf-8")
    preview = (title or body[:60].replace("\n", " "))[:70]
    print(f"[{ch}/{end}] SAVED {out_path.name}  —  {preview!r}")
    return "saved"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
async def scrape_novel(
    slug: str,
    output_dir: Path,
    start: int,
    end: int,
    delay: float,
    workers: int,
    browser_bin: str,
    profile_dir: str | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    total = end - start + 1
    counts = {"saved": 0, "skip": 0, "error": 0}

    # Spin up the real browser with CDP enabled
    tmp_profile = None
    if profile_dir is None:
        tmp_profile = tempfile.mkdtemp(prefix="cf_scraper_profile_")
        profile_dir = tmp_profile

    browser_args = [
        browser_bin,
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
    ]

    print(f"Launching browser: {browser_bin}")
    print(f"Profile dir:       {profile_dir}")
    print(f"Scraping {start}–{end} ({total} chapters) | workers={workers} delay={delay}s\n")

    browser_proc = subprocess.Popen(
        browser_args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the browser a moment to start and open the CDP endpoint
    await asyncio.sleep(3)

    try:
        async with async_playwright() as pw:
            # Connect to the already-running real browser
            browser = await pw.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
            contexts = browser.contexts
            context = contexts[0] if contexts else await browser.new_context()

            # Warmup: solve CF on the index page (human-visible browser window)
            print("Warming up — visiting novel index to solve Cloudflare challenge...")
            print("(The browser window is open. If you see a checkbox, click it.)\n")
            warmup = await context.new_page()
            await wait_for_page(warmup, f"{BASE_URL}/{slug}/", cf_timeout=90.0)
            await warmup.close()
            print("Challenge cleared! Starting parallel downloads...\n")

            sem = asyncio.Semaphore(workers)
            tasks = [
                scrape_chapter(context, slug, ch, end, output_dir / f"{ch}.txt", delay, sem)
                for ch in range(start, end + 1)
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                counts[r] += 1

            await browser.close()

    finally:
        browser_proc.terminate()
        try:
            browser_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            browser_proc.kill()

        if tmp_profile:
            shutil.rmtree(tmp_profile, ignore_errors=True)

    print(
        f"\nDone. {counts['saved']} saved, {counts['skip']} skipped, "
        f"{counts['error']} errors (out of {total} chapters)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scrape empirenovel.com via your real Brave/Chrome browser (bypasses Cloudflare)."
    )
    parser.add_argument("slug",       help="Novel slug, e.g. tondemo-skill-de-isekai-hourou-meshi")
    parser.add_argument("output_dir", help="Directory to save chapter .txt files")
    parser.add_argument("start", type=int, help="First chapter index (inclusive)")
    parser.add_argument("end",   type=int, help="Last chapter index (inclusive)")
    parser.add_argument("--workers", type=int,   default=3,   help="Parallel tabs (default: 3)")
    parser.add_argument("--delay",   type=float, default=1.5, help="Delay between requests per tab (default: 1.5s)")
    parser.add_argument("--browser", default=None, help="Explicit path to Brave/Chrome binary")
    parser.add_argument("--profile", default=None, help="Browser profile dir (default: fresh temp dir)")
    args = parser.parse_args()

    if args.start < 1:
        parser.error("start must be >= 1")
    if args.end < args.start:
        parser.error("end must be >= start")

    try:
        browser_bin = find_browser(args.browser)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(scrape_novel(
        slug=args.slug,
        output_dir=Path(args.output_dir),
        start=args.start,
        end=args.end,
        delay=args.delay,
        workers=args.workers,
        browser_bin=browser_bin,
        profile_dir=args.profile,
    ))


if __name__ == "__main__":
    main()