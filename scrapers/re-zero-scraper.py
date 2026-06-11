#!/usr/bin/env python3
"""
Scraper for witchculttranslation.com Re:Zero chapters.
Starts from the prologue and follows "Next Post" links until exhausted.
Saves each chapter as a .txt file and a combined output.
"""

import os
import re
import time
import json
import logging
import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START_URL = "https://witchculttranslation.com/2021/05/19/prologue-waste-heat-of-the-beginning/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
    "Referer": "https://witchculttranslation.com/table-of-content/",
    "Upgrade-Insecure-Requests": "1",
    "sec-ch-ua": '"Chromium";v="148", "Brave";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
}

REQUEST_DELAY   = 0.5   # seconds between requests (be polite)
MAX_RETRIES     = 3
RETRY_BACKOFF   = 5     # seconds to wait before retry

OUTPUT_DIR      = Path("ReZero_JP")
COMBINED_FILE   = Path("rezero_all_chapters.txt")
STATE_FILE      = Path("scrape_state.json")  # for resuming

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch(url: str, session: requests.Session) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    log.error("Giving up on %s after %d attempts.", url, MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def get_title(soup: BeautifulSoup) -> str:
    tag = soup.find("h1", class_="entry-title")
    if tag:
        return tag.get_text(strip=True)
    tag = soup.find("title")
    return tag.get_text(strip=True) if tag else "Untitled"


def get_content(soup: BeautifulSoup) -> str:
    """
    Extract readable story text from .entry-content.
    Strips translator notes headers, decorative separators, image tags,
    and meta lines like 'ALL RIGHTS BELONG TO …'.
    """
    content_div = soup.find("div", class_="entry-content")
    if not content_div:
        return ""

    paragraphs = []
    for tag in content_div.find_all(["p", "li"]):
        text = tag.get_text(separator=" ", strip=True)

        # Skip empty
        if not text:
            continue
        # Skip decorative separator lines (※ repeated)
        if re.fullmatch(r"[※\s]+", text):
            continue
        # Skip ALL-CAPS rights / source lines
        if re.search(r"ALL RIGHTS BELONG|JAPANESE WEB NOVEL SOURCE|FREE.*JAPANESE", text):
            continue
        # Skip translator credit lines
        if re.search(r"Translated By|SNUserTL|SNUser Translations", text, re.I):
            continue
        # Skip bare URLs
        if re.fullmatch(r"https?://\S+", text):
            continue

        paragraphs.append(text)

    return "\n\n".join(paragraphs)


def get_next_url(soup: BeautifulSoup, current_url: str) -> str | None:
    """
    Find the 'Next Post' link in the post-navigation block.
    Returns the absolute URL, or None if this is the last chapter.
    """
    nav = soup.find("nav", class_="post-navigation")
    if not nav:
        return None
    next_div = nav.find("div", class_="nav-next")
    if not next_div:
        return None
    a_tag = next_div.find("a", href=True)
    if not a_tag:
        return None
    return urljoin(current_url, a_tag["href"])


# ---------------------------------------------------------------------------
# State / resume support
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open() as f:
            return json.load(f)
    return {"visited": [], "next_url": START_URL}


def save_state(state: dict):
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Main scraping loop
# ---------------------------------------------------------------------------

def scrape(resume: bool = True, combine: bool = True):
    OUTPUT_DIR.mkdir(exist_ok=True)

    state = load_state() if resume else {"visited": [], "next_url": START_URL}
    visited: set[str] = set(state["visited"])
    current_url: str | None = state["next_url"]

    session = requests.Session()

    chapter_index = len(visited) + 1
    log.info("Starting scrape. Already visited %d chapters.", len(visited))
    if current_url:
        log.info("Resuming from: %s", current_url)

    while current_url:
        if current_url in visited:
            log.warning("Loop detected — already visited %s. Stopping.", current_url)
            break

        log.info("[Chapter %d] Fetching: %s", chapter_index, current_url)
        soup = fetch(current_url, session)

        if soup is None:
            log.error("Could not fetch chapter %d. Saving state and stopping.", chapter_index)
            state["next_url"] = current_url
            save_state(state)
            break

        title   = get_title(soup)
        content = get_content(soup)
        next_url = get_next_url(soup, current_url)

        # Save individual chapter file
        safe_title = re.sub(r'[^\w\s\-]', '', title).strip().replace(" ", "_")[:80]
        chapter_filename = OUTPUT_DIR / f"{chapter_index}.txt"
        with chapter_filename.open("w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            f.write(content)
            f.write("\n")

        log.info("  Saved: %s  (%d chars)", chapter_filename.name, len(content))

        # Update state
        visited.add(current_url)
        state["visited"] = list(visited)
        state["next_url"] = next_url
        save_state(state)

        current_url = next_url
        chapter_index += 1

        if current_url:
            time.sleep(REQUEST_DELAY)

    log.info("Scraping complete. %d chapters collected.", chapter_index - 1)

    if combine:
        combine_chapters()


def combine_chapters():
    """Merge all individual chapter files into one big text file."""
    files = sorted(OUTPUT_DIR.glob("*.txt"))
    if not files:
        log.warning("No chapter files found to combine.")
        return

    log.info("Combining %d chapters into %s …", len(files), COMBINED_FILE)
    with COMBINED_FILE.open("w", encoding="utf-8") as out:
        for path in files:
            with path.open(encoding="utf-8") as f:
                out.write(f.read())
            out.write("\n\n" + "─" * 80 + "\n\n")

    log.info("Combined file written: %s (%.1f MB)", COMBINED_FILE, COMBINED_FILE.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Re:Zero chapters from witchculttranslation.com"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any saved state",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip scraping; only merge already-downloaded chapter files",
    )
    parser.add_argument(
        "--start-url",
        default=None,
        help="Override the starting URL (useful for scraping a specific arc)",
    )
    args = parser.parse_args()

    if args.start_url:
        START_URL = args.start_url

    if args.combine_only:
        combine_chapters()
    else:
        scrape(resume=not args.no_resume)