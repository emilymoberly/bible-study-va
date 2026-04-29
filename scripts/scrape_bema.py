"""
Scrape BEMA Discipleship podcast transcripts.

Why this script exists
----------------------
BEMA episodes have transcripts published as Google Docs linked from each
episode. We need them as local plain-text files so the BEMA tool can do
TF-IDF retrieval over them without hitting the network at query time.

How it works (v2 — 2026-04)
---------------------------
The static episode archive page (bemadiscipleship.com/episodes) only
exposes ~16 episodes in its initial HTML; the rest are JS-paginated. So
instead we use BEMA's official **JSON feed**:

    https://bema.fireside.fm/json

…which lists *all* episodes (currently 507). For each episode we:

  1. Look in the show-notes HTML inside the feed itself for a
     `docs.google.com/document/...` link. This works for ~50% of episodes
     and saves a network round-trip.
  2. Otherwise, fetch the episode page (e.g. https://www.bemadiscipleship.com/257)
     and look for the "Transcript for BEMA N" link there.
  3. If a transcript URL is found, fetch the published Google Doc HTML,
     strip Google's chrome and Boilerplate, and save the body as
     `data/bema/transcripts/<number>.txt`.
  4. Save all metadata to `data/bema/episodes.json`.

Be polite
---------
- Custom User-Agent
- Small delay between requests
- --max-episodes flag for testing
- Skips episodes whose transcript file already exists
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

FEED_URL = "https://bema.fireside.fm/json"
HEADERS = {
    "User-Agent": (
        "BibleStudyVA-ClassProject/0.1 "
        "(educational use; contact: student@example.edu)"
    )
}
SLEEP_SECONDS = 0.4  # polite-ish

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = ROOT / "data" / "bema" / "transcripts"
EPISODES_JSON = ROOT / "data" / "bema" / "episodes.json"

GOOGLE_DOC_RE = re.compile(
    r"https://docs\.google\.com/document/d/(?:e/)?[A-Za-z0-9_\-]+(?:/pub)?",
    re.IGNORECASE,
)
# Episode numbers in URLs may be negative (-1, 0, 1, ..., 502)
# or have a letter suffix (e.g. "12b"). We extract whatever follows the
# final "/" of the URL.
URL_NUMBER_RE = re.compile(r"/(-?\d+[a-zA-Z]?)/?$")


@dataclass
class Episode:
    number: str
    title: str
    url: str
    date_published: str = ""
    transcript_url: str | None = None
    transcript_path: str | None = None
    note: str = ""  # short status: "ok", "no-transcript", "skipped", etc.


# ---------------------------------------------------------------------------
# Feed → episodes
# ---------------------------------------------------------------------------

def fetch(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def episode_number_from_url(url: str) -> str | None:
    m = URL_NUMBER_RE.search(url.rstrip("/"))
    return m.group(1) if m else None


def load_episodes_from_feed() -> list[Episode]:
    print(f"[bema] fetching feed: {FEED_URL}")
    data = fetch(FEED_URL).json()
    items = data.get("items", [])
    episodes: list[Episode] = []
    for item in items:
        url = item.get("url", "")
        number = episode_number_from_url(url)
        if number is None:
            continue
        episodes.append(Episode(
            number=number,
            title=item.get("title", "").strip(),
            url=url,
            date_published=item.get("date_published", ""),
        ))
    # The feed lists newest first; sort old-to-new so progress feels natural.
    def sort_key(ep: Episode):
        # numeric first (with letter suffix grouped), then by published date
        m = re.match(r"(-?\d+)([a-zA-Z]?)$", ep.number)
        if m:
            return (int(m.group(1)), m.group(2))
        return (10**9, ep.number)
    episodes.sort(key=sort_key)
    return episodes


def find_transcript_in_html(html: str) -> str | None:
    m = GOOGLE_DOC_RE.search(html)
    return m.group(0) if m else None


def find_transcript_on_episode_page(episode_url: str) -> str | None:
    """Fetch the episode page and look for a 'Transcript for BEMA N' link."""
    try:
        html = fetch(episode_url).text
    except Exception:
        return None
    soup = BeautifulSoup(html, "lxml")
    # Prefer anchor tags whose label mentions transcript.
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "docs.google.com/document" not in href:
            continue
        label = (a.get_text() or "").lower()
        if "transcript" in label:
            return href
    # Fallback: any Google Doc link on the page.
    return find_transcript_in_html(html)


# ---------------------------------------------------------------------------
# Google Doc cleaning
# ---------------------------------------------------------------------------

GDOC_BOILERPLATE = (
    "Published by Google Docs",
    "Mit Google Docs veröffentlicht",
    "Report Abuse",
    "Missbrauch melden",
    "Updated automatically every 5 minutes",
    "Automatisch alle 5 Minuten aktualisiert",
    "Learn more",
    "Weitere Informationen",
)


def clean_gdoc(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    body = soup.select_one("#contents") or soup.body or soup
    text = body.get_text("\n", strip=True)

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(b in line for b in GDOC_BOILERPLATE):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def ensure_pub_url(url: str) -> str:
    """Make sure the URL ends in /pub so we get the published HTML view."""
    if "/pub" in url:
        return url
    return url.rstrip("/") + "/pub"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def safe_filename(number: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", number) + ".txt"


def scrape(max_episodes: int | None, force: bool, only_missing: bool) -> list[Episode]:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes_from_feed()
    print(f"[bema] feed listed {len(episodes)} episodes")

    if max_episodes:
        episodes = episodes[:max_episodes]
        print(f"[bema] limiting to first {max_episodes}")

    # We need each episode's show-notes HTML for the in-feed transcript hint.
    # Re-fetch the feed once and index by URL.
    feed_html_by_url: dict[str, str] = {}
    feed = fetch(FEED_URL).json()
    for item in feed.get("items", []):
        feed_html_by_url[item.get("url", "")] = item.get("content_html", "")

    saved = 0
    skipped_existing = 0
    no_transcript = 0
    errors = 0

    for ep in tqdm(episodes, desc="episodes"):
        out_path = TRANSCRIPT_DIR / safe_filename(ep.number)
        ep.transcript_path = str(out_path.relative_to(ROOT))

        if out_path.exists() and not force:
            ep.note = "cached"
            skipped_existing += 1
            continue

        if only_missing and out_path.exists():
            ep.note = "cached"
            skipped_existing += 1
            continue

        try:
            # Step 1: try to find transcript URL inside the feed's show notes.
            t_url = find_transcript_in_html(feed_html_by_url.get(ep.url, ""))

            # Step 2: fall back to fetching the episode page.
            if not t_url:
                t_url = find_transcript_on_episode_page(ep.url)
                time.sleep(SLEEP_SECONDS)

            if not t_url:
                ep.note = "no-transcript"
                ep.transcript_path = None
                no_transcript += 1
                continue

            ep.transcript_url = t_url
            doc_html = fetch(ensure_pub_url(t_url)).text
            time.sleep(SLEEP_SECONDS)
            text = clean_gdoc(doc_html)

            if len(text) < 300:
                ep.note = f"short ({len(text)} chars)"
            else:
                ep.note = "ok"
            out_path.write_text(text, encoding="utf-8")
            saved += 1
        except Exception as e:  # noqa: BLE001
            ep.note = f"error: {e}"
            ep.transcript_path = None
            errors += 1
            tqdm.write(f"[bema] episode {ep.number}: {e}")

    EPISODES_JSON.write_text(
        json.dumps([asdict(e) for e in episodes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"\n[bema] summary: saved={saved} skipped_existing={skipped_existing} "
        f"no_transcript={no_transcript} errors={errors} of {len(episodes)} total"
    )
    return episodes


def main() -> int:
    p = argparse.ArgumentParser(description="Scrape BEMA podcast transcripts")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Only process the first N episodes (good for testing).")
    p.add_argument("--force", action="store_true",
                   help="Re-download transcripts that are already on disk.")
    p.add_argument("--only-missing", action="store_true",
                   help="Process only episodes whose transcript file is missing.")
    args = p.parse_args()
    scrape(args.max_episodes, args.force, args.only_missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
