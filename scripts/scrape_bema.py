"""
Scrape BEMA Discipleship podcast transcripts.

Why this script exists
----------------------
BEMA episodes have transcripts published as Google Docs linked from each
episode page on bemadiscipleship.com. We need them as local plain-text files
so the BEMA tool can do TF-IDF retrieval over them without hitting the
network at query time.

What it does
------------
1. Fetches the episode index (https://www.bemadiscipleship.com/episodes)
   and parses out (number, title) for every episode.
2. For each episode, fetches https://www.bemadiscipleship.com/{number} and
   looks for a "Transcript for BEMA N" link pointing to docs.google.com.
3. Downloads the published Google Doc HTML, extracts clean text, and writes
   it to data/bema/transcripts/{number}.txt.
4. Saves all episode metadata to data/bema/episodes.json.

Be polite
---------
- Custom User-Agent
- 1 second delay between requests
- --max-episodes flag so you can test with a small batch first
- Skips episodes whose transcript file already exists
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://www.bemadiscipleship.com"
INDEX_URL = f"{BASE}/episodes"
HEADERS = {
    "User-Agent": (
        "BibleStudyVA-ClassProject/0.1 "
        "(educational use; contact: student@example.edu)"
    )
}
SLEEP_SECONDS = 1.0  # be polite

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = ROOT / "data" / "bema" / "transcripts"
EPISODES_JSON = ROOT / "data" / "bema" / "episodes.json"


@dataclass
class Episode:
    number: str           # keep as string; some episodes are "-1", "0", "12b"
    title: str
    url: str
    transcript_url: str | None = None
    transcript_path: str | None = None


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------

# Episode headings on the archive look like: "<h3>1: Trust the Story</h3>"
EPISODE_HEADING_RE = re.compile(r"^\s*(-?\d+[a-zA-Z]?)\s*:\s*(.+?)\s*$")


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def parse_episode_index(html: str) -> list[Episode]:
    soup = BeautifulSoup(html, "lxml")
    episodes: list[Episode] = []
    seen: set[str] = set()
    # Headings on the archive page are h3; titles are "{number}: {title}".
    for h in soup.find_all(["h2", "h3", "h4"]):
        text = h.get_text(strip=True)
        m = EPISODE_HEADING_RE.match(text)
        if not m:
            continue
        number, title = m.group(1), m.group(2)
        if number in seen:
            continue
        seen.add(number)
        episodes.append(Episode(
            number=number,
            title=title,
            url=f"{BASE}/{number}",
        ))
    return episodes


# ---------------------------------------------------------------------------
# Per-episode transcript discovery
# ---------------------------------------------------------------------------

TRANSCRIPT_LINK_RE = re.compile(r"transcript", re.IGNORECASE)


def find_transcript_url(episode_html: str) -> str | None:
    """
    Episode pages list 1+ Google Docs transcript links labeled
    "Transcript for BEMA N". When multiple exist (legacy + current), prefer
    the first one because the page lists current/preferred transcripts first.
    """
    soup = BeautifulSoup(episode_html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "docs.google.com/document" not in href:
            continue
        if TRANSCRIPT_LINK_RE.search(a.get_text() or ""):
            return href
    return None


# ---------------------------------------------------------------------------
# Google Doc cleaning
# ---------------------------------------------------------------------------

# Lines that appear in every published Google Doc and are not transcript content.
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

    # Strip script/style/nav noise
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Google Docs publishes the body inside #contents; fall back to body.
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

    # Collapse runs of blank-ish formatting; rejoin with single newlines.
    cleaned = "\n".join(lines)
    # Collapse runs of 3+ newlines down to 2.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def safe_filename(number: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", number) + ".txt"


def scrape(max_episodes: int | None, force: bool) -> list[Episode]:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[bema] fetching episode index: {INDEX_URL}")
    episodes = parse_episode_index(fetch(INDEX_URL))
    print(f"[bema] found {len(episodes)} episodes")

    if max_episodes:
        episodes = episodes[:max_episodes]
        print(f"[bema] limiting to first {max_episodes} for this run")

    for ep in tqdm(episodes, desc="episodes"):
        out_path = TRANSCRIPT_DIR / safe_filename(ep.number)
        ep.transcript_path = str(out_path.relative_to(ROOT))

        if out_path.exists() and not force:
            # Already scraped — still record transcript_url if missing
            ep.transcript_url = ep.transcript_url or "<cached>"
            continue

        try:
            episode_html = fetch(ep.url)
            time.sleep(SLEEP_SECONDS)

            t_url = find_transcript_url(episode_html)
            if not t_url:
                tqdm.write(f"[bema] no transcript link on {ep.url}; skipping")
                ep.transcript_path = None
                continue
            ep.transcript_url = t_url

            doc_html = fetch(t_url)
            time.sleep(SLEEP_SECONDS)

            text = clean_gdoc(doc_html)
            if len(text) < 500:
                tqdm.write(f"[bema] suspiciously short transcript for {ep.number} "
                           f"({len(text)} chars); keeping anyway")
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"[bema] failed on episode {ep.number}: {e}")
            ep.transcript_path = None

    EPISODES_JSON.write_text(
        json.dumps([asdict(e) for e in episodes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    saved = sum(1 for e in episodes if e.transcript_path)
    print(f"[bema] done. {saved}/{len(episodes)} transcripts saved under {TRANSCRIPT_DIR}")
    return episodes


def main() -> int:
    p = argparse.ArgumentParser(description="Scrape BEMA podcast transcripts")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Only scrape the first N episodes (good for testing).")
    p.add_argument("--force", action="store_true",
                   help="Re-download transcripts that are already on disk.")
    args = p.parse_args()
    scrape(args.max_episodes, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
