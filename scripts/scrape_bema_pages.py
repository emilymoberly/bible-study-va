"""
Scrape BEMA episode-page bodies + a small set of static site pages.

What it captures (per episode page, e.g. https://www.bemadiscipleship.com/1)
-------------------------------------------------------------------------
- summary text (the lead paragraph(s) above the link list)
- list of "study tools" links (PDFs, companion docs, discussion videos)
- list of YouTube video IDs found anywhere on the page (used later by
  scrape_youtube.py to fetch captions)

What it captures (static site pages)
------------------------------------
A short hard-coded list of the public, content-rich pages on the BEMA
site: /about, /support, /companion, /sermons-of-jesus, etc. Anything that
404s is silently skipped.

Outputs
-------
- data/bema/episode_pages.json   (one record per episode)
- data/bema/site_pages.json      (one record per static page)
- data/bema/log.txt              (append-only log of run)

Polite by default: 0.4s between requests, custom UA, --max-episodes flag.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
EPISODES_JSON = ROOT / "data" / "bema" / "episodes.json"
EPISODE_PAGES_JSON = ROOT / "data" / "bema" / "episode_pages.json"
SITE_PAGES_JSON = ROOT / "data" / "bema" / "site_pages.json"
LOG_PATH = ROOT / "data" / "bema" / "log.txt"

HEADERS = {
    "User-Agent": (
        "BibleStudyVA-ClassProject/0.1 "
        "(educational use; contact: student@example.edu)"
    )
}
SLEEP = 0.4

YT_VIDEO_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)


@dataclass
class StudyLink:
    text: str
    url: str
    kind: str = "link"  # "pdf", "youtube", "transcript", "companion", "link"


@dataclass
class EpisodePage:
    number: str
    url: str
    summary: str = ""
    study_links: list[StudyLink] = field(default_factory=list)
    youtube_video_ids: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class SitePage:
    slug: str
    url: str
    title: str = ""
    text: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# Static site pages worth grabbing. We probe each one and skip 404s.
# ---------------------------------------------------------------------------
STATIC_SLUGS = [
    "about",
    "support",
    "companion",
    "groups",
    "slack",
    "subscribe",
    "speaking",
    "sermons-of-jesus",
    "podcast",
    "resources",
    "covenant",
]


# ---------------------------------------------------------------------------
def fetch(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def classify_link(text: str, url: str) -> str:
    text_l = (text or "").lower()
    url_l = url.lower()
    if "youtube" in url_l or "youtu.be" in url_l:
        return "youtube"
    if url_l.endswith(".pdf") or ".pdf" in url_l:
        return "pdf"
    if "docs.google.com/document" in url_l and "transcript" in text_l:
        return "transcript"
    if "companion" in text_l:
        return "companion"
    return "link"


def extract_summary(soup: BeautifulSoup) -> str:
    """Return the first 1-3 paragraphs of human-readable text on the page."""
    paragraphs: list[str] = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text or len(text) < 40:
            continue
        # Stop once we hit obvious link-list content (short bullet lines).
        paragraphs.append(text)
        if len(paragraphs) >= 3:
            break
    return "\n\n".join(paragraphs)


def parse_episode_page(html: str, url: str, number: str) -> EpisodePage:
    soup = BeautifulSoup(html, "lxml")
    page = EpisodePage(number=number, url=url)

    page.summary = extract_summary(soup)

    seen_urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True)
        if not href or href.startswith("#"):
            continue
        # Resolve relative URLs naively.
        if href.startswith("/"):
            href = "https://www.bemadiscipleship.com" + href
        if href in seen_urls:
            continue
        seen_urls.add(href)
        kind = classify_link(text, href)
        # We're really only interested in study-tool-ish links: docs, PDFs,
        # YouTube, companion. Skip pure social / nav / footer links.
        if kind in {"link"} and "bemadiscipleship.com" in href:
            continue
        page.study_links.append(StudyLink(text=text[:120], url=href, kind=kind))

    # Pull all YouTube video IDs from the raw HTML (more permissive than the
    # link-walker — also catches embedded iframes).
    page.youtube_video_ids = list(dict.fromkeys(YT_VIDEO_RE.findall(html)))
    return page


def parse_site_page(html: str, slug: str, url: str) -> SitePage:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else slug
    # Remove obvious chrome before extracting body text.
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text_chunks: list[str] = []
    for p in soup.find_all(["p", "li", "h1", "h2", "h3"]):
        t = p.get_text(" ", strip=True)
        if t and len(t) > 20:
            text_chunks.append(t)
    text = "\n".join(text_chunks)
    return SitePage(slug=slug, url=url, title=title, text=text)


# ---------------------------------------------------------------------------
def log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {line}\n")


def scrape_episode_pages(episodes: list[dict], max_episodes: int | None) -> list[EpisodePage]:
    if max_episodes:
        episodes = episodes[:max_episodes]

    pages: list[EpisodePage] = []
    errors = 0
    no_content = 0
    for ep in tqdm(episodes, desc="episode pages"):
        url = ep["url"]
        number = str(ep["number"])
        try:
            html = fetch(url).text
            time.sleep(SLEEP)
            page = parse_episode_page(html, url, number)
            if not page.summary and not page.study_links:
                page.note = "empty"
                no_content += 1
            else:
                page.note = "ok"
            pages.append(page)
        except Exception as e:  # noqa: BLE001
            errors += 1
            tqdm.write(f"[bema-pages] {url} -> {e}")
            log(f"episode {number} ({url}) ERROR: {e}")

    log(
        f"episode pages: total={len(episodes)} parsed={len(pages)} "
        f"empty={no_content} errors={errors}"
    )
    return pages


def scrape_static_pages() -> list[SitePage]:
    out: list[SitePage] = []
    for slug in STATIC_SLUGS:
        url = f"https://www.bemadiscipleship.com/{slug}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            time.sleep(SLEEP)
            if r.status_code != 200:
                log(f"static {slug}: status {r.status_code}, skipped")
                continue
            page = parse_site_page(r.text, slug, url)
            if len(page.text) < 200:
                page.note = "thin"
            else:
                page.note = "ok"
            out.append(page)
        except Exception as e:  # noqa: BLE001
            log(f"static {slug}: ERROR {e}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Process only the first N episodes (good for testing)")
    args = p.parse_args()

    if not EPISODES_JSON.exists():
        print("No episodes.json — run scripts/scrape_bema.py first.", file=sys.stderr)
        return 1

    log(f"--- run started, max-episodes={args.max_episodes} ---")
    episodes = json.loads(EPISODES_JSON.read_text(encoding="utf-8"))
    print(f"[bema-pages] loaded {len(episodes)} episodes from feed metadata")

    print("[bema-pages] scraping episode pages…")
    ep_pages = scrape_episode_pages(episodes, args.max_episodes)
    EPISODE_PAGES_JSON.parent.mkdir(parents=True, exist_ok=True)
    EPISODE_PAGES_JSON.write_text(
        json.dumps([_serialize(p) for p in ep_pages], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[bema-pages] wrote {EPISODE_PAGES_JSON} ({len(ep_pages)} records)")

    print("[bema-pages] scraping static site pages…")
    site = scrape_static_pages()
    SITE_PAGES_JSON.write_text(
        json.dumps([asdict(p) for p in site], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[bema-pages] wrote {SITE_PAGES_JSON} ({len(site)} records)")

    total_yt = sum(len(p.youtube_video_ids) for p in ep_pages)
    unique_yt = len({vid for p in ep_pages for vid in p.youtube_video_ids})
    print(f"[bema-pages] discovered {total_yt} YouTube video links "
          f"({unique_yt} unique)")
    log(f"static pages saved: {len(site)}")
    log(f"unique youtube ids discovered: {unique_yt}")
    return 0


def _serialize(p: EpisodePage) -> dict:
    d = asdict(p)
    return d


if __name__ == "__main__":
    sys.exit(main())
