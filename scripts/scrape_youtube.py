"""
Scrape YouTube transcripts for videos referenced from BEMA episode pages.

Why this is the YouTube source we picked
----------------------------------------
The BEMA episode pages already link to "Discussion Video for BEMA N"
YouTube videos (Marty's official discussion videos) plus other Marty/BEMA
content (sermons, talks). Rather than enumerate Marty's whole YouTube
channel (which is fragile and requires more permissions), we restrict
ourselves to videos already linked from the public BEMA site. This is a
focused, relevant subset of "Marty Solomon YouTube content".

What it does
------------
1. Reads `data/bema/episode_pages.json` (produced by scrape_bema_pages.py)
   and collects the unique YouTube video IDs found on those pages.
2. For each video ID, tries to:
     a. Fetch the public title via YouTube's oEmbed endpoint
        (no API key required; just an HTTP GET).
     b. Fetch the publicly available captions via youtube-transcript-api.
        Skips videos that have no public captions.
3. Writes:
     data/youtube/videos.json           — metadata per video
     data/youtube/transcripts/<id>.txt  — caption text per video
     data/youtube/log.txt               — append-only log of run

Polite by default: small delay between requests, --max-videos for testing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
EPISODE_PAGES_JSON = ROOT / "data" / "bema" / "episode_pages.json"
YT_DIR = ROOT / "data" / "youtube"
YT_TRANSCRIPTS = YT_DIR / "transcripts"
VIDEOS_JSON = YT_DIR / "videos.json"
LOG_PATH = YT_DIR / "log.txt"

HEADERS = {
    "User-Agent": (
        "BibleStudyVA-ClassProject/0.1 "
        "(educational use; contact: student@example.edu)"
    )
}
SLEEP = 0.4


@dataclass
class Video:
    video_id: str
    url: str
    title: str = ""
    referenced_by_episodes: list[str] = field(default_factory=list)
    transcript_path: str | None = None
    transcript_chars: int = 0
    note: str = ""  # ok | no-captions | error: ...


def log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {line}\n")


def collect_video_ids() -> dict[str, list[str]]:
    """Return {video_id: [episode_numbers that referenced it]}."""
    if not EPISODE_PAGES_JSON.exists():
        raise FileNotFoundError(
            f"{EPISODE_PAGES_JSON} not found. "
            "Run scripts/scrape_bema_pages.py first."
        )
    pages = json.loads(EPISODE_PAGES_JSON.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for page in pages:
        for vid in page.get("youtube_video_ids", []):
            out.setdefault(vid, []).append(str(page["number"]))
    return out


def fetch_title(video_id: str) -> str:
    """Use YouTube's public oEmbed endpoint to get the video title."""
    url = (
        "https://www.youtube.com/oembed?url="
        f"https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return ""
        return r.json().get("title", "")
    except Exception:  # noqa: BLE001
        return ""


_YT_API_SINGLETON = None


def _yt_api():
    """The 0.6+ youtube-transcript-api uses an instance, not a classmethod."""
    global _YT_API_SINGLETON
    if _YT_API_SINGLETON is None:
        from youtube_transcript_api import YouTubeTranscriptApi
        _YT_API_SINGLETON = YouTubeTranscriptApi()
    return _YT_API_SINGLETON


def fetch_transcript(video_id: str) -> str | None:
    """Return concatenated caption text, or None if no public captions."""
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    api = _yt_api()
    try:
        try:
            transcript = api.fetch(video_id, languages=["en"])
        except NoTranscriptFound:
            transcript = api.fetch(video_id)
        # FetchedTranscript is iterable of FetchedTranscriptSnippet (.text/.start/.duration)
        parts: list[str] = []
        for snippet in transcript:
            text = getattr(snippet, "text", None)
            if not text:
                # Fallback in case the lib returns plain dicts in some version
                text = (snippet or {}).get("text") if isinstance(snippet, dict) else None
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:  # noqa: BLE001
        log(f"transcript {video_id}: ERROR {e}")
        return None


def scrape(max_videos: int | None) -> list[Video]:
    YT_TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    id_to_eps = collect_video_ids()
    print(f"[yt] discovered {len(id_to_eps)} unique YouTube video IDs from BEMA pages")
    log(f"--- run started, total candidates={len(id_to_eps)}, max_videos={max_videos} ---")

    items = list(id_to_eps.items())
    if max_videos:
        items = items[:max_videos]

    videos: list[Video] = []
    saved_with_transcript = 0
    no_captions = 0
    errors = 0

    for video_id, episodes in tqdm(items, desc="videos"):
        out_path = YT_TRANSCRIPTS / f"{video_id}.txt"
        url = f"https://www.youtube.com/watch?v={video_id}"
        v = Video(video_id=video_id, url=url, referenced_by_episodes=episodes)

        # Reuse cached transcript if present
        cached = out_path.exists() and out_path.stat().st_size > 200
        if cached:
            v.transcript_path = str(out_path.relative_to(ROOT))
            v.transcript_chars = out_path.stat().st_size
            v.note = "cached"
            v.title = fetch_title(video_id)
            time.sleep(SLEEP / 2)
            videos.append(v)
            saved_with_transcript += 1
            continue

        v.title = fetch_title(video_id)
        time.sleep(SLEEP / 2)
        text = fetch_transcript(video_id)
        time.sleep(SLEEP)

        if not text:
            v.note = "no-captions"
            no_captions += 1
        else:
            out_path.write_text(text, encoding="utf-8")
            v.transcript_path = str(out_path.relative_to(ROOT))
            v.transcript_chars = len(text)
            v.note = "ok"
            saved_with_transcript += 1

        videos.append(v)

    VIDEOS_JSON.write_text(
        json.dumps([asdict(v) for v in videos], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"\n[yt] summary: with_transcript={saved_with_transcript} "
        f"no_captions={no_captions} errors={errors} of {len(items)} candidates"
    )
    log(
        f"summary: with_transcript={saved_with_transcript} "
        f"no_captions={no_captions} errors={errors} total={len(items)}"
    )
    return videos


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-videos", type=int, default=None,
                   help="Process only the first N videos (good for testing).")
    args = p.parse_args()
    scrape(args.max_videos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
