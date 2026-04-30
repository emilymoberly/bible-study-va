"""
Unify all data sources into a single labeled chunk corpus.

What gets indexed
-----------------
Every chunk in `data/corpus/corpus.jsonl` looks like:

    {
      "id":         "bema-transcript-100-3",
      "source_type":"bema_transcript",   # or bema_summary | bema_studytool |
                                         #   bema_site | youtube | bible
      "title":      "BEMA 100: Healing at Great Cost",
      "url":        "https://www.bemadiscipleship.com/100",
      "episode":    "100",               # if applicable, else null
      "date":       "2023-08-04",        # if applicable, else ""
      "chunk_index":3,
      "text":       "...",               # cleaned text
      "verse_refs": ["Matthew 9:1", "Mark 2"]   # detected scripture refs
    }

Source types
------------
- bible:           1 verse per chunk
- bema_transcript: ~400-word windows of episode transcripts
- bema_summary:    1 chunk per episode (the show-notes summary)
- bema_studytool:  1 chunk per study-tool link (text + URL)
- bema_site:       chunks of static BEMA pages (about, resources, etc.)
- youtube:         ~400-word windows of YouTube captions

Dedup
-----
Within each source type we hash a normalized prefix of every chunk
(lowercased, whitespace-collapsed, first 200 chars) and skip duplicates.
The Bible is exempt from this because every verse is unique by design.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bible_refs import find_refs  # noqa: E402

# ---------------------------------------------------------------------------
DATA = ROOT / "data"
BIBLE_JSON = DATA / "bible" / "bible.json"
EPISODES_JSON = DATA / "bema" / "episodes.json"
EPISODE_PAGES_JSON = DATA / "bema" / "episode_pages.json"
SITE_PAGES_JSON = DATA / "bema" / "site_pages.json"
TRANSCRIPT_DIR = DATA / "bema" / "transcripts"
YT_VIDEOS_JSON = DATA / "youtube" / "videos.json"
YT_TRANSCRIPT_DIR = DATA / "youtube" / "transcripts"

CORPUS_DIR = DATA / "corpus"
CORPUS_JSONL = CORPUS_DIR / "corpus.jsonl"
STATS_JSON = CORPUS_DIR / "stats.json"
LOG_PATH = CORPUS_DIR / "log.txt"


# ---------------------------------------------------------------------------
# Chunking + cleaning helpers
# ---------------------------------------------------------------------------

WHITESPACE_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    """Normalize quotes and collapse runs of whitespace."""
    if not text:
        return ""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def chunk_text(text: str, words_per_chunk: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        end = start + words_per_chunk
        out.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return out


def dedup_key(text: str) -> str:
    norm = WHITESPACE_RE.sub(" ", text.lower().strip())
    return norm[:200]


# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str
    source_type: str
    title: str
    url: str
    episode: str | None
    date: str
    chunk_index: int
    text: str
    verse_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
def emit_bible(out: list[Chunk]) -> tuple[int, int]:
    if not BIBLE_JSON.exists():
        return 0, 0
    verses = json.loads(BIBLE_JSON.read_text(encoding="utf-8"))
    count = 0
    for v in verses:
        ref = f"{v['book']} {v['chapter']}:{v['verse']}"
        out.append(Chunk(
            id=f"bible-{ref.replace(' ', '_').replace(':', '_')}",
            source_type="bible",
            title=ref,
            url="",  # no canonical URL
            episode=None,
            date="",
            chunk_index=0,
            text=clean(v["text"]),
            verse_refs=[ref],
        ))
        count += 1
    return count, count


def emit_bema_transcripts(out: list[Chunk], episodes_meta: dict[str, dict]) -> tuple[int, int]:
    if not TRANSCRIPT_DIR.exists():
        return 0, 0
    seen: set[str] = set()
    files = 0
    chunks = 0
    for path in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        number = path.stem
        meta = episodes_meta.get(number, {})
        title = meta.get("title", f"BEMA {number}")
        url = meta.get("url", f"https://www.bemadiscipleship.com/{number}")
        date = meta.get("date_published", "")[:10]
        text = clean(path.read_text(encoding="utf-8"))
        if not text:
            continue
        files += 1
        for i, c in enumerate(chunk_text(text)):
            k = dedup_key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append(Chunk(
                id=f"bema-transcript-{number}-{i}",
                source_type="bema_transcript",
                title=title,
                url=url,
                episode=number,
                date=date,
                chunk_index=i,
                text=c,
                verse_refs=[r.normalized for r in find_refs(c)],
            ))
            chunks += 1
    return files, chunks


def emit_bema_summaries(out: list[Chunk], episodes_meta: dict[str, dict]) -> tuple[int, int]:
    if not EPISODE_PAGES_JSON.exists():
        return 0, 0
    pages = json.loads(EPISODE_PAGES_JSON.read_text(encoding="utf-8"))
    seen: set[str] = set()
    n_pages = 0
    n_chunks = 0
    for p in pages:
        summary = clean(p.get("summary", ""))
        if not summary:
            continue
        n_pages += 1
        k = dedup_key(summary)
        if k in seen:
            continue
        seen.add(k)
        number = str(p["number"])
        meta = episodes_meta.get(number, {})
        out.append(Chunk(
            id=f"bema-summary-{number}",
            source_type="bema_summary",
            title=meta.get("title", f"BEMA {number} (summary)"),
            url=p["url"],
            episode=number,
            date=meta.get("date_published", "")[:10],
            chunk_index=0,
            text=summary,
            verse_refs=[r.normalized for r in find_refs(summary)],
        ))
        n_chunks += 1
    return n_pages, n_chunks


def emit_bema_studytools(out: list[Chunk], episodes_meta: dict[str, dict]) -> tuple[int, int]:
    if not EPISODE_PAGES_JSON.exists():
        return 0, 0
    pages = json.loads(EPISODE_PAGES_JSON.read_text(encoding="utf-8"))
    seen: set[str] = set()
    n_links = 0
    n_chunks = 0
    for p in pages:
        number = str(p["number"])
        meta = episodes_meta.get(number, {})
        ep_title = meta.get("title", f"BEMA {number}")
        for j, link in enumerate(p.get("study_links", [])):
            n_links += 1
            link_text = clean(link.get("text") or "")
            kind = link.get("kind", "link")
            url = link.get("url", "")
            # Build a small chunk so retrieval can surface study-tool links.
            text = f"[{kind}] {link_text} — referenced in {ep_title}"
            k = dedup_key(text + url)
            if k in seen:
                continue
            seen.add(k)
            out.append(Chunk(
                id=f"bema-studytool-{number}-{j}",
                source_type="bema_studytool",
                title=f"{ep_title} — {kind}: {link_text[:60]}",
                url=url,
                episode=number,
                date=meta.get("date_published", "")[:10],
                chunk_index=j,
                text=text,
                verse_refs=[],
            ))
            n_chunks += 1
    return n_links, n_chunks


def emit_bema_site(out: list[Chunk]) -> tuple[int, int]:
    if not SITE_PAGES_JSON.exists():
        return 0, 0
    pages = json.loads(SITE_PAGES_JSON.read_text(encoding="utf-8"))
    seen: set[str] = set()
    n_pages = 0
    n_chunks = 0
    for p in pages:
        text = clean(p.get("text", ""))
        if not text:
            continue
        n_pages += 1
        for i, c in enumerate(chunk_text(text)):
            k = dedup_key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append(Chunk(
                id=f"bema-site-{p['slug']}-{i}",
                source_type="bema_site",
                title=p.get("title") or p["slug"],
                url=p["url"],
                episode=None,
                date="",
                chunk_index=i,
                text=c,
                verse_refs=[r.normalized for r in find_refs(c)],
            ))
            n_chunks += 1
    return n_pages, n_chunks


def emit_youtube(out: list[Chunk]) -> tuple[int, int]:
    if not YT_VIDEOS_JSON.exists() or not YT_TRANSCRIPT_DIR.exists():
        return 0, 0
    videos = json.loads(YT_VIDEOS_JSON.read_text(encoding="utf-8"))
    seen: set[str] = set()
    n_vids = 0
    n_chunks = 0
    for v in videos:
        path_rel = v.get("transcript_path")
        if not path_rel:
            continue
        path = ROOT / path_rel
        if not path.exists():
            continue
        text = clean(path.read_text(encoding="utf-8"))
        if not text:
            continue
        n_vids += 1
        for i, c in enumerate(chunk_text(text)):
            k = dedup_key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append(Chunk(
                id=f"youtube-{v['video_id']}-{i}",
                source_type="youtube",
                title=v.get("title") or f"YouTube {v['video_id']}",
                url=v["url"],
                episode=(v.get("referenced_by_episodes") or [None])[0],
                date="",
                chunk_index=i,
                text=c,
                verse_refs=[r.normalized for r in find_refs(c)],
            ))
            n_chunks += 1
    return n_vids, n_chunks


# ---------------------------------------------------------------------------
def log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {line}\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-bible", action="store_true",
                   help="Don't include the 31K Bible verses (faster, smaller).")
    args = p.parse_args()

    log("--- build started ---")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    episodes_meta: dict[str, dict] = {}
    if EPISODES_JSON.exists():
        for ep in json.loads(EPISODES_JSON.read_text(encoding="utf-8")):
            episodes_meta[str(ep["number"])] = ep

    chunks: list[Chunk] = []
    stats: dict = {"by_source_type": {}, "totals": {}}

    if not args.skip_bible:
        n_items, n_chunks = emit_bible(chunks)
        stats["by_source_type"]["bible"] = {"items": n_items, "chunks": n_chunks}
        print(f"[corpus] bible:           {n_chunks:,} chunks ({n_items:,} verses)")

    n_items, n_chunks = emit_bema_transcripts(chunks, episodes_meta)
    stats["by_source_type"]["bema_transcript"] = {"items": n_items, "chunks": n_chunks}
    print(f"[corpus] bema_transcript: {n_chunks:,} chunks from {n_items:,} episodes")

    n_items, n_chunks = emit_bema_summaries(chunks, episodes_meta)
    stats["by_source_type"]["bema_summary"] = {"items": n_items, "chunks": n_chunks}
    print(f"[corpus] bema_summary:    {n_chunks:,} chunks from {n_items:,} pages")

    n_items, n_chunks = emit_bema_studytools(chunks, episodes_meta)
    stats["by_source_type"]["bema_studytool"] = {"items": n_items, "chunks": n_chunks}
    print(f"[corpus] bema_studytool:  {n_chunks:,} chunks from {n_items:,} links")

    n_items, n_chunks = emit_bema_site(chunks)
    stats["by_source_type"]["bema_site"] = {"items": n_items, "chunks": n_chunks}
    print(f"[corpus] bema_site:       {n_chunks:,} chunks from {n_items:,} pages")

    n_items, n_chunks = emit_youtube(chunks)
    stats["by_source_type"]["youtube"] = {"items": n_items, "chunks": n_chunks}
    print(f"[corpus] youtube:         {n_chunks:,} chunks from {n_items:,} videos")

    # Write JSONL
    with CORPUS_JSONL.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    stats["totals"] = {
        "total_chunks": len(chunks),
        "total_chars": sum(len(c.text) for c in chunks),
        "with_verse_refs": sum(1 for c in chunks if c.verse_refs),
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"\n[corpus] DONE. "
        f"{stats['totals']['total_chunks']:,} chunks "
        f"({stats['totals']['total_chars']/1e6:.1f} MB of text), "
        f"{stats['totals']['with_verse_refs']:,} chunks with verse refs"
    )
    print(f"[corpus] wrote {CORPUS_JSONL}")
    print(f"[corpus] wrote {STATS_JSON}")
    log(f"build done: {stats['totals']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
