"""
Unified retrieval over the labeled corpus produced by scripts/build_corpus.py.

Why a separate module
---------------------
Earlier the project used three independent tools (Bible / BEMA / Web). Now
we have one chunk store with rich metadata (source_type, episode, date,
verse_refs). This module gives the agent and Gradio UI a single API to:

  - rank chunks by TF-IDF relevance to a query
  - filter by source_type ("bible", "bema_transcript", ...)
  - filter by episode number
  - filter by Bible verse reference (normalized form like "John 3:16")

Scope deliberately stays small (TF-IDF, no embeddings) so the class
project remains lightweight.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Matches normalized range refs like "Genesis 1:1-3" or "1 Corinthians 13:1-13".
# Single-verse and chapter-only refs do NOT match and are returned unchanged.
_RANGE_RE = re.compile(r"^(.+?)\s+(\d+):(\d+)-(\d+)$")

# Safety cap on range expansion. Prevents pathological inputs like
# "Psalms 119:1-176" from blowing up the filter.
_MAX_RANGE_EXPAND = 50


def _expand_verse_range(verse_ref: str) -> list[str]:
    """Expand a verse-range ref into one entry per verse.

    "Genesis 1:1-3"  -> ["Genesis 1:1", "Genesis 1:2", "Genesis 1:3"]
    "John 3:16"      -> ["John 3:16"]
    "Psalms 23"      -> ["Psalms 23"]   (chapter-only, not a range)
    """
    ref = verse_ref.strip()
    m = _RANGE_RE.match(ref)
    if not m:
        return [ref]
    book, ch, vs, ve = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if ve < vs:
        return [f"{book} {ch}:{vs}"]
    if ve - vs + 1 > _MAX_RANGE_EXPAND:
        ve = vs + _MAX_RANGE_EXPAND - 1
    return [f"{book} {ch}:{v}" for v in range(vs, ve + 1)]


@dataclass(frozen=True)
class CorpusChunk:
    id: str
    source_type: str
    title: str
    url: str
    episode: str | None
    date: str
    chunk_index: int
    text: str
    verse_refs: tuple[str, ...]

    @property
    def reference(self) -> str:
        if self.source_type == "bible":
            return self.title
        if self.episode:
            return f"BEMA {self.episode}: {self.title}"
        return self.title

    def excerpt(self, n: int = 240) -> str:
        t = self.text.strip()
        return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


@dataclass(frozen=True)
class CorpusHit:
    chunk: CorpusChunk
    score: float


SOURCE_TYPES = (
    "bible",
    "bema_transcript",
    "bema_summary",
    "bema_studytool",
    "bema_site",
    "youtube",
)


class Corpus:
    """In-memory TF-IDF index over a corpus.jsonl file."""

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.chunks: list[CorpusChunk] = []
        self._load()

    def _load(self) -> None:
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.chunks.append(CorpusChunk(
                    id=d["id"],
                    source_type=d["source_type"],
                    title=d.get("title", ""),
                    url=d.get("url", ""),
                    episode=d.get("episode"),
                    date=d.get("date", ""),
                    chunk_index=int(d.get("chunk_index", 0)),
                    text=d.get("text", ""),
                    verse_refs=tuple(d.get("verse_refs", []) or []),
                ))

    # ------------------------------------------------------------------ index
    @cached_property
    def _vectorizer(self) -> TfidfVectorizer:
        v = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words="english",
        )
        v.fit([c.text for c in self.chunks])
        return v

    @cached_property
    def _matrix(self):
        return self._vectorizer.transform([c.text for c in self.chunks])

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        out: dict = {"total_chunks": len(self.chunks), "by_source_type": {}}
        for st in SOURCE_TYPES:
            out["by_source_type"][st] = sum(1 for c in self.chunks if c.source_type == st)
        out["with_verse_refs"] = sum(1 for c in self.chunks if c.verse_refs)
        out["episodes_covered"] = len({c.episode for c in self.chunks if c.episode})
        return out

    # ----------------------------------------------------------------- filters
    def _candidate_indices(
        self,
        source_types: Iterable[str] | None,
        episode: str | None,
        verse_ref: str | None,
    ) -> np.ndarray:
        """Return indices into self.chunks that pass all filters."""
        if source_types:
            allowed = set(source_types)
            mask = np.array([c.source_type in allowed for c in self.chunks])
        else:
            mask = np.ones(len(self.chunks), dtype=bool)

        if episode is not None:
            ep_str = str(episode)
            mask &= np.array([c.episode == ep_str for c in self.chunks])

        if verse_ref:
            # Expand verse ranges so e.g. "Genesis 1:1-3" matches any chunk
            # whose verse_refs include Gen 1:1, 1:2, or 1:3 individually.
            wanted = set(_expand_verse_range(verse_ref))
            mask &= np.array([
                bool(wanted.intersection(c.verse_refs)) for c in self.chunks
            ])

        return np.where(mask)[0]

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        k: int = 5,
        source_types: Iterable[str] | None = None,
        episode: str | None = None,
        verse_ref: str | None = None,
    ) -> list[CorpusHit]:
        if not query.strip():
            return []
        cand = self._candidate_indices(source_types, episode, verse_ref)
        if cand.size == 0:
            return []

        q_vec = self._vectorizer.transform([query])
        sub_matrix = self._matrix[cand]
        sims = cosine_similarity(q_vec, sub_matrix).ravel()

        if k >= sims.size:
            order = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, k)[:k]
            order = top[np.argsort(-sims[top])]

        out: list[CorpusHit] = []
        for j in order:
            score = float(sims[j])
            if score <= 0:
                continue
            idx = int(cand[j])
            out.append(CorpusHit(chunk=self.chunks[idx], score=score))
        return out

    # ------------------------------------------------------------------ exact
    def lookup_verse(self, verse_ref: str) -> list[CorpusChunk]:
        """Return bible chunks whose title matches the given normalized ref.

        Verse ranges are expanded — `lookup_verse("Genesis 1:1-3")` returns
        the three chunks for Gen 1:1, 1:2, 1:3 in canonical order. Missing
        verses (e.g. a typo'd reference) are silently skipped.
        """
        wanted = _expand_verse_range(verse_ref)
        by_title: dict[str, CorpusChunk] = {
            c.title: c for c in self.chunks if c.source_type == "bible"
        }
        return [by_title[r] for r in wanted if r in by_title]
