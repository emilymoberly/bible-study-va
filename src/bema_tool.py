"""
BEMA podcast transcript retrieval.

Loads scraped transcripts from data/bema/transcripts/, splits them into
fixed-size chunks, and exposes TF-IDF search.

Public API:
    BemaTool(transcripts_dir, episodes_json).search("Babylon", k=5)
        -> list[SearchHit[BemaChunk]]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .retriever import SearchHit, TfidfIndex


@dataclass(frozen=True)
class BemaChunk:
    episode_number: str
    episode_title: str
    chunk_index: int
    text: str

    @property
    def reference(self) -> str:
        return f"BEMA {self.episode_number} ({self.episode_title}) — chunk {self.chunk_index}"

    def __str__(self) -> str:
        snippet = self.text[:200].replace("\n", " ")
        return f"{self.reference}: {snippet}..."


def chunk_text(text: str, words_per_chunk: int = 400, overlap: int = 50) -> list[str]:
    """
    Split a transcript into overlapping word-windows.

    400 words ≈ ~3 paragraphs of conversation, which is a good unit for
    podcast retrieval — small enough to be focused, large enough to keep
    context. The 50-word overlap helps when relevant info straddles a boundary.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        end = start + words_per_chunk
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks


class BemaTool:
    def __init__(self, transcripts_dir: str | Path, episodes_json: str | Path | None = None):
        self.transcripts_dir = Path(transcripts_dir)
        self.episodes_json = Path(episodes_json) if episodes_json else None

        self.title_by_number: dict[str, str] = {}
        if self.episodes_json and self.episodes_json.exists():
            for ep in json.loads(self.episodes_json.read_text(encoding="utf-8")):
                self.title_by_number[str(ep["number"])] = ep.get("title", "")

        self.chunks: list[BemaChunk] = []
        for path in sorted(self.transcripts_dir.glob("*.txt")):
            number = path.stem
            title = self.title_by_number.get(number, "")
            text = path.read_text(encoding="utf-8")
            for i, c in enumerate(chunk_text(text)):
                self.chunks.append(BemaChunk(
                    episode_number=number,
                    episode_title=title,
                    chunk_index=i,
                    text=c,
                ))

    @cached_property
    def _index(self) -> TfidfIndex[BemaChunk]:
        if not self.chunks:
            raise RuntimeError(
                "No BEMA transcripts loaded. Run scripts/scrape_bema.py first."
            )
        return TfidfIndex(self.chunks, [c.text for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[SearchHit[BemaChunk]]:
        if not self.chunks:
            return []
        return self._index.search(query, k=k)
