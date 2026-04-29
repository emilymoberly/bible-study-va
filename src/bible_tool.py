"""
Bible verse lookup + keyword search.

Public API:
    BibleTool(json_path).lookup("John 3:16")          -> list[Verse]
    BibleTool(json_path).lookup("Genesis 1:1-5")      -> list[Verse]
    BibleTool(json_path).search("babylon", k=5)       -> list[SearchHit[Verse]]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .retriever import SearchHit, TfidfIndex


@dataclass(frozen=True)
class Verse:
    book: str
    chapter: int
    verse: int
    text: str

    @property
    def reference(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"

    def __str__(self) -> str:
        return f"{self.reference} — {self.text}"


# Common book-name aliases. Keep small; users can always type the full name.
ALIASES = {
    "gen": "Genesis", "ex": "Exodus", "lev": "Leviticus",
    "num": "Numbers", "deut": "Deuteronomy", "dt": "Deuteronomy",
    "josh": "Joshua", "judg": "Judges", "ps": "Psalms", "psalm": "Psalms",
    "prov": "Proverbs", "eccl": "Ecclesiastes", "song": "Song of Solomon",
    "isa": "Isaiah", "jer": "Jeremiah", "lam": "Lamentations",
    "ezek": "Ezekiel", "dan": "Daniel", "hos": "Hosea",
    "matt": "Matthew", "mt": "Matthew", "mk": "Mark", "lk": "Luke",
    "jn": "John", "acts": "Acts", "rom": "Romans",
    "1 cor": "1 Corinthians", "2 cor": "2 Corinthians",
    "gal": "Galatians", "eph": "Ephesians", "phil": "Philippians",
    "col": "Colossians",
    "1 thess": "1 Thessalonians", "2 thess": "2 Thessalonians",
    "1 tim": "1 Timothy", "2 tim": "2 Timothy",
    "tit": "Titus", "phlm": "Philemon", "heb": "Hebrews",
    "jas": "James", "1 pet": "1 Peter", "2 pet": "2 Peter",
    "1 jn": "1 John", "2 jn": "2 John", "3 jn": "3 John",
    "rev": "Revelation",
}

REF_RE = re.compile(
    r"""
    ^\s*
    (?P<book>(?:\d\s*)?[A-Za-z]+(?:\s+[A-Za-z]+)*)   # book name (may start with 1/2/3)
    \s+
    (?P<chap>\d+)
    (?:\s*:\s*(?P<v_start>\d+)(?:\s*-\s*(?P<v_end>\d+))?)?
    \s*$
    """,
    re.VERBOSE,
)


class BibleTool:
    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        with self.json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.verses: list[Verse] = [
            Verse(book=v["book"], chapter=int(v["chapter"]),
                  verse=int(v["verse"]), text=v["text"])
            for v in raw
        ]
        # Build a lookup index keyed on lowercased book name -> verses
        self._by_book: dict[str, list[Verse]] = {}
        for v in self.verses:
            self._by_book.setdefault(v.book.lower(), []).append(v)

    # ------------------------------------------------------------------ lookup
    def _resolve_book(self, raw: str) -> str | None:
        key = raw.strip().lower()
        if key in self._by_book:
            return key
        if key in ALIASES:
            return ALIASES[key].lower()
        # Try fuzzy startswith
        for known in self._by_book:
            if known.startswith(key):
                return known
        return None

    def lookup(self, reference: str) -> list[Verse]:
        """Resolve "John 3:16" or "Genesis 1:1-5" to verse objects."""
        m = REF_RE.match(reference)
        if not m:
            return []
        book_key = self._resolve_book(m.group("book"))
        if book_key is None:
            return []
        chapter = int(m.group("chap"))
        v_start = m.group("v_start")
        v_end = m.group("v_end")
        verses = [v for v in self._by_book[book_key] if v.chapter == chapter]
        if v_start is None:
            return verses  # whole chapter
        v_start = int(v_start)
        v_end = int(v_end) if v_end else v_start
        return [v for v in verses if v_start <= v.verse <= v_end]

    # ------------------------------------------------------------------ search
    @cached_property
    def _index(self) -> TfidfIndex[Verse]:
        return TfidfIndex(self.verses, [v.text for v in self.verses])

    def search(self, query: str, k: int = 5) -> list[SearchHit[Verse]]:
        return self._index.search(query, k=k)
