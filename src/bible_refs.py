"""
Bible-reference detector + normalizer.

Given an arbitrary text blob, find every Bible citation in it and return a
list of normalized references like:

    [{"book": "John", "chapter": 3, "verse_start": 16, "verse_end": 16,
      "raw": "John 3:16", "normalized": "John 3:16"}]

This is deliberately conservative — we only match references whose book
name is in our known list, so that random capitalized words ("Marty 4:00")
don't get treated as scripture.

Used by:
- scripts/build_corpus.py to attach `verse_refs` metadata to every chunk
- src/corpus.py to filter retrieval by verse reference
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Canonical book names (KJV ordering) plus very common short aliases.
# Each entry: canonical name, aliases (lowercased, no punctuation).
BOOKS: list[tuple[str, list[str]]] = [
    # Pentateuch
    ("Genesis",        ["genesis", "gen", "gn"]),
    ("Exodus",         ["exodus", "exod", "ex"]),
    ("Leviticus",      ["leviticus", "lev", "lv"]),
    ("Numbers",        ["numbers", "num", "nm"]),
    ("Deuteronomy",    ["deuteronomy", "deut", "deu", "dt"]),
    # History
    ("Joshua",         ["joshua", "josh", "jos"]),
    ("Judges",         ["judges", "judg", "jdg"]),
    ("Ruth",           ["ruth", "rth"]),
    ("1 Samuel",       ["1 samuel", "1 sam", "1sam", "i samuel", "i sam", "1 sm"]),
    ("2 Samuel",       ["2 samuel", "2 sam", "2sam", "ii samuel", "ii sam", "2 sm"]),
    ("1 Kings",        ["1 kings", "1 kgs", "1kgs", "i kings", "i kgs", "1 ki"]),
    ("2 Kings",        ["2 kings", "2 kgs", "2kgs", "ii kings", "ii kgs", "2 ki"]),
    ("1 Chronicles",   ["1 chronicles", "1 chron", "1 chr", "i chron"]),
    ("2 Chronicles",   ["2 chronicles", "2 chron", "2 chr", "ii chron"]),
    ("Ezra",           ["ezra", "ezr"]),
    ("Nehemiah",       ["nehemiah", "neh"]),
    ("Esther",         ["esther", "est", "esth"]),
    # Wisdom
    ("Job",            ["job"]),
    ("Psalms",         ["psalms", "psalm", "ps", "psa", "pss"]),
    ("Proverbs",       ["proverbs", "prov", "prv", "pr"]),
    ("Ecclesiastes",   ["ecclesiastes", "eccl", "eccles", "ec", "qoh", "qoheleth"]),
    ("Song of Solomon", ["song of solomon", "song of songs", "song", "sos", "canticles"]),
    # Major prophets
    ("Isaiah",         ["isaiah", "isa", "is"]),
    ("Jeremiah",       ["jeremiah", "jer"]),
    ("Lamentations",   ["lamentations", "lam"]),
    ("Ezekiel",        ["ezekiel", "ezek", "eze"]),
    ("Daniel",         ["daniel", "dan"]),
    # Minor prophets
    ("Hosea",          ["hosea", "hos"]),
    ("Joel",           ["joel"]),
    ("Amos",           ["amos"]),
    ("Obadiah",        ["obadiah", "obad", "oba"]),
    ("Jonah",          ["jonah", "jon"]),
    ("Micah",          ["micah", "mic"]),
    ("Nahum",          ["nahum", "nah"]),
    ("Habakkuk",       ["habakkuk", "hab", "hbk"]),
    ("Zephaniah",      ["zephaniah", "zeph", "zep"]),
    ("Haggai",         ["haggai", "hag"]),
    ("Zechariah",      ["zechariah", "zech", "zec"]),
    ("Malachi",        ["malachi", "mal"]),
    # Gospels + Acts
    ("Matthew",        ["matthew", "matt", "mt"]),
    ("Mark",           ["mark", "mk"]),
    ("Luke",           ["luke", "lk"]),
    ("John",           ["john", "jn"]),
    ("Acts",           ["acts", "act"]),
    # Pauline epistles
    ("Romans",         ["romans", "rom"]),
    ("1 Corinthians",  ["1 corinthians", "1 cor", "1cor", "i corinthians", "i cor"]),
    ("2 Corinthians",  ["2 corinthians", "2 cor", "2cor", "ii corinthians", "ii cor"]),
    ("Galatians",      ["galatians", "gal"]),
    ("Ephesians",      ["ephesians", "eph"]),
    ("Philippians",    ["philippians", "phil", "php"]),
    ("Colossians",     ["colossians", "col"]),
    ("1 Thessalonians", ["1 thessalonians", "1 thess", "1 th", "i thessalonians", "i thess"]),
    ("2 Thessalonians", ["2 thessalonians", "2 thess", "2 th", "ii thessalonians", "ii thess"]),
    ("1 Timothy",      ["1 timothy", "1 tim", "1 tm", "i timothy", "i tim"]),
    ("2 Timothy",      ["2 timothy", "2 tim", "2 tm", "ii timothy", "ii tim"]),
    ("Titus",          ["titus", "tit"]),
    ("Philemon",       ["philemon", "phlm", "phm"]),
    # General epistles
    ("Hebrews",        ["hebrews", "heb"]),
    ("James",          ["james", "jas", "jam"]),
    ("1 Peter",        ["1 peter", "1 pet", "1 pt", "i peter", "i pet"]),
    ("2 Peter",        ["2 peter", "2 pet", "2 pt", "ii peter", "ii pet"]),
    ("1 John",         ["1 john", "1 jn", "i john"]),
    ("2 John",         ["2 john", "2 jn", "ii john"]),
    ("3 John",         ["3 john", "3 jn", "iii john"]),
    ("Jude",           ["jude"]),
    ("Revelation",     ["revelation", "rev", "rv", "apocalypse"]),
]

# Build alias -> canonical lookup.
ALIAS_TO_BOOK: dict[str, str] = {}
for canon, aliases in BOOKS:
    for a in aliases:
        ALIAS_TO_BOOK[a] = canon

# Regex: match (optional 1/2/3/I/II/III prefix)? + book word(s) + space + ch[:vs[-vs]]
# Build the alternation from longest aliases first so e.g. "Song of Songs" wins
# over "Song".
_aliases_sorted = sorted(ALIAS_TO_BOOK.keys(), key=len, reverse=True)
_alias_pattern = "|".join(re.escape(a) for a in _aliases_sorted)

REF_RE = re.compile(
    rf"\b({_alias_pattern})\.?\s*(\d+)(?:\s*[:.,]\s*(\d+)(?:\s*[\-\u2013]\s*(\d+))?)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BibleRef:
    book: str
    chapter: int
    verse_start: int | None
    verse_end: int | None
    raw: str
    normalized: str

    def to_dict(self) -> dict:
        return {
            "book": self.book,
            "chapter": self.chapter,
            "verse_start": self.verse_start,
            "verse_end": self.verse_end,
            "raw": self.raw,
            "normalized": self.normalized,
        }


def _normalize(book: str, chapter: int, vs: int | None, ve: int | None) -> str:
    if vs is None:
        return f"{book} {chapter}"
    if ve is None or ve == vs:
        return f"{book} {chapter}:{vs}"
    return f"{book} {chapter}:{vs}-{ve}"


def find_refs(text: str) -> list[BibleRef]:
    if not text:
        return []
    out: list[BibleRef] = []
    seen: set[str] = set()
    for m in REF_RE.finditer(text):
        alias_raw = m.group(1).lower().strip()
        book = ALIAS_TO_BOOK.get(alias_raw)
        if not book:
            continue
        chapter = int(m.group(2))
        vs = int(m.group(3)) if m.group(3) else None
        ve = int(m.group(4)) if m.group(4) else None
        if ve is not None and vs is not None and ve < vs:
            ve = None
        normalized = _normalize(book, chapter, vs, ve)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(BibleRef(
            book=book,
            chapter=chapter,
            verse_start=vs,
            verse_end=ve,
            raw=m.group(0),
            normalized=normalized,
        ))
    return out


def normalize_query_ref(query: str) -> str | None:
    """If `query` looks like a single Bible reference, return its normalized form."""
    refs = find_refs(query.strip())
    if len(refs) == 1 and refs[0].raw.strip().lower() == query.strip().lower():
        return refs[0].normalized
    if len(refs) == 1:
        # Allow some surrounding text but only if the ref dominates
        return refs[0].normalized
    return None


def all_book_names() -> list[str]:
    return [canon for canon, _ in BOOKS]
