"""
Download a public-domain Bible and normalize it to a flat JSON file.

Why this script exists
----------------------
Our Bible tool needs a *structured* representation of the whole Bible so we
can do verse lookups ("John 3:16") and TF-IDF search across all verses.
Different sources publish the Bible in slightly different shapes; this script
hides those differences from the rest of the codebase by writing a single
canonical file:

    data/bible/bible.json

with one JSON object per verse:

    {"book": "John", "chapter": 3, "verse": 16, "text": "For God so loved..."}

Run it once. The notebook also calls it on first launch in Colab.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import requests

# Public-domain sources, tried in order. Each entry is (url, parser_name).
# We use the King James Version (KJV) by default — it's public domain and
# universally available. The thiagobodruk/bible repo bundles the entire
# Bible into a single JSON file, which keeps this script trivial.
SOURCES = [
    (
        "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json",
        "thiagobodruk",
    ),
    (
        "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_bbe.json",
        "thiagobodruk",
    ),
]

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "bible" / "bible.json"


def parse_thiagobodruk(raw: list) -> Iterable[dict]:
    """
    Source schema:
        [
          {"abbrev": "gn", "book": "Genesis",
           "chapters": [["v1 text", "v2 text", ...], [...]]},
          ...
        ]
    """
    for book in raw:
        book_name = book["book"]
        for chap_idx, chapter in enumerate(book["chapters"], start=1):
            for verse_idx, text in enumerate(chapter, start=1):
                yield {
                    "book": book_name,
                    "chapter": chap_idx,
                    "verse": verse_idx,
                    "text": text.strip(),
                }


PARSERS = {"thiagobodruk": parse_thiagobodruk}


def download_bible() -> list[dict]:
    last_err: Exception | None = None
    for url, parser_name in SOURCES:
        try:
            print(f"[load_bible] downloading {url}")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            # Some mirrors serve UTF-8 with BOM; json.loads handles it via
            # response.content -> decoded string.
            raw = json.loads(r.content.decode("utf-8-sig"))
            verses = list(PARSERS[parser_name](raw))
            print(f"[load_bible] parsed {len(verses):,} verses from {url}")
            return verses
        except Exception as e:  # noqa: BLE001 - we want to fall through to next mirror
            print(f"[load_bible] source failed ({e}); trying next mirror")
            last_err = e
    raise RuntimeError(f"All Bible sources failed. Last error: {last_err}")


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        print(f"[load_bible] {OUTPUT} already exists — skipping. "
              "Delete it and rerun to refresh.")
        return 0

    verses = download_bible()
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"[load_bible] wrote {len(verses):,} verses to {OUTPUT} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
