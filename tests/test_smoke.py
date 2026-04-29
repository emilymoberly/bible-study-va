"""
Smoke tests — fast, no LLMs required.

Run with:  pytest -v

What these tests prove:
- Bible JSON loaded and lookup/search work
- BEMA transcripts loaded and retrieval finds reasonable hits
- Prompting templates render
- Routing picks tools sensibly

If Bible or BEMA data is missing, the relevant tests are skipped (not failed)
so a fresh checkout can run `pytest -v` without first running the data scripts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BIBLE_JSON = REPO / "data" / "bible" / "bible.json"
BEMA_DIR = REPO / "data" / "bema" / "transcripts"
BEMA_JSON = REPO / "data" / "bema" / "episodes.json"


# ---------------------------------------------------------------------------
# Bible tool
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bible():
    if not BIBLE_JSON.exists():
        pytest.skip("Bible JSON missing — run `python scripts/load_bible.py` first")
    from src.bible_tool import BibleTool
    return BibleTool(BIBLE_JSON)


def test_bible_loaded(bible):
    assert len(bible.verses) > 30_000  # KJV has 31,102 verses


def test_bible_lookup_john_3_16(bible):
    verses = bible.lookup("John 3:16")
    assert len(verses) == 1
    assert "loved the world" in verses[0].text.lower()


def test_bible_lookup_range(bible):
    verses = bible.lookup("Genesis 1:1-3")
    assert len(verses) == 3
    assert "in the beginning" in verses[0].text.lower()


def test_bible_lookup_alias(bible):
    verses = bible.lookup("Gen 1:1")
    assert len(verses) == 1
    assert verses[0].book == "Genesis"


def test_bible_search_babylon(bible):
    hits = bible.search("babylon great fallen", k=5)
    assert hits, "expected non-empty results"
    refs = [h.item.reference for h in hits]
    # Revelation 18:2 is the canonical "Babylon the great is fallen" verse.
    assert "Revelation 18:2" in refs


# ---------------------------------------------------------------------------
# BEMA tool
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bema():
    if not any(BEMA_DIR.glob("*.txt")):
        pytest.skip("BEMA transcripts missing — run `python scripts/scrape_bema.py`")
    from src.bema_tool import BemaTool
    return BemaTool(BEMA_DIR, BEMA_JSON if BEMA_JSON.exists() else None)


def test_bema_chunks_loaded(bema):
    assert len(bema.chunks) > 0
    assert all(len(c.text) > 100 for c in bema.chunks[:5])


def test_bema_search_chiasm(bema):
    hits = bema.search("chiasm structure parallel literary", k=5)
    assert hits
    # BEMA 1 "Trust the Story" introduces chiasms.
    top_episodes = {h.item.episode_number for h in hits[:3]}
    assert "1" in top_episodes or "7" in top_episodes


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_prompt_techniques_render():
    from src.prompts import TECHNIQUES, zero_shot
    pair = zero_shot("What is a chiasm?", "[BEMA 1] chiasm is a literary structure...")
    assert "QUESTION" in pair.user
    assert "EVIDENCE" in pair.user
    assert pair.system  # non-empty
    assert set(TECHNIQUES) == {"zero_shot", "few_shot", "chain_of_thought"}


# ---------------------------------------------------------------------------
# Agent routing
# ---------------------------------------------------------------------------

def test_routing_recognizes_bible_keywords():
    from src.agent import route
    d = route("What is the difference between a Pharisee and a teacher of the law?")
    assert d.use_bible


def test_routing_extracts_direct_refs():
    from src.agent import route
    d = route("Explain John 3:16 and Genesis 1:1")
    assert "John 3:16" in d.direct_refs
    assert "Genesis 1:1" in d.direct_refs


def test_routing_uses_web_for_history():
    from src.agent import route
    d = route("How did the Roman Empire clash with the Jews?")
    assert d.use_web
