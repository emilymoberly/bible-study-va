"""
Smoke tests — fast, no LLMs required.

Run with:  pytest -v

Skips gracefully if a particular data file isn't present yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BIBLE_JSON = REPO / "data" / "bible" / "bible.json"
BEMA_DIR = REPO / "data" / "bema" / "transcripts"
BEMA_JSON = REPO / "data" / "bema" / "episodes.json"
CORPUS_JSONL = REPO / "data" / "corpus" / "corpus.jsonl"


# ---------------------------------------------------------------------------
# bible_tool (legacy single-source tool — still ships in src/)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bible():
    if not BIBLE_JSON.exists():
        pytest.skip("Bible JSON missing — run `python scripts/load_bible.py` first")
    from src.bible_tool import BibleTool
    return BibleTool(BIBLE_JSON)


def test_bible_loaded(bible):
    assert len(bible.verses) > 30_000


def test_bible_lookup_john_3_16(bible):
    verses = bible.lookup("John 3:16")
    assert len(verses) == 1
    assert "loved the world" in verses[0].text.lower()


def test_bible_lookup_range(bible):
    verses = bible.lookup("Genesis 1:1-3")
    assert len(verses) == 3


def test_bible_lookup_alias(bible):
    verses = bible.lookup("Gen 1:1")
    assert len(verses) == 1


def test_bible_search_babylon(bible):
    hits = bible.search("babylon great fallen", k=5)
    assert hits
    refs = [h.item.reference for h in hits]
    assert "Revelation 18:2" in refs


# ---------------------------------------------------------------------------
# bema_tool (legacy)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bema():
    if not any(BEMA_DIR.glob("*.txt")):
        pytest.skip("BEMA transcripts missing")
    from src.bema_tool import BemaTool
    return BemaTool(BEMA_DIR, BEMA_JSON if BEMA_JSON.exists() else None)


def test_bema_chunks_loaded(bema):
    assert len(bema.chunks) > 0


def test_bema_search_chiasm(bema):
    hits = bema.search("chiasm structure parallel literary", k=5)
    assert hits


# ---------------------------------------------------------------------------
# Bible reference detector
# ---------------------------------------------------------------------------

def test_bible_refs_simple():
    from src.bible_refs import find_refs
    refs = find_refs("see John 3:16 and Genesis 1:1-3")
    norms = {r.normalized for r in refs}
    assert "John 3:16" in norms
    assert "Genesis 1:1-3" in norms


def test_bible_refs_no_false_positives():
    from src.bible_refs import find_refs
    refs = find_refs("Marty was on at 4:00 today")
    assert refs == []


def test_bible_refs_song_of_solomon():
    from src.bible_refs import find_refs
    refs = find_refs("read Song of Solomon 4:1 today")
    assert any(r.normalized == "Song of Solomon 4:1" for r in refs)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_prompt_techniques_render():
    from src.prompts import TECHNIQUES, zero_shot
    pair = zero_shot("What is a chiasm?", "[BEMA 1] chiasm is a literary structure...")
    assert "QUESTION" in pair.user
    assert "EVIDENCE" in pair.user
    assert pair.system
    assert set(TECHNIQUES) == {"zero_shot", "few_shot", "chain_of_thought"}


# ---------------------------------------------------------------------------
# New agent routing (Corpus-based)
# ---------------------------------------------------------------------------

def test_routing_default_includes_bible_and_bema():
    from src.agent import route
    d = route("What does the bible say about Pharisees?")
    assert "bible" in d.source_types
    assert "bema_transcript" in d.source_types


def test_routing_detects_verse_ref():
    from src.agent import route
    d = route("Explain John 3:16")
    assert d.detected_verse_ref == "John 3:16"


def test_routing_uses_web_for_history():
    from src.agent import route
    d = route("How did the Roman Empire clash with the Jews?")
    assert d.use_web


def test_routing_explicit_source_types():
    from src.agent import route
    d = route("anything", explicit_source_types=["bible"])
    assert d.source_types == ["bible"]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    if not CORPUS_JSONL.exists():
        pytest.skip("corpus.jsonl missing — run `python scripts/build_corpus.py`")
    from src.corpus import Corpus
    return Corpus(CORPUS_JSONL)


def test_corpus_loaded(corpus):
    assert len(corpus.chunks) > 1000  # at least bible + some bema


def test_corpus_stats(corpus):
    stats = corpus.stats()
    assert stats["total_chunks"] == len(corpus.chunks)
    assert "bible" in stats["by_source_type"]


def test_corpus_search_bible_only(corpus):
    hits = corpus.search("babylon great fallen", k=3, source_types=["bible"])
    assert hits
    assert all(h.chunk.source_type == "bible" for h in hits)


def test_corpus_search_bema_only(corpus):
    hits = corpus.search("chiasm structure", k=3,
                         source_types=["bema_transcript"])
    assert hits
    assert all(h.chunk.source_type == "bema_transcript" for h in hits)


def test_corpus_filter_by_episode(corpus):
    hits = corpus.search("anything", k=3,
                         source_types=["bema_transcript"], episode="1")
    # All returned chunks should be from episode 1 (or empty if ep 1 not in corpus)
    assert all(h.chunk.episode == "1" for h in hits)


def test_corpus_filter_by_verse_ref(corpus):
    hits = corpus.lookup_verse("John 3:16")
    assert len(hits) == 1
    assert hits[0].text.lower().startswith("for god so loved")
