# Bible Study Virtual Assistant

A class project: an LLM-based virtual assistant that answers Bible-study
questions by combining open-source LLMs with structured retrieval over a
unified, labeled corpus (Bible + BEMA podcast transcripts + BEMA show
notes + BEMA site pages + Marty Solomon YouTube captions + web search).

The goal is **grounded answers**, not free-form recall: every response is
generated from evidence retrieved through tools, not the model's memory.

## Status

| Component | State |
|---|---|
| Folder structure + 10 source modules + 4 scripts | done |
| Bible JSON (KJV, 31,100 verses) | done |
| BEMA transcripts (259 episodes, ~1.65 M words) | done |
| BEMA episode-page summaries + study tools (507 pages) | done |
| BEMA static site pages (5 pages) | done |
| YouTube captions for videos linked from BEMA pages (~400 candidates) | done |
| Unified labeled corpus (`data/corpus/corpus.jsonl`) | done |
| Bible-reference detector (66 books, 200+ aliases) | done |
| Per-chunk metadata: `source_type`, `title`, `url`, `episode`, `date`, `chunk_index`, `verse_refs` | done |
| Dedup of near-identical chunks | done |
| Filterable retrieval (by `source_type`, `episode`, `verse_ref`) | done |
| Gradio UI with filters + technique + model selector | done |
| Agent routing + prompt assembly + 3 prompting techniques | done |
| 5 prompt-injection security tests | done |
| MockLLM (no-GPU stand-in for the agent pipeline) | done |
| Real LLMs: Phi-3.5-mini + Mistral-7B | run in Colab |
| pytest smoke suite | passing |

## Features

- **Two open-source LLMs** for comparison
  - Smaller: `microsoft/Phi-3.5-mini-instruct`
  - Larger:  `mistralai/Mistral-7B-Instruct-v0.3` (4-bit on Colab GPU)
- **Tools**
  - Unified TF-IDF retriever over a labeled corpus
  - Bible verse lookup (structured KJV JSON, 31 K verses)
  - BEMA Discipleship Podcast transcripts, show-notes summaries, study-tool links, static site pages
  - YouTube captions for videos linked from BEMA episode pages
  - Web search (DuckDuckGo, no API key required)
  - Filter retrieval by `source_type`, `episode`, or `verse_ref`
- **3 prompting techniques** explored: zero-shot, few-shot, chain-of-thought
- **Quality / latency benchmarking** across models
- **5 prompt-injection security tests**
- **Gradio UI** (`src/app.py`)
- Final deliverable runs in **Google Colab** (free T4 GPU is enough)

## Repo layout

```
bible-study-va/
├── README.md
├── requirements.txt
├── TODO.md
├── data/                              (everything in here is gitignored)
│   ├── bible/
│   │   └── bible.json                 # 31,100 verses
│   ├── bema/
│   │   ├── episodes.json              # metadata for all 507 episodes
│   │   ├── transcripts/               # 259 .txt transcripts
│   │   ├── episode_pages.json         # summary + study tools per page
│   │   ├── site_pages.json            # static site pages (about, resources, ...)
│   │   └── log.txt                    # scrape log
│   ├── youtube/
│   │   ├── videos.json                # video metadata
│   │   ├── transcripts/               # .txt captions per video
│   │   └── log.txt
│   └── corpus/
│       ├── corpus.jsonl               # unified labeled chunks
│       ├── stats.json                 # build summary
│       └── log.txt
├── src/
│   ├── bible_tool.py                  # legacy single-source tool (still tested)
│   ├── bema_tool.py                   # legacy single-source tool (still tested)
│   ├── web_tool.py                    # DuckDuckGo
│   ├── retriever.py                   # shared TF-IDF helper
│   ├── corpus.py                      # unified labeled corpus retriever
│   ├── bible_refs.py                  # Bible-reference detection / normalization
│   ├── llm.py                         # Phi-3.5 / Mistral / MockLLM loaders
│   ├── prompts.py                     # zero/few-shot + CoT
│   ├── agent.py                       # router + answer pipeline
│   ├── security.py                    # 5 prompt-injection attacks
│   └── app.py                         # Gradio UI
├── scripts/
│   ├── load_bible.py                  # downloads + normalizes Bible JSON
│   ├── scrape_bema.py                 # scrapes BEMA transcripts (Google Docs)
│   ├── scrape_bema_pages.py           # scrapes BEMA episode pages + static site pages
│   ├── scrape_youtube.py              # captions for videos linked from BEMA pages
│   └── build_corpus.py                # unifies all sources into corpus.jsonl
├── tests/
│   ├── conftest.py                    # makes `src` importable for pytest
│   └── test_smoke.py                  # fast tests, no LLM needed
└── notebooks/
    └── bible_study_va.ipynb           # the deliverable
```

## Quickstart — local (Mac / any laptop, no GPU)

This gets you a working agent pipeline + Gradio UI with the **MockLLM**
stand-in. Real LLMs need Colab (next section).

```bash
# 1) sandbox + deps  (~90s, ~2 GB)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) data  (one-time, total ~30 min)
python scripts/load_bible.py            # ~10s
python scripts/scrape_bema.py           # ~9 min (or --max-episodes N for a quick test)
python scripts/scrape_bema_pages.py     # ~12 min
python scripts/scrape_youtube.py        # ~7 min (skips videos without captions)
python scripts/build_corpus.py          # ~10s

# 3) sanity check
pytest tests/ -v                        # ~5 seconds

# 4) launch the Gradio UI
python -m src.app                       # http://127.0.0.1:7860
```

## Quickstart — Google Colab (real LLMs)

1. Open `notebooks/bible_study_va.ipynb` in Colab
   (`File → Open notebook → GitHub → emilymoberly/bible-study-va`).
2. **Runtime → Change runtime type → T4 GPU.**
3. Run all cells top to bottom. The first run downloads ~10 GB of model
   weights from Hugging Face.

To launch the Gradio UI in Colab and get a public URL:

```python
!pip install -q -r requirements.txt
!python scripts/load_bible.py
!python scripts/scrape_bema.py
!python scripts/scrape_bema_pages.py
!python scripts/scrape_youtube.py
!python scripts/build_corpus.py
from src.app import build_app
build_app().launch(share=True)
```

## Example questions

- What are hidden meanings and symbols behind Babylon in the Bible?
- What was the common Jewish cultural acceptance of wine?
- How did the Roman Empire clash with the Jews?
- What is a chiasm and what are examples in Deuteronomy?
- What is the difference between a Pharisee and a teacher of the law?
- John 3:16     (filter: Bible only)

## Known limitations

- **BEMA corpus is 259 / 507 episodes.** The remaining 248 episodes do
  not have transcripts published. We still index their show-notes summary
  and study-tool links from the episode pages.
- **YouTube coverage** is limited to videos linked from BEMA episode
  pages, with only those that have public captions enabled. We don't
  enumerate Marty Solomon's full YouTube channel — that would require
  a different API and isn't necessary for the class project.
- **TF-IDF is keyword-based.** Vague questions ("hidden meanings behind
  X") may retrieve weak matches because TF-IDF over-weights the vague
  words. Sentence-transformer embeddings would help and are an obvious
  v2 upgrade.
- **Phi-3.5 / Mistral-7B do not run on a Mac with < 16 GB RAM.** Use Colab.

## Notes for graders

- All models are open source and downloaded from Hugging Face at runtime.
- The Bible text used is the public-domain **King James Version (KJV)**.
- BEMA transcripts and episode pages are scraped from publicly-available
  Google Docs and `bemadiscipleship.com` pages, for educational,
  non-commercial use, with polite delays between requests.
- YouTube captions are fetched only when publicly enabled by the video
  owner; no API key is used.
