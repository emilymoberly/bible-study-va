# Bible Study Virtual Assistant

A class project: an LLM-based virtual assistant that answers Bible-study
questions by combining open-source LLMs with structured retrieval tools
(Bible verse lookup, BEMA podcast transcripts, web search).

The goal is **grounded answers**, not free-form recall: every response is
generated from evidence retrieved through tools, not the model's memory.

## Status (snapshot)

| Component | State |
|---|---|
| Folder structure + 8 source modules + 2 scripts | ✅ written |
| Bible JSON (KJV, 31,100 verses, 6.3 MB) | ✅ downloaded |
| BEMA transcripts (16 episodes, ~126 K words) | ✅ scraped |
| Bible tool (lookup + TF-IDF search) | ✅ tested |
| BEMA tool (chunked TF-IDF search) | ✅ tested |
| Web tool (DuckDuckGo) | ✅ written, untested locally |
| Agent routing + prompt assembly | ✅ tested |
| 3 prompting techniques | ✅ written |
| 5 prompt-injection tests | ✅ framework verified (real test runs in Colab) |
| MockLLM (no-GPU stand-in for the agent pipeline) | ✅ works |
| Real LLMs: Phi-3.5-mini + Mistral-7B | ⏳ run in Colab (need GPU) |
| pytest smoke suite | ✅ 11/11 passing |

## Features

- **Two open-source LLMs** for comparison
  - Smaller: `microsoft/Phi-3.5-mini-instruct`
  - Larger:  `mistralai/Mistral-7B-Instruct-v0.3` (4-bit on Colab GPU)
- **Tools**
  - Bible verse lookup (structured KJV JSON, 31 K verses)
  - BEMA Discipleship Podcast transcript retrieval (TF-IDF over scraped transcripts)
  - Web search (DuckDuckGo, no API key required)
- **3 prompting techniques** explored: zero-shot, few-shot, chain-of-thought
- **Quality / latency benchmarking** across models
- **5 prompt-injection security tests**
- Final deliverable runs in **Google Colab** (free T4 GPU is enough)

## Repo layout

```
bible-study-va/
├── README.md
├── requirements.txt
├── TODO.md
├── data/
│   ├── bible/
│   │   └── bible.json            # 31,100 verses, ~6 MB (gitignored)
│   └── bema/
│       ├── episodes.json         # episode metadata (gitignored)
│       └── transcripts/          # 16 .txt files (gitignored)
├── src/
│   ├── bible_tool.py             # verse lookup + search
│   ├── bema_tool.py              # transcript chunking + TF-IDF
│   ├── web_tool.py               # DuckDuckGo
│   ├── retriever.py              # shared TF-IDF helper
│   ├── llm.py                    # Phi-3.5 / Mistral / MockLLM loaders
│   ├── prompts.py                # zero/few-shot + CoT
│   ├── agent.py                  # rule-based router + answer pipeline
│   └── security.py               # 5 prompt-injection attacks
├── scripts/
│   ├── load_bible.py             # downloads + normalizes Bible JSON
│   └── scrape_bema.py            # scrapes BEMA transcripts
├── tests/
│   ├── conftest.py               # makes `src` importable for pytest
│   └── test_smoke.py             # 11 fast tests, no LLM needed
└── notebooks/
    └── bible_study_va.ipynb      # the deliverable
```

## Quickstart — local (Mac / any laptop, no GPU)

This gets you a working agent pipeline with the **MockLLM** stand-in. Real
LLMs need Colab (next section).

```bash
# 1) sandbox + deps  (~90s, ~2 GB)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) data  (~10s + ~35s)
python scripts/load_bible.py
python scripts/scrape_bema.py --max-episodes 30

# 3) sanity check
pytest tests/ -v        # 11 tests, ~3 seconds

# 4) interactive demo with MockLLM
python -c "
from src.agent import Agent
from src.llm import MockLLM
agent = Agent('data/bible/bible.json',
              'data/bema/transcripts',
              'data/bema/episodes.json',
              enable_web=False)
print(agent.answer('What is a chiasm?', MockLLM(), 'zero_shot').text)
"
```

## Quickstart — Google Colab (real LLMs)

1. Open `notebooks/bible_study_va.ipynb` in Colab
   (`File → Open notebook → GitHub → emilymoberly/bible-study-va`).
2. **Runtime → Change runtime type → T4 GPU.**
3. Run all cells top to bottom. The first run downloads ~10 GB of model
   weights from Hugging Face.

## Example questions

- What are hidden meanings and symbols behind Babylon in the Bible?
- What was the common Jewish cultural acceptance of wine?
- How did the Roman Empire clash with the Jews?
- What is a chiasm and what are examples in Deuteronomy?
- What is the difference between a Pharisee and a teacher of the law?

## Known limitations

- **BEMA corpus is partial** (16 of ~400 episodes). The bemadiscipleship.com
  archive lazy-loads with JS; only the static portion was scraped. Adding
  Selenium/Playwright to the scraper would unlock the full corpus.
- **TF-IDF is keyword-based.** Vague questions ("hidden meanings behind X")
  may retrieve weak matches because TF-IDF over-weights the vague words. A
  switch to sentence-transformer embeddings would help.
- **Phi-3.5 / Mistral-7B do not run on a Mac with < 16 GB RAM.** Use Colab.

## Notes for graders

- All models are open source and downloaded from Hugging Face at runtime.
- The Bible text used is the public-domain **King James Version (KJV)**.
- BEMA transcripts are scraped from publicly-available Google Docs linked
  from [BEMA Discipleship](https://www.bemadiscipleship.com/) episode pages,
  for educational, non-commercial use.
