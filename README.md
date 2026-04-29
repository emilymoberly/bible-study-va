# Bible Study Virtual Assistant

A class project: an LLM-based virtual assistant that answers Bible-study
questions by combining open-source LLMs with structured retrieval tools
(Bible verse lookup, BEMA podcast transcripts, web search).

The goal is **grounded answers**, not free-form recall: every response is
generated from evidence retrieved through tools, not the model's memory.

## Features

- **Two open-source LLMs** for comparison
  - Smaller: `microsoft/Phi-3.5-mini-instruct`
  - Larger:  `mistralai/Mistral-7B-Instruct-v0.3`
- **Tools**
  - Bible verse lookup (structured JSON Bible)
  - BEMA Discipleship Podcast transcript retrieval (TF-IDF over scraped transcripts)
  - Web search (DuckDuckGo, no API key required)
- **3 prompting techniques** explored: zero-shot, few-shot, chain-of-thought
- **Quality / latency benchmarking** across models
- **5 prompt-injection security tests**
- Final deliverable runs in **Google Colab**

## Repo layout

```
bible-study-va/
├── README.md
├── requirements.txt
├── TODO.md
├── data/
│   ├── bible/              # bible.json (verses)
│   └── bema/
│       ├── episodes.json   # episode metadata
│       └── transcripts/    # one .txt per episode
├── src/
│   ├── bible_tool.py       # verse lookup
│   ├── bema_tool.py        # transcript retrieval (TF-IDF)
│   ├── web_tool.py         # DuckDuckGo web search
│   ├── retriever.py        # shared TF-IDF helper
│   ├── llm.py              # load + run Phi-3.5 / Mistral-7B
│   ├── prompts.py          # 3 prompting techniques
│   ├── agent.py            # tool router + answer generator
│   └── security.py         # 5 prompt-injection tests
├── scripts/
│   ├── load_bible.py       # download + normalize Bible JSON
│   └── scrape_bema.py      # scrape BEMA transcripts
└── notebooks/
    └── bible_study_va.ipynb  # the deliverable
```

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) load Bible into data/bible/bible.json
python scripts/load_bible.py

# 2) scrape a few BEMA episodes (start small to verify pipeline)
python scripts/scrape_bema.py --max-episodes 5

# 3) open the notebook
jupyter lab notebooks/bible_study_va.ipynb
```

## Quick start (Google Colab)

Open `notebooks/bible_study_va.ipynb` in Colab. The first cells clone this
repo, install dependencies, and download the Bible + BEMA data.

## Example questions

- What are hidden meanings and symbols behind Babylon in the Bible?
- What was the common Jewish cultural acceptance of wine?
- How did the Roman Empire clash with the Jews?
- What is a chiasm and what are examples in Deuteronomy?
- What is the difference between a Pharisee and a teacher of the law?

## Notes for graders

- All models are open source and downloaded from Hugging Face at runtime.
- The Bible text used is the public-domain **World English Bible (WEB)**.
- BEMA transcripts are scraped from the publicly-available
  [BEMA Discipleship](https://www.bemadiscipleship.com/) site for
  educational, non-commercial use.
