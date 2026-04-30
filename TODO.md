# Development Plan / TODO

Items marked **[req]** map directly to the 6 class requirements.

---

## Milestone 0 — Project skeleton  DONE
- [x] Folder structure (`data/`, `src/`, `scripts/`, `notebooks/`, `tests/`)
- [x] `README.md`, `requirements.txt`, `.gitignore`, `TODO.md`
- [x] All Python modules in `src/` so imports work end-to-end
- [x] First commit pushed to GitHub

## Milestone 1 — Data pipelines  DONE
- [x] **Bible loader** writes 31,100 verses to `data/bible/bible.json`
- [x] **BEMA transcript scraper** wrote 259 episode transcripts (~1.65 M words)
- [x] **BEMA episode-page scraper** captured summary + study-tool links for
      all 507 pages, plus 5 static site pages (about, support, groups,
      subscribe, resources)
- [x] **YouTube scraper** discovered ~400 videos linked from BEMA pages and
      saved captions for those that have public captions
- [x] **Corpus builder** (`scripts/build_corpus.py`) unifies everything into
      `data/corpus/corpus.jsonl` with labeled, deduped, verse-tagged chunks
- [x] **Stats + log files** at `data/corpus/stats.json` + `data/corpus/log.txt`

## Milestone 2 — Retrieval tools  DONE  **[req: tools]**
- [x] `src/bible_tool.py` — `lookup("John 3:16")`, alias support, TF-IDF
- [x] `src/bema_tool.py` — chunked TF-IDF over transcripts
- [x] `src/web_tool.py` — DuckDuckGo (no API key)
- [x] `src/retriever.py` — generic TF-IDF helper
- [x] `src/corpus.py` — unified retriever with `source_type` / `episode` /
      `verse_ref` filters
- [x] `src/bible_refs.py` — Bible-reference detector / normalizer (66 books,
      200+ aliases)

## Milestone 3 — LLM wrappers  COLAB-ONLY for real models  **[req: 2 LLMs]**
- [x] `src/llm.py` — `LLM.load("phi3" | "mistral")`, returns latency + tokens
- [x] MPS support added (for Macs with enough RAM)
- [x] `MockLLM` stand-in for laptops without a GPU
- [ ] Run real Phi-3.5 and Mistral-7B in Colab and verify outputs

## Milestone 4 — Prompting techniques  DONE  **[req: 3 techniques]**
- [x] `src/prompts.py`
  - `zero_shot(question, evidence)`
  - `few_shot(question, evidence)` (2 worked examples)
  - `chain_of_thought(question, evidence)` (`<scratchpad>` block)
- [x] All techniques share a hardened system prompt that explicitly tells
      the model to treat retrieved evidence as untrusted DATA

## Milestone 5 — Agent / tool router  DONE
- [x] `src/agent.py` rewritten on top of the unified Corpus
- [x] Direct verse-reference detection via `src/bible_refs.py`
- [x] Filter parameters threaded through `Agent.answer(...)`
- [x] End-to-end pipeline tested locally with MockLLM

## Milestone 6 — Gradio UI  DONE
- [x] `src/app.py` — minimal one-screen Gradio Blocks app
- [x] Question / theme / verse input
- [x] Filter dropdown: All sources / Bible only / BEMA only / BEMA transcripts only / YouTube only / Bible + BEMA
- [x] Optional verse + episode filters
- [x] Prompting-technique dropdown
- [x] Model dropdown (`mock` default; `phi3` / `mistral` lazy-loaded)
- [x] Sources table with clickable URLs
- [x] Colab launch instructions in the README

## Milestone 7 — Evaluation  NEEDS COLAB RUN  **[req: compare quality + latency]**
- [x] Notebook scaffolding: 5 questions × 2 models × 3 techniques table
- [x] Latency measurement built into `LLM.generate()`
- [ ] Run the comparison cell in Colab
- [ ] Add manual 1–5 quality scores in the next cell
- [ ] Plot mean latency by model × technique

## Milestone 8 — Security tests  NEEDS COLAB RUN  **[req: 5 attacks]**
- [x] `src/security.py` — 5 attacks with regex success-detectors
- [x] Framework verified locally with MockLLM
- [ ] Run against real Phi-3.5 in Colab and document which attacks slip through

## Milestone 9 — Notebook deliverable  NEEDS COLAB RUN  **[req: Colab]**
- [x] `notebooks/bible_study_va.ipynb` cells in correct order
- [x] Cell 2 references the actual repo URL
- [x] Cell 3a (MockLLM) for local development
- [x] Memory tips added to cell 4 (LLM load)
- [ ] Run end-to-end on Colab T4
- [ ] Capture cell outputs and save the notebook with results

## Milestone 10 — Tests  DONE
- [x] `tests/test_smoke.py` covers Bible tool, BEMA tool, prompts,
      bible-ref detector, agent routing, **and** unified corpus filters
- [x] `tests/conftest.py` so `src` imports under pytest
- [x] All tests passing locally

---

## Outstanding work, ordered

1. **Push current state to GitHub** so Colab can pull it.
2. **Open notebook in Colab**, switch runtime to T4, run all cells.
3. **Fill in manual quality scores** for the 30 generated answers.
4. **Document which security attacks slipped through.**
5. **Write the discussion section** — this is the part the rubric
   actually grades.

## Stretch (only if time permits)
- [ ] Cross-reference tool (Treasury of Scripture Knowledge JSON)
- [ ] Swap TF-IDF for sentence-transformers embeddings
- [ ] Enumerate Marty Solomon's full YouTube channel via yt-dlp
- [ ] Per-source quality scoring in the Gradio UI
