# Development Plan / TODO

Items marked **[req]** map directly to the 6 class requirements.

---

## Milestone 0 — Project skeleton  ✅ DONE
- [x] Folder structure (`data/`, `src/`, `scripts/`, `notebooks/`, `tests/`)
- [x] `README.md`, `requirements.txt`, `.gitignore`, `TODO.md`
- [x] All Python modules in `src/` so imports work end-to-end
- [x] First commit pushed to GitHub

## Milestone 1 — Data pipelines  ✅ DONE
- [x] **Bible loader** writes 31,100 verses to `data/bible/bible.json`
- [x] **BEMA scraper** wrote 16 episode transcripts (~126 K words)
  - Note: only the first 16 episodes are in the static HTML on
    bemadiscipleship.com; the full archive uses JS pagination.
    Stretch goal: switch to Playwright to scrape all ~400 episodes.

## Milestone 2 — Retrieval tools  ✅ DONE  **[req: tools]**
- [x] `src/bible_tool.py` — `lookup("John 3:16")`, `lookup("Gen 1:1-3")`,
      alias support, TF-IDF `search(query, k)`
- [x] `src/bema_tool.py` — chunks transcripts (400 words / 50 overlap)
      and TF-IDF search across them
- [x] `src/web_tool.py` — DuckDuckGo (no API key)
- [x] `src/retriever.py` — generic TF-IDF helper

## Milestone 3 — LLM wrappers  ⏳ COLAB-ONLY for real models  **[req: 2 LLMs]**
- [x] `src/llm.py` — `LLM.load("phi3" | "mistral")`, returns latency + tokens
- [x] MPS support added (for Macs with enough RAM)
- [x] `MockLLM` stand-in for laptops without a GPU
- [ ] Run real Phi-3.5 and Mistral-7B in Colab and verify outputs

## Milestone 4 — Prompting techniques  ✅ DONE  **[req: 3 techniques]**
- [x] `src/prompts.py`
  - `zero_shot(question, evidence)`
  - `few_shot(question, evidence)` (2 worked examples)
  - `chain_of_thought(question, evidence)` (`<scratchpad>` block)
- [x] All techniques share a hardened system prompt that explicitly tells
      the model to treat retrieved evidence as untrusted DATA

## Milestone 5 — Agent / tool router  ✅ DONE
- [x] `src/agent.py` — keyword-based routing
- [x] Direct verse-reference detection (`John 3:16`, `1 Cor 13:4-7`)
- [x] End-to-end pipeline tested locally with MockLLM

## Milestone 6 — Evaluation  ⏳ NEEDS COLAB RUN  **[req: compare quality + latency]**
- [x] Notebook scaffolding: 5 questions × 2 models × 3 techniques table
- [x] Latency measurement built into `LLM.generate()`
- [ ] Run the comparison cell in Colab
- [ ] Add manual 1–5 quality scores in the next cell
- [ ] Plot mean latency by model × technique

## Milestone 7 — Security tests  ⏳ NEEDS COLAB RUN  **[req: 5 attacks]**
- [x] `src/security.py` — 5 attacks with regex success-detectors:
  1. Direct override ("Ignore previous instructions…")
  2. Role hijack ("BibleDAN")
  3. Indirect injection (poisoned BEMA chunk)
  4. System-prompt exfiltration
  5. Off-topic abuse (network-scanning code)
- [x] Framework verified locally with MockLLM (5/5 trivially blocked)
- [ ] Run against real Phi-3.5 in Colab and document which attacks slip through

## Milestone 8 — Notebook deliverable  ⏳ NEEDS COLAB RUN  **[req: Colab]**
- [x] `notebooks/bible_study_va.ipynb` cells in correct order
- [x] Cell 2 references the actual repo URL
- [x] Cell 3a (MockLLM) for local development
- [x] Memory tips added to cell 4 (LLM load)
- [ ] Run end-to-end on Colab T4
- [ ] Capture cell outputs and save the notebook with results

## Milestone 9 — Tests  ✅ DONE
- [x] `tests/test_smoke.py` (11 tests, ~3s, no LLM needed)
- [x] `tests/conftest.py` so `src` imports under pytest
- [x] All 11 tests passing locally

---

## Outstanding work, ordered

1. **Push current state to GitHub** so Colab can pull it.
2. **Open notebook in Colab**, switch runtime to T4, run all cells.
3. **Fill in manual quality scores** (cell 17) for the 30 generated answers.
4. **Document which security attacks slipped through** (cell 19 output).
5. **Write the discussion section** (cell 20) — this is the part the rubric
   actually grades.

## Stretch (only if time permits)
- [ ] Cross-reference tool (Treasury of Scripture Knowledge JSON)
- [ ] Swap TF-IDF for sentence-transformers embeddings
- [ ] Playwright BEMA scraper for the full ~400-episode corpus
- [ ] Streamlit/Gradio UI
