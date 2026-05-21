# Bible Study Virtual Assistant — Class Project Writeup

**Emily Moberly** | Repo: [github.com/emilymoberly/bible-study-va](https://github.com/emilymoberly/bible-study-va) | Public demo: [Gradio share URL — paste once §11 prints it]

## Design of the VA

I built the Bible Study Virtual Assistant as a tool-augmented language model: every answer the system produces is grounded in evidence that it *retrieves before generating*, rather than relying on the LLM's parametric memory. The pipeline has four layers — router, retriever, prompt assembler, and language model — and a few specific decisions in each layer drove most of the project's character.

**Why a unified corpus instead of three separate tools.** My first version split the data into three independent tools (Bible / BEMA / Web). I rewrote that into a single labeled corpus of 45,289 chunks (KJV verses, BEMA transcripts, episode pages, study tools, site pages, and YouTube captions) because the three-tool dispatch was making routing decisions *before* the system actually knew what evidence existed. With a unified corpus, the router only has to set filters (source type, episode, verse reference) and the retriever does the real ranking work. That single change made the system noticeably easier to debug and explain.

**Why TF-IDF instead of embeddings.** A 45K-chunk corpus fits comfortably in TF-IDF with 1–2 grams and English stopword filtering. It indexes in 4 seconds, queries in milliseconds, and is fully transparent — if a chunk was retrieved, I can point at the word overlap that caused it. Embeddings would have added a 5-minute build step, 600 MB of vectors, and a black box that's harder to defend in a class writeup. The trade-off (queries that paraphrase without lexical overlap will under-retrieve) is real but acceptable for a study tool.

**Why sequential model loading.** I picked Phi-3.5-mini (3.8 B params, fp16, ~7.6 GB VRAM) and Mistral-7B (4-bit NF4 via bitsandbytes, ~5 GB VRAM) because that pair was the largest combination that fits a free Colab T4 — sequentially. Loading both at once OOMs at ~14.5 GB VRAM, so the notebook loads Phi-3.5, runs its work, frees it with `del + gc.collect() + empty_cache()`, then loads Mistral. (In practice — see Results — Mistral was ultimately excluded due to a *system*-RAM constraint, not a VRAM one.)

**Why four prompting techniques.** Zero-shot is the baseline. Few-shot adds two worked Q/A exemplars so the model learns to cite by source label. Chain-of-thought adds an internal `<scratchpad>` before the final answer. Prompt chaining runs *two* LLM calls per question — step 1 extracts factual bullets from the raw evidence, step 2 synthesizes the answer using only those bullets. The chained variant is structurally more robust to prompt injection (any instruction-shaped text inside the evidence is dropped during the extraction step) at the cost of about 2× latency.

**Why defense-in-depth on security.** The system prompt explicitly treats the EVIDENCE block as untrusted data and forbids the model from following commands embedded inside it. Routing never executes anything from the user's text — it only sets filters. The chained technique adds a second filtering layer. These were not afterthoughts: they were designed in because the corpus includes user-facing content (BEMA transcripts) that a malicious actor could in principle have poisoned.

*Quantitative results (20-query showcase, model × technique comparison, security tests, caching speedup) are reported separately in `RESULTS.md`.*

## Reflection

The Bible Study VA does what I set out to build: it answers Bible-study questions with grounded citations from a labeled corpus, and it's transparent enough that a user can trace any claim back to its source. I'm proud of the unified-corpus design and the four-layer architecture — those are decisions I'd make the same way again.

The most challenging part was not the LLMs but the infrastructure around them. Free Colab is generous but unstable, and a class-project budget doesn't cover paid alternatives. I spent more time managing Drive uploads, runtime resets, silent CPU downgrades, and crash recovery than I did tuning prompts. That said, working through those constraints taught me more than a smooth run would have: I now understand exactly which system-RAM costs are hidden inside a model load, why sequential loading is necessary but not always sufficient, and how to design defensive cells — an idempotent scrape, a Drive fast path, a `SKIP_MISTRAL` toggle — that survive partial crashes without losing prior work.

If I were to extend this beyond class, I would: (1) swap TF-IDF for a small embedding retriever once GPU stops being a constraint, (2) add per-source confidence calibration so a question with weak evidence triggers an "I'm not sure" response instead of a confident citation of a marginal source, (3) re-run the comparison on a paid T4 or A10 to actually complete the Phi-3.5 vs Mistral-7B axis and add a third axis against an instruction-tuned 13 B model, and (4) build a small evaluation harness so prompt-technique changes can be regression-tested instead of eyeballed. The architecture supports all four of these — they're blocked by the same infrastructure I struggled with, not by the design.
