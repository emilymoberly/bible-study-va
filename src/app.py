"""
Minimal Gradio UI for the Bible Study Virtual Assistant.

Usage
-----
Local:
    python -m src.app
    # then open http://127.0.0.1:7860

Colab:
    !git clone https://github.com/emilymoberly/bible-study-va.git
    %cd bible-study-va
    !pip install -q -r requirements.txt
    !python scripts/load_bible.py
    !python scripts/scrape_bema.py            # ~9 min
    !python scripts/scrape_bema_pages.py      # ~12 min
    !python scripts/scrape_youtube.py         # ~7 min
    !python scripts/build_corpus.py           # ~10 sec
    from src.app import build_app
    build_app().launch(share=True)            # share=True gives a public URL

Design notes
------------
- One-screen layout (no tabs). Class-project simple.
- The model selector defaults to MockLLM so the app loads in any environment
  (CPU laptop, Colab without GPU). Switching to phi3 or mistral lazily
  loads the real model on first use.
- Filters: Bible only / BEMA only / YouTube only / All sources.
- Answer area + a sources table with clickable URLs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gradio as gr

from .agent import Agent, Answer
from .llm import MockLLM


REPO = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO / "data" / "corpus" / "corpus.jsonl"


# ---------------------------------------------------------------------------
# Lazy LLM cache so we don't reload weights on every request.
# ---------------------------------------------------------------------------

_LLM_CACHE: dict[str, object] = {}


def get_model(choice: str):
    """Return a model object by short name. Loads on first request only."""
    if choice in _LLM_CACHE:
        return _LLM_CACHE[choice]
    if choice == "mock":
        m = MockLLM()
    else:
        # Lazy import — transformers + torch is heavy to import.
        from .llm import LLM
        m = LLM.load(choice)  # "phi3" or "mistral"
    _LLM_CACHE[choice] = m
    return m


# ---------------------------------------------------------------------------
# Filter mapping. UI label -> list[source_type] (or None == all)
# ---------------------------------------------------------------------------

FILTER_PRESETS = {
    "All sources": None,
    "Bible only": ["bible"],
    "BEMA only": ["bema_transcript", "bema_summary",
                  "bema_studytool", "bema_site"],
    "BEMA transcripts only": ["bema_transcript"],
    "YouTube only": ["youtube"],
    "Bible + BEMA transcripts": ["bible", "bema_transcript"],
}


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------

def build_app(corpus_path: Optional[Path] = None) -> gr.Blocks:
    corpus_path = Path(corpus_path or CORPUS_PATH)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"corpus.jsonl not found at {corpus_path}. "
            f"Run `python scripts/build_corpus.py` first."
        )

    agent = Agent(corpus_path, enable_web=False)
    stats = agent.corpus.stats()

    intro = (
        "## Bible Study Virtual Assistant\n"
        "Ask a question, theme, or paste a verse like `John 3:16`. "
        "The assistant retrieves grounded evidence from the Bible (KJV), "
        "BEMA Discipleship Podcast transcripts, BEMA show notes, BEMA site "
        "pages, and Marty Solomon's linked YouTube videos.\n\n"
        f"**Corpus**: {stats['total_chunks']:,} chunks across "
        f"{len([k for k,v in stats['by_source_type'].items() if v])} source types."
    )

    def run(question: str, filter_choice: str, technique: str, model_choice: str,
            verse_filter: str, episode_filter: str):
        if not question.strip():
            return "Please ask a question.", _empty_table()

        source_types = FILTER_PRESETS.get(filter_choice)
        verse_ref = verse_filter.strip() or None
        episode = episode_filter.strip() or None

        try:
            model = get_model(model_choice)
        except Exception as e:  # noqa: BLE001
            return (f"⚠️ Could not load model `{model_choice}`: {e}\n\n"
                    f"Falling back to MockLLM.", _empty_table())

        try:
            ans: Answer = agent.answer(
                question=question,
                model=model,
                technique=technique,
                source_types=source_types,
                episode=episode,
                verse_ref=verse_ref,
            )
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}", _empty_table()

        header = (
            f"**Routing:** {ans.routing.reason}  \n"
            f"**Latency:** {ans.latency_s:.2f}s  |  "
            f"tokens in/out: {ans.tokens_in}/{ans.tokens_out}  |  "
            f"model: `{ans.model_id}`\n\n---\n"
        )

        rows = []
        for s in ans.sources[:10]:
            link = f"[{s.title[:80]}]({s.url})" if s.url else s.title[:80]
            excerpt = (s.text[:240] + "…") if len(s.text) > 240 else s.text
            rows.append([s.source_type, link, f"{s.score:.3f}", excerpt])
        return header + ans.text, rows

    with gr.Blocks(title="Bible Study VA") as demo:
        gr.Markdown(intro)

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Your question, theme, or verse",
                    placeholder="e.g. What is a chiasm? or John 3:16",
                    lines=2,
                )
            with gr.Column(scale=1):
                technique = gr.Dropdown(
                    ["zero_shot", "few_shot", "chain_of_thought"],
                    value="zero_shot",
                    label="Prompting technique",
                )
                model_choice = gr.Dropdown(
                    ["mock", "phi3", "mistral"],
                    value="mock",
                    label="Model (mock = no GPU needed)",
                )

        with gr.Row():
            filter_choice = gr.Dropdown(
                list(FILTER_PRESETS.keys()),
                value="All sources",
                label="Filter sources",
            )
            verse_filter = gr.Textbox(
                label="Optional: only chunks tagged with this verse ref",
                placeholder="e.g. John 3:16",
            )
            episode_filter = gr.Textbox(
                label="Optional: only this BEMA episode #",
                placeholder="e.g. 100",
            )

        ask_btn = gr.Button("Ask", variant="primary")
        answer_md = gr.Markdown(label="Answer")
        sources_tbl = gr.Dataframe(
            headers=["source_type", "title (link)", "score", "excerpt"],
            label="Top sources used as evidence",
            wrap=True,
            interactive=False,
        )

        ask_btn.click(
            run,
            inputs=[question, filter_choice, technique, model_choice,
                    verse_filter, episode_filter],
            outputs=[answer_md, sources_tbl],
        )
        question.submit(
            run,
            inputs=[question, filter_choice, technique, model_choice,
                    verse_filter, episode_filter],
            outputs=[answer_md, sources_tbl],
        )

        gr.Examples(
            examples=[
                ["What are hidden meanings and symbols behind Babylon in the Bible?",
                 "All sources", "zero_shot", "mock", "", ""],
                ["What is a chiasm and what are examples in Deuteronomy?",
                 "BEMA only", "zero_shot", "mock", "", ""],
                ["What is the difference between a Pharisee and a teacher of the law?",
                 "All sources", "few_shot", "mock", "", ""],
                ["John 3:16", "Bible only", "zero_shot", "mock", "John 3:16", ""],
            ],
            inputs=[question, filter_choice, technique, model_choice,
                    verse_filter, episode_filter],
        )

    return demo


def _empty_table() -> list:
    return []


def main() -> None:
    demo = build_app()
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
