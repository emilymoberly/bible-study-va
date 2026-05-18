"""
Gradio UI for the Bible Study Virtual Assistant.

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
- One-screen, vertical layout. Charcoal background, warm sand accent,
  off-white text. No purple, no playful styling.
- The model selector defaults to MockLLM so the app loads in any
  environment (CPU laptop, Colab without GPU). Switching to phi3 or
  mistral lazily loads the real model on first use.
- Filters: All sources / Bible only / BEMA only / YouTube only / etc.
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
# Visual design
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ---------- palette ---------- */
:root {
    --bg-base:        #161618;
    --bg-surface:     #1f1f22;
    --bg-elevated:    #26262a;
    --border-subtle:  rgba(236, 230, 220, 0.06);
    --border-stronger:rgba(236, 230, 220, 0.10);
    --text-primary:   #ece6dc;
    --text-secondary: #8b857d;
    --text-muted:     #6a6660;
    --accent:         #c9a87c;
    --accent-hover:   #d6b78b;
    --accent-soft:    rgba(201, 168, 124, 0.10);
}

/* ---------- override Gradio's own CSS variables (light + dark) -------- */
:root, .dark, html, body, .gradio-container, gradio-app {
    /* core fills */
    --body-background-fill:        #161618 !important;
    --body-background-fill-dark:   #161618 !important;
    --background-fill-primary:     #161618 !important;
    --background-fill-secondary:   #1f1f22 !important;
    --block-background-fill:       #1f1f22 !important;
    --block-label-background-fill: transparent !important;
    --block-title-background-fill: transparent !important;
    --panel-background-fill:       #1f1f22 !important;

    /* borders */
    --block-border-color:          rgba(236,230,220,0.06) !important;
    --block-label-border-color:    transparent !important;
    --block-title-border-color:    transparent !important;
    --border-color-primary:        rgba(236,230,220,0.06) !important;
    --border-color-accent:         #c9a87c !important;
    --border-color-accent-subdued: rgba(201,168,124,0.30) !important;
    --panel-border-color:          rgba(236,230,220,0.06) !important;

    /* inputs */
    --input-background-fill:       #1f1f22 !important;
    --input-background-fill-focus: #26262a !important;
    --input-border-color:          rgba(236,230,220,0.08) !important;
    --input-border-color-focus:    #c9a87c !important;
    --input-placeholder-color:     #6a6660 !important;

    /* text */
    --body-text-color:             #ece6dc !important;
    --body-text-color-subdued:     #8b857d !important;
    --block-label-text-color:      #8b857d !important;
    --block-title-text-color:      #ece6dc !important;

    /* tables */
    --table-border-color:          rgba(236,230,220,0.06) !important;
    --table-even-background-fill:  #1f1f22 !important;
    --table-odd-background-fill:   #1f1f22 !important;
    --table-row-focus:             rgba(201,168,124,0.10) !important;

    /* buttons */
    --button-primary-background-fill:        #c9a87c !important;
    --button-primary-background-fill-hover:  #d6b78b !important;
    --button-primary-text-color:             #161618 !important;
    --button-primary-border-color:           transparent !important;
    --button-secondary-background-fill:      transparent !important;
    --button-secondary-background-fill-hover:#26262a !important;
    --button-secondary-text-color:           #8b857d !important;
    --button-secondary-border-color:         rgba(236,230,220,0.08) !important;

    /* accents */
    --color-accent:                #c9a87c !important;
    --color-accent-soft:           rgba(201,168,124,0.10) !important;
    --link-text-color:             #c9a87c !important;
    --link-text-color-hover:       #d6b78b !important;

    /* neutral ramp — Gradio uses this for many internal surfaces */
    --neutral-50:  #161618 !important;
    --neutral-100: #1f1f22 !important;
    --neutral-200: #26262a !important;
    --neutral-300: #2e2e32 !important;
    --neutral-400: #6a6660 !important;
    --neutral-500: #8b857d !important;
    --neutral-600: #b9b3a9 !important;
    --neutral-700: #ece6dc !important;
    --neutral-800: #ece6dc !important;
    --neutral-900: #ece6dc !important;
    --neutral-950: #ece6dc !important;

    /* shadows */
    --shadow-drop: none !important;
    --shadow-drop-lg: none !important;
}

/* ---------- page: dark from edge to edge ---------- */
html, body {
    background: #161618 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Inter", system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* every Gradio shell wrapper inherits the page color */
gradio-app, gradio-app > *,
.gradio-container, .gradio-container .main, .gradio-container .contain,
.app, .app > .main, .wrap, .wrapper, .fillable {
    background: #161618 !important;
    color: var(--text-primary) !important;
    border: none !important;
    box-shadow: none !important;
}

/* the centered column — wider than before so it doesn't feel like a panel */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 56px 48px 96px !important;
}

/* hide gradio chrome we don't want */
footer { display: none !important; }
.show-api, .built-with, .api-docs { display: none !important; }

/* ---------- header ---------- */
.bsa-header {
    text-align: center;
    margin-bottom: 48px;
    padding-bottom: 36px;
    border-bottom: 1px solid var(--border-subtle);
}
.bsa-title {
    font-size: 34px;
    font-weight: 300;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0 0 14px 0;
    line-height: 1.15;
}
.bsa-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0 0 22px 0;
    font-weight: 400;
    line-height: 1.5;
    max-width: 520px;
    margin-left: auto;
    margin-right: auto;
}
.bsa-tagline {
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 500;
    margin: 0;
    opacity: 0.85;
}

/* ---------- section labels ---------- */
.bsa-section-label {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 500;
    margin: 28px 0 12px 0;
}

/* ---------- block defaults: transparent, no white cards ---------- */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .gr-form,
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-padded,
.gradio-container [class^="block-"],
.gradio-container [class*=" block-"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ---------- inputs (textarea + text input + number input) ---------- */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input[type="search"] {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    padding: 16px 18px !important;
    transition: border-color 0.18s ease, background 0.18s ease;
    box-shadow: none !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}
.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container input:focus-visible {
    background: var(--bg-elevated) !important;
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* big primary question textbox */
.bsa-question textarea {
    font-size: 17px !important;
    line-height: 1.55 !important;
    padding: 22px 24px !important;
    border-radius: 14px !important;
    min-height: 120px !important;
}

/* ---------- labels ---------- */
.gradio-container label,
.gradio-container label > span,
.gradio-container .label-wrap span,
.gradio-container .label-wrap > span,
.gradio-container [data-testid="block-label"] {
    color: var(--text-secondary) !important;
    background: transparent !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* ---------- dropdowns (closed state + the popup menu) ---------- */
.gradio-container .gr-dropdown,
.gradio-container [data-testid="dropdown"],
.gradio-container .dropdown,
.gradio-container .wrap-inner,
.gradio-container .secondary-wrap {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
/* the dropdown's inner search input shouldn't add a second border */
.gradio-container .gr-dropdown input,
.gradio-container [data-testid="dropdown"] input {
    background: transparent !important;
    border: none !important;
    padding: 12px 14px !important;
}

/* the floating menu of options (Gradio renders this as .options) */
.gradio-container .options,
.gradio-container ul[role="listbox"],
.gradio-container .options-list,
body > .options {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35) !important;
}
.gradio-container .options .item,
.gradio-container .options li,
.gradio-container [role="option"],
body > .options .item,
body > .options li {
    background: transparent !important;
    color: var(--text-primary) !important;
}
.gradio-container .options .item:hover,
.gradio-container [role="option"]:hover,
.gradio-container .options .item.selected,
.gradio-container [role="option"][aria-selected="true"],
body > .options .item:hover,
body > .options .item.selected {
    background: var(--accent-soft) !important;
    color: var(--text-primary) !important;
}

/* dropdown chevron / clear icons */
.gradio-container .gr-dropdown svg,
.gradio-container [data-testid="dropdown"] svg {
    color: var(--text-secondary) !important;
    fill: var(--text-secondary) !important;
}

/* ---------- primary Ask button ---------- */
.bsa-ask-row { justify-content: flex-end !important; gap: 0 !important; }
.bsa-ask {
    flex: 0 0 auto !important;
}
.bsa-ask button, button.bsa-ask, .bsa-ask > button {
    background: var(--accent) !important;
    color: #1a1a1c !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    box-shadow: none !important;
    transition: background 0.15s ease, transform 0.05s ease;
    min-width: 140px;
}
.bsa-ask button:hover {
    background: var(--accent-hover) !important;
}
.bsa-ask button:active {
    transform: translateY(1px);
}

/* generic button reset for everything else (e.g. Examples) */
button:not(.bsa-ask button) {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    font-size: 13px !important;
    box-shadow: none !important;
}
button:not(.bsa-ask button):hover {
    border-color: var(--accent) !important;
    color: var(--text-primary) !important;
}

/* ---------- compact controls row ---------- */
.bsa-controls {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 22px 24px !important;
    gap: 18px !important;
}
.bsa-controls .gr-form,
.bsa-controls .form { gap: 14px !important; background: transparent !important; }
.bsa-controls input,
.bsa-controls textarea,
.bsa-controls .gr-dropdown {
    background: var(--bg-elevated) !important;
    padding: 12px 14px !important;
    font-size: 14px !important;
    min-height: 0 !important;
}

/* ---------- answer card ---------- */
.bsa-answer, .bsa-answer .prose {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 28px 32px !important;
    color: var(--text-primary) !important;
    line-height: 1.7 !important;
    font-size: 15px !important;
    min-height: 88px;
}
.bsa-answer p { margin: 0 0 12px 0; color: var(--text-primary) !important; }
.bsa-answer hr {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 18px 0;
}
.bsa-answer code {
    background: var(--bg-elevated) !important;
    color: var(--accent) !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
}
.bsa-answer pre, .bsa-answer pre code {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px;
}
.bsa-answer strong { color: var(--text-primary); font-weight: 600; }
.bsa-answer em { color: var(--text-secondary); }

/* ---------- sources table ---------- */
.bsa-sources, .bsa-sources > * {
    background: transparent !important;
    border: none !important;
}
.bsa-sources table {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    overflow: hidden;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
}
.bsa-sources th {
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 14px 16px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    text-align: left !important;
}
.bsa-sources td {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 14px 16px !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
}
.bsa-sources tr:last-child td { border-bottom: none !important; }
.bsa-sources tr:hover td { background: var(--bg-elevated) !important; }

a, .bsa-answer a, .bsa-sources a {
    color: var(--accent) !important;
    text-decoration: none !important;
    border-bottom: 1px solid transparent;
    transition: border-color 0.15s ease;
}
a:hover, .bsa-answer a:hover, .bsa-sources a:hover {
    border-bottom-color: var(--accent) !important;
}

/* ---------- examples ---------- */
#bsa-examples,
#bsa-examples > *,
.gradio-container [class*="examples"],
.gradio-container .gallery {
    background: transparent !important;
    border: none !important;
    margin-top: 8px !important;
}
#bsa-examples table,
.gradio-container .examples-table,
.gradio-container [class*="examples"] table {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
#bsa-examples td,
.gradio-container .examples-table td,
.gradio-container [class*="examples"] td {
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
    border-color: var(--border-subtle) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}
#bsa-examples tr:hover td,
.gradio-container .examples-table tr:hover td,
.gradio-container [class*="examples"] tr:hover td {
    background: var(--accent-soft) !important;
    color: var(--text-primary) !important;
    cursor: pointer;
}

/* ---------- scrollbars ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb {
    background: var(--bg-elevated);
    border-radius: 6px;
    border: 2px solid var(--bg-base);
}
::-webkit-scrollbar-thumb:hover { background: #36363b; }
::selection { background: var(--accent-soft); color: var(--text-primary); }
"""

HEADER_HTML = """
<div class="bsa-header">
  <h1 class="bsa-title">Bible Study Assistant</h1>
  <p class="bsa-subtitle">
    Grounded answers from scripture, the BEMA Discipleship Podcast,
    show notes, and Marty Solomon's teaching videos.
  </p>
  <p class="bsa-tagline">Love God. Love others. Become People of the Text.</p>
</div>
"""


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
    n_types = len([k for k, v in stats["by_source_type"].items() if v])
    corpus_line = (
        f"<span style='color:var(--text-muted); font-size:12px; "
        f"letter-spacing:0.06em; display:block; margin-top:8px;'>"
        f"{stats['total_chunks']:,} indexed chunks · {n_types} source types"
        f"</span>"
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

    # Build a fully dark theme so Gradio's own machinery never paints white.
    # We set both the light and dark variants of every relevant color so the
    # interface is identical regardless of the user's system mode.
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.stone,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
    ).set(
        body_background_fill="#161618",
        body_background_fill_dark="#161618",
        background_fill_primary="#161618",
        background_fill_primary_dark="#161618",
        background_fill_secondary="#1f1f22",
        background_fill_secondary_dark="#1f1f22",
        block_background_fill="#1f1f22",
        block_background_fill_dark="#1f1f22",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_title_background_fill="transparent",
        block_title_background_fill_dark="transparent",
        panel_background_fill="#1f1f22",
        panel_background_fill_dark="#1f1f22",

        block_border_color="rgba(236,230,220,0.06)",
        block_border_color_dark="rgba(236,230,220,0.06)",
        border_color_primary="rgba(236,230,220,0.06)",
        border_color_primary_dark="rgba(236,230,220,0.06)",
        border_color_accent="#c9a87c",
        border_color_accent_dark="#c9a87c",
        input_background_fill="#1f1f22",
        input_background_fill_dark="#1f1f22",
        input_background_fill_focus="#26262a",
        input_background_fill_focus_dark="#26262a",
        input_border_color="rgba(236,230,220,0.08)",
        input_border_color_dark="rgba(236,230,220,0.08)",
        input_border_color_focus="#c9a87c",
        input_border_color_focus_dark="#c9a87c",
        input_placeholder_color="#6a6660",
        input_placeholder_color_dark="#6a6660",

        body_text_color="#ece6dc",
        body_text_color_dark="#ece6dc",
        body_text_color_subdued="#8b857d",
        body_text_color_subdued_dark="#8b857d",
        block_label_text_color="#8b857d",
        block_label_text_color_dark="#8b857d",
        block_title_text_color="#ece6dc",
        block_title_text_color_dark="#ece6dc",

        table_border_color="rgba(236,230,220,0.06)",
        table_border_color_dark="rgba(236,230,220,0.06)",
        table_even_background_fill="#1f1f22",
        table_even_background_fill_dark="#1f1f22",
        table_odd_background_fill="#1f1f22",
        table_odd_background_fill_dark="#1f1f22",
        table_row_focus="rgba(201,168,124,0.10)",
        table_row_focus_dark="rgba(201,168,124,0.10)",

        button_primary_background_fill="#c9a87c",
        button_primary_background_fill_dark="#c9a87c",
        button_primary_background_fill_hover="#d6b78b",
        button_primary_background_fill_hover_dark="#d6b78b",
        button_primary_text_color="#161618",
        button_primary_text_color_dark="#161618",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="#26262a",
        button_secondary_background_fill_hover_dark="#26262a",
        button_secondary_text_color="#8b857d",
        button_secondary_text_color_dark="#8b857d",
        button_secondary_border_color="rgba(236,230,220,0.08)",
        button_secondary_border_color_dark="rgba(236,230,220,0.08)",

        link_text_color="#c9a87c",
        link_text_color_dark="#c9a87c",
        link_text_color_hover="#d6b78b",
        link_text_color_hover_dark="#d6b78b",

        shadow_drop="none",
        shadow_drop_lg="none",
    )

    # NOTE on Gradio 6: `theme`, `css`, and `js` were moved off the Blocks
    # constructor and onto `.launch(...)`. If you pass them here Gradio
    # silently ignores them. We capture them now and inject into launch()
    # via a small wrapper at the bottom of build_app(), so every entrypoint
    # (`python -m src.app`, `build_app().launch()`, etc.) gets the styling.
    force_dark_js = (
        "() => {"
        "  document.documentElement.classList.add('dark');"
        "  document.body.classList.add('dark');"
        "  document.documentElement.style.background = '#161618';"
        "  document.body.style.background = '#161618';"
        "}"
    )

    with gr.Blocks(title="Bible Study Assistant") as demo:
        gr.HTML(HEADER_HTML + corpus_line)

        # --- main question ---
        question = gr.Textbox(
            placeholder="Ask a question, explore a theme, or paste a verse like John 3:16…",
            lines=3,
            show_label=False,
            elem_classes=["bsa-question"],
        )

        with gr.Row(elem_classes=["bsa-ask-row"]):
            ask_btn = gr.Button("Ask", variant="primary",
                                elem_classes=["bsa-ask"])

        # --- compact controls row ---
        gr.HTML('<div class="bsa-section-label">Settings</div>')
        with gr.Group(elem_classes=["bsa-controls"]):
            with gr.Row():
                filter_choice = gr.Dropdown(
                    list(FILTER_PRESETS.keys()),
                    value="All sources",
                    label="Source filter",
                )
                technique = gr.Dropdown(
                    ["zero_shot", "few_shot", "chain_of_thought", "prompt_chaining"],
                    value="zero_shot",
                    label="Prompting technique",
                )
                model_choice = gr.Dropdown(
                    ["mock", "phi3", "mistral"],
                    value="mock",
                    label="Model",
                )
            with gr.Row():
                verse_filter = gr.Textbox(
                    label="Verse reference (optional)",
                    placeholder="John 3:16",
                )
                episode_filter = gr.Textbox(
                    label="BEMA episode # (optional)",
                    placeholder="100",
                )

        # --- answer ---
        gr.HTML('<div class="bsa-section-label">Answer</div>')
        answer_md = gr.Markdown(
            value="*Your grounded answer will appear here.*",
            elem_classes=["bsa-answer"],
        )

        # --- sources ---
        gr.HTML('<div class="bsa-section-label">Sources</div>')
        sources_tbl = gr.Dataframe(
            headers=["source", "title", "score", "excerpt"],
            datatype=["str", "markdown", "str", "str"],
            wrap=True,
            interactive=False,
            elem_classes=["bsa-sources"],
            show_label=False,
        )

        # --- examples (subtle, at the bottom) ---
        gr.HTML('<div class="bsa-section-label">Try one of these</div>')
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
            elem_id="bsa-examples",
        )

        # --- wiring (unchanged) ---
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

    # --- inject theme / css / js into demo.launch (Gradio 6 requirement) ---
    _original_launch = demo.launch

    def _launch(*args, **kwargs):
        kwargs.setdefault("theme", theme)
        kwargs.setdefault("css", CUSTOM_CSS)
        kwargs.setdefault("js", force_dark_js)
        return _original_launch(*args, **kwargs)

    demo.launch = _launch  # type: ignore[assignment]
    return demo


def _empty_table() -> list:
    return []


def main() -> None:
    demo = build_app()
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    # Force dark mode so the dark CSS variables are the ones in effect
    # regardless of the user's system theme.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
