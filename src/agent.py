"""
The agent: routes a question, retrieves grounded evidence, asks the LLM.

Architectural change vs. v1
---------------------------
v1 routed between three independent tools (Bible / BEMA / Web). v2 uses
the unified corpus (`src/corpus.py`) which already contains every Bible
verse, every BEMA transcript chunk, every BEMA show-notes summary, every
study-tool link, every BEMA static page, and every YouTube caption chunk
— all labeled by `source_type` and tagged with detected `verse_refs`.

The agent now:
  1. Parses the question for an explicit Bible reference and a hint about
     which source types to focus on.
  2. Calls `Corpus.search(...)` with the chosen filters.
  3. (Optionally) augments with a DuckDuckGo web hit for "history"-flavored
     questions, since web context isn't in our corpus.
  4. Builds an EVIDENCE block + asks the LLM via one of the prompting
     techniques.

Public types and functions:
  RoutingDecision, Source, Answer
  Agent(corpus, enable_web=True).answer(question, model, technique=...,
                                       source_types=..., episode=..., verse_ref=...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .bible_refs import find_refs, normalize_query_ref
from .corpus import Corpus, CorpusHit
from .prompts import TECHNIQUES
from .web_tool import WebTool

if TYPE_CHECKING:
    from .llm import LLM, GenerationResult


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

WEB_KEYWORDS = (
    "history", "historical", "roman", "greek", "egypt", "persian",
    "babylonian", "archaeolog", "scholar", "modern", "today", "current",
)
BEMA_HINT = ("bema", "podcast", "marty solomon", "brent billings",
             "discipleship podcast")
YOUTUBE_HINT = ("youtube", "video", "talk", "sermon")


@dataclass
class RoutingDecision:
    source_types: list[str]
    detected_verse_ref: str | None
    use_web: bool
    reason: str


def route(
    question: str,
    explicit_source_types: Iterable[str] | None = None,
) -> RoutingDecision:
    q = question.lower()
    detected_ref = normalize_query_ref(question)

    if explicit_source_types:
        chosen = list(explicit_source_types)
        reason_bits = ["explicit"]
    else:
        # Default: search bible + bema_transcript + bema_summary + youtube.
        # Add bema_studytool / bema_site only if the user mentions them.
        chosen = ["bible", "bema_transcript", "bema_summary", "youtube"]
        if any(k in q for k in BEMA_HINT):
            chosen.append("bema_studytool")
            chosen.append("bema_site")
        reason_bits = ["default"]

    use_web = any(k in q for k in WEB_KEYWORDS)
    if use_web:
        reason_bits.append("web")
    if detected_ref:
        reason_bits.append(f"verse:{detected_ref}")

    return RoutingDecision(
        source_types=chosen,
        detected_verse_ref=detected_ref,
        use_web=use_web,
        reason=", ".join(reason_bits),
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class Source:
    label: str        # short tag used inside the prompt, e.g. "Bible: John 3:16"
    title: str        # human-readable title for the UI
    url: str          # back-link for the UI
    text: str         # excerpt to splice into the prompt
    source_type: str
    score: float = 0.0

    def render_for_prompt(self) -> str:
        return f"[{self.label}] {self.text}"


def _hit_to_source(hit: CorpusHit) -> Source:
    c = hit.chunk
    if c.source_type == "bible":
        label = f"Bible: {c.title}"
    elif c.source_type.startswith("bema_"):
        ep = f"BEMA {c.episode}" if c.episode else "BEMA"
        kind = c.source_type.split("_", 1)[1]
        label = f"{ep} {kind}: {c.title[:60]}"
    elif c.source_type == "youtube":
        label = f"YouTube: {c.title[:60]}"
    else:
        label = c.title[:80]
    return Source(
        label=label,
        title=c.title,
        url=c.url,
        text=c.text,
        source_type=c.source_type,
        score=hit.score,
    )


def format_evidence(sources: list[Source], max_chars: int = 6000) -> str:
    out: list[str] = []
    used = 0
    for s in sources:
        rendered = s.render_for_prompt()
        if used + len(rendered) > max_chars:
            break
        out.append(rendered)
        used += len(rendered) + 1
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    question: str
    text: str
    technique: str
    routing: RoutingDecision
    sources: list[Source]
    latency_s: float
    tokens_in: int
    tokens_out: int
    model_id: str


class Agent:
    def __init__(
        self,
        corpus_path: str | Path | Corpus,
        enable_web: bool = True,
    ):
        if isinstance(corpus_path, Corpus):
            self.corpus = corpus_path
        else:
            self.corpus = Corpus(corpus_path)
        self.web: WebTool | None = WebTool() if enable_web else None

    # -------------------------------------------------------- retrieval steps
    def gather(
        self,
        question: str,
        source_types: Iterable[str] | None = None,
        episode: str | None = None,
        verse_ref: str | None = None,
        k: int = 8,
    ) -> tuple[RoutingDecision, list[Source]]:
        decision = route(question, explicit_source_types=source_types)
        ref = verse_ref or decision.detected_verse_ref

        # Primary corpus search
        hits = self.corpus.search(
            question,
            k=k,
            source_types=decision.source_types,
            episode=episode,
            verse_ref=ref,
        )
        sources = [_hit_to_source(h) for h in hits]

        # If a verse ref was detected and we have it in the bible source,
        # ensure that verse is included as evidence even if it didn't rank.
        if ref:
            verse_chunks = self.corpus.lookup_verse(ref)
            for vc in verse_chunks:
                # avoid duplicate
                if any(s.label == f"Bible: {vc.title}" for s in sources):
                    continue
                sources.insert(0, Source(
                    label=f"Bible: {vc.title}",
                    title=vc.title,
                    url=vc.url,
                    text=vc.text,
                    source_type="bible",
                    score=1.0,
                ))

        # Web augmentation for "history" flavored questions
        if decision.use_web and self.web is not None:
            try:
                web_hits = self.web.search(question, k=2)
                for h in web_hits:
                    sources.append(Source(
                        label=f"Web: {h.url}",
                        title=h.title,
                        url=h.url,
                        text=f"{h.title} — {h.snippet}",
                        source_type="web",
                        score=0.0,
                    ))
            except Exception as e:  # noqa: BLE001
                sources.append(Source(
                    label="Web error", title="(web error)", url="",
                    text=str(e), source_type="web",
                ))

        return decision, sources

    # ----------------------------------------------------------------- answer
    def answer(
        self,
        question: str,
        model: "LLM",
        technique: str = "zero_shot",
        max_new_tokens: int = 400,
        source_types: Iterable[str] | None = None,
        episode: str | None = None,
        verse_ref: str | None = None,
        k: int = 8,
    ) -> Answer:
        if technique not in TECHNIQUES:
            raise ValueError(
                f"Unknown technique '{technique}'. Choose from {list(TECHNIQUES)}."
            )

        decision, sources = self.gather(
            question, source_types=source_types, episode=episode,
            verse_ref=verse_ref, k=k,
        )
        evidence = format_evidence(sources)
        pair = TECHNIQUES[technique](question, evidence)
        result: "GenerationResult" = model.generate(
            user_prompt=pair.user,
            system_prompt=pair.system,
            max_new_tokens=max_new_tokens,
        )

        return Answer(
            question=question,
            text=result.text,
            technique=technique,
            routing=decision,
            sources=sources,
            latency_s=result.latency_s,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            model_id=model.model_id,
        )
