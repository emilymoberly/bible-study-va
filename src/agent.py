"""
The agent: rule-based tool router + grounded answer generator.

Pipeline for `answer(question, model)`:

    1. route(question) decides which tools to call
    2. each tool returns small structured hits
    3. format_evidence() turns hits into a single labeled EVIDENCE block
    4. prompts.<technique>() builds (system, user) prompt
    5. model.generate() runs the LLM and times it
    6. return Answer(text, sources, latency_s, ...)

Why rule-based routing?
We deliberately avoid full ReAct-style tool-use loops because:
  (a) small open-source models are unreliable at producing valid tool calls
  (b) the project rubric just requires "use tools"; a router meets that
  (c) it's far easier to demo / debug / explain
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .bible_tool import BibleTool, Verse
from .bema_tool import BemaTool, BemaChunk
from .prompts import TECHNIQUES
from .web_tool import WebHit, WebTool

if TYPE_CHECKING:
    from .llm import LLM, GenerationResult


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# Direct verse references like "John 3:16", "Genesis 1", "1 Cor 13:4-7".
# We deliberately match only:
#   - a single capitalized word ("John", "Genesis"), optionally with a 1/2/3 prefix
#   - or "Song of Solomon" / "Song of Songs" specifically
# This prevents over-matching e.g. "Explain John 3:16" -> "Explain John 3:16".
VERSE_REF_RE = re.compile(
    r"\b(?:(?:[123]\s+)?[A-Z][a-z]+|Song\s+of\s+(?:Solomon|Songs))"
    r"\s+\d+(?::\d+(?:-\d+)?)?\b"
)

BEMA_KEYWORDS = ("bema", "podcast", "marty solomon", "brent billings",
                 "chiasm", "discipleship podcast")

WEB_KEYWORDS = ("history", "historical", "roman", "greek", "egypt",
                "persian", "babylonian", "archaeolog", "scholar", "modern",
                "today", "current")

# These topics are clearly biblical and should always pull bible_tool +
# usually bema_tool (BEMA covers them deeply).
BIBLE_KEYWORDS = ("verse", "passage", "scripture", "bible", "torah",
                  "gospel", "psalm", "proverb", "covenant", "messiah",
                  "jesus", "yahweh", "god", "moses", "david", "abraham",
                  "pharisee", "sadducee", "babylon", "exodus")


@dataclass
class RoutingDecision:
    use_bible: bool
    use_bema: bool
    use_web: bool
    direct_refs: list[str] = field(default_factory=list)
    reason: str = ""


def route(question: str) -> RoutingDecision:
    q = question.lower()

    direct_refs = VERSE_REF_RE.findall(question)
    use_bible = bool(direct_refs) or any(k in q for k in BIBLE_KEYWORDS)
    use_bema = any(k in q for k in BEMA_KEYWORDS) or use_bible
    use_web = any(k in q for k in WEB_KEYWORDS)

    # Always try Bible if nothing matched — it's our primary data source.
    if not (use_bible or use_bema or use_web):
        use_bible = True
        use_bema = True

    reason_bits = []
    if direct_refs:
        reason_bits.append(f"direct refs: {direct_refs}")
    if use_bible:
        reason_bits.append("bible")
    if use_bema:
        reason_bits.append("bema")
    if use_web:
        reason_bits.append("web")

    return RoutingDecision(
        use_bible=use_bible,
        use_bema=use_bema,
        use_web=use_web,
        direct_refs=direct_refs,
        reason=", ".join(reason_bits),
    )


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------

@dataclass
class Source:
    label: str
    text: str

    def render(self) -> str:
        return f"[{self.label}] {self.text}"


def format_evidence(sources: list[Source], max_chars: int = 6000) -> str:
    """Concatenate sources into a single block, truncating to fit the prompt."""
    out: list[str] = []
    used = 0
    for s in sources:
        rendered = s.render()
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
        bible_path: str | Path,
        bema_transcripts_dir: str | Path,
        bema_episodes_json: str | Path,
        enable_web: bool = True,
    ):
        self.bible = BibleTool(bible_path)
        self.bema = BemaTool(bema_transcripts_dir, bema_episodes_json)
        self.web: WebTool | None = WebTool() if enable_web else None

    # -------------------------------------------------------- retrieval steps
    def _bible_evidence(self, question: str, decision: RoutingDecision,
                        k: int = 5) -> list[Source]:
        sources: list[Source] = []
        for ref in decision.direct_refs:
            for v in self.bible.lookup(ref):
                sources.append(Source(label=f"Bible: {v.reference}", text=v.text))
        if not sources:  # fall back to TF-IDF search
            for hit in self.bible.search(question, k=k):
                v: Verse = hit.item
                sources.append(Source(label=f"Bible: {v.reference}", text=v.text))
        return sources

    def _bema_evidence(self, question: str, k: int = 3) -> list[Source]:
        out: list[Source] = []
        try:
            hits = self.bema.search(question, k=k)
        except RuntimeError:
            return out
        for hit in hits:
            c: BemaChunk = hit.item
            label = f"BEMA {c.episode_number}: {c.episode_title}".strip(": ")
            out.append(Source(label=label, text=c.text))
        return out

    def _web_evidence(self, question: str, k: int = 3) -> list[Source]:
        if self.web is None:
            return []
        try:
            hits = self.web.search(question, k=k)
        except Exception as e:  # noqa: BLE001
            return [Source(label="Web error", text=str(e))]
        return [
            Source(label=f"Web: {h.url}", text=f"{h.title} — {h.snippet}")
            for h in hits
        ]

    # ----------------------------------------------------------------- public
    def gather(self, question: str) -> tuple[RoutingDecision, list[Source]]:
        decision = route(question)
        sources: list[Source] = []
        if decision.use_bible:
            sources.extend(self._bible_evidence(question, decision))
        if decision.use_bema:
            sources.extend(self._bema_evidence(question))
        if decision.use_web:
            sources.extend(self._web_evidence(question))
        return decision, sources

    def answer(
        self,
        question: str,
        model: "LLM",
        technique: str = "zero_shot",
        max_new_tokens: int = 400,
    ) -> Answer:
        if technique not in TECHNIQUES:
            raise ValueError(
                f"Unknown technique '{technique}'. "
                f"Choose from {list(TECHNIQUES)}."
            )
        decision, sources = self.gather(question)
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
