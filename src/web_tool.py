"""
Web search tool — DuckDuckGo via the `duckduckgo-search` package.

We use DuckDuckGo because it requires no API key and is friendly to
educational/class-project use. Results are returned as small dicts so the
agent can splice them into prompts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    snippet: str

    @property
    def reference(self) -> str:
        return self.url

    def __str__(self) -> str:
        return f"{self.title} ({self.url}): {self.snippet}"


class WebTool:
    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate"):
        self.region = region
        self.safesearch = safesearch

    def search(self, query: str, k: int = 3) -> list[WebHit]:
        # Imported lazily so that loading the project doesn't require the
        # network library if the user only wants offline tools.
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "duckduckgo-search not installed. `pip install duckduckgo-search`."
            ) from e

        hits: list[WebHit] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=self.region,
                               safesearch=self.safesearch, max_results=k):
                hits.append(WebHit(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))
        return hits
