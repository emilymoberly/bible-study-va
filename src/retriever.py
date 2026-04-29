"""
Tiny TF-IDF retriever shared by the Bible and BEMA tools.

Why a custom thin wrapper?
- We want to keep dependencies light (no FAISS / sentence-transformers).
- For class-project scale (~31k Bible verses, a few hundred BEMA chunks)
  scikit-learn's TfidfVectorizer + cosine similarity is fast and zero setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

T = TypeVar("T")


@dataclass
class SearchHit(Generic[T]):
    item: T
    score: float


class TfidfIndex(Generic[T]):
    """Generic TF-IDF index over arbitrary items + their text fields."""

    def __init__(self, items: Sequence[T], texts: Sequence[str]):
        if len(items) != len(texts):
            raise ValueError("items and texts must be the same length")
        self.items = list(items)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 5) -> list[SearchHit[T]]:
        if not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        if k >= len(sims):
            top = np.argsort(-sims)
        else:
            # argpartition for speed, then sort the small slice
            top = np.argpartition(-sims, k)[:k]
            top = top[np.argsort(-sims[top])]
        return [SearchHit(item=self.items[i], score=float(sims[i]))
                for i in top if sims[i] > 0]
