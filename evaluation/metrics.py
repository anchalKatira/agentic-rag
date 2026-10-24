"""Pure metric functions (no LLM / DB needed, so they're easy to unit test)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


def _matches(doc, keywords: Optional[Sequence[str]], pages: Optional[Sequence[int]]) -> bool:
    text = doc.page_content.lower()
    if keywords and any(k.lower() in text for k in keywords):
        return True
    if pages and doc.metadata.get("page") in pages:
        return True
    return False


def retrieval_hit(docs, keywords=None, pages=None) -> bool:
    """Did ANY retrieved chunk contain an expected keyword / come from an expected page?"""
    return any(_matches(d, keywords, pages) for d in docs)


def reciprocal_rank(docs, keywords=None, pages=None) -> float:
    """1 / rank of the first matching chunk (0 if none)."""
    for rank, d in enumerate(docs, start=1):
        if _matches(d, keywords, pages):
            return 1.0 / rank
    return 0.0


def answer_keyword_hit(answer: str, keywords: Optional[Iterable[str]]) -> Optional[bool]:
    """Does the final answer mention an expected keyword? None if no keywords given."""
    if not keywords:
        return None
    a = answer.lower()
    return any(k.lower() in a for k in keywords)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
