"""Tests for context builder."""
import pytest
from app.models.schema import RetrievedChunk
from app.research.context_builder import build_context, _count_tokens


def _make_chunks(n: int, text_len: int = 100) -> list:
    return [
        RetrievedChunk(
            chunk_id=i,
            source_url=f"https://source{i}.com/article",
            title=f"Source {i}",
            domain=f"source{i}.com",
            text="word " * text_len,
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


def test_build_context_returns_prompt_and_citations():
    chunks = _make_chunks(3)
    prompt, citations = build_context("test query", chunks)
    assert "test query" in prompt
    assert len(citations) == 3
    assert "[S1]" in prompt


def test_citation_labels_are_unique():
    chunks = _make_chunks(5)
    _, citations = build_context("q", chunks)
    labels = [c.label for c in citations]
    assert len(labels) == len(set(labels))


def test_token_budget_enforced():
    # Create many large chunks; budget should cut them off
    chunks = _make_chunks(20, text_len=500)
    prompt, _ = build_context("q", chunks, max_tokens=1000)
    assert _count_tokens(prompt) <= 1200  # allow small tolerance


def test_same_url_gets_same_label():
    # Two chunks from the same URL should share a citation label
    chunk_a = RetrievedChunk(
        chunk_id=1, source_url="https://same.com", title="Same",
        domain="same.com", text="text a", score=0.9,
    )
    chunk_b = RetrievedChunk(
        chunk_id=2, source_url="https://same.com", title="Same",
        domain="same.com", text="text b", score=0.8,
    )
    prompt, citations = build_context("q", [chunk_a, chunk_b])
    assert len(citations) == 1   # deduplicated
    assert chunk_a.citation_label == chunk_b.citation_label
