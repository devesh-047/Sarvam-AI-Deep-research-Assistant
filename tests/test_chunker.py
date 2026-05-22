"""Tests for the token-aware chunker."""
import pytest
from app.research.chunker import chunk_document, _count_tokens
from app.models.schema import ExtractedDocument


def _make_doc(text: str) -> ExtractedDocument:
    return ExtractedDocument(
        url="https://example.com/article",
        title="Test Article",
        domain="example.com",
        text=text,
        word_count=len(text.split()),
        extraction_successful=True,
    )


def test_chunker_produces_chunks():
    doc = _make_doc("word " * 300)  # ~300 tokens
    chunks = chunk_document(doc, source_id=1, session_id=1, turn_id=1, chunk_size=128, overlap=20)
    assert len(chunks) >= 1


def test_chunker_metadata_preserved():
    doc = _make_doc("word " * 300)
    chunks = chunk_document(doc, source_id=1, session_id=1, turn_id=1, chunk_size=128, overlap=20)
    for c in chunks:
        assert c.source_url == doc.url
        assert c.title == doc.title
        assert c.domain == doc.domain
        assert c.session_id == 1
        assert c.turn_id == 1


def test_chunker_overlap_creates_more_chunks():
    doc = _make_doc("word " * 600)
    chunks_no_overlap = chunk_document(doc, source_id=1, session_id=1, turn_id=1, chunk_size=256, overlap=0)
    chunks_with_overlap = chunk_document(doc, source_id=1, session_id=1, turn_id=1, chunk_size=256, overlap=64)
    # With overlap we expect at least as many or more chunks
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_chunker_skips_failed_doc():
    doc = ExtractedDocument(
        url="https://example.com/bad",
        title="Bad",
        domain="example.com",
        text="",
        word_count=0,
        extraction_successful=False,
        error="failed",
    )
    chunks = chunk_document(doc, source_id=1, session_id=1, turn_id=1)
    assert chunks == []


def test_chunk_index_sequential():
    doc = _make_doc("word " * 600)
    chunks = chunk_document(doc, source_id=1, session_id=1, turn_id=1, chunk_size=128, overlap=20)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_token_count_within_limit():
    chunk_size = 128
    doc = _make_doc("word " * 600)
    chunks = chunk_document(doc, source_id=1, session_id=1, turn_id=1, chunk_size=chunk_size, overlap=20)
    for c in chunks:
        assert c.token_count <= chunk_size
