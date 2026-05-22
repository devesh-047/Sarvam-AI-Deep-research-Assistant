"""Tests for Stage 3 context builder: memory injection and token budget."""
import pytest
from app.models.schema import RetrievedChunk
from app.research.context_builder import build_context

def test_build_context_with_memory_and_summary():
    chunks = [
        RetrievedChunk(
            chunk_id=1, source_id=1, session_id=1, turn_id=1,
            text="Evidence text.", chunk_index=0, token_count=5, embedding_id=0,
            source_url="http://example.com", title="Title", domain="example.com", score=0.9
        )
    ]
    prompt, citations = build_context(
        query="User Query",
        retrieved_chunks=chunks,
        rolling_summary="Summary text.",
        memory_block="User: Q1\nAssistant: A1"
    )
    
    assert "Summary text." in prompt
    assert "User: Q1\nAssistant: A1" in prompt
    assert "Evidence text." in prompt
    assert "User Question:\nUser Query" in prompt
