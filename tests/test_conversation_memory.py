"""Tests for conversation memory helpers."""
import pytest
from app.models.schema import ResearchTurn
from app.memory.conversation_memory import format_memory_block

def test_format_memory_block_empty():
    assert format_memory_block([], max_tokens=100) == ""

def test_format_memory_block_within_budget():
    turns = [
        ResearchTurn(session_id=1, user_query="Q1", search_queries_json="[]", opened_urls_json="[]", final_answer="A1"),
        ResearchTurn(session_id=1, user_query="Q2", search_queries_json="[]", opened_urls_json="[]", final_answer="A2"),
    ]
    block = format_memory_block(turns, max_tokens=100)
    assert "User: Q1\nAssistant: A1" in block
    assert "User: Q2\nAssistant: A2" in block

def test_format_memory_block_exceeds_budget():
    turns = [
        ResearchTurn(session_id=1, user_query="Q1", search_queries_json="[]", opened_urls_json="[]", final_answer="A1 " * 100),
        ResearchTurn(session_id=1, user_query="Q2", search_queries_json="[]", opened_urls_json="[]", final_answer="A2"),
    ]
    # Small token budget should drop the older turn (Q1) and only keep the newer one (Q2)
    block = format_memory_block(turns, max_tokens=30)
    assert "User: Q2\nAssistant: A2" in block
    assert "User: Q1" not in block
