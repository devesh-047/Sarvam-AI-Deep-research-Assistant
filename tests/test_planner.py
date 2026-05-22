"""Tests for app/research/planner.py"""
import pytest
from app.research.planner import (
    generate_plan,
    _build_search_queries,
    _extract_keywords,
    _generate_plan_deterministic,
    _PLAN_TEMPLATE,
)


def test_generate_plan_deterministic_returns_plan():
    plan = _generate_plan_deterministic("What is quantum computing?")
    assert plan.plan_text, "plan_text should be non-empty"
    assert _PLAN_TEMPLATE in plan.plan_text or plan.plan_text == _PLAN_TEMPLATE


def test_generate_plan_search_queries_non_empty():
    plan = _generate_plan_deterministic("What is quantum computing?")
    assert len(plan.search_queries) >= 1, "At least one search query must be generated"


def test_generate_plan_deterministic():
    """Same input → same output."""
    p1 = _generate_plan_deterministic("How does FAISS work?")
    p2 = _generate_plan_deterministic("How does FAISS work?")
    assert p1.plan_text == p2.plan_text
    assert p1.search_queries == p2.search_queries


def test_generate_plan_max_three_queries():
    plan = _generate_plan_deterministic("Tell me about the latest advancements in large language models")
    assert len(plan.search_queries) <= 3


def test_extract_keywords_strips_stopwords():
    keywords = _extract_keywords("What is the capital of France?")
    assert "what" not in keywords
    assert "is" not in keywords
    assert "the" not in keywords
    assert "of" not in keywords


def test_extract_keywords_keeps_meaningful_words():
    keywords = _extract_keywords("quantum computing breakthroughs 2024")
    assert "quantum" in keywords or "computing" in keywords


def test_build_search_queries_returns_list():
    queries = _build_search_queries("What are the latest AI breakthroughs?")
    assert isinstance(queries, list)
    assert len(queries) >= 1


def test_build_search_queries_first_is_original():
    q = "How does BERT work?"
    queries = _build_search_queries(q)
    assert queries[0] == q


@pytest.mark.asyncio
async def test_generate_plan_async():
    plan = await generate_plan("What is the speed of light?")
    assert plan.plan_text
    assert isinstance(plan.search_queries, list)
    assert len(plan.search_queries) >= 1


@pytest.mark.asyncio
async def test_generate_plan_with_memory_block():
    plan = await generate_plan(
        "How many times should I consume it?",
        memory_block="User: Is drinking coffee good?\nAssistant: Yes, coffee has benefits."
    )
    assert plan.plan_text
    assert isinstance(plan.search_queries, list)
    assert len(plan.search_queries) >= 1
