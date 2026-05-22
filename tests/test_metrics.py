"""Tests for evaluation/metrics.py"""
import pytest
from evaluation.metrics import (
    citation_presence,
    citation_count,
    sources_cited,
    answer_completeness,
    answer_length,
    retrieval_hit_count,
    uncertainty_acknowledged,
    conflict_acknowledged,
    grounding_ratio,
    score_result,
    aggregate_scores,
)


def _make_result(**kwargs):
    base = {
        "id": "T001",
        "category": "factual",
        "question": "Test question?",
        "answer": "",
        "citations": [],
        "retrieved_chunks": [],
        "plan_text": "",
        "search_queries": [],
        "latency_ms": 100.0,
        "error": None,
    }
    base.update(kwargs)
    return base


# ── Citation metrics ──────────────────────────────────────────────────────────

def test_citation_presence_found():
    r = _make_result(answer="Quantum computing has made strides [S1] in recent years [S2].")
    assert citation_presence(r) == 1.0


def test_citation_presence_missing():
    r = _make_result(answer="This answer has no citation markers at all.")
    assert citation_presence(r) == 0.0


def test_citation_count_multiple():
    r = _make_result(answer="Point A [S1]. Point B [S2]. Also [S1] again.")
    assert citation_count(r) == 2  # unique only


def test_sources_cited():
    r = _make_result(citations=[
        {"label": "[S1]", "title": "T1", "url": "http://a.com", "domain": "a.com"},
        {"label": "[S2]", "title": "T2", "url": "http://b.com", "domain": "b.com"},
    ])
    assert sources_cited(r) == 2


# ── Completeness ──────────────────────────────────────────────────────────────

def test_answer_completeness_empty():
    r = _make_result(answer="")
    assert answer_completeness(r) == "empty"


def test_answer_completeness_short():
    r = _make_result(answer="Short answer.")
    assert answer_completeness(r) == "short"


def test_answer_completeness_complete():
    r = _make_result(answer="A" * 250)
    assert answer_completeness(r) == "complete"


def test_answer_length():
    r = _make_result(answer="Hello world")
    assert answer_length(r) == len("Hello world")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def test_retrieval_hit_count():
    r = _make_result(retrieved_chunks=[
        {"domain": "a.com", "score": 0.8},
        {"domain": "b.com", "score": 0.1},
        {"domain": "c.com", "score": 0.5},
    ])
    assert retrieval_hit_count(r, min_score=0.3) == 2


# ── Uncertainty and conflict ──────────────────────────────────────────────────

def test_uncertainty_acknowledged_true():
    r = _make_result(answer="There is insufficient evidence to confirm this claim.")
    assert uncertainty_acknowledged(r) is True


def test_uncertainty_acknowledged_false():
    r = _make_result(answer="Quantum computing is definitely the future.")
    assert uncertainty_acknowledged(r) is False


def test_conflict_acknowledged_true():
    r = _make_result(answer="While some studies show benefit, others contradict these findings.")
    assert conflict_acknowledged(r) is True


def test_conflict_acknowledged_false():
    r = _make_result(answer="All studies agree that the treatment is effective [S1].")
    assert conflict_acknowledged(r) is False


# ── Grounding ratio ───────────────────────────────────────────────────────────

def test_grounding_ratio_all_cited():
    r = _make_result(answer="First point [S1]. Second point [S2]. Third [S3].")
    ratio = grounding_ratio(r)
    assert ratio == 1.0


def test_grounding_ratio_none_cited():
    r = _make_result(answer="First point. Second point. Third point.")
    assert grounding_ratio(r) == 0.0


def test_grounding_ratio_partial():
    r = _make_result(answer="First point [S1]. Second point. Third point.")
    ratio = grounding_ratio(r)
    assert 0.0 < ratio < 1.0


# ── Aggregate scorer ──────────────────────────────────────────────────────────

def test_score_result_has_all_keys():
    r = _make_result(answer="Some answer [S1].", citations=[{"label": "[S1]", "title": "T", "url": "http://x.com", "domain": "x.com"}])
    scored = score_result(r)
    expected_keys = [
        "id", "category", "has_error", "citation_presence", "citation_count",
        "sources_cited", "answer_completeness", "answer_length",
        "retrieval_hit_count", "retrieval_source_count",
        "uncertainty_acknowledged", "conflict_acknowledged",
        "grounding_ratio", "latency_ms",
    ]
    for key in expected_keys:
        assert key in scored, f"Missing key: {key}"


def test_aggregate_scores_summary():
    results = [
        score_result(_make_result(id="A1", answer="Answer [S1]." * 20, citations=[{"label":"[S1]","title":"T","url":"http://x.com","domain":"x.com"}])),
        score_result(_make_result(id="A2", answer="", error="failed")),
    ]
    agg = aggregate_scores(results)
    assert agg["total"] == 2
    assert agg["errors"] == 1
    assert 0.0 <= agg["success_rate"] <= 1.0
