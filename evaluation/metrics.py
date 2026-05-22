"""
Evaluation metrics — Stage 4.

All metrics operate on a single result dict (from the runner).
No LLM calls required. Simple, defensible, reproducible.

Result dict schema:
  {
    "id": str,
    "category": str,
    "question": str,
    "answer": str,
    "citations": list[dict],           # [{"label", "title", "url", "domain"}]
    "retrieved_chunks": list[dict],    # [{"title", "domain", "score", ...}]
    "plan_text": str,
    "search_queries": list[str],
    "latency_ms": float,
    "error": str | None,
  }
"""
import re
from typing import Any, Dict, List


# ── Citation metrics ──────────────────────────────────────────────────────────

def citation_presence(result: Dict[str, Any]) -> float:
    """1.0 if the answer contains at least one [Sn] citation marker, else 0.0."""
    answer = result.get("answer") or ""
    return 1.0 if re.search(r"\[S\d+\]", answer) else 0.0


def citation_count(result: Dict[str, Any]) -> int:
    """Number of unique [Sn] markers referenced in the answer."""
    answer = result.get("answer") or ""
    return len(set(re.findall(r"\[S\d+\]", answer)))


def sources_cited(result: Dict[str, Any]) -> int:
    """Number of distinct sources in the citations list."""
    return len(result.get("citations") or [])


def citation_reference_integrity(result: Dict[str, Any]) -> float:
    """
    Fraction of answer citation markers that exist in the citations payload.

    This catches hallucinated labels such as `[S9]` when only `[S1]` and `[S2]`
    were actually produced by the context builder.
    """
    answer_labels = set(re.findall(r"\[S\d+\]", result.get("answer") or ""))
    if not answer_labels:
        return 0.0
    declared_labels = {c.get("label") for c in (result.get("citations") or []) if c.get("label")}
    return round(len(answer_labels & declared_labels) / len(answer_labels), 3)


def citation_source_traceability(result: Dict[str, Any]) -> float:
    """
    Fraction of declared citation URLs that came from retrieved chunks.

    A grounded answer should cite only sources that were actually selected from
    retrieval, not arbitrary URLs introduced during generation or formatting.
    """
    citations = result.get("citations") or []
    if not citations:
        return 0.0
    retrieved_urls = {c.get("source_url") for c in (result.get("retrieved_chunks") or []) if c.get("source_url")}
    citation_urls = [c.get("url") for c in citations if c.get("url")]
    if not citation_urls:
        return 0.0
    return round(sum(1 for url in citation_urls if url in retrieved_urls) / len(citation_urls), 3)


# ── Answer completeness ───────────────────────────────────────────────────────

def answer_completeness(result: Dict[str, Any], min_chars: int = 200) -> str:
    """
    Returns:
      "empty"    — answer is blank or only whitespace
      "short"    — answer is non-empty but below min_chars
      "complete" — answer is at least min_chars characters
    """
    answer = (result.get("answer") or "").strip()
    if not answer:
        return "empty"
    if len(answer) < min_chars:
        return "short"
    return "complete"


def answer_length(result: Dict[str, Any]) -> int:
    """Character length of the answer."""
    return len((result.get("answer") or "").strip())


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def retrieval_hit_count(result: Dict[str, Any], min_score: float = 0.3) -> int:
    """Number of retrieved chunks with score above min_score."""
    chunks = result.get("retrieved_chunks") or []
    return sum(1 for c in chunks if (c.get("score") or 0.0) >= min_score)


def retrieval_source_count(result: Dict[str, Any]) -> int:
    """Number of unique domains in retrieved chunks."""
    chunks = result.get("retrieved_chunks") or []
    return len({c.get("domain", "") for c in chunks if c.get("domain")})


def retrieval_method_mix(result: Dict[str, Any]) -> str:
    """Compact summary of dense/BM25/hybrid retrieval methods represented."""
    chunks = result.get("retrieved_chunks") or []
    methods = sorted({c.get("retrieval_method", "dense") for c in chunks})
    return ",".join(methods) if methods else "none"


# ── Uncertainty and conflict ─────────────────────────────────────────────────

_UNCERTAINTY_PHRASES = [
    "insufficient", "not enough information", "cannot determine",
    "unclear", "uncertain", "limited evidence", "no evidence",
    "could not find", "not available", "not publicly", "unknown",
    "not confirmed", "unverified",
]

_CONFLICT_PHRASES = [
    "conflict", "contradict", "disagree", "inconsistent", "differ",
    "on the other hand", "however", "while some", "while others",
    "debate", "disputed", "controversy", "mixed evidence",
]


def uncertainty_acknowledged(result: Dict[str, Any]) -> bool:
    """True if the answer contains phrases indicating epistemic humility."""
    answer = (result.get("answer") or "").lower()
    return any(phrase in answer for phrase in _UNCERTAINTY_PHRASES)


def conflict_acknowledged(result: Dict[str, Any]) -> bool:
    """True if the answer acknowledges conflicting or contradictory sources."""
    answer = (result.get("answer") or "").lower()
    return any(phrase in answer for phrase in _CONFLICT_PHRASES)


# ── Grounding ratio ───────────────────────────────────────────────────────────

def grounding_ratio(result: Dict[str, Any]) -> float:
    """
    Fraction of sentences that contain at least one citation marker.
    0.0–1.0. Higher is better for factual claims.
    """
    answer = (result.get("answer") or "").strip()
    if not answer:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[S\d+\]", s))
    return round(cited / len(sentences), 3)


# ── Aggregate scorer ──────────────────────────────────────────────────────────

def score_result(result: Dict[str, Any], min_chars: int = 200, min_score: float = 0.3) -> Dict[str, Any]:
    """
    Compute all metrics for a single result and return as a flat dict.
    """
    has_error = bool(result.get("error"))
    return {
        "id": result.get("id"),
        "category": result.get("category"),
        "has_error": has_error,
        "citation_presence": citation_presence(result),
        "citation_count": citation_count(result),
        "sources_cited": sources_cited(result),
        "citation_reference_integrity": citation_reference_integrity(result),
        "citation_source_traceability": citation_source_traceability(result),
        "answer_completeness": answer_completeness(result, min_chars=min_chars),
        "answer_length": answer_length(result),
        "retrieval_hit_count": retrieval_hit_count(result, min_score=min_score),
        "retrieval_source_count": retrieval_source_count(result),
        "retrieval_method_mix": retrieval_method_mix(result),
        "uncertainty_acknowledged": uncertainty_acknowledged(result),
        "conflict_acknowledged": conflict_acknowledged(result),
        "grounding_ratio": grounding_ratio(result),
        "latency_ms": result.get("latency_ms") or 0.0,
    }


def aggregate_scores(scored_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a list of scored results into summary statistics.
    """
    if not scored_results:
        return {}

    n = len(scored_results)
    errors = sum(1 for r in scored_results if r.get("has_error"))

    def _avg(key):
        vals = [r[key] for r in scored_results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    complete = sum(1 for r in scored_results if r.get("answer_completeness") == "complete")
    short = sum(1 for r in scored_results if r.get("answer_completeness") == "short")
    empty = sum(1 for r in scored_results if r.get("answer_completeness") == "empty")

    return {
        "total": n,
        "errors": errors,
        "success_rate": round((n - errors) / n, 3),
        "avg_citation_presence": _avg("citation_presence"),
        "avg_citation_count": _avg("citation_count"),
        "avg_citation_reference_integrity": _avg("citation_reference_integrity"),
        "avg_citation_source_traceability": _avg("citation_source_traceability"),
        "avg_grounding_ratio": _avg("grounding_ratio"),
        "avg_latency_ms": _avg("latency_ms"),
        "avg_answer_length": _avg("answer_length"),
        "complete_answers": complete,
        "short_answers": short,
        "empty_answers": empty,
        "uncertainty_rate": round(sum(1 for r in scored_results if r.get("uncertainty_acknowledged")) / n, 3),
        "conflict_rate": round(sum(1 for r in scored_results if r.get("conflict_acknowledged")) / n, 3),
    }
