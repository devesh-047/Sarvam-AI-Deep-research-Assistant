"""
Research Planner — Stage 4 (ReAct UX enhancement).

Generates a short, visible research plan and targeted search queries before
Tavily search begins.

Design:
  - LLM planning enabled by default — produces a ReAct-style user-visible plan.
  - Safe deterministic fallback if LLM planning fails.
  - Output is ALWAYS a pre-scripted safe summary — never raw chain-of-thought.
  - Plan text uses "Thought:" / "Planned Actions:" labels for ReAct-style UX.
"""
import re
from typing import List

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schema import ResearchPlan

logger = get_logger(__name__)


# ── Stopwords for keyword extraction ─────────────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "what", "which", "who",
    "how", "why", "when", "where", "and", "or", "but", "if", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "about", "into",
    "through", "tell", "me", "explain", "describe", "i", "my", "we", "our",
    "give", "list", "compare", "difference", "between", "vs",
}

# ── Fixed visible plan template (deterministic fallback) ──────────────────────
_PLAN_TEMPLATE = """\
Thought:
To answer this question well, I need authoritative and recent web sources. \
I will search for specific evidence, retrieve the most relevant passages, \
and synthesise a grounded answer with citations.

Planned Actions:
1. Search the web for authoritative sources relevant to the query.
2. Fetch and extract readable content from the top results.
3. Retrieve the most relevant evidence passages via dense retrieval.
4. Build a grounded context and generate a cited answer."""


def _extract_keywords(query: str, max_keywords: int = 5) -> List[str]:
    """Extract meaningful keywords from the query for search query generation."""
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2][:max_keywords]


def _build_search_queries(query: str) -> List[str]:
    """
    Derive 1–3 targeted Tavily search queries from the user's question.

    Strategy:
    - Primary:  the raw query as-is (always reliable).
    - Secondary: keyword-focused variant (strips question framing).
    - Tertiary:  "latest" variant if the query seems time-sensitive.
    """
    queries = [query]

    keywords = _extract_keywords(query)
    if keywords:
        keyword_query = " ".join(keywords)
        if keyword_query.lower() != query.lower()[:len(keyword_query)]:
            queries.append(keyword_query)

    time_signals = {"latest", "recent", "2024", "2025", "new", "current", "now"}
    has_time = any(w in query.lower() for w in time_signals)
    if not has_time and keywords:
        queries.append(f"latest {' '.join(keywords[:3])}")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return deduped[:3]


async def _generate_plan_llm(query: str) -> ResearchPlan:
    """
    LLM-based planning — enabled by default.

    Produces a ReAct-style plan with visible "Thought:" and "Planned Actions:"
    sections. Output is always a safe, user-visible research summary.
    Never exposes raw model reasoning or hidden chain-of-thought.
    """
    from app.research.generator import generate_answer

    planning_prompt = (
        "You are a research planning assistant. Your job is to create a concise, "
        "user-visible research plan for the given question.\n\n"
        "Output format (use EXACTLY — no extra text before or after):\n\n"
        "PLAN:\n"
        "Thought:\n"
        "<One sentence: what kind of evidence is needed and why>\n\n"
        "Planned Actions:\n"
        "1. <Action 1>\n"
        "2. <Action 2>\n"
        "3. <Action 3>\n"
        "4. <Action 4>\n\n"
        "QUERIES:\n"
        "- <targeted web search query 1>\n"
        "- <targeted web search query 2>\n"
        "- <targeted web search query 3>\n\n"
        "Rules:\n"
        "- The Thought must be ONE sentence only.\n"
        "- Actions must be operational steps (search, retrieve, compare, synthesise).\n"
        "- Queries must be concrete, concise, web-searchable phrases.\n"
        "- Do NOT include reasoning, explanation, or chain-of-thought.\n"
        "- Do NOT include any text outside the PLAN/QUERIES blocks.\n\n"
        f"Question: {query}\n"
    )
    try:
        raw = await generate_answer(planning_prompt)

        # Parse PLAN section
        plan_match = re.search(r"PLAN:\s*(.*?)(?:QUERIES:|$)", raw, re.DOTALL)
        plan_text = plan_match.group(1).strip() if plan_match else _PLAN_TEMPLATE

        # Parse QUERIES section
        queries_match = re.search(r"QUERIES:\s*(.*)", raw, re.DOTALL)
        queries: List[str] = []
        if queries_match:
            lines = queries_match.group(1).strip().splitlines()
            for line in lines:
                line = re.sub(r"^[-*\d.]+\s*", "", line).strip()
                if line and len(line) > 3:
                    queries.append(line)

        if not queries:
            queries = _build_search_queries(query)

        return ResearchPlan(plan_text=plan_text, search_queries=queries[:3])

    except Exception as e:
        logger.warning(f"[Planner] LLM planning failed, using deterministic fallback: {e}")
        return _generate_plan_deterministic(query)


def _generate_plan_deterministic(query: str) -> ResearchPlan:
    """Fast, deterministic planner — no LLM required. Used as fallback."""
    return ResearchPlan(
        plan_text=_PLAN_TEMPLATE,
        search_queries=_build_search_queries(query),
    )


async def generate_plan(query: str) -> ResearchPlan:
    """
    Generate a research plan for the given query.

    Returns a ResearchPlan with:
      - plan_text: short ReAct-style plan visible to the user
      - search_queries: 1–3 targeted search strings for Tavily

    Uses LLM planning by default; falls back to deterministic on failure.
    """
    if settings.enable_llm_planning:
        return await _generate_plan_llm(query)
    return _generate_plan_deterministic(query)
