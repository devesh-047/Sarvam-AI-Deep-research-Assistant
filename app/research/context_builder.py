"""
Context builder: turns retrieved chunks into a bounded, well-formatted LLM prompt.

Responsibilities:
- Assign citation labels [S1], [S2], …
- Enforce token budget (max_context_tokens).
- De-duplicate by source URL so one source doesn't dominate.
- Return both the formatted prompt string and the Citation list.
"""
from typing import List, Tuple

import tiktoken

from app.models.schema import RetrievedChunk, Citation
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def _count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


SYSTEM_PROMPT = """You are a grounded deep research assistant.

Your job is to answer the user's question using ONLY the evidence provided below.
Rules:
- Cite every factual claim using the source label, e.g. [S1], [S2].
- If the evidence is insufficient, say so explicitly.
- If sources conflict, mention the disagreement and cite both.
- Do NOT invent facts or cite sources not listed in the evidence.
- Write a clear, well-structured answer."""


def build_context(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    max_tokens: int = None,
    rolling_summary: str = "",
    memory_block: str = "",
) -> Tuple[str, List[Citation]]:
    """
    Build the final prompt string and a list of citations.

    Args:
        query: The original user question.
        retrieved_chunks: Ranked list of RetrievedChunk objects.
        max_tokens: Token budget for the entire context.
        rolling_summary: Rolling summary of the conversation so far.
        memory_block: Formatted string of recent turns.

    Returns:
        (prompt_text, citations)
    """
    max_tokens = max_tokens or settings.max_context_tokens

    # --- Build evidence section within token budget ---
    url_to_label: dict = {}
    citations: List[Citation] = []
    evidence_parts: List[str] = []
    used_tokens = 0

    summary_section = f"Conversation Summary:\n{rolling_summary}\n\n" if rolling_summary else ""
    memory_section = f"Recent Conversation Turns:\n{memory_block}\n\n" if memory_block else ""

    # Reserve tokens for the system + question preamble + memory + summary
    preamble = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{summary_section}"
        f"{memory_section}"
        f"User Question:\n{query}\n\n"
        f"Evidence:\n"
    )
    preamble_tokens = _count_tokens(preamble)
    budget = max_tokens - preamble_tokens - 200  # 200-token buffer for answer instructions

    for chunk in retrieved_chunks:
        url = chunk.source_url
        label = url_to_label.get(url) or f"[S{len(url_to_label) + 1}]"
        block = f"{label} {chunk.title} | {chunk.domain} | {chunk.source_url}\n{chunk.text}\n"
        block_tokens = _count_tokens(block)
        if used_tokens + block_tokens > budget:
            logger.info(f"Token budget reached at chunk {label}. Stopping evidence selection.")
            break
        if url not in url_to_label:
            url_to_label[url] = label
            citations.append(Citation(
                label=label,
                title=chunk.title,
                url=url,
                domain=chunk.domain,
            ))
        chunk.citation_label = label
        evidence_parts.append(block)
        used_tokens += block_tokens

    if not evidence_parts:
        evidence_section = "(No evidence retrieved. State that you cannot answer this question.)\n"
    else:
        evidence_section = "\n".join(evidence_parts)

    # --- Assemble final prompt ---
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{summary_section}"
        f"{memory_section}"
        f"User Question:\n{query}\n\n"
        f"Evidence:\n{evidence_section}\n\n"
        f"Answer Requirements:\n"
        f"- Cite claims with [S1], [S2], … as shown in the evidence.\n"
        f"- Mention uncertainty or insufficient evidence where applicable.\n"
        f"- Do not invent sources.\n"
    )

    logger.info(
        f"Built context: {len(evidence_parts)} evidence blocks, "
        f"~{_count_tokens(prompt)} tokens, {len(citations)} unique sources."
    )
    return prompt, citations
