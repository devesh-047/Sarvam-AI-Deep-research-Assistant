"""
Rolling conversation summarizer (Stage 3).

Calls the LLM to generate/update a rolling summary of the research conversation.
"""
from app.core.logging import get_logger
from app.research.generator import generate_answer

logger = get_logger(__name__)

async def update_rolling_summary(old_summary: str, last_query: str, last_answer: str) -> str:
    """
    Updates the conversation summary incorporating the latest turn.
    """
    prompt = (
        "You are an expert research assistant. Your task is to update a rolling summary of the research conversation so far.\n"
        f"Existing Summary: {old_summary or 'No summary yet.'}\n\n"
        "Latest Turn:\n"
        f"User Query: {last_query}\n"
        f"Assistant Answer: {last_answer}\n\n"
        "Write an updated summary incorporating the latest turn. Keep it concise, factual, and under 300 words. Do not use prefixes like 'Summary:'."
    )
    
    try:
        new_summary = await generate_answer(prompt)
        new_summary = new_summary.strip()
        # Strip common prefixes
        for prefix in ["Summary:", "Updated Summary:", "rolling summary:", "Rolling Summary:"]:
            if new_summary.startswith(prefix):
                new_summary = new_summary[len(prefix):].strip()
        return new_summary
    except Exception as e:
        logger.error(f"[Summarizer] Failed to generate rolling summary: {e}")
        # Fallback summary
        fallback = f"{old_summary or ''}\n- Added query: {last_query}".strip()
        return fallback
