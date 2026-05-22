"""
Query Normalizer — Phase 5 Multilingual Layer.

Converts multilingual / Romanized Indic research queries into clean
English research queries suitable for the planning + retrieval pipeline.

Strategy (waterfall):
  1. If English → return as-is (no API call)
  2. Try Sarvam AI translate to English
  3. Fall back to LLM normalization prompt (generate_answer)
  4. Fall back to raw query
  → Never raises; always returns a usable string
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.multilingual.language_detector import detect_language, is_non_english, DETECTED_TO_SARVAM_SOURCE

logger = get_logger(__name__)

_LLM_NORMALIZE_PROMPT = """\
You are a multilingual research query normalization assistant.
The user has typed a research question in a mixed or non-English language.

Convert the following query into a clean, concise English research question
that preserves the user's intent, entities, and technical terms.

Rules:
- Output ONLY the normalized English query — no explanation.
- Keep it concise (under 25 words).
- Preserve named entities (companies, places, technologies).
- Make it suitable for academic/web research.

Query: {query}

Normalized English query:"""


async def normalize_query(query: str) -> tuple[str, str]:
    """
    Normalize a potentially multilingual query into English.

    Returns:
        (normalized_query, detected_lang)
        normalized_query: clean English research query
        detected_lang:    detected language code string

    Never raises — always falls back gracefully.
    """
    detected_lang = detect_language(query)

    # Fast path: already English
    if not is_non_english(detected_lang):
        logger.info(f"[Normalizer] Query is English — no normalization needed.")
        return query, detected_lang

    logger.info(f"[Normalizer] Detected '{detected_lang}' — attempting normalization.")

    # ── Step 1: Try Sarvam ────────────────────────────────────────────────────
    if settings.sarvam_api_key and settings.enable_multilingual:
        try:
            from app.multilingual.sarvam_client import transliterate_to_english
            source_lang = DETECTED_TO_SARVAM_SOURCE.get(detected_lang, "hi-IN")
            normalized = await transliterate_to_english(query, source_lang)
            normalized = normalized.strip()
            if normalized and len(normalized) > 3:
                logger.info(f"[Normalizer] Sarvam normalized: '{normalized}'")
                return normalized, detected_lang
        except Exception as e:
            logger.warning(f"[Normalizer] Sarvam normalization failed: {e}. Trying LLM fallback.")

    # ── Step 2: LLM fallback ──────────────────────────────────────────────────
    try:
        from app.research.generator import generate_answer
        prompt = _LLM_NORMALIZE_PROMPT.format(query=query)
        normalized = await generate_answer(prompt)
        normalized = normalized.strip()
        # Basic sanity: output should be shorter than 200 chars and not start with "I"/"Sorry"
        if normalized and len(normalized) < 200 and not normalized.lower().startswith(("sorry", "i cannot", "i'm")):
            # Clean up any trailing explanation
            normalized = normalized.split("\n")[0].strip().strip('"').strip("'")
            logger.info(f"[Normalizer] LLM normalized: '{normalized}'")
            return normalized, detected_lang
    except Exception as e:
        logger.warning(f"[Normalizer] LLM normalization failed: {e}. Using raw query.")

    # ── Step 3: Raw query fallback ────────────────────────────────────────────
    logger.info(f"[Normalizer] Falling back to raw query.")
    return query, detected_lang
