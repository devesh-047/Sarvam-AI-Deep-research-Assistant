"""
Response Localizer — Phase 5 Multilingual Layer.

Translates the grounded English answer into the user's selected target
language while preserving all citations, URLs, and source labels.

Strategy:
  1. If target_language == "en" → return unchanged (no API call)
  2. Extract citations block from answer (keep citations in English always)
  3. Translate the body text via Sarvam AI
  4. Re-append the original English citations block
  5. Fall back to English answer on any failure

CRITICAL: URLs, citation labels ([S1], [S2]...), and domain names
are NEVER translated. Only human-readable prose is localized.
"""
import re

from app.core.config import settings
from app.core.logging import get_logger
from app.multilingual.language_detector import LANG_CODE_MAP

logger = get_logger(__name__)

# Pattern to split off the "**Sources:**" block at the end of an answer
_SOURCES_SPLIT_RE = re.compile(
    r"\n\*\*Sources:\*\*.*$",
    re.DOTALL
)
_CITATION_RE = re.compile(r"\[S\d+\]")


def _split_answer_and_citations(answer: str) -> tuple[str, str]:
    """
    Split the answer into (prose, citations_block).
    Citations block includes the '**Sources:**' header and all bullet points.
    """
    match = _SOURCES_SPLIT_RE.search(answer)
    if match:
        prose = answer[:match.start()]
        citations_block = answer[match.start():]
        return prose.strip(), citations_block.strip()
    return answer.strip(), ""


def _protect_inline_citations(text: str) -> tuple[str, dict[str, str]]:
    """Replace citation labels with stable placeholders before translation."""
    replacements: dict[str, str] = {}

    def repl(match: re.Match) -> str:
        label = match.group(0)
        placeholder = f"CITATIONTOKEN{len(replacements)}"
        replacements[placeholder] = label
        return placeholder

    return _CITATION_RE.sub(repl, text), replacements


def _restore_inline_citations(text: str, replacements: dict[str, str]) -> str:
    """Restore citation placeholders and append any labels removed by translation."""
    restored = text
    missing_labels = []
    for placeholder, label in replacements.items():
        if placeholder in restored:
            restored = restored.replace(placeholder, label)
        elif label not in restored:
            missing_labels.append(label)

    if missing_labels:
        restored = restored.rstrip() + " " + " ".join(missing_labels)
    return restored


async def localize_response(
    answer: str,
    target_language: str,
) -> str:
    """
    Localize the grounded English answer to the target language.

    Args:
        answer:          Grounded English answer (may include **Sources:** block)
        target_language: ISO 639-1 code, e.g. "hi", "bn", "ta". "en" = no-op.

    Returns:
        Localized answer string with English citations re-appended.
        Falls back to original English answer on any failure.
    """
    if not target_language or target_language == "en":
        return answer

    if not settings.sarvam_api_key or not settings.enable_multilingual:
        logger.info("[Localizer] Sarvam API key not set or multilingual disabled. Returning English.")
        return answer

    target_sarvam = LANG_CODE_MAP.get(target_language)
    if not target_sarvam:
        logger.warning(f"[Localizer] Unknown target language '{target_language}'. Returning English.")
        return answer

    try:
        from app.multilingual.sarvam_client import translate

        # Split prose from citations
        prose, citations_block = _split_answer_and_citations(answer)

        if not prose:
            return answer

        # Translate prose only. Inline citations are protected because the
        # localized answer must preserve the exact labels used by the evidence.
        protected_prose, citation_tokens = _protect_inline_citations(prose)
        localized_prose = await translate(
            text=protected_prose,
            source_lang="en-IN",
            target_lang=target_sarvam,
        )
        localized_prose = _restore_inline_citations(localized_prose, citation_tokens)

        # Re-assemble: localized prose + original English citations
        if citations_block:
            result = localized_prose.strip() + "\n\n" + citations_block
        else:
            result = localized_prose.strip()

        logger.info(
            f"[Localizer] Localized to {target_language} "
            f"({len(prose)} → {len(localized_prose)} chars). "
            f"Citations preserved: {bool(citations_block)}"
        )
        return result

    except Exception as e:
        logger.warning(f"[Localizer] Localization to '{target_language}' failed: {e}. Returning English.")
        return answer
