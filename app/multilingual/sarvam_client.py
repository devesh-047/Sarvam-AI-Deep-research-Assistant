"""
Sarvam AI API Client — Phase 5 Multilingual Layer.

Provides async wrappers around the Sarvam AI translation and
transliteration endpoints.

Design:
  - Async-first (httpx.AsyncClient)
  - Timeout: 10 seconds per request
  - Retries: up to 2 attempts with 1s delay
  - Graceful failure: raises SarvamUnavailableError on all failures
    → callers catch this and fall back to English

Environment variable:
  SARVAM_API_KEY  — required for live API calls
"""
import asyncio
import os
from typing import Optional

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(10.0)
_MAX_RETRIES = 2


class SarvamUnavailableError(Exception):
    """Raised when Sarvam API is unavailable or returns an error."""
    pass


async def _post_with_retry(url: str, payload: dict, api_key: str) -> dict:
    """POST to Sarvam API with retry logic. Raises SarvamUnavailableError on failure."""
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    last_err: Optional[Exception] = None

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    return response.json()
                else:
                    last_err = SarvamUnavailableError(
                        f"Sarvam API returned {response.status_code}: {response.text[:200]}"
                    )
                    logger.warning(f"[Sarvam] Attempt {attempt + 1} failed: {last_err}")
            except httpx.TimeoutException as e:
                last_err = SarvamUnavailableError(f"Sarvam API timed out: {e}")
                logger.warning(f"[Sarvam] Timeout on attempt {attempt + 1}")
            except Exception as e:
                last_err = SarvamUnavailableError(f"Sarvam API error: {e}")
                logger.warning(f"[Sarvam] Error on attempt {attempt + 1}: {e}")

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1.0)

    raise last_err or SarvamUnavailableError("Sarvam API failed after retries.")


def _get_api_key() -> str:
    key = settings.sarvam_api_key or os.environ.get("SARVAM_API_KEY", "")
    if not key:
        raise SarvamUnavailableError("SARVAM_API_KEY is not configured.")
    return key


async def _translate_chunk(chunk: str, source_lang: str, target_lang: str, api_key: str) -> str:
    """Helper to translate a single chunk of text."""
    if not chunk.strip():
        return ""
    payload = {
        "input": chunk,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }
    result = await _post_with_retry(settings.sarvam_translate_url, payload, api_key)
    translated = result.get("translated_text") or result.get("translation") or ""
    return translated


async def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source_lang to target_lang using Sarvam AI.
    Automatically handles Sarvam's 2000 character limit by chunking.

    Args:
        text:        Text to translate
        source_lang: Sarvam language code, e.g. "en-IN", "hi-IN"
        target_lang: Sarvam language code, e.g. "hi-IN", "bn-IN"

    Returns:
        Translated text string.

    Raises:
        SarvamUnavailableError on any failure.
    """
    if not text.strip():
        return text

    api_key = _get_api_key()

    # Split by paragraphs to preserve markdown structure
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 > 1800 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = p
        else:
            if current_chunk:
                current_chunk += "\n\n" + p
            else:
                current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)

    # Fallback: Further split any single paragraph that exceeds 1800 chars
    final_chunks = []
    for c in chunks:
        while len(c) > 1800:
            split_idx = c.rfind(' ', 0, 1800)
            if split_idx <= 0:
                split_idx = 1800
            final_chunks.append(c[:split_idx])
            c = c[split_idx:].strip()
        if c:
            final_chunks.append(c)

    # Translate all chunks concurrently for speed
    tasks = [_translate_chunk(c, source_lang, target_lang, api_key) for c in final_chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    translated_pieces = []
    for res in results:
        if isinstance(res, Exception):
            logger.error(f"[Sarvam] Chunk translation failed: {res}")
            raise res
        translated_pieces.append(res)

    translated_final = '\n\n'.join(translated_pieces)
    if not translated_final.strip():
        raise SarvamUnavailableError("Sarvam translate returned empty text.")

    logger.info(f"[Sarvam] Translated ({source_lang}→{target_lang}): {len(translated_final)} chars (in {len(final_chunks)} chunks)")
    return translated_final

async def transliterate_to_english(text: str, source_lang: str) -> str:
    """
    Transliterate Romanized Indic text to proper English interpretation
    using Sarvam's transliterate endpoint.

    For query normalization: converts Romanized Indic input into a
    clean English research query.

    Args:
        text:        Romanized Indic text (e.g. "bharat me AI startups ka future")
        source_lang: Sarvam source language code (e.g. "hi-IN")

    Returns:
        English interpretation of the query.

    Raises:
        SarvamUnavailableError on any failure.
    """
    if not text.strip():
        return text

    api_key = _get_api_key()

    # Sarvam translate endpoint: translate from Indic → English
    # (transliterate endpoint converts Roman script → native script;
    #  we actually want translate to English for query normalization)
    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": "en-IN",
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }

    result = await _post_with_retry(settings.sarvam_translate_url, payload, api_key)
    english_text = result.get("translated_text") or result.get("translation") or ""
    if not english_text:
        raise SarvamUnavailableError("Sarvam transliterate returned empty text.")
    logger.info(f"[Sarvam] Transliterated to English: '{english_text[:80]}'")
    return english_text
