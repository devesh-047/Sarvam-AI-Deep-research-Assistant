"""
LLM answer generator — Stage 4 (ReAct UX enhancement).

Supports:
  - gemini   (Google Gemini API via google-genai SDK)
  - openai   (OpenAI / any OpenAI-compatible endpoint)

New in this version:
  - generate_answer_stream(): async generator that yields string chunks.
    Gemini: uses generate_content_stream().
    OpenAI: uses stream=True + delta.content.
    Graceful fallback: yields the whole answer as a single chunk on error.
"""
import os
from typing import List, AsyncGenerator

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schema import Citation

logger = get_logger(__name__)


def _make_openai_client() -> AsyncOpenAI:
    api_key = settings.llm_api_key or os.environ.get("OPENAI_API_KEY")
    kwargs = {"api_key": api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return AsyncOpenAI(**kwargs)


# ── Non-streaming generation ──────────────────────────────────────────────────

async def _generate_openai(prompt: str) -> str:
    api_key = settings.llm_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not set.")
        return "⚠️ OpenAI API key not set. Set LLM_API_KEY/OPENAI_API_KEY."

    client = _make_openai_client()
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=settings.max_output_tokens,
    )
    return response.choices[0].message.content or ""


async def _generate_gemini(prompt: str) -> str:
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set.")
        return "⚠️ GEMINI_API_KEY not set in .env."

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    thinking_config = None
    if settings.gemini_thinking_budget is not None:
        thinking_config = types.ThinkingConfig(
            thinking_budget=settings.gemini_thinking_budget
        )

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=settings.max_output_tokens,
        thinking_config=thinking_config,
    )
    response = await client.aio.models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=config
    )
    return response.text or ""


async def generate_answer(prompt: str) -> str:
    """
    Send the assembled prompt to the LLM and return the full answer text.
    """
    provider = settings.llm_provider.lower()
    try:
        if provider == "gemini":
            answer = await _generate_gemini(prompt)
        elif provider == "openai":
            answer = await _generate_openai(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'")

        logger.info(f"LLM generated answer ({len(answer)} chars) via {provider}.")
        return answer.strip()
    except Exception as e:
        logger.error(f"LLM generation failed for provider {provider}: {e}")
        return f"⚠️ Answer generation failed: {e}"


# ── Streaming generation ──────────────────────────────────────────────────────

async def _stream_gemini(prompt: str) -> AsyncGenerator[str, None]:
    """Yield answer chunks from Gemini using generate_content_stream."""
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        yield "⚠️ GEMINI_API_KEY not set in .env."
        return

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    thinking_config = None
    if settings.gemini_thinking_budget is not None:
        thinking_config = types.ThinkingConfig(
            thinking_budget=settings.gemini_thinking_budget
        )

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=settings.max_output_tokens,
        thinking_config=thinking_config,
    )

    try:
        async for chunk in await client.aio.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
            config=config,
        ):
            # Only yield non-thinking text parts
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logger.warning(f"[Generator] Gemini streaming failed, falling back: {e}")
        # Fallback: yield whole answer
        answer = await _generate_gemini(prompt)
        yield answer


async def _stream_openai(prompt: str) -> AsyncGenerator[str, None]:
    """Yield answer delta chunks from OpenAI streaming."""
    api_key = settings.llm_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        yield "⚠️ OpenAI API key not set."
        return

    client = _make_openai_client()
    try:
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=settings.max_output_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        logger.warning(f"[Generator] OpenAI streaming failed, falling back: {e}")
        answer = await _generate_openai(prompt)
        yield answer


async def generate_answer_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Stream the answer from the LLM — yields string chunks as they arrive.

    The caller should accumulate chunks to build the full answer text.
    Gracefully falls back to whole-answer if streaming is unavailable.
    """
    provider = settings.llm_provider.lower()
    try:
        if provider == "gemini":
            async for chunk in _stream_gemini(prompt):
                yield chunk
        elif provider == "openai":
            async for chunk in _stream_openai(prompt):
                yield chunk
        else:
            # Unknown provider — fall back to non-streaming
            answer = await generate_answer(prompt)
            yield answer
    except Exception as e:
        logger.error(f"[Generator] Streaming failed entirely: {e}")
        yield f"⚠️ Answer generation failed: {e}"


def format_citations(citations: List[Citation]) -> str:
    """Return a human-readable citation list for display below the answer."""
    if not citations:
        return ""
    lines = ["\n**Sources:**"]
    for c in citations:
        lines.append(f"- {c.label} [{c.title}]({c.url}) — {c.domain}")
    return "\n".join(lines)
