"""
LLM answer generator.

Calls the configured LLM provider with the prompt built by context_builder
and returns the raw answer text.

Currently supports:
  - gemini   (Google Gemini API, using google-genai SDK)
  - openai   (OpenAI API or any OpenAI-compatible endpoint, e.g. Azure, Groq, Together)

The provider is selected via settings.llm_provider.
"""
import os
from typing import List

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
    Send the assembled prompt to the LLM and return the answer text.

    Args:
        prompt: The full context + instructions string from context_builder.

    Returns:
        The answer string from the LLM.
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


def format_citations(citations: List[Citation]) -> str:
    """Return a human-readable citation list for display below the answer."""
    if not citations:
        return ""
    lines = ["\n**Sources:**"]
    for c in citations:
        lines.append(f"- {c.label} [{c.title}]({c.url}) — {c.domain}")
    return "\n".join(lines)

