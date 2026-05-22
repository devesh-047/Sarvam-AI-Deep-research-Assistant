import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.config import settings
from app.research.generator import generate_answer, format_citations
from app.models.schema import Citation


@pytest.mark.asyncio
async def test_generate_answer_gemini():
    # Setup settings for Gemini
    settings.llm_provider = "gemini"
    settings.gemini_api_key = "test-gemini-key"
    settings.llm_model = "gemini-2.5-flash"

    # Mock google-genai Client and generate_content call
    mock_response = MagicMock()
    mock_response.text = "Grounded response from Gemini."
    
    mock_aio = MagicMock()
    mock_aio.models = MagicMock()
    mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    mock_client = MagicMock()
    mock_client.aio = mock_aio

    with patch("google.genai.Client", return_value=mock_client):
        answer = await generate_answer("Please research quantum computing.")
        assert answer == "Grounded response from Gemini."
        mock_aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_openai():
    # Setup settings for OpenAI
    settings.llm_provider = "openai"
    settings.llm_api_key = "test-openai-key"
    settings.llm_model = "gpt-4o-mini"

    # Mock openai.AsyncOpenAI
    mock_choice = MagicMock()
    mock_choice.message.content = "Grounded response from OpenAI."
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()
    mock_chat.completions.create = AsyncMock(return_value=mock_response)
    
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("app.research.generator._make_openai_client", return_value=mock_client):
        answer = await generate_answer("Please research fusion energy.")
        assert answer == "Grounded response from OpenAI."
        mock_chat.completions.create.assert_called_once()


def test_format_citations():
    citations = [
        Citation(label="[S1]", title="Quantum breakthrough", url="http://quantum.com", domain="quantum.com")
    ]
    formatted = format_citations(citations)
    assert "[S1]" in formatted
    assert "Quantum breakthrough" in formatted
    assert "http://quantum.com" in formatted
