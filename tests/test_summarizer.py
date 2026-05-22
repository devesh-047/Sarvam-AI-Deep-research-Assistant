"""Tests for the rolling summarizer."""
import pytest
from unittest.mock import patch, AsyncMock
from app.research.summarizer import update_rolling_summary

@pytest.mark.asyncio
async def test_update_rolling_summary_success():
    with patch("app.research.summarizer.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Summary: This conversation is about quantum physics."
        summary = await update_rolling_summary("Old summary", "What is quantum computing?", "It is computing using quantum bits.")
        
        mock_gen.assert_called_once()
        assert summary == "This conversation is about quantum physics."

@pytest.mark.asyncio
async def test_update_rolling_summary_failure_fallback():
    with patch("app.research.summarizer.generate_answer", side_effect=Exception("API Error")):
        summary = await update_rolling_summary("Old summary", "What is quantum computing?", "It is computing using quantum bits.")
        assert "Old summary" in summary
        assert "What is quantum computing?" in summary
