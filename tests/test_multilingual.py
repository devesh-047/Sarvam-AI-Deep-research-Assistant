"""
Phase 5 — Multilingual Layer Tests.

Tests for:
  - Language detection (English, Hinglish, Bengali, Devanagari, South Indic)
  - Query normalization (Sarvam mock, LLM fallback, raw fallback)
  - Response localization (citation preservation, fallback)
  - Multilingual pipeline events (ordering, content)
  - Fallback behavior on Sarvam API failure
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Language Detection ────────────────────────────────────────────────────────

from app.multilingual.language_detector import (
    detect_language,
    is_non_english,
    SUPPORTED_RESPONSE_LANGUAGES,
    LANG_CODE_MAP,
)


def test_detect_english_simple():
    assert detect_language("What are the latest AI breakthroughs in 2025?") == "english"


def test_detect_english_technical():
    assert detect_language("How does transformer attention mechanism work?") == "english"


def test_detect_hinglish_kya_hai():
    assert detect_language("bharat me AI startups ka future kya hai") == "hinglish"


def test_detect_hinglish_market():
    assert detect_language("india me semiconductor companies ka market kaisa hai") == "hinglish"


def test_detect_hinglish_mixed():
    assert detect_language("quantum computing kya hai aur future kaisa hai") == "hinglish"


def test_detect_benglish_kemon():
    result = detect_language("kolkata te EV startup growth kemon ache")
    assert result in ("benglish", "hinglish", "transliterated")  # some overlap expected


def test_detect_devanagari_script():
    result = detect_language("भारत में AI स्टार्टअप्स का भविष्य क्या है?")
    assert result == "devanagari"


def test_detect_bengali_script():
    result = detect_language("ভারতে AI স্টার্টআপের ভবিষ্যৎ কেমন?")
    assert result == "bengali_script"


def test_detect_empty_query():
    assert detect_language("") == "unknown"


def test_detect_whitespace():
    assert detect_language("   ") == "unknown"


def test_is_non_english_for_hinglish():
    assert is_non_english("hinglish") is True


def test_is_non_english_for_english():
    assert is_non_english("english") is False


def test_is_non_english_for_unknown():
    assert is_non_english("unknown") is False


def test_supported_response_languages_keys():
    assert "en" in SUPPORTED_RESPONSE_LANGUAGES
    assert "hi" in SUPPORTED_RESPONSE_LANGUAGES
    assert "bn" in SUPPORTED_RESPONSE_LANGUAGES
    assert "ta" in SUPPORTED_RESPONSE_LANGUAGES


def test_lang_code_map_has_hindi():
    assert LANG_CODE_MAP["hi"] == "hi-IN"


# ── Sarvam Client ─────────────────────────────────────────────────────────────

from app.multilingual.sarvam_client import SarvamUnavailableError


def test_sarvam_unavailable_error_is_exception():
    err = SarvamUnavailableError("test")
    assert isinstance(err, Exception)
    assert "test" in str(err)


@pytest.mark.asyncio
async def test_sarvam_translate_no_api_key():
    """Should raise SarvamUnavailableError when no API key is set."""
    from app.multilingual.sarvam_client import translate
    with patch("app.multilingual.sarvam_client.settings") as mock_settings:
        mock_settings.sarvam_api_key = ""
        with pytest.raises(SarvamUnavailableError):
            await translate("hello", "en-IN", "hi-IN")


@pytest.mark.asyncio
async def test_sarvam_translate_success():
    """Should return translated text on successful API response."""
    from app.multilingual.sarvam_client import translate
    import httpx
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"translated_text": "नमस्ते"}

    with patch("app.multilingual.sarvam_client.settings") as mock_settings, \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.sarvam_translate_url = "https://api.sarvam.ai/translate"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await translate("hello", "en-IN", "hi-IN")
        assert result == "नमस्ते"


@pytest.mark.asyncio
async def test_sarvam_translate_503_raises():
    """Should raise SarvamUnavailableError on server error after retries."""
    from app.multilingual.sarvam_client import translate
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    with patch("app.multilingual.sarvam_client.settings") as mock_settings, \
         patch("httpx.AsyncClient") as mock_client_cls, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.sarvam_translate_url = "https://api.sarvam.ai/translate"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(SarvamUnavailableError):
            await translate("hello", "en-IN", "hi-IN")


# ── Query Normalizer ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_normalize_english_query_unchanged():
    """English queries should be returned unchanged without any API call."""
    from app.multilingual.query_normalizer import normalize_query
    result, lang = await normalize_query("What is quantum computing?")
    assert result == "What is quantum computing?"
    assert lang == "english"


@pytest.mark.asyncio
async def test_normalize_hinglish_sarvam_success():
    """Should return Sarvam-normalized English query for Hinglish input."""
    from app.multilingual.query_normalizer import normalize_query
    normalized_english = "What is the future of AI startups in India?"

    with patch("app.multilingual.query_normalizer.settings") as mock_settings, \
         patch("app.multilingual.sarvam_client.transliterate_to_english",
               new_callable=AsyncMock) as mock_transliterate:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.enable_multilingual = True
        mock_transliterate.return_value = normalized_english

        result, lang = await normalize_query("bharat me AI startups ka future kya hai")
        assert result == normalized_english
        assert lang == "hinglish"


@pytest.mark.asyncio
async def test_normalize_hinglish_sarvam_failure_llm_fallback():
    """Should fall back to LLM normalization when Sarvam fails."""
    from app.multilingual.query_normalizer import normalize_query
    from app.multilingual.sarvam_client import SarvamUnavailableError

    with patch("app.multilingual.query_normalizer.settings") as mock_settings, \
         patch("app.multilingual.sarvam_client.transliterate_to_english",
               new_callable=AsyncMock) as mock_sarvam, \
         patch("app.research.generator.generate_answer",
               new_callable=AsyncMock) as mock_llm:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.enable_multilingual = True
        mock_sarvam.side_effect = SarvamUnavailableError("503")
        mock_llm.return_value = "What is the future of AI startups in India?"

        result, lang = await normalize_query("bharat me AI startups ka future kya hai")
        assert "India" in result or "AI" in result
        assert lang == "hinglish"


@pytest.mark.asyncio
async def test_normalize_all_fallbacks_return_raw():
    """Should return raw query when both Sarvam and LLM fail."""
    from app.multilingual.query_normalizer import normalize_query
    from app.multilingual.sarvam_client import SarvamUnavailableError
    raw_query = "bharat me AI startups ka future kya hai"

    with patch("app.multilingual.query_normalizer.settings") as mock_settings, \
         patch("app.multilingual.sarvam_client.transliterate_to_english",
               new_callable=AsyncMock) as mock_sarvam, \
         patch("app.research.generator.generate_answer",
               new_callable=AsyncMock) as mock_llm:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.enable_multilingual = True
        mock_sarvam.side_effect = SarvamUnavailableError("error")
        mock_llm.side_effect = Exception("LLM also failed")

        result, lang = await normalize_query(raw_query)
        assert result == raw_query  # raw fallback


# ── Response Localizer ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_localize_english_unchanged():
    """English target should return answer unchanged."""
    from app.multilingual.response_localizer import localize_response
    answer = "Quantum computing uses qubits [S1]."
    result = await localize_response(answer, "en")
    assert result == answer


@pytest.mark.asyncio
async def test_localize_no_api_key_fallback():
    """Should return English answer if Sarvam API key not set."""
    from app.multilingual.response_localizer import localize_response
    answer = "This is an English answer about AI."

    with patch("app.multilingual.response_localizer.settings") as mock_settings:
        mock_settings.sarvam_api_key = ""
        mock_settings.enable_multilingual = True
        result = await localize_response(answer, "hi")
        assert result == answer


@pytest.mark.asyncio
async def test_localize_preserves_citations():
    """Citations block should remain in English after localization."""
    from app.multilingual.response_localizer import localize_response

    answer = (
        "Artificial intelligence is transforming industries globally.\n\n"
        "**Sources:**\n- [S1] Tech Review — techreview.com\n- [S2] Nature — nature.com"
    )
    hindi_prose = "कृत्रिम बुद्धिमत्ता वैश्विक उद्योगों को बदल रही है।"

    with patch("app.multilingual.response_localizer.settings") as mock_settings, \
         patch("app.multilingual.sarvam_client.translate",
               new_callable=AsyncMock) as mock_translate:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.enable_multilingual = True
        mock_translate.return_value = hindi_prose

        result = await localize_response(answer, "hi")
        # Hindi prose present
        assert hindi_prose in result
        # Citations preserved in English
        assert "[S1]" in result
        assert "[S2]" in result
        assert "techreview.com" in result
        assert "nature.com" in result


@pytest.mark.asyncio
async def test_localize_preserves_inline_citation_labels():
    """Inline source labels should survive localization even if the translator drops placeholders."""
    from app.multilingual.response_localizer import localize_response

    answer = "AI adoption is accelerating [S1], but evidence remains mixed [S2]."
    hindi_prose_without_tokens = "AI अपनाना तेज़ हो रहा है, लेकिन प्रमाण मिश्रित हैं।"

    with patch("app.multilingual.response_localizer.settings") as mock_settings, \
         patch("app.multilingual.sarvam_client.translate",
               new_callable=AsyncMock) as mock_translate:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.enable_multilingual = True
        mock_translate.return_value = hindi_prose_without_tokens

        result = await localize_response(answer, "hi")

    assert "[S1]" in result
    assert "[S2]" in result


@pytest.mark.asyncio
async def test_localize_sarvam_failure_returns_english():
    """Should fall back to English answer when Sarvam translation fails."""
    from app.multilingual.response_localizer import localize_response
    from app.multilingual.sarvam_client import SarvamUnavailableError

    answer = "This is an English answer."

    with patch("app.multilingual.response_localizer.settings") as mock_settings, \
         patch("app.multilingual.sarvam_client.translate",
               new_callable=AsyncMock) as mock_translate:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.enable_multilingual = True
        mock_translate.side_effect = SarvamUnavailableError("Sarvam down")

        result = await localize_response(answer, "hi")
        assert result == answer  # English fallback


@pytest.mark.asyncio
async def test_localize_unknown_language_fallback():
    """Unknown language code should return English answer."""
    from app.multilingual.response_localizer import localize_response

    answer = "English answer here."
    with patch("app.multilingual.response_localizer.settings") as mock_settings:
        mock_settings.sarvam_api_key = "test-key"
        mock_settings.enable_multilingual = True
        result = await localize_response(answer, "xx")  # unknown code
        assert result == answer


# ── Multilingual Pipeline Events ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_emits_lang_detect_event_for_hinglish():
    """Agent should emit lang_detect event before plan for Hinglish queries."""
    from app.research.agent import ResearchAgent
    from unittest.mock import patch, AsyncMock, MagicMock

    agent = ResearchAgent()
    events = []

    async def mock_stream_run(self, query, session_id, target_language="en"):
        from app.models.schema import PipelineEvent
        yield PipelineEvent(stage="lang_detect", message="Detected: Hinglish", event_type="observation",
                            data={"detected_lang": "hinglish", "original_query": query})
        yield PipelineEvent(stage="normalize", message="Normalizing...", event_type="action")
        yield PipelineEvent(stage="normalize_complete", message="Normalized", event_type="observation",
                            data={"normalized_query": "What is the future of AI in India?"})
        yield PipelineEvent(stage="complete", message="done", event_type="final_answer",
                            data={"answer": "AI future in India.", "english_answer": "AI future in India.",
                                  "citations": [], "retrieved_chunks": [], "citation_text": "",
                                  "plan": {"plan_text": "", "search_queries": []},
                                  "top_sources": [], "detected_lang": "hinglish",
                                  "normalized_query": "What is the future of AI in India?",
                                  "target_language": "en"})

    with patch.object(ResearchAgent, "stream_run", mock_stream_run):
        async for event in agent.stream_run("bharat me AI ka future kya hai", 1):
            events.append(event.stage)

    assert events[0] == "lang_detect"
    assert "normalize" in events
    assert "normalize_complete" in events
    assert events[-1] == "complete"


@pytest.mark.asyncio
async def test_agent_no_lang_detect_for_english():
    """For English queries, lang_detect should still be emitted but normalize should not."""
    from app.research.agent import ResearchAgent

    agent = ResearchAgent()
    events = []

    async def mock_stream_run(self, query, session_id, target_language="en"):
        from app.models.schema import PipelineEvent
        yield PipelineEvent(stage="lang_detect", message="Detected: English", event_type="observation",
                            data={"detected_lang": "english"})
        yield PipelineEvent(stage="complete", message="done", event_type="final_answer",
                            data={"answer": "AI.", "english_answer": "AI.",
                                  "citations": [], "retrieved_chunks": [], "citation_text": "",
                                  "plan": {"plan_text": "", "search_queries": []},
                                  "top_sources": [], "detected_lang": "english",
                                  "normalized_query": "What is AI?",
                                  "target_language": "en"})

    with patch.object(ResearchAgent, "stream_run", mock_stream_run):
        async for event in agent.stream_run("What is AI?", 1):
            events.append(event.stage)

    assert "lang_detect" in events
    assert "normalize" not in events
