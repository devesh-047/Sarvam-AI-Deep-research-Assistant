"""Tests for ReAct-style workflow event enhancement."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

from app.models.schema import PipelineEvent, RetrievedChunk
from app.research.agent import ResearchAgent, _evt


# ── PipelineEvent field tests ─────────────────────────────────────────────────

def test_pipeline_event_default_event_type():
    e = PipelineEvent(stage="test", message="hello")
    assert e.event_type == "system"


def test_pipeline_event_custom_event_type():
    e = PipelineEvent(stage="plan", message="planning", event_type="thought")
    assert e.event_type == "thought"


def test_evt_helper_system():
    e = _evt("faiss", "Indexing...")
    assert e.stage == "faiss"
    assert e.event_type == "system"


def test_evt_helper_thought():
    e = _evt("plan", "Planning...", "thought")
    assert e.event_type == "thought"


def test_evt_helper_action():
    e = _evt("search", "Searching...", "action")
    assert e.event_type == "action"


def test_evt_helper_observation():
    e = _evt("select_evidence", "Found 5 passages.", "observation")
    assert e.event_type == "observation"


def test_evt_helper_token():
    e = _evt("token", "Hello ", "token")
    assert e.event_type == "token"


# ── Conflict detection tests ──────────────────────────────────────────────────

@pytest.fixture
def agent():
    a = ResearchAgent()
    a.session_repo = MagicMock()
    a.session_repo.get_summary.return_value = ""
    a.turn_repo = MagicMock()
    a.turn_repo.get_recent_turns.return_value = []
    a.turn_repo.get_all_for_session.return_value = []
    a.turn_repo.create.return_value = MagicMock(id=1)
    a.source_repo = MagicMock()
    a.chunk_repo = MagicMock()
    a.vector_store = MagicMock()
    a.retriever = MagicMock()
    a.retriever.retrieve.return_value = []
    return a


def _make_chunk(domain: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        source_url=f"http://{domain}/page",
        title=f"Page from {domain}",
        domain=domain,
        text="Some evidence text.",
        score=0.8,
        citation_label="[S1]",
    )


def test_no_conflict_single_domain(agent):
    chunks = [_make_chunk("example.com")] * 3
    conflicts = agent._detect_conflicts(chunks)
    assert conflicts == []


def test_conflict_detected_three_domains(agent):
    chunks = [
        _make_chunk("alpha.com"),
        _make_chunk("beta.com"),
        _make_chunk("gamma.com"),
    ]
    conflicts = agent._detect_conflicts(chunks)
    assert len(conflicts) >= 1


def test_conflict_gov_vs_other(agent):
    chunks = [
        _make_chunk("cdc.gov"),
        _make_chunk("healthblog.com"),
    ]
    conflicts = agent._detect_conflicts(chunks)
    # May or may not fire 3-domain rule but should fire gov rule
    assert any("cdc.gov" in c or "Official sources" in c for c in conflicts)


# ── Event ordering tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_react_event_order(agent):
    """Verify thought→action→observation order in the pipeline."""
    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.agent.AsyncFetcher") as mock_fetcher, \
         patch("app.research.agent.ContentExtractor") as mock_extractor, \
         patch("app.research.agent.chunk_document") as mock_chunk_fn, \
         patch("app.research.agent.embed_texts") as mock_embed, \
         patch("app.research.agent.build_context") as mock_context, \
         patch("app.research.agent.generate_answer_stream") as mock_stream, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock):

        mock_tavily.return_value.search = AsyncMock(return_value=[
            MagicMock(url="http://example.com", title="Example")
        ])
        mock_fetcher.return_value.fetch_all = AsyncMock(return_value=[
            MagicMock(url="http://example.com", status_code=200,
                      content_type="text/html", html="<p>text</p>", error=None)
        ])

        mock_doc = MagicMock(
            url="http://example.com", title="Example", domain="example.com",
            text="Content here.", word_count=2, extraction_successful=True, error=None
        )
        mock_extractor.return_value.extract.return_value = mock_doc

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = 1
        mock_chunk.text = "Some content."
        mock_chunk_fn.return_value = [mock_chunk]
        agent.chunk_repo.bulk_insert.return_value = [mock_chunk]
        agent.vector_store.add.return_value = [0]
        mock_embed.return_value = np.zeros((1, 384), dtype="float32")
        agent.retriever.retrieve.return_value = []
        mock_context.return_value = ("prompt text", [])

        # Mock streaming generator
        async def fake_stream(prompt):
            yield "Hello "
            yield "world."
        mock_stream.return_value = fake_stream("prompt text")

        events = []
        async for event in agent.stream_run("What is quantum computing?", session_id=1):
            events.append(event)

        stages = [e.stage for e in events]
        etypes = [e.event_type for e in events]

        # Plan stage should be "thought"
        plan_idx = stages.index("plan")
        assert events[plan_idx].event_type == "thought"

        # Search stage should be "action"
        search_idx = stages.index("search")
        assert events[search_idx].event_type == "action"

        # search_complete should be "observation"
        sc_idx = stages.index("search_complete")
        assert events[sc_idx].event_type == "observation"

        # Plan must come before search
        assert plan_idx < search_idx < sc_idx

        # Token events should be present
        assert "token" in stages
        token_events = [e for e in events if e.stage == "token"]
        combined = "".join(e.message for e in token_events)
        assert "Hello" in combined


@pytest.mark.asyncio
async def test_token_events_reconstruct_answer(agent):
    """Token events concatenated should equal the complete event's answer."""
    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.agent.AsyncFetcher") as mock_fetcher, \
         patch("app.research.agent.ContentExtractor") as mock_extractor, \
         patch("app.research.agent.chunk_document") as mock_chunk_fn, \
         patch("app.research.agent.embed_texts") as mock_embed, \
         patch("app.research.agent.build_context") as mock_context, \
         patch("app.research.agent.generate_answer_stream") as mock_stream, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock):

        mock_tavily.return_value.search = AsyncMock(return_value=[
            MagicMock(url="http://x.com", title="X")
        ])
        mock_fetcher.return_value.fetch_all = AsyncMock(return_value=[
            MagicMock(url="http://x.com", status_code=200,
                      content_type="text/html", html="<p>x</p>", error=None)
        ])
        mock_extractor.return_value.extract.return_value = MagicMock(
            url="http://x.com", title="X", domain="x.com",
            text="Content.", word_count=1, extraction_successful=True, error=None
        )
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = 2
        mock_chunk.text = "Content."
        mock_chunk_fn.return_value = [mock_chunk]
        agent.chunk_repo.bulk_insert.return_value = [mock_chunk]
        agent.vector_store.add.return_value = [0]
        mock_embed.return_value = np.zeros((1, 384), dtype="float32")
        agent.retriever.retrieve.return_value = []
        mock_context.return_value = ("prompt", [])

        EXPECTED = "The answer is 42."

        async def fake_stream(prompt):
            for ch in EXPECTED:
                yield ch
        mock_stream.return_value = fake_stream("prompt")

        token_texts = []
        complete_answer = ""
        async for event in agent.stream_run("test", session_id=1):
            if event.stage == "token":
                token_texts.append(event.message)
            elif event.stage == "complete":
                complete_answer = (event.data or {}).get("answer", "")

        assert "".join(token_texts) == EXPECTED
        assert complete_answer == EXPECTED


@pytest.mark.asyncio
async def test_conflict_event_emitted_for_diverse_domains(agent):
    """If 3+ unique domains are retrieved, a conflict observation must be emitted."""
    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.agent.AsyncFetcher") as mock_fetcher, \
         patch("app.research.agent.ContentExtractor") as mock_extractor, \
         patch("app.research.agent.chunk_document") as mock_chunk_fn, \
         patch("app.research.agent.embed_texts") as mock_embed, \
         patch("app.research.agent.build_context") as mock_context, \
         patch("app.research.agent.generate_answer_stream") as mock_stream, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock):

        mock_tavily.return_value.search = AsyncMock(return_value=[
            MagicMock(url="http://a.com", title="A")
        ])
        mock_fetcher.return_value.fetch_all = AsyncMock(return_value=[
            MagicMock(url="http://a.com", status_code=200,
                      content_type="text/html", html="<p>x</p>", error=None)
        ])
        mock_extractor.return_value.extract.return_value = MagicMock(
            url="http://a.com", title="A", domain="a.com",
            text="Content.", word_count=1, extraction_successful=True, error=None
        )
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = 3
        mock_chunk.text = "content"
        mock_chunk_fn.return_value = [mock_chunk]
        agent.chunk_repo.bulk_insert.return_value = [mock_chunk]
        agent.vector_store.add.return_value = [0]
        mock_embed.return_value = np.zeros((1, 384), dtype="float32")

        # 3 diverse domain chunks → should trigger conflict observation
        agent.retriever.retrieve.return_value = [
            _make_chunk("alpha.com"),
            _make_chunk("beta.org"),
            _make_chunk("gamma.net"),
        ]
        mock_context.return_value = ("prompt", [])

        async def fake_stream(prompt):
            yield "Answer."
        mock_stream.return_value = fake_stream("prompt")

        events = []
        async for event in agent.stream_run("test conflict", session_id=1):
            events.append(event)

        conflict_events = [e for e in events if e.stage == "conflict"]
        assert len(conflict_events) >= 1
        assert all(e.event_type == "observation" for e in conflict_events)


# ── Streaming generator tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_answer_stream_fallback():
    """If Gemini streaming fails, fallback should yield the whole answer."""
    with patch("app.research.generator._stream_gemini") as mock_gemini_stream:
        async def failing_stream(prompt):
            raise RuntimeError("stream broken")
            yield  # make it an async generator

        mock_gemini_stream.side_effect = failing_stream

        # Should not raise
        from app.research.generator import generate_answer_stream
        from app.core.config import settings
        original = settings.llm_provider

        settings.llm_provider = "gemini"
        chunks = []
        try:
            async for chunk in generate_answer_stream("test"):
                chunks.append(chunk)
        finally:
            settings.llm_provider = original

        # Should have received something (either error message or fallback)
        assert len(chunks) >= 1
