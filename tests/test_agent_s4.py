"""Stage 4 agent orchestrator tests."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.research.agent import ResearchAgent
from app.models.schema import PipelineEvent, ResearchPlan


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


@pytest.mark.asyncio
async def test_plan_event_is_emitted(agent):
    """Plan and plan_complete events must be emitted before search."""
    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.agent.AsyncFetcher") as mock_fetcher, \
         patch("app.research.agent.ContentExtractor") as mock_extractor, \
         patch("app.research.agent.chunk_document") as mock_chunk, \
         patch("app.research.agent.embed_texts") as mock_embed, \
         patch("app.research.agent.build_context") as mock_context, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock):

        mock_tavily.return_value.search = AsyncMock(return_value=[
            MagicMock(url="http://example.com", title="Example")
        ])
        mock_fetcher.return_value.fetch_all = AsyncMock(return_value=[
            MagicMock(url="http://example.com", status_code=200, content_type="text/html", html="<p>hello</p>", error=None)
        ])
        mock_extractor.return_value.extract.return_value = MagicMock(
            url="http://example.com", title="Example", domain="example.com",
            text="Some text about things.", word_count=5, extraction_successful=True, error=None
        )
        mock_chunk.return_value = []
        mock_embed.return_value = __import__("numpy").array([]).reshape(0, 384)
        mock_context.return_value = ("prompt", [])

        events = []
        async for event in agent.stream_run("What is quantum computing?", session_id=1):
            events.append(event)

        stages = [e.stage for e in events]
        assert "plan" in stages, "plan event must be emitted"
        assert "plan_complete" in stages, "plan_complete event must be emitted"

        # Plan must come before search
        plan_idx = stages.index("plan")
        search_idx = stages.index("search") if "search" in stages else len(stages)
        assert plan_idx < search_idx, "plan event must precede search event"


@pytest.mark.asyncio
async def test_select_evidence_event_is_emitted(agent):
    """select_evidence event must appear between retrieve and context."""
    import numpy as np
    from app.models.schema import DocumentChunk, RetrievedChunk

    mock_chunk = MagicMock(spec=DocumentChunk)
    mock_chunk.chunk_id = 1
    mock_chunk.text = "Some content."

    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.agent.AsyncFetcher") as mock_fetcher, \
         patch("app.research.agent.ContentExtractor") as mock_extractor, \
         patch("app.research.agent.chunk_document") as mock_chunk_fn, \
         patch("app.research.agent.embed_texts") as mock_embed, \
         patch("app.research.agent.build_context") as mock_context, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock):

        mock_tavily.return_value.search = AsyncMock(return_value=[
            MagicMock(url="http://example.com", title="Example")
        ])
        mock_fetcher.return_value.fetch_all = AsyncMock(return_value=[
            MagicMock(url="http://example.com", status_code=200, content_type="text/html",
                      html="<p>text</p>", error=None)
        ])
        mock_extractor.return_value.extract.return_value = MagicMock(
            url="http://example.com", title="Example", domain="example.com",
            text="Content here.", word_count=2, extraction_successful=True, error=None
        )
        # Return a real-ish chunk so pipeline doesn't exit early
        mock_chunk_fn.return_value = [mock_chunk]
        agent.chunk_repo.bulk_insert.return_value = [mock_chunk]
        agent.vector_store.add.return_value = [0]
        mock_embed.return_value = np.zeros((1, 384), dtype="float32")

        retrieved_chunk = RetrievedChunk(
            chunk_id=1,
            source_url="http://example.com",
            title="Example",
            domain="example.com",
            text="Content here.",
            score=0.9,
            citation_label="[S1]",
        )
        agent.retriever.retrieve.return_value = [retrieved_chunk]

        mock_context.return_value = ("prompt", [])
        agent.turn_repo.update_turn_results = MagicMock()

        events = []
        async for event in agent.stream_run("test query", session_id=1):
            events.append(event)

        stages = [e.stage for e in events]
        assert "select_evidence" in stages, f"select_evidence event must be emitted. Got: {stages}"

        if "retrieve" in stages and "context" in stages:
            retrieve_idx = stages.index("retrieve")
            select_idx = stages.index("select_evidence")
            context_idx = stages.index("context")
            assert retrieve_idx < select_idx < context_idx


@pytest.mark.asyncio
async def test_graceful_search_failure(agent):
    """If Tavily raises, agent should yield an error event without crashing."""
    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock):

        mock_tavily.return_value.search = AsyncMock(side_effect=Exception("API timeout"))

        events = []
        async for event in agent.stream_run("test query", session_id=1):
            events.append(event)

        stages = [e.stage for e in events]
        assert "error" in stages, "An error event must be emitted on search failure"
        # Pipeline should not raise — it should complete the generator
        error_events = [e for e in events if e.stage == "error"]
        assert len(error_events) >= 1


@pytest.mark.asyncio
async def test_plan_complete_data_has_queries(agent):
    """plan_complete event must carry search_queries in its data."""
    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.agent.AsyncFetcher") as mock_fetcher, \
         patch("app.research.agent.ContentExtractor") as mock_extractor, \
         patch("app.research.agent.chunk_document") as mock_chunk, \
         patch("app.research.agent.embed_texts") as mock_embed, \
         patch("app.research.agent.build_context") as mock_context, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock):

        mock_tavily.return_value.search = AsyncMock(return_value=[
            MagicMock(url="http://example.com", title="Example")
        ])
        mock_fetcher.return_value.fetch_all = AsyncMock(return_value=[
            MagicMock(url="http://example.com", status_code=200, content_type="text/html", html="<p>x</p>", error=None)
        ])
        mock_extractor.return_value.extract.return_value = MagicMock(
            url="http://example.com", title="Example", domain="example.com",
            text="Content.", word_count=1, extraction_successful=True, error=None
        )
        mock_chunk.return_value = []
        mock_embed.return_value = __import__("numpy").array([]).reshape(0, 384)
        mock_context.return_value = ("prompt", [])

        plan_complete_event = None
        async for event in agent.stream_run("What is FAISS?", session_id=1):
            if event.stage == "plan_complete":
                plan_complete_event = event
                break

        assert plan_complete_event is not None
        assert "search_queries" in (plan_complete_event.data or {})
        assert isinstance(plan_complete_event.data["search_queries"], list)
        assert len(plan_complete_event.data["search_queries"]) >= 1
