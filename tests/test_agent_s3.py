"""Tests for the Stage 3 agent: stream_run() event sequence."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.research.agent import ResearchAgent
from app.models.schema import PipelineEvent

@pytest.mark.asyncio
async def test_agent_stream_run_event_sequence():
    agent = ResearchAgent()
    
    # Mock repositories and pipeline components
    agent.session_repo = MagicMock()
    agent.session_repo.get_summary.return_value = "Mock summary"
    
    agent.turn_repo = MagicMock()
    agent.turn_repo.get_recent_turns.return_value = []
    agent.turn_repo.create.return_value = MagicMock(id=1)
    
    agent.source_repo = MagicMock()
    agent.chunk_repo = MagicMock()
    agent.vector_store = MagicMock()
    agent.retriever = MagicMock()
    agent.retriever.retrieve.return_value = []
    
    with patch("app.research.agent.TavilyClient") as mock_tavily, \
         patch("app.research.agent.AsyncFetcher") as mock_fetcher, \
         patch("app.research.agent.ContentExtractor") as mock_extractor, \
         patch("app.research.agent.chunk_document") as mock_chunk, \
         patch("app.research.agent.embed_texts") as mock_embed, \
         patch("app.research.agent.build_context") as mock_context, \
         patch("app.research.summarizer.update_rolling_summary", new_callable=AsyncMock) as mock_summary:
         
        # Mocking returns
        mock_tavily_inst = MagicMock()
        mock_tavily_inst.search = AsyncMock(return_value=[MagicMock(url="http://example.com", title="Title")])
        mock_tavily.return_value = mock_tavily_inst
        
        mock_fetcher_inst = MagicMock()
        mock_fetcher_inst.fetch_all = AsyncMock(return_value=[MagicMock(status_code=200)])
        mock_fetcher.return_value = mock_fetcher_inst
        
        mock_extractor_inst = MagicMock()
        mock_extractor_inst.extract.return_value = MagicMock(url="http://example.com", title="Title", extraction_successful=True, domain="example.com")
        mock_extractor.return_value = mock_extractor_inst
        
        mock_chunk.return_value = [MagicMock(text="chunk", source_id=1, session_id=1, turn_id=1, chunk_index=0, token_count=1, embedding_id=None)]
        mock_embed.return_value = [[0.1] * 384]
        
        agent.chunk_repo.bulk_insert.return_value = [MagicMock(chunk_id=1, text="chunk", source_id=1, session_id=1, turn_id=1, chunk_index=0, token_count=1, embedding_id=None)]
        agent.vector_store.add.return_value = [0]
        
        mock_context.return_value = ("prompt", [])
        mock_summary.return_value = "New rolling summary"
        
        # Collect events
        events = []
        async for event in agent.stream_run("Query", session_id=1):
            events.append(event)
            
        stages = [e.stage for e in events]
        assert "start" in stages
        assert "memory" in stages
        assert "search" in stages
        assert "search_complete" in stages
        assert "fetch" in stages
        assert "extract" in stages
        assert "extract_complete" in stages
        assert "persist_turn" in stages
        assert "chunk" in stages
        assert "chunk_complete" in stages
        assert "embed" in stages
        assert "faiss" in stages
        assert "retrieve" in stages
        assert "context" in stages
        assert "llm" in stages
        assert "summary" in stages
        assert "complete" in stages
