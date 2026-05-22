"""
ResearchAgent: explicit manual orchestrator for the Stage 2 pipeline.

Flow:
  1.  Search Tavily
  2.  Fetch pages concurrently
  3.  Extract readable text
  4.  Chunk documents
  5.  Embed chunks
  6.  Add vectors to FAISS
  7.  Persist chunks (with embedding_ids) to SQLite
  8.  Retrieve top-k relevant chunks
  9.  Build grounded context
  10. Generate answer
  11. Persist turn artefacts (turn, sources, chunks already done above)
  12. Return answer + citations

No frameworks. No recursion. Pure explicit Python.
"""
import json
from dataclasses import dataclass, field
from typing import List

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.db import init_db
from app.memory.repositories import (
    SessionRepository, ResearchTurnRepository,
    SourceRepository, ChunkRepository
)
from app.models.schema import Citation, RetrievedChunk, PipelineEvent
from app.research.search import TavilyClient
from app.research.fetcher import AsyncFetcher
from app.research.extractor import ContentExtractor
from app.research.chunker import chunk_document
from app.research.embeddings import embed_texts
from app.research.vector_store import FAISSVectorStore
from app.research.retriever import DenseRetriever
from app.research.context_builder import build_context
from app.research.generator import generate_answer, format_citations

logger = get_logger(__name__)


@dataclass
class ResearchResult:
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    citation_text: str = ""
    error: str = ""


class ResearchAgent:
    def __init__(self):
        init_db()
        self.session_repo = SessionRepository()
        self.turn_repo = ResearchTurnRepository()
        self.source_repo = SourceRepository()
        self.chunk_repo = ChunkRepository()
        self.vector_store = FAISSVectorStore()
        self.retriever = DenseRetriever(
            vector_store=self.vector_store,
            chunk_repo=self.chunk_repo,
            top_k=settings.retrieval_top_k,
        )

    async def stream_run(self, query: str, session_id: int):
        """
        Run the research pipeline yielding PipelineEvent objects at each stage.
        Injects rolling summary and recent memory context, updates them on completion.
        """
        yield PipelineEvent(stage="start", message="Initializing deep research pipeline...", data={"query": query})
        
        # ── Step 0: Fetch memory & summary ────────────────────────────────
        from app.memory.conversation_memory import format_memory_block
        from app.research.summarizer import update_rolling_summary
        
        rolling_summary = self.session_repo.get_summary(session_id) or ""
        recent_turns = self.turn_repo.get_recent_turns(session_id, limit=settings.max_recent_turns)
        memory_block = format_memory_block(recent_turns, max_tokens=settings.max_summary_tokens)
        
        yield PipelineEvent(
            stage="memory", 
            message="Retrieved conversation context and rolling summary.", 
            data={"rolling_summary": rolling_summary, "recent_turns_count": len(recent_turns)}
        )

        # ── Step 1: Tavily search ─────────────────────────────────────────
        yield PipelineEvent(stage="search", message="Searching the web via Tavily...", data={"query": query})
        search_client = TavilyClient()
        try:
            search_results = await search_client.search(query, max_results=5)
        except Exception as e:
            yield PipelineEvent(stage="error", message=f"Search failed: {e}")
            return

        if not search_results:
            yield PipelineEvent(stage="error", message="No search results returned.")
            return

        urls = [r.url for r in search_results]
        yield PipelineEvent(stage="search_complete", message=f"Found {len(search_results)} search results.", data={"urls": urls})

        # ── Step 2: Fetch pages ───────────────────────────────────────────
        yield PipelineEvent(stage="fetch", message=f"Fetching {len(urls)} webpages concurrently...")
        fetcher = AsyncFetcher()
        fetched_pages = await fetcher.fetch_all(urls)
        
        # ── Step 3: Extract readable text ─────────────────────────────────
        yield PipelineEvent(stage="extract", message="Extracting main text from fetched webpages...")
        extractor = ContentExtractor()
        extracted_docs = [
            extractor.extract(page, default_title=result.title)
            for page, result in zip(fetched_pages, search_results)
        ]
        successful_docs = [d for d in extracted_docs if d.extraction_successful]
        yield PipelineEvent(
            stage="extract_complete", 
            message=f"Successfully extracted content from {len(successful_docs)}/{len(extracted_docs)} pages.",
            data={"successful_count": len(successful_docs)}
        )

        # ── Step 4: Persist turn and sources ─────────────────────────────
        yield PipelineEvent(stage="persist_turn", message="Saving research turn and sources to database...")
        turn = self.turn_repo.create(
            session_id=session_id,
            user_query=query,
            search_queries_json=json.dumps([query]),
            opened_urls_json=json.dumps(urls),
        )

        source_map: dict = {}
        for doc in extracted_docs:
            status = "success" if doc.extraction_successful else "failed"
            source_record = self.source_repo.create(
                session_id=session_id,
                turn_id=turn.id,
                url=doc.url,
                title=doc.title,
                domain=doc.domain,
                extraction_status=status,
            )
            source_map[doc.url] = source_record

        if not successful_docs:
            yield PipelineEvent(stage="error", message="Could not extract content from any retrieved page.")
            return

        # ── Step 5: Chunk documents ───────────────────────────────────────
        yield PipelineEvent(stage="chunk", message="Chunking extracted documents for retrieval...")
        all_chunks = []
        for doc in successful_docs:
            source_record = source_map[doc.url]
            chunks = chunk_document(
                doc=doc,
                source_id=source_record.id,
                session_id=session_id,
                turn_id=turn.id,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            yield PipelineEvent(stage="error", message="No document chunks produced.")
            return

        yield PipelineEvent(stage="chunk_complete", message=f"Generated {len(all_chunks)} chunks.")

        # ── Step 6: Embed chunks ──────────────────────────────────────────
        yield PipelineEvent(stage="embed", message="Generating embeddings for chunks...")
        texts = [c.text for c in all_chunks]
        embeddings = embed_texts(texts)

        # ── Step 7: Add to FAISS & persist chunks ─────────────────────────
        yield PipelineEvent(stage="faiss", message="Adding vectors to FAISS index...")
        persisted_chunks = self.chunk_repo.bulk_insert(all_chunks)
        chunk_ids = [c.chunk_id for c in persisted_chunks]
        embedding_ids = self.vector_store.add(embeddings, chunk_ids)

        for chunk, emb_id in zip(persisted_chunks, embedding_ids):
            self.chunk_repo.update_embedding_id(chunk.chunk_id, emb_id)
            chunk.embedding_id = emb_id

        self.vector_store.save()

        # ── Step 8: Retrieve relevant chunks ─────────────────────────────
        yield PipelineEvent(stage="retrieve", message="Retrieving relevant chunks from vector store...")
        retrieved = self.retriever.retrieve(query)

        # ── Step 9: Build context ─────────────────────────────────────────
        yield PipelineEvent(stage="context", message="Building prompt context with memory & evidence...")
        prompt, citations = build_context(
            query=query,
            retrieved_chunks=retrieved,
            rolling_summary=rolling_summary,
            memory_block=memory_block
        )

        # ── Step 10: Generate answer ──────────────────────────────────────
        yield PipelineEvent(stage="llm", message="Generating grounded answer from Google Gemini...")
        answer = await generate_answer(prompt)
        
        # Save final answer, citations, and retrieved chunks to SQLite
        citations_json = json.dumps([c.model_dump() for c in citations])
        retrieved_chunks_json = json.dumps([c.model_dump() for c in retrieved])
        self.turn_repo.update_turn_results(
            turn_id=turn.id,
            final_answer=answer,
            citations_json=citations_json,
            retrieved_chunks_json=retrieved_chunks_json
        )

        # ── Step 11: Update rolling summary ───────────────────────────────
        yield PipelineEvent(stage="summary", message="Updating rolling conversation summary...")
        try:
            new_summary = await update_rolling_summary(rolling_summary, query, answer)
            self.session_repo.update_summary(session_id, new_summary)
            logger.info(f"[Agent] Rolling summary updated for session {session_id}")
        except Exception as e:
            logger.error(f"[Agent] Failed to update rolling summary: {e}")

        citation_text = format_citations(citations)
        
        yield PipelineEvent(
            stage="complete", 
            message="Deep research completed successfully.",
            data={
                "answer": answer,
                "citations": [c.model_dump() for c in citations],
                "retrieved_chunks": [c.model_dump() for c in retrieved],
                "citation_text": citation_text
            }
        )

    async def run(self, query: str, session_id: int) -> ResearchResult:
        """Run the full research pipeline synchronously for a user query."""
        logger.info(f"[Agent] Starting synchronous research for query: '{query}'")
        
        result = None
        async for event in self.stream_run(query, session_id):
            if event.stage == "complete":
                data = event.data
                citations = [Citation(**c) for c in data["citations"]]
                retrieved_chunks = [RetrievedChunk(**c) for c in data["retrieved_chunks"]]
                result = ResearchResult(
                    answer=data["answer"],
                    citations=citations,
                    retrieved_chunks=retrieved_chunks,
                    citation_text=data["citation_text"]
                )
            elif event.stage == "error":
                return ResearchResult(
                    answer=event.message,
                    citations=[],
                    retrieved_chunks=[],
                    error="failed"
                )
        return result or ResearchResult(answer="Error running pipeline.", citations=[], retrieved_chunks=[], error="error")

