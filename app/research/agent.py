"""
ResearchAgent — Stage 4 (ReAct UX enhancement).

Explicit deterministic flow with ReAct-style visible workflow events:

  Thought → Action → Observation → Thought → Action → Observation → Final Answer

Pipeline steps:
  0.  Load conversation memory
  1.  Plan  (Thought: what evidence is needed)
  2.  Search (Action: web search)
  3.  Fetch + Extract (Action: acquire content)
  4.  Persist (system)
  5.  Chunk + Embed + Index (system)
  6.  Retrieve (Action: evidence retrieval)
  7.  Select Evidence (Observation: what was found)
  8.  Conflict Detection (Observation: source disagreements)
  9.  Build Context (system)
  10. Generate Answer — streamed token-by-token
  11. Persist results (system)
  12. Update rolling summary (system)

No frameworks. No recursion. Pure deterministic Python.
ReAct-style events are pre-scripted safe summaries — never raw chain-of-thought.
"""
import asyncio
import json
from dataclasses import dataclass
from typing import List, AsyncGenerator

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.db import init_db
from app.memory.repositories import (
    SessionRepository, ResearchTurnRepository,
    SourceRepository, ChunkRepository
)
from app.models.schema import Citation, RetrievedChunk, PipelineEvent, ResearchPlan
from app.research.planner import generate_plan
from app.research.search import TavilyClient
from app.research.fetcher import AsyncFetcher
from app.research.extractor import ContentExtractor
from app.research.chunker import chunk_document
from app.research.embeddings import embed_texts
from app.research.vector_store import FAISSVectorStore
from app.research.retriever import DenseRetriever
from app.research.context_builder import build_context
from app.research.generator import generate_answer_stream, format_citations
from app.multilingual.language_detector import detect_language, is_non_english, SUPPORTED_RESPONSE_LANGUAGES
from app.multilingual.query_normalizer import normalize_query
from app.multilingual.response_localizer import localize_response

logger = get_logger(__name__)


@dataclass
class ResearchResult:
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    citation_text: str = ""
    plan: ResearchPlan = None
    error: str = ""


def _evt(stage: str, message: str, event_type: str = "system", data: dict = None) -> PipelineEvent:
    """Convenience constructor for PipelineEvent."""
    return PipelineEvent(stage=stage, message=message, event_type=event_type, data=data or {})


class ResearchAgent:
    def __init__(self):
        init_db()
        self.session_repo = SessionRepository()
        self.turn_repo = ResearchTurnRepository()
        self.source_repo = SourceRepository()
        self.chunk_repo = ChunkRepository()

        if not settings.gemini_api_key and settings.llm_provider == "gemini":
            logger.warning("[Agent] GEMINI_API_KEY is not set. Answer generation will fail.")
        if not settings.tavily_api_key:
            logger.warning("[Agent] TAVILY_API_KEY is not set. Web search will fail.")

        try:
            self.vector_store = FAISSVectorStore()
        except Exception as e:
            logger.warning(f"[Agent] FAISS load failed ({e}). Starting with empty index.")
            self.vector_store = FAISSVectorStore.__new__(FAISSVectorStore)
            self.vector_store._init_empty()

        self.retriever = DenseRetriever(
            vector_store=self.vector_store,
            chunk_repo=self.chunk_repo,
            top_k=settings.retrieval_top_k,
        )

    async def _search_with_retry(self, client, query, max_results=5, retries=2):
        last_err = None
        for attempt in range(retries + 1):
            try:
                return await client.search(query, max_results=max_results)
            except Exception as e:
                last_err = e
                if attempt < retries:
                    logger.warning(f"[Agent] Search attempt {attempt + 1} failed: {e}. Retrying...")
                    await asyncio.sleep(1.0)
        raise last_err

    def _get_seen_urls(self, session_id: int) -> set:
        turns = self.turn_repo.get_all_for_session(session_id)
        seen = set()
        for t in turns:
            try:
                urls = json.loads(t.opened_urls_json or "[]")
                seen.update(urls)
            except Exception:
                pass
        return seen

    def _detect_conflicts(self, retrieved: List[RetrievedChunk]) -> List[str]:
        """
        Lightweight conflict detection: checks if retrieved chunks come from
        sufficiently diverse domains. Returns a list of human-readable
        conflict observation strings (never hallucinated — only structural signals).
        """
        conflicts = []
        domains = [c.domain for c in retrieved]
        unique_domains = set(domains)

        # Structural signal: multiple domains — potential for varied perspectives
        if len(unique_domains) >= 3:
            conflicts.append(
                f"Evidence was gathered from {len(unique_domains)} different sources "
                f"({', '.join(list(unique_domains)[:3])}{'...' if len(unique_domains) > 3 else ''}). "
                "Multiple perspectives may be present — the answer will synthesise all viewpoints."
            )

        # Check for known conflicting domain types (e.g., .gov vs blog vs wiki)
        gov_sources = [d for d in unique_domains if ".gov" in d or ".edu" in d]
        other_sources = [d for d in unique_domains if d not in gov_sources]
        if gov_sources and other_sources:
            conflicts.append(
                f"Official sources ({', '.join(gov_sources[:2])}) and secondary sources "
                f"({', '.join(other_sources[:2])}) were both retrieved. "
                "Conflicting claims will be noted in the answer."
            )

        return conflicts

    async def stream_run(
        self,
        query: str,
        session_id: int,
        target_language: str = "en",
    ) -> AsyncGenerator[PipelineEvent, None]:
        """
        Run the full research pipeline, yielding ReAct-style PipelineEvents.

        Args:
            query:           Raw user query (any language / Romanized Indic)
            session_id:      Active session ID
            target_language: ISO 639-1 code for response localization (default "en")
        """
        yield _evt("start", "Initializing deep research pipeline...", "system", {"query": query})

        # ── Phase 5: Language detection + query normalization ──────────────────
        detected_lang = "english"
        normalized_query = query

        if settings.enable_multilingual:
            try:
                detected_lang_raw = detect_language(query)
                yield _evt(
                    "lang_detect",
                    f"Detected input language: {detected_lang_raw.replace('_', ' ').title()}",
                    "observation",
                    {"detected_lang": detected_lang_raw, "original_query": query}
                )

                if is_non_english(detected_lang_raw):
                    yield _evt("normalize", "Normalizing multilingual query into English research form...", "action")
                    normalized_query, detected_lang = await normalize_query(query)
                    yield _evt(
                        "normalize_complete",
                        f"Normalized research query: \"{normalized_query}\"",
                        "observation",
                        {"normalized_query": normalized_query, "detected_lang": detected_lang_raw}
                    )
                else:
                    detected_lang = detected_lang_raw
            except Exception as e:
                logger.warning(f"[Agent] Multilingual normalization failed: {e}. Using raw query.")
                normalized_query = query

        # Use normalized query for the rest of the pipeline
        query = normalized_query

        # ── Step 0: Load conversation memory ──────────────────────────────────
        from app.memory.conversation_memory import format_memory_block
        from app.research.summarizer import update_rolling_summary

        rolling_summary = self.session_repo.get_summary(session_id) or ""
        recent_turns = self.turn_repo.get_recent_turns(session_id, limit=settings.max_recent_turns)
        memory_block = format_memory_block(recent_turns, max_tokens=settings.max_summary_tokens)

        yield _evt(
            "memory",
            f"Loaded {len(recent_turns)} recent turn(s) and conversation summary.",
            "system",
            {"rolling_summary": rolling_summary, "recent_turns_count": len(recent_turns)}
        )

        # ── Step 1: Plan ──────────────────────────────────────────────────────
        yield _evt("plan", "Analysing question and planning research strategy...", "thought")
        plan = await generate_plan(query, memory_block=memory_block)
        logger.info(f"[Agent] Plan generated. Queries: {plan.search_queries}")

        yield _evt(
            "plan_complete",
            "Research plan ready. Proceeding to web search.",
            "thought",
            {"plan_text": plan.plan_text, "search_queries": plan.search_queries}
        )

        # ── Step 2: Search ────────────────────────────────────────────────────
        primary_query = plan.search_queries[0] if plan.search_queries else query
        yield _evt(
            "search",
            f"Searching the web for: \"{primary_query}\"",
            "action",
            {"query": primary_query, "all_queries": plan.search_queries}
        )

        search_client = TavilyClient()
        try:
            search_results = await self._search_with_retry(search_client, primary_query, max_results=5)
        except Exception as e:
            yield _evt("error", f"Web search failed: {e}", "system")
            return

        if not search_results:
            yield _evt("error", "No search results returned.", "system")
            return

        seen_urls = self._get_seen_urls(session_id)
        all_urls = [r.url for r in search_results]
        new_results = [r for r in search_results if r.url not in seen_urls]
        if not new_results:
            new_results = search_results
        else:
            search_results = new_results

        urls = [r.url for r in search_results]
        yield _evt(
            "search_complete",
            f"Found {len(search_results)} source(s) to investigate.",
            "observation",
            {"urls": urls, "all_candidate_urls": all_urls}
        )

        # ── Step 3: Fetch ─────────────────────────────────────────────────────
        yield _evt("fetch", f"Fetching content from {len(urls)} webpage(s)...", "action")
        fetcher = AsyncFetcher()
        fetched_pages = await fetcher.fetch_all(urls)

        # ── Step 4: Extract ───────────────────────────────────────────────────
        yield _evt("extract", "Extracting readable text from fetched pages...", "action")
        extractor = ContentExtractor()
        extracted_docs = [
            extractor.extract(page, default_title=result.title)
            for page, result in zip(fetched_pages, search_results)
        ]
        successful_docs = [d for d in extracted_docs if d.extraction_successful]
        failed_count = len(extracted_docs) - len(successful_docs)
        if failed_count:
            logger.warning(f"[Agent] {failed_count} page(s) failed extraction.")

        yield _evt(
            "extract_complete",
            f"Successfully extracted content from {len(successful_docs)} of {len(extracted_docs)} page(s).",
            "observation",
            {"successful_count": len(successful_docs), "failed_count": failed_count}
        )

        # ── Step 5: Persist turn + sources ────────────────────────────────────
        yield _evt("persist_turn", "Saving research turn to database...", "system")
        turn = self.turn_repo.create(
            session_id=session_id,
            user_query=query,
            search_queries_json=json.dumps(plan.search_queries),
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
            err_msg = "Could not extract content from any page."
            self.turn_repo.update_turn_results(turn.id, final_answer=f"⚠️ Research failed: {err_msg}", citations_json="[]", retrieved_chunks_json="[]")
            yield _evt("error", err_msg, "system")
            return

        # ── Step 6: Chunk ─────────────────────────────────────────────────────
        yield _evt("chunk", "Chunking documents into overlapping passages...", "action")
        all_chunks = []
        for doc in successful_docs:
            source_record = source_map.get(doc.url)
            if not source_record:
                continue
            chunks = chunk_document(
                doc=doc,
                source_id=source_record.id,
                session_id=session_id,
                turn_id=turn.id,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            err_msg = "No document chunks produced."
            self.turn_repo.update_turn_results(turn.id, final_answer=f"⚠️ Research failed: {err_msg}", citations_json="[]", retrieved_chunks_json="[]")
            yield _evt("error", err_msg, "system")
            return

        yield _evt("chunk_complete", f"Produced {len(all_chunks)} passage(s) across {len(successful_docs)} document(s).", "system")

        # ── Step 7: Embed ─────────────────────────────────────────────────────
        yield _evt("embed", f"Generating semantic embeddings for {len(all_chunks)} passage(s)...", "action")
        texts = [c.text for c in all_chunks]
        embeddings = embed_texts(texts)

        # ── Step 8: Index in FAISS ────────────────────────────────────────────
        yield _evt("faiss", "Indexing passage vectors in FAISS for similarity retrieval...", "system")
        persisted_chunks = self.chunk_repo.bulk_insert(all_chunks)
        chunk_ids = [c.chunk_id for c in persisted_chunks]
        embedding_ids = self.vector_store.add(embeddings, chunk_ids)

        for chunk, emb_id in zip(persisted_chunks, embedding_ids):
            self.chunk_repo.update_embedding_id(chunk.chunk_id, emb_id)
            chunk.embedding_id = emb_id

        try:
            self.vector_store.save()
        except Exception as e:
            logger.warning(f"[Agent] FAISS save failed: {e}. Continuing without persistence.")

        # ── Step 9: Retrieve ──────────────────────────────────────────────────
        yield _evt(
            "retrieve",
            "Retrieving the most relevant evidence passages via hybrid dense and lexical search...",
            "action"
        )
        retrieved = self.retriever.retrieve(query)

        # ── Step 10: Select Evidence ──────────────────────────────────────────
        top_sources = []
        seen_domains = set()
        for chunk in retrieved:
            if chunk.domain not in seen_domains:
                top_sources.append({"title": chunk.title, "domain": chunk.domain, "url": chunk.source_url})
                seen_domains.add(chunk.domain)

        yield _evt(
            "select_evidence",
            f"Selected {len(retrieved)} evidence passage(s) from {len(top_sources)} unique source(s).",
            "observation",
            {
                "top_sources": top_sources,
                "chunk_count": len(retrieved),
                "chunks": [
                    {
                        "title": c.title,
                        "domain": c.domain,
                        "source_url": c.source_url,
                        "score": round(c.score, 3),
                        "retrieval_method": c.retrieval_method,
                        "dense_score": round(c.dense_score, 3) if c.dense_score is not None else None,
                        "lexical_score": round(c.lexical_score, 3) if c.lexical_score is not None else None,
                        "preview": c.text[:200] + ("…" if len(c.text) > 200 else ""),
                    }
                    for c in retrieved
                ],
            }
        )

        # ── Step 10b: Conflict detection ──────────────────────────────────────
        conflicts = self._detect_conflicts(retrieved)
        for conflict_msg in conflicts:
            yield _evt("conflict", conflict_msg, "observation", {"conflict_text": conflict_msg})

        # ── Step 11: Build context ────────────────────────────────────────────
        yield _evt("context", "Assembling grounded prompt with memory and retrieved evidence...", "system")
        prompt, citations = build_context(
            query=query,
            retrieved_chunks=retrieved,
            rolling_summary=rolling_summary,
            memory_block=memory_block,
        )

        # ── Step 12: Generate answer — streamed ───────────────────────────────
        yield _evt("llm", "Generating grounded answer with citations...", "action")

        answer_chunks = []
        async for token in generate_answer_stream(prompt):
            answer_chunks.append(token)
            yield _evt(
                "token",
                token,
                "token",
                {"chunk": token}
            )

        answer = "".join(answer_chunks).strip()
        logger.info(f"[Agent] Answer generated ({len(answer)} chars).")

        # ── Phase 5: Optional response localization ───────────────────────────
        localized_answer = answer
        if target_language and target_language != "en" and settings.enable_multilingual:
            yield _evt(
                "localize",
                f"Localizing answer to {SUPPORTED_RESPONSE_LANGUAGES.get(target_language, target_language)}...",
                "action",
                {"target_language": target_language}
            )
            localized_answer = await localize_response(answer, target_language)
            yield _evt(
                "localize_complete",
                f"Answer localized to {SUPPORTED_RESPONSE_LANGUAGES.get(target_language, target_language)}.",
                "observation",
                {"target_language": target_language, "localized": localized_answer != answer}
            )

        # ── Step 13: Persist results ──────────────────────────────────────────
        citations_json = json.dumps([c.model_dump() for c in citations])
        retrieved_chunks_json = json.dumps([c.model_dump() for c in retrieved])
        self.turn_repo.update_turn_results(
            turn_id=turn.id,
            final_answer=localized_answer,
            citations_json=citations_json,
            retrieved_chunks_json=retrieved_chunks_json,
        )

        # ── Step 14: Update rolling summary ──────────────────────────────────
        yield _evt("summary", "Updating conversation summary...", "system")
        try:
            new_summary = await update_rolling_summary(rolling_summary, query, answer)
            self.session_repo.update_summary(session_id, new_summary)
        except Exception as e:
            logger.error(f"[Agent] Rolling summary update failed: {e}.")

        citation_text = format_citations(citations)

        yield _evt(
            "complete",
            "Research complete.",
            "final_answer",
            {
                "answer": localized_answer,
                "english_answer": answer,  # always English for evaluation harness
                "citations": [c.model_dump() for c in citations],
                "retrieved_chunks": [c.model_dump() for c in retrieved],
                "citation_text": citation_text,
                "plan": {"plan_text": plan.plan_text, "search_queries": plan.search_queries},
                "top_sources": top_sources,
                "detected_lang": detected_lang,
                "normalized_query": normalized_query,
                "target_language": target_language,
            }
        )

    async def run(
        self,
        query: str,
        session_id: int,
        target_language: str = "en",
    ) -> ResearchResult:
        """Synchronous wrapper — runs full pipeline and returns final ResearchResult."""
        logger.info(f"[Agent] Starting research: '{query}'")
        result = None
        plan = None
        answer_chunks = []

        async for event in self.stream_run(query, session_id, target_language=target_language):
            if event.stage == "plan_complete" and event.data:
                plan = ResearchPlan(
                    plan_text=event.data.get("plan_text", ""),
                    search_queries=event.data.get("search_queries", []),
                )
            elif event.stage == "token":
                answer_chunks.append(event.message)
            elif event.stage == "complete":
                data = event.data
                citations = [Citation(**c) for c in data["citations"]]
                retrieved_chunks = [RetrievedChunk(**c) for c in data["retrieved_chunks"]]
                # Evaluation harness always uses the English answer for metrics
                eval_answer = data.get("english_answer") or data["answer"]
                result = ResearchResult(
                    answer=eval_answer,
                    citations=citations,
                    retrieved_chunks=retrieved_chunks,
                    citation_text=data["citation_text"],
                    plan=plan,
                )
            elif event.stage == "error":
                return ResearchResult(
                    answer=event.message,
                    citations=[],
                    retrieved_chunks=[],
                    error="failed",
                )

        return result or ResearchResult(
            answer="Pipeline did not produce a result.",
            citations=[],
            retrieved_chunks=[],
            error="error",
        )
