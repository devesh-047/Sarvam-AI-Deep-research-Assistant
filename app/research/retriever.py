"""
Dense retriever: embed the query → search FAISS → restore metadata from SQLite.

Returns a ranked list of RetrievedChunk objects ready for the context builder.
"""
from typing import List

from app.research.embeddings import embed_query
from app.research.vector_store import FAISSVectorStore
from app.memory.repositories import ChunkRepository
from app.models.schema import RetrievedChunk
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    def __init__(
        self,
        vector_store: FAISSVectorStore = None,
        chunk_repo: ChunkRepository = None,
        top_k: int = None,
    ):
        self.vector_store = vector_store or FAISSVectorStore()
        self.chunk_repo = chunk_repo or ChunkRepository()
        self.top_k = top_k or settings.retrieval_top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Embed the query, search FAISS, and return annotated chunks.

        1. Embed query.
        2. Search FAISS for top-k nearest chunk ids + scores.
        3. Fetch chunk text + metadata from SQLite.
        4. Return RetrievedChunk list sorted by descending score.
        """
        if self.vector_store.total_vectors == 0:
            logger.warning("Vector store is empty — no retrieval possible.")
            return []

        # 1. Embed query
        query_vec = embed_query(query)

        # 2. FAISS search
        raw_results = self.vector_store.search(query_vec, top_k=self.top_k)
        if not raw_results:
            return []

        # 3. Restore metadata from SQLite
        chunk_id_to_score = {chunk_id: score for chunk_id, score in raw_results}
        chunk_ids = list(chunk_id_to_score.keys())
        db_chunks = self.chunk_repo.get_by_ids(chunk_ids)

        # 4. Build RetrievedChunk list
        retrieved: List[RetrievedChunk] = []
        for chunk in db_chunks:
            retrieved.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source_url=chunk.source_url,
                title=chunk.title,
                domain=chunk.domain,
                text=chunk.text,
                score=chunk_id_to_score.get(chunk.chunk_id, 0.0),
            ))

        # Sort descending by score
        retrieved.sort(key=lambda c: c.score, reverse=True)
        logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:60]}…'")
        return retrieved
