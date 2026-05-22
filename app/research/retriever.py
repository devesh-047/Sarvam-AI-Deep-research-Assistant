"""
Hybrid retriever: FAISS cosine search + lightweight BM25 lexical search.

Dense retrieval handles semantic similarity; BM25 protects exact entities,
acronyms, dates, and technical terms that embedding search can smooth over.
Both ranked lists are combined with reciprocal rank fusion so raw FAISS and
BM25 scores never need to be compared directly.
"""
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from app.research.embeddings import embed_query
from app.research.vector_store import FAISSVectorStore
from app.memory.repositories import ChunkRepository
from app.models.schema import DocumentChunk, RetrievedChunk
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _bm25_rank(
    query: str,
    chunks: List[DocumentChunk],
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    """Return BM25-ranked `(chunk_id, score)` pairs for the query."""
    if not query or not chunks:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    doc_tokens = [_tokenize(c.text) for c in chunks]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    if avgdl == 0.0:
        return []

    document_frequency: Dict[str, int] = defaultdict(int)
    term_counts = []
    for tokens in doc_tokens:
        counts = Counter(tokens)
        term_counts.append(counts)
        for term in set(tokens):
            document_frequency[term] += 1

    n_docs = len(chunks)
    scores: List[Tuple[int, float]] = []
    for chunk, counts, doc_len in zip(chunks, term_counts, doc_lengths):
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (n_docs - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (freq * (k1 + 1) / denom)
        if score > 0 and chunk.chunk_id is not None:
            scores.append((chunk.chunk_id, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def _reciprocal_rank_fusion(
    dense_results: List[Tuple[int, float]],
    lexical_results: List[Tuple[int, float]],
    rrf_k: int,
) -> List[Tuple[int, float, float, float, str]]:
    """
    Fuse dense and lexical rankings.

    Returns tuples of:
      `(chunk_id, fused_score, dense_score, lexical_score, retrieval_method)`.
    """
    fused: Dict[int, float] = defaultdict(float)
    dense_scores = {chunk_id: score for chunk_id, score in dense_results}
    lexical_scores = {chunk_id: score for chunk_id, score in lexical_results}

    for rank, (chunk_id, _) in enumerate(dense_results, start=1):
        fused[chunk_id] += 1.0 / (rrf_k + rank)
    for rank, (chunk_id, _) in enumerate(lexical_results, start=1):
        fused[chunk_id] += 1.0 / (rrf_k + rank)

    ranked = []
    for chunk_id, score in fused.items():
        has_dense = chunk_id in dense_scores
        has_lexical = chunk_id in lexical_scores
        if has_dense and has_lexical:
            method = "hybrid"
        elif has_lexical:
            method = "bm25"
        else:
            method = "dense"
        ranked.append((
            chunk_id,
            score,
            dense_scores.get(chunk_id, 0.0),
            lexical_scores.get(chunk_id, 0.0),
            method,
        ))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked

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
        self.dense_top_k = max(self.top_k, settings.dense_retrieval_top_k)
        self.lexical_top_k = max(self.top_k, settings.lexical_retrieval_top_k)
        self.rrf_k = settings.rrf_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Search dense and lexical indexes, fuse rankings, and return chunks.

        1. Embed query and search FAISS by cosine similarity.
        2. Run BM25 over stored chunk text.
        3. Fuse both ranked lists with reciprocal rank fusion.
        4. Fetch chunk text + metadata from SQLite.
        5. Return RetrievedChunk list sorted by fused score.
        """
        all_chunks = self.chunk_repo.list_all()
        if not all_chunks:
            logger.warning("No chunks are available — no retrieval possible.")
            return []

        dense_results: List[Tuple[int, float]] = []
        if self.vector_store.total_vectors > 0:
            query_vec = embed_query(query)
            dense_results = self.vector_store.search(query_vec, top_k=self.dense_top_k)
        else:
            logger.warning("Vector store is empty — falling back to BM25 retrieval only.")

        lexical_results = _bm25_rank(query, all_chunks, top_k=self.lexical_top_k)
        fused_results = _reciprocal_rank_fusion(dense_results, lexical_results, rrf_k=self.rrf_k)
        if not fused_results:
            return []

        fused_results = fused_results[:self.top_k]
        chunk_ids = [chunk_id for chunk_id, *_ in fused_results]
        score_map = {
            chunk_id: {
                "score": fused_score,
                "dense_score": dense_score,
                "lexical_score": lexical_score,
                "retrieval_method": method,
            }
            for chunk_id, fused_score, dense_score, lexical_score, method in fused_results
        }
        db_chunks = self.chunk_repo.get_by_ids(chunk_ids)

        chunks_by_id = {chunk.chunk_id: chunk for chunk in db_chunks}
        retrieved: List[RetrievedChunk] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                logger.warning(f"Retrieved chunk_id={chunk_id} but metadata was missing in SQLite.")
                continue
            scores = score_map[chunk_id]
            retrieved.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source_url=chunk.source_url,
                title=chunk.title,
                domain=chunk.domain,
                text=chunk.text,
                score=scores["score"],
                retrieval_method=scores["retrieval_method"],
                dense_score=scores["dense_score"],
                lexical_score=scores["lexical_score"],
            ))

        logger.info(
            f"Retrieved {len(retrieved)} chunks for query: '{query[:60]}…' "
            f"(dense={len(dense_results)}, bm25={len(lexical_results)})."
        )
        return retrieved
