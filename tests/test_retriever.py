import pytest
from unittest.mock import MagicMock, patch
from app.research.retriever import DenseRetriever
from app.models.schema import DocumentChunk, RetrievedChunk


def test_dense_retriever_retrieve():
    mock_vector_store = MagicMock()
    mock_vector_store.total_vectors = 10
    # FAISS search returns (chunk_id, score)
    mock_vector_store.search.return_value = [(42, 0.95), (101, 0.88)]

    mock_chunk_repo = MagicMock()
    # Repository get_by_ids returns full DocumentChunks
    mock_chunk_repo.get_by_ids.return_value = [
        DocumentChunk(
            chunk_id=42,
            source_id=1,
            session_id=1,
            turn_id=1,
            source_url="https://url1.com",
            title="Title 1",
            domain="url1.com",
            chunk_index=0,
            text="chunk text 1",
            token_count=10,
            embedding_id=0,
        ),
        DocumentChunk(
            chunk_id=101,
            source_id=1,
            session_id=1,
            turn_id=1,
            source_url="https://url2.com",
            title="Title 2",
            domain="url2.com",
            chunk_index=3,
            text="chunk text 2",
            token_count=12,
            embedding_id=1,
        ),
    ]

    retriever = DenseRetriever(
        vector_store=mock_vector_store,
        chunk_repo=mock_chunk_repo,
        top_k=2,
    )

    with patch("app.research.retriever.embed_query", return_value=MagicMock()):
        results = retriever.retrieve("test query")

    assert len(results) == 2
    assert results[0].chunk_id == 42
    assert results[0].score == 0.95
    assert results[0].source_url == "https://url1.com"
    assert results[0].text == "chunk text 1"
    
    assert results[1].chunk_id == 101
    assert results[1].score == 0.88
    assert results[1].source_url == "https://url2.com"
    assert results[1].text == "chunk text 2"
