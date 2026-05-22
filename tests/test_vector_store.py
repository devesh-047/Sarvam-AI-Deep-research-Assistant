"""Tests for FAISS vector store."""
import numpy as np
import pytest
import tempfile
import os


def _random_vecs(n: int, dim: int = 64) -> np.ndarray:
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_add_and_search(tmp_path):
    from app.research.vector_store import FAISSVectorStore
    store = FAISSVectorStore(
        index_path=str(tmp_path / "index.faiss"),
        map_path=str(tmp_path / "map.json"),
    )
    vecs = _random_vecs(10)
    chunk_ids = list(range(100, 110))
    embedding_ids = store.add(vecs, chunk_ids)

    assert len(embedding_ids) == 10
    assert store.total_vectors == 10

    # Search with exact same vector — should get highest score back
    result = store.search(vecs[0], top_k=3)
    assert len(result) == 3
    top_chunk_id, top_score = result[0]
    assert top_chunk_id == 100
    assert top_score > 0.99  # cosine similarity with itself ≈ 1.0


def test_save_and_reload(tmp_path):
    from app.research.vector_store import FAISSVectorStore
    idx_path = str(tmp_path / "index.faiss")
    map_path = str(tmp_path / "map.json")

    store = FAISSVectorStore(index_path=idx_path, map_path=map_path)
    vecs = _random_vecs(5)
    store.add(vecs, list(range(10, 15)))
    store.save()

    # Reload
    store2 = FAISSVectorStore(index_path=idx_path, map_path=map_path)
    assert store2.total_vectors == 5
    results = store2.search(vecs[0], top_k=1)
    assert results[0][0] == 10


def test_search_empty_store(tmp_path):
    from app.research.vector_store import FAISSVectorStore
    store = FAISSVectorStore(
        index_path=str(tmp_path / "index.faiss"),
        map_path=str(tmp_path / "map.json"),
    )
    result = store.search(_random_vecs(1)[0], top_k=5)
    assert result == []
