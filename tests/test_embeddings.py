"""Tests for the embedding service."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_embed_texts_shape():
    from app.research.embeddings import embed_texts
    vecs = embed_texts(["hello world", "deep research"])
    assert vecs.shape[0] == 2
    assert vecs.ndim == 2


def test_embed_texts_normalised():
    from app.research.embeddings import embed_texts
    vecs = embed_texts(["test sentence"])
    norms = np.linalg.norm(vecs, axis=1)
    assert abs(norms[0] - 1.0) < 1e-5


def test_embed_query_shape():
    from app.research.embeddings import embed_query
    vec = embed_query("what is AI?")
    assert vec.ndim == 1


def test_embed_empty_list():
    from app.research.embeddings import embed_texts
    result = embed_texts([])
    assert result.shape[0] == 0
