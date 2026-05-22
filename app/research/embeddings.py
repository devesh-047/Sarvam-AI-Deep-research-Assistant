"""
Embedding service using sentence-transformers.

The model is loaded lazily on first use to avoid slowing down import time
and to keep the module safe for testing without GPU/model downloads.
"""
from typing import List
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts.

    Returns a float32 numpy array of shape (len(texts), embedding_dim).
    Vectors are L2-normalised so dot-product == cosine similarity.
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # L2-normalise so FAISS IndexFlatIP gives cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
    embeddings = embeddings / norms

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns a 1-D float32 numpy array (embedding_dim,), L2-normalised.
    """
    result = embed_texts([query])
    return result[0]
