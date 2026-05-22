"""
Embedding service using sentence-transformers.

The model is loaded lazily on first use to avoid slowing down import time
and to keep the module safe for testing without GPU/model downloads.
"""
from typing import List
import hashlib
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model = None
_model_load_failed = False
_FALLBACK_DIM = 384


def _get_model():
    global _model, _model_load_failed
    if _model is None and not _model_load_failed:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.embedding_model_name)
        except Exception as e:
            _model_load_failed = True
            logger.warning(
                "Embedding model load failed; using deterministic fallback embeddings. "
                f"Retrieval quality will be degraded until the model is available. Error: {e}"
            )
    return _model


def _fallback_embed_text(text: str, dim: int = _FALLBACK_DIM) -> np.ndarray:
    """Create a deterministic hashed bag-of-words vector for degraded mode."""
    vector = np.zeros(dim, dtype=np.float32)
    tokens = (text or "").lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    return vector


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts.

    Returns a float32 numpy array of shape (len(texts), embedding_dim).
    Vectors are L2-normalised so dot-product == cosine similarity.
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    model = _get_model()
    if model is None:
        embeddings = np.vstack([_fallback_embed_text(text) for text in texts]).astype(np.float32)
    else:
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
