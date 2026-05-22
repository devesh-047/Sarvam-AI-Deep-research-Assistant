"""
FAISS vector store for dense cosine similarity retrieval.

Design:
- Uses IndexFlatIP (inner product) with L2-normalised vectors → cosine similarity.
- Maintains an in-memory list that maps FAISS position (embedding_id) → SQLite chunk_id.
- Persists the FAISS index and the id map to disk so they survive restarts.
- Metadata (text, title, url, domain) is never stored here; it lives in SQLite.
"""
import os
import json
from typing import List, Tuple, Optional

import numpy as np
import faiss

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    def __init__(
        self,
        index_path: str = None,
        map_path: str = None,
    ):
        self.index_path = index_path or settings.faiss_index_path
        self.map_path = map_path or settings.faiss_map_path

        # embedding_id (FAISS row position) → SQLite chunk_id
        self._id_map: List[int] = []
        self._index: Optional[faiss.Index] = None

        # Try to load from disk if files already exist
        self._try_load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_index(self, dim: int) -> None:
        """Create a new index if one doesn't exist yet."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
            logger.info(f"Created new FAISS IndexFlatIP with dim={dim}")

    def _try_load(self) -> None:
        """Load index and id map from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            try:
                self._index = faiss.read_index(self.index_path)
                with open(self.map_path, "r") as f:
                    self._id_map = json.load(f)
                logger.info(f"Loaded FAISS index ({self._index.ntotal} vectors) from {self.index_path}")
            except Exception as e:
                logger.warning(f"Could not load FAISS index: {e}. Starting fresh.")
                self._index = None
                self._id_map = []

    def save(self) -> None:
        """Persist index and id map to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self._index, self.index_path)
        with open(self.map_path, "w") as f:
            json.dump(self._id_map, f)
        logger.info(f"Saved FAISS index ({self._index.ntotal} vectors) to {self.index_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, chunk_ids: List[int]) -> List[int]:
        """
        Add embeddings to the index.

        Args:
            embeddings: float32 array of shape (n, dim), already L2-normalised.
            chunk_ids: SQLite chunk ids corresponding to each row.

        Returns:
            List of embedding_ids (FAISS row positions) assigned to each chunk.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            return []

        dim = embeddings.shape[1]
        self._ensure_index(dim)

        # Record the FAISS positions we are about to assign
        start = len(self._id_map)
        embedding_ids = list(range(start, start + len(chunk_ids)))

        self._index.add(embeddings)
        self._id_map.extend(chunk_ids)

        return embedding_ids

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """
        Find the top-k nearest chunks by cosine similarity.

        Args:
            query_vector: 1-D float32 array, L2-normalised.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, score) tuples sorted descending by score.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self._id_map[idx]
            results.append((chunk_id, float(score)))

        return results

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
