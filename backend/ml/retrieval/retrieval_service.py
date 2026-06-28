from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

_default_index_path = Path(__file__).resolve().parents[2] / "indices" / "candidate_index.faiss"
_default_ids_path = Path(__file__).resolve().parents[2] / "indices" / "candidate_ids.npy"

_index: Optional[faiss.Index] = None
_candidate_ids: Optional[np.ndarray] = None
_model: Optional[SentenceTransformer] = None
_model_name = "BAAI/bge-large-en-v1.5"


def load_index(index_path: str = str(_default_index_path), ids_path: str = str(_default_ids_path)) -> Tuple[faiss.Index, np.ndarray]:
    """Load FAISS index and candidate IDs from disk; cached after first call."""
    global _index, _candidate_ids

    if _index is not None and _candidate_ids is not None:
        return _index, _candidate_ids

    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not Path(ids_path).exists():
        raise FileNotFoundError(f"Candidate IDs not found: {ids_path}")

    print(f"Loading FAISS index from {index_path}...")
    _index = faiss.read_index(index_path)

    print(f"Loading candidate IDs from {ids_path}...")
    _candidate_ids = np.load(ids_path, allow_pickle=True)

    print(f"Loaded index: {_index.ntotal} vectors, {len(_candidate_ids)} IDs")
    return _index, _candidate_ids


def get_model(model_name: str = _model_name) -> SentenceTransformer:
    """Load the BGE model on first call, then return the cached instance."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        _model_name = model_name
        _model = SentenceTransformer(model_name)
    return _model


def encode_query(query_text: str, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """Encode a query text into an embedding vector."""
    model = model or get_model()
    embedding = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    return embedding.astype(np.float32)


def retrieve_top_candidates(
    query_text: str,
    top_k: int = 2000,
    index: Optional[faiss.Index] = None,
    candidate_ids: Optional[np.ndarray] = None,
    model: Optional[SentenceTransformer] = None,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k similar candidates for a query text.

    Args:
        query_text: Job description or search query
        top_k: Number of candidates to retrieve (default 2000)
        index: Pre-loaded FAISS index (optional)
        candidate_ids: Pre-loaded candidate ID mapping (optional)
        model: Pre-loaded SentenceTransformer model (optional)

    Returns:
        List of (candidate_id, similarity_score) tuples, sorted by score descending
    """
    if index is None or candidate_ids is None:
        index, candidate_ids = load_index()

    query_embedding = encode_query(query_text, model)

    # Ensure top_k doesn't exceed index size
    top_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(candidate_ids):
            candidate_id = candidate_ids[idx]
            score = float(scores[0][i])
            results.append((candidate_id, score))

    return results


def retrieve_by_embedding(
    query_embedding: np.ndarray,
    top_k: int = 2000,
    index: Optional[faiss.Index] = None,
    candidate_ids: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k similar candidates using a pre-computed embedding.

    Args:
        query_embedding: Pre-computed normalized embedding vector (1024-dim)
        top_k: Number of candidates to retrieve
        index: Pre-loaded FAISS index (optional)
        candidate_ids: Pre-loaded candidate ID mapping (optional)

    Returns:
        List of (candidate_id, similarity_score) tuples
    """
    if index is None or candidate_ids is None:
        index, candidate_ids = load_index()

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = query_embedding.astype(np.float32)

    top_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(candidate_ids):
            candidate_id = candidate_ids[idx]
            score = float(scores[0][i])
            results.append((candidate_id, score))

    return results


def get_candidate_embedding(candidate_id: str, index: Optional[faiss.Index] = None, candidate_ids: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Retrieve the embedding vector for a specific candidate ID."""
    if index is None or candidate_ids is None:
        index, candidate_ids = load_index()

    try:
        idx = np.where(candidate_ids == candidate_id)[0][0]
        # Reconstruct vector from index (only works for IndexFlatIP/L2)
        if hasattr(index, 'reconstruct'):
            return index.reconstruct(int(idx))
    except (IndexError, AttributeError):
        pass
    return None


def search_similar_to_candidate(
    candidate_id: str,
    top_k: int = 2000,
    index: Optional[faiss.Index] = None,
    candidate_ids: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    """Find candidates similar to a given candidate."""
    if index is None or candidate_ids is None:
        index, candidate_ids = load_index()
    embedding = get_candidate_embedding(candidate_id, index, candidate_ids)
    if embedding is None:
        return []
    return retrieve_by_embedding(embedding, top_k, index, candidate_ids)


class RetrievalService:
    """High-level retrieval service with cached resources."""

    def __init__(
        self,
        index_path: str = str(_default_index_path),
        ids_path: str = str(_default_ids_path),
        model_name: str = _model_name,
    ):
        self.index_path = index_path
        self.ids_path = ids_path
        self.model_name = model_name
        self._index: Optional[faiss.Index] = None
        self._candidate_ids: Optional[np.ndarray] = None
        self._model: Optional[SentenceTransformer] = None

    def _ensure_loaded(self):
        if self._index is None or self._candidate_ids is None:
            self._index, self._candidate_ids = load_index(self.index_path, self.ids_path)
        if self._model is None:
            self._model = get_model(self.model_name)

    def retrieve(self, query_text: str, top_k: int = 2000) -> List[Tuple[str, float]]:
        self._ensure_loaded()
        return retrieve_top_candidates(query_text, top_k, self._index, self._candidate_ids, self._model)

    def retrieve_by_embedding(self, query_embedding: np.ndarray, top_k: int = 2000) -> List[Tuple[str, float]]:
        self._ensure_loaded()
        return retrieve_by_embedding(query_embedding, top_k, self._index, self._candidate_ids)

    def similar_to_candidate(self, candidate_id: str, top_k: int = 2000) -> List[Tuple[str, float]]:
        self._ensure_loaded()
        return search_similar_to_candidate(candidate_id, top_k, self._index, self._candidate_ids)

    @property
    def index_size(self) -> int:
        self._ensure_loaded()
        return self._index.ntotal

    @property
    def candidate_count(self) -> int:
        self._ensure_loaded()
        return len(self._candidate_ids)