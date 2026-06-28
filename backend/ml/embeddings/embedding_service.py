from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_model: Optional[SentenceTransformer] = None
_model_name = "BAAI/bge-large-en-v1.5"


def get_model(model_name: str = _model_name) -> SentenceTransformer:
    """Load or return cached SentenceTransformer model."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        _model_name = model_name
        _model = SentenceTransformer(model_name)
    return _model


def encode_text(text: str, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """Encode a single text string into an embedding vector."""
    model = model or get_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding


def encode_batch(texts: List[str], model: Optional[SentenceTransformer] = None, batch_size: int = 32) -> np.ndarray:
    """Encode a batch of texts into embedding vectors."""
    model = model or get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def generate_candidate_embedding(candidate: Dict[str, Any], model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """Generate embedding for a candidate profile."""
    from .build_candidate_text import build_candidate_text

    text = build_candidate_text(candidate)
    return encode_text(text, model)


def generate_job_embedding(job: Dict[str, Any], model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """Generate embedding for a job description."""
    from .build_candidate_text import build_job_text

    text = build_job_text(job)
    return encode_text(text, model)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def search_similar(
    query_embedding: np.ndarray,
    candidate_embeddings: Dict[str, np.ndarray],
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """Search for most similar candidates by cosine similarity."""
    scores = []
    for candidate_id, emb in candidate_embeddings.items():
        score = cosine_similarity(query_embedding, emb)
        scores.append((candidate_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def embeddings_to_mongo_doc(candidate_id: str, embedding: np.ndarray, model_name: str = _model_name) -> Dict[str, Any]:
    """Convert embedding to MongoDB document format."""
    return {
        "candidate_id": candidate_id,
        "embedding": embedding.tolist(),
        "embedding_model": model_name,
        "embedding_dim": len(embedding),
    }


def mongo_doc_to_embedding(doc: Dict[str, Any]) -> tuple[str, np.ndarray]:
    """Convert MongoDB document back to candidate_id and embedding array."""
    return doc["candidate_id"], np.array(doc["embedding"], dtype=np.float32)