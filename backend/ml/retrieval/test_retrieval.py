from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.database.data_loader import load_candidates
from backend.ml.embeddings.embedding_service import generate_candidate_embedding, get_model
from backend.ml.embeddings.build_candidate_text import build_candidate_text
from backend.ml.retrieval.retrieval_service import RetrievalService, retrieve_top_candidates


def main():
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    print(f"Loading candidates from {raw_dir}...")
    candidates = load_candidates(raw_dir)
    candidates = candidates[:100]  # Test with 100 candidates
    print(f"Processing {len(candidates)} candidates...")

    print("Loading model...")
    model = get_model("BAAI/bge-large-en-v1.5")

    print("Generating embeddings...")
    embeddings = []
    candidate_ids = []
    for c in candidates:
        emb = generate_candidate_embedding(c, model)
        embeddings.append(emb)
        candidate_ids.append(c.get("candidate_id"))

    embeddings = np.array(embeddings, dtype=np.float32)
    candidate_ids = np.array(candidate_ids, dtype=object)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index in memory
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"Index size: {index.ntotal}")

    # Save to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index.faiss"
        ids_path = Path(tmpdir) / "test_ids.npy"

        faiss.write_index(index, str(index_path))
        np.save(str(ids_path), candidate_ids)

        print(f"\nTesting RetrievalService...")
        service = RetrievalService(index_path=str(index_path), ids_path=str(ids_path))
        print(f"Index size: {service.index_size}")
        print(f"Candidate count: {service.candidate_count}")

        # Test query
        query = "Backend engineer with Python, SQL, Spark experience building data pipelines"
        print(f"\nQuery: {query}")
        results = service.retrieve(query, top_k=10)
        print(f"Top 10 results:")
        for i, (cid, score) in enumerate(results):
            print(f"  {i+1}. {cid}: {score:.4f}")

        # Test similar_to_candidate
        print(f"\nSimilar to {candidate_ids[0]}:")
        similar = service.similar_to_candidate(candidate_ids[0], top_k=5)
        for i, (cid, score) in enumerate(similar):
            print(f"  {i+1}. {cid}: {score:.4f}")

        # Test retrieve_by_embedding
        print(f"\nRetrieve by embedding (first candidate):")
        query_emb = embeddings[0:1]
        results2 = service.retrieve_by_embedding(query_emb, top_k=5)
        for i, (cid, score) in enumerate(results2):
            print(f"  {i+1}. {cid}: {score:.4f}")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()