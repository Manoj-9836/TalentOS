from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.database.data_loader import load_candidates
from backend.ml.embeddings.embedding_service import (
    generate_candidate_embedding,
    get_model,
)
from backend.ml.embeddings.build_candidate_text import build_candidate_text


def main():
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    print(f"Loading candidates from {raw_dir}...")
    candidates = load_candidates(raw_dir)

    candidates = candidates[:5]
    print(f"Processing {len(candidates)} candidates...")

    print(f"Loading model: BAAI/bge-large-en-v1.5")
    model = get_model("BAAI/bge-large-en-v1.5")

    print("Generating embeddings...")
    for c in candidates:
        candidate_id = c.get("candidate_id")
        text = build_candidate_text(c)
        embedding = generate_candidate_embedding(c, model)

        import numpy as np
        print(f"\n--- {candidate_id} ---")
        print(f"Text length: {len(text)} chars")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"Embedding (first 5): {embedding[:5]}")

    print("\n=== Batch encoding test ===")
    texts = [build_candidate_text(c) for c in candidates]
    from backend.ml.embeddings.embedding_service import encode_batch
    embeddings = encode_batch(texts, model, batch_size=2)
    import numpy as np
    print(f"Batch embeddings shape: {embeddings.shape}")
    print(f"All normalized: {all(abs(np.linalg.norm(e) - 1.0) < 1e-6 for e in embeddings)}")


if __name__ == "__main__":
    main()