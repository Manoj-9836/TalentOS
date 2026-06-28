"""Build a FAISS index from candidate embeddings stored as JSONL.

Reads candidate_embeddings.jsonl (produced by generate_embeddings.py) and
writes candidate_index.faiss + candidate_ids.npy. No database required.

Usage:
    python -m backend.ml.retrieval.build_index [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from candidate embeddings stored as JSONL.")
    parser.add_argument(
        "--embeddings-path",
        default=str(Path(__file__).resolve().parents[2] / "data" / "embeddings" / "candidate_embeddings.jsonl"),
        help="Path to the candidate embeddings JSONL",
    )
    parser.add_argument(
        "--index-path",
        default=str(Path(__file__).resolve().parents[2] / "indices" / "candidate_index.faiss"),
        help="Path to save FAISS index",
    )
    parser.add_argument(
        "--ids-path",
        default=str(Path(__file__).resolve().parents[2] / "indices" / "candidate_ids.npy"),
        help="Path to save candidate ID mapping",
    )
    parser.add_argument("--index-type", default="IndexFlatIP", help="FAISS index type")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of embeddings to load (0 = all)")
    return parser.parse_args()


def load_embeddings_from_jsonl(
    embeddings_path: Path, limit: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Read {candidate_id, embedding} records from a JSONL file."""
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings JSONL not found: {embeddings_path}")

    embeddings_list: list[list[float]] = []
    candidate_ids: list[str] = []
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            candidate_ids.append(record["candidate_id"])
            embeddings_list.append(record["embedding"])
            if limit and len(candidate_ids) >= limit:
                break

    embeddings = np.asarray(embeddings_list, dtype=np.float32)
    candidate_id_array = np.asarray(candidate_ids, dtype=object)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings, candidate_id_array


def build_faiss_index(embeddings: np.ndarray, index_type: str = "IndexFlatIP") -> faiss.Index:
    """Wrap the embedding matrix in a FAISS index of the requested type."""
    dim = embeddings.shape[1]
    print(f"Building FAISS index: {index_type}, dim={dim}, n={embeddings.shape[0]}")

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dim)
    elif index_type == "IndexHNSWFlat":
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
    elif index_type == "IndexIVFFlat":
        nlist = min(4096, max(1, embeddings.shape[0] // 100))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"Training IVF index with nlist={nlist}...")
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    print("Adding embeddings to index...")
    index.add(embeddings)
    print(f"Index built. Total vectors: {index.ntotal}")
    return index


def save_index(index: faiss.Index, candidate_ids: np.ndarray, index_path: str, ids_path: str) -> None:
    """Write the FAISS index and the candidate-ID numpy array to disk."""
    index_path_obj = Path(index_path)
    ids_path_obj = Path(ids_path)
    index_path_obj.parent.mkdir(parents=True, exist_ok=True)
    ids_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)

    print(f"Saving candidate IDs to {ids_path}...")
    np.save(ids_path, candidate_ids)

    print("Done.")


def main() -> None:
    args = parse_args()

    start_time = time.time()

    embeddings, candidate_ids = load_embeddings_from_jsonl(
        Path(args.embeddings_path), limit=args.limit
    )

    index = build_faiss_index(embeddings, args.index_type)

    save_index(index, candidate_ids, args.index_path, args.ids_path)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Index: {args.index_path}")
    print(f"IDs: {args.ids_path}")


if __name__ == "__main__":
    main()
