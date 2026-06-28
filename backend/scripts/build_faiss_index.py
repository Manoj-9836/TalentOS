"""Phase 7 — Build the FAISS retrieval index from pre-computed candidate embeddings.

Reads candidate_embeddings.pkl (built by scripts/build_embeddings.py) and
writes candidate_index.faiss + candidate_ids.npy at the default paths that
retrieval_service.load_index already looks at.

Usage:
    python -m backend.scripts.build_faiss_index [--limit N]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_EMBEDDINGS_PATH = ROOT_DIR / "backend" / "data" / "embeddings" / "candidate_embeddings.pkl"
DEFAULT_INDICES_DIR = ROOT_DIR / "backend" / "indices"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from candidate embeddings.")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS_PATH), help="Path to candidate_embeddings.pkl")
    parser.add_argument("--indices-dir", default=str(DEFAULT_INDICES_DIR), help="Where to write the index and ID array")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on candidates (0 = no cap)")
    parser.add_argument(
        "--index-type",
        default="IndexFlatIP",
        choices=["IndexFlatIP", "IndexFlatL2", "IndexHNSWFlat", "IndexIVFFlat"],
        help="FAISS index flavor",
    )
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings file: {embeddings_path}. "
            "Run `python -m backend.scripts.build_embeddings` first."
        )

    with embeddings_path.open("rb") as handle:
        candidate_map: dict[str, np.ndarray] = pickle.load(handle)

    if not candidate_map:
        raise RuntimeError("candidate_embeddings.pkl is empty")

    candidate_ids = list(candidate_map.keys())
    if args.limit and args.limit > 0:
        candidate_ids = candidate_ids[: args.limit]

    embeddings = np.stack([candidate_map[cid] for cid in candidate_ids]).astype(np.float32)
    n, dim = embeddings.shape
    print(f"Loaded {n} embeddings, dim={dim}")

    if args.index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)
    elif args.index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dim)
    elif args.index_type == "IndexHNSWFlat":
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
    elif args.index_type == "IndexIVFFlat":
        nlist = min(4096, max(1, n // 100))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"Training IVF index with nlist={nlist}...")
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {args.index_type}")

    print(f"Adding {n} vectors to {args.index_type}...")
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors")

    indices_dir = Path(args.indices_dir)
    indices_dir.mkdir(parents=True, exist_ok=True)

    index_path = indices_dir / "candidate_index.faiss"
    ids_path = indices_dir / "candidate_ids.npy"
    faiss.write_index(index, str(index_path))
    np.save(ids_path, np.array(candidate_ids, dtype=object))
    print(f"Wrote index -> {index_path}")
    print(f"Wrote IDs   -> {ids_path}")

    # Sanity check: reload through the retrieval service
    print("Sanity check: loading via retrieval_service.load_index()...")
    from backend.ml.retrieval.retrieval_service import load_index

    index_loaded, ids_loaded = load_index(str(index_path), str(ids_path))
    print(f"  Reloaded: {index_loaded.ntotal} vectors, {len(ids_loaded)} IDs")
    assert index_loaded.ntotal == n, "Index size mismatch on reload"
    print("OK")


if __name__ == "__main__":
    main()
