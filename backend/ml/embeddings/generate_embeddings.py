"""Generate BGE (BAAI/bge-large-en-v1.5) embeddings for candidates.

Reads candidates.jsonl (or sample_candidates.json with --sample) and writes
candidate_embeddings.jsonl. The JSONL is consumed by build_index.py; no
database is required.

Usage:
    python -m backend.ml.embeddings.generate_embeddings [--sample] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.database.data_loader import load_candidates
from backend.ml.embeddings.build_candidate_text import build_candidate_text

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BGE embeddings for candidates and write them to JSONL.")
    parser.add_argument(
        "--raw-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw"),
        help="Path to the raw dataset directory",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path(__file__).resolve().parents[2] / "data" / "embeddings" / "candidate_embeddings.jsonl"),
        help="Where to write the embedding JSONL",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of candidates to process (0 = all)")
    parser.add_argument("--sample", action="store_true", help="Use sample_candidates.json instead of candidates.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Print candidate texts without generating embeddings")
    return parser.parse_args()


def _load_candidates(raw_dir: Path, sample: bool) -> list[dict[str, Any]]:
    """Load the full JSONL or the small sample, depending on --sample."""
    if sample:
        sample_path = raw_dir / "sample_candidates.json"
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample dataset: {sample_path}")
        with sample_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "candidates" in data:
            return data["candidates"]
        if isinstance(data, list):
            return data
        raise ValueError("sample_candidates.json has unexpected shape")

    jsonl_path = raw_dir / "candidates.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing candidates dataset: {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write one JSON record per line, UTF-8 encoded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output_path)

    print(f"Loading candidates from {raw_dir}...")
    candidates = load_candidates(raw_dir) if not args.sample else _load_candidates(raw_dir, sample=True)

    if args.limit > 0:
        candidates = candidates[: args.limit]

    print(f"Processing {len(candidates)} candidates")

    if args.dry_run:
        print("\n=== DRY RUN: Sample candidate texts ===")
        for i, c in enumerate(candidates[:3]):
            print(f"\n--- Candidate {c.get('candidate_id')} ---")
            text = build_candidate_text(c)
            print(text[:500] + ("..." if len(text) > 500 else ""))
        return

    print(f"Loading model: {args.model}")
    from backend.ml.embeddings.embedding_service import get_model

    model = get_model(args.model)

    print("Generating embeddings...")
    candidate_texts: list[str] = []
    candidate_ids: list[str] = []
    for c in candidates:
        candidate_texts.append(build_candidate_text(c))
        candidate_ids.append(c.get("candidate_id") or "")

    if any(not cid for cid in candidate_ids):
        raise ValueError("All candidates must have a candidate_id.")

    embeddings = model.encode(
        candidate_texts,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"Generated embeddings shape: {embeddings.shape}")

    records: list[dict[str, Any]] = []
    for cid, vector in zip(candidate_ids, embeddings):
        records.append({
            "candidate_id": cid,
            "embedding": vector.tolist(),
            "embedding_model": args.model,
            "embedding_dim": int(vector.shape[0]),
        })

    print(f"Writing {len(records)} embedding records to {output_path}...")
    _write_jsonl(records, output_path)

    print(f"\n=== Sample record ===")
    sample_record = records[0]
    print(f"candidate_id: {sample_record['candidate_id']}")
    print(f"embedding_model: {sample_record['embedding_model']}")
    print(f"embedding_dim: {sample_record['embedding_dim']}")
    print(f"embedding (first 5): {sample_record['embedding'][:5]}")


if __name__ == "__main__":
    main()
