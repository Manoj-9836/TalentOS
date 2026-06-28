"""Phase 6 — Build BGE (BAAI/bge-large-en-v1.5) embeddings for candidates and the job description.

Reads candidates.jsonl (or sample_candidates.json with --sample) and
job_description.docx. Writes candidate_embeddings.pkl and job_embeddings.pkl
under backend/data/embeddings/.

Usage:
    python -m backend.scripts.build_embeddings [--sample]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "embeddings"

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def _load_candidates(raw_dir: Path, sample: bool) -> list[dict[str, Any]]:
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


def _read_docx_text(path: Path) -> str:
    """Extract plain text from a .docx without requiring python-docx.

    .docx files are zipped XML; paragraph text lives in w:t elements inside
    w:document/w:body.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    if not path.exists():
        raise FileNotFoundError(f"Missing document: {path}")

    with zipfile.ZipFile(path) as zf:
        with zf.open("word/document.xml") as handle:
            xml_bytes = handle.read()

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    lines: list[str] = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text_parts = [
            node.text for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
            if node.text
        ]
        if text_parts:
            lines.append("".join(text_parts))
    return "\n".join(lines)


def _build_candidate_texts(candidates: list[dict[str, Any]]) -> list[str]:
    from backend.ml.embeddings.build_candidate_text import build_candidate_text

    return [build_candidate_text(c) for c in candidates]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BGE embeddings for candidates and the job description.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw inputs")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to write the .pkl files")
    parser.add_argument("--sample", action="store_true", help="Use sample_candidates.json instead of the full JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of candidates (0 = no cap)")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--model", default=MODEL_NAME, help=f"SentenceTransformer model name (default: {MODEL_NAME})")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(raw_dir, sample=args.sample)
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]
    print(f"Loaded {len(candidates)} candidates")

    print("Loading SentenceTransformer model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    print("Building candidate text representations...")
    candidate_texts = _build_candidate_texts(candidates)
    candidate_ids = [c.get("candidate_id") for c in candidates]

    print(f"Encoding {len(candidate_texts)} candidates (batch_size={args.batch_size})...")
    candidate_embeddings = model.encode(
        candidate_texts,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    candidate_map: dict[str, np.ndarray] = {
        cid: emb for cid, emb in zip(candidate_ids, candidate_embeddings) if cid is not None
    }
    candidate_path = output_dir / "candidate_embeddings.pkl"
    with candidate_path.open("wb") as handle:
        pickle.dump(candidate_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(candidate_map)} candidate embeddings -> {candidate_path}")

    print("Extracting and embedding job description...")
    job_docx_path = raw_dir / "job_description.docx"
    job_text = _read_docx_text(job_docx_path)
    if not job_text.strip():
        raise RuntimeError(f"No text extracted from {job_docx_path}")
    job_embedding = model.encode(
        [job_text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    )[0].astype(np.float32)

    job_map = {"primary": job_embedding, "job_description.docx": job_embedding}
    job_path = output_dir / "job_embeddings.pkl"
    with job_path.open("wb") as handle:
        pickle.dump(job_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote 1 job embedding (dim={job_embedding.shape[0]}) -> {job_path}")

    print("Done.")


if __name__ == "__main__":
    main()
