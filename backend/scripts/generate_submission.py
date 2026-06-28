"""Phase 12 — Ranked Output Generator.

End-to-end run: JD -> BGE embedding -> FAISS top-2000 -> features -> rank -> CSV.

Output schema (matches `sample_submission.csv`):

    candidate_id, rank, score, reasoning

Usage:
    python -m backend.scripts.generate_submission [--top-k 100] [--no-index]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"
EMBEDDINGS_DIR = ROOT_DIR / "backend" / "data" / "embeddings"
INDICES_DIR = ROOT_DIR / "backend" / "indices"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "output" / "ranked_candidates.csv"

FAISS_INDEX_PATH = INDICES_DIR / "candidate_index.faiss"
FAISS_IDS_PATH = INDICES_DIR / "candidate_ids.npy"
JOB_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "job_embeddings.pkl"
CANDIDATE_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "candidate_embeddings.pkl"


def _read_docx_text(path: Path) -> str:
    """Extract plain text from a .docx without requiring python-docx."""
    if not path.exists():
        raise FileNotFoundError(f"Missing document: {path}")
    with zipfile.ZipFile(path) as zf:
        with zf.open("word/document.xml") as handle:
            xml_bytes = handle.read()
    root = ET.fromstring(xml_bytes)
    lines: list[str] = []
    for paragraph in root.iter(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
    ):
        text_parts = [
            node.text
            for node in paragraph.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
            )
            if node.text
        ]
        if text_parts:
            lines.append("".join(text_parts))
    return "\n".join(lines)


def _ai_core_skill_count(candidate: dict[str, Any]) -> int:
    """Count of AI core skills — mirrors the heuristic used in sample_submission.csv."""
    ai_keywords = (
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "data science",
        "neural network", "transformer", "llm", "artificial intelligence",
    )
    skills = candidate.get("skills", []) or []
    count = 0
    for skill in skills:
        if isinstance(skill, dict):
            name = (skill.get("name") or "").lower()
        else:
            name = str(skill).lower()
        if any(kw in name for kw in ai_keywords):
            count += 1
    return count


def _build_reasoning(candidate: dict[str, Any], candidate_full: dict[str, Any], rank_score: float) -> str:
    """Generate a one-line reasoning string similar to sample_submission.csv."""
    profile = candidate_full.get("profile", {}) or {}
    title = profile.get("current_title") or "Professional"
    years = profile.get("years_of_experience") or 0
    years_str = f"{years:.1f}" if isinstance(years, (int, float)) else str(years)
    ai_count = _ai_core_skill_count(candidate)
    response_rate = candidate_full.get("redrob_signals", {}).get("recruiter_response_rate")
    if response_rate is None:
        response_rate = 0.0
    return f"{title} with {years_str} yrs; {ai_count} AI core skills; response rate {response_rate:.2f}; rank_score {rank_score:.4f}."


# ---------- core pipeline --------------------------------------------------


def _assemble_ranking_inputs(
    candidate_ids: list[str],
    semantic_scores: dict[str, float],
    full_by_id: dict[str, dict[str, Any]],
    profiles_by_id: dict[str, dict[str, Any]],
    recruiter,
) -> list[dict[str, Any]]:
    from backend.ml.feature_engineering.profile_builder import (
        calculate_growth_score,
        calculate_stability_score,
        calculate_learning_velocity,
        calculate_skill_credibility,
    )

    inputs: list[dict[str, Any]] = []
    for cid in candidate_ids:
        raw = full_by_id.get(cid)
        profile = profiles_by_id.get(cid)
        if not raw or not profile:
            continue
        payload = dict(profile)
        payload["candidate_id"] = cid
        try:
            recruitability = recruiter.predict(payload)
        except Exception:
            recruitability = 0.5
        inputs.append({
            "candidate_id": cid,
            "semantic_score": float(semantic_scores.get(cid, 0.0)),
            "growth_score": float(calculate_growth_score(payload)),
            "stability_score": float(calculate_stability_score(payload)),
            "learning_velocity": float(calculate_learning_velocity(payload)),
            "recruitability_prediction": float(recruitability),
            "skill_credibility": float(calculate_skill_credibility(payload)),
        })
    return inputs


def _rank_with_index(job_embedding: np.ndarray, top_k_retrieve: int) -> list[tuple[str, float]]:
    """Use FAISS to retrieve top-K candidates by semantic similarity."""
    from backend.ml.retrieval.retrieval_service import (
        load_index,
        retrieve_by_embedding,
    )

    index, ids = load_index(str(FAISS_INDEX_PATH), str(FAISS_IDS_PATH))
    results = retrieve_by_embedding(job_embedding, top_k=top_k_retrieve, index=index, candidate_ids=ids)
    return [(str(cid), float(score)) for cid, score in results]


def _rank_without_index(job_embedding: np.ndarray, candidate_embeddings: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """Brute-force cosine similarity over the in-memory candidate embedding map."""
    if not candidate_embeddings:
        raise RuntimeError("No candidate embeddings to rank against")
    job_norm = job_embedding / max(np.linalg.norm(job_embedding), 1e-9)
    scored = []
    for cid, emb in candidate_embeddings.items():
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        score = float(np.dot(job_norm, emb / norm))
        scored.append((cid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------- main -----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ranked_candidates.csv from the TalentOS pipeline.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing raw inputs")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write ranked_candidates.csv")
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to include in the output")
    parser.add_argument("--retrieve-k", type=int, default=2000, help="Number of candidates to retrieve before ranking")
    parser.add_argument("--no-index", action="store_true", help="Skip FAISS, brute-force against the embedding map")
    args = parser.parse_args()

    print("=== TalentOS submission generator ===\n")

    # ---- 1. Load job embedding ----
    if not JOB_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {JOB_EMBEDDINGS_PATH}. Run `python -m backend.scripts.build_embeddings` first."
        )
    with JOB_EMBEDDINGS_PATH.open("rb") as handle:
        job_map = pickle.load(handle)
    job_embedding = job_map.get("primary")
    if job_embedding is None:
        raise RuntimeError("job_embeddings.pkl has no 'primary' key")
    print(f"[1/5] Loaded job embedding (dim={job_embedding.shape[0]})")

    # ---- 2. Retrieve top-K ----
    if args.no_index:
        if not CANDIDATE_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Missing {CANDIDATE_EMBEDDINGS_PATH}")
        with CANDIDATE_EMBEDDINGS_PATH.open("rb") as handle:
            candidate_embeddings = pickle.load(handle)
        retrieved = _rank_without_index(job_embedding, candidate_embeddings)[: args.retrieve_k]
        print(f"[2/5] Brute-force retrieved {len(retrieved)} candidates from {len(candidate_embeddings)} embeddings")
    else:
        if not (FAISS_INDEX_PATH.exists() and FAISS_IDS_PATH.exists()):
            raise FileNotFoundError(
                f"Missing FAISS index at {FAISS_INDEX_PATH}. "
                "Run `python -m backend.scripts.build_faiss_index` first."
            )
        retrieved = _rank_with_index(job_embedding, args.retrieve_k)
        print(f"[2/5] FAISS retrieved {len(retrieved)} candidates")

    # ---- 3. Load candidate data + build features ----
    from backend.app.database.data_loader import load_candidates_full
    from backend.ml.feature_engineering.profile_builder import build_candidate_profiles
    from backend.ml.recruitability.recruitability_service import get_recruitability_predictor

    full = load_candidates_full(RAW_DIR)
    full_by_id = {c.get("candidate_id"): c for c in full}
    profiles = build_candidate_profiles(full)
    profiles_by_id = {p.get("candidate_id"): p for p in profiles}
    recruiter = get_recruitability_predictor()

    semantic_scores = {cid: score for cid, score in retrieved}
    candidate_ids = [cid for cid, _ in retrieved]
    ranking_inputs = _assemble_ranking_inputs(
        candidate_ids, semantic_scores, full_by_id, profiles_by_id, recruiter
    )
    print(f"[3/5] Assembled features for {len(ranking_inputs)} candidates")

    # ---- 4. Rank ----
    from backend.ml.ranking.ranking_engine import rank_candidates_from_dicts

    ranked = rank_candidates_from_dicts(ranking_inputs, top_k=args.top_k)
    print(f"[4/5] Ranked top {len(ranked)} candidates")

    # ---- 5. Write CSV ----
    rows = []
    for rank_idx, r in enumerate(ranked, start=1):
        cid = r["candidate_id"]
        full_record = full_by_id.get(cid, {})
        reasoning = _build_reasoning(
            {"skills": full_record.get("skills", [])},
            full_record,
            r["candidate_rank_score"],
        )
        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(float(r["candidate_rank_score"]), 4),
            "reasoning": reasoning,
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df.to_csv(output_path, index=False)
    print(f"[5/5] Wrote {len(df)} rows -> {output_path}")

    # ---- Sanity prints ----
    print("\nTop 5:")
    print(df.head().to_string(index=False))

    # Score sanity: must be strictly descending in rank order
    score_diffs = df["score"].diff().dropna()
    if (score_diffs > 1e-9).any():
        print(f"\nWARNING: scores are not strictly descending (max diff = {score_diffs.max():.6f})")


if __name__ == "__main__":
    main()
