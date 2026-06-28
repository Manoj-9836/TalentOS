"""Evaluation harness.

Computes three metric groups and writes docs/reports/metrics_report.md:

    1. Retrieval quality: Recall@K against the sample_submission top-N.
    2. Ranking quality: Pearson / Spearman of rank_score vs response_rate.
    3. Recruitability: training-time MAE / RMSE / R^2 + feature importance.

Run:
    python -m backend.scripts.run_evaluation
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"
EMBEDDINGS_DIR = ROOT_DIR / "backend" / "data" / "embeddings"
INDICES_DIR = ROOT_DIR / "backend" / "indices"
RECRUITABILITY_METRICS_PATH = ROOT_DIR / "backend" / "ml" / "recruitability" / "recruitability_metrics.json"
METRICS_REPORT_PATH = ROOT_DIR / "docs" / "reports" / "metrics_report.md"


def _response_rate_from_reasoning(reasoning: str) -> float | None:
    """Pull the response_rate field out of the sample_submission reasoning string."""
    if not isinstance(reasoning, str):
        return None
    match = re.search(r"response rate\s+([0-9.]+)", reasoning)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _load_reference() -> pd.DataFrame:
    """Load sample_submission.csv as the reference ranking + proxy target."""
    ref = pd.read_csv(RAW_DIR / "sample_submission.csv")
    ref["response_rate"] = ref["reasoning"].apply(_response_rate_from_reasoning)
    return ref


def _load_full_candidates() -> list[dict]:
    """Load all candidates from candidates.jsonl."""
    from backend.app.database.data_loader import load_candidates_full
    return load_candidates_full(RAW_DIR)


def retrieval_quality() -> dict:
    """Top-K recall of the FAISS index against the reference top-N (default 50)."""
    from backend.ml.retrieval.retrieval_service import load_index, retrieve_top_candidates

    index, ids = load_index(
        str(INDICES_DIR / "candidate_index.faiss"),
        str(INDICES_DIR / "candidate_ids.npy"),
    )
    faiss_ids = set(map(str, ids.tolist()))

    job_pkl = EMBEDDINGS_DIR / "job_embeddings.pkl"
    if not job_pkl.exists():
        return {"available": False, "reason": f"Missing {job_pkl}"}
    with job_pkl.open("rb") as handle:
        job_map = pickle_load(handle)
    job_embedding = job_map.get("primary")
    if job_embedding is None:
        return {"available": False, "reason": "job_embeddings.pkl has no 'primary' key"}

    # Top-50 reference set comes from sample_submission.csv.
    ref = _load_reference()
    ground_truth = set(ref.head(50)["candidate_id"].astype(str))

    # Retrieval uses the precomputed job embedding; no model needed at query time.
    from backend.ml.retrieval.retrieval_service import retrieve_by_embedding
    results = retrieve_by_embedding(job_embedding, top_k=500, index=index, candidate_ids=ids)
    retrieved_ids = [str(cid) for cid, _ in results]

    recall_at_k: dict[str, float] = {}
    for k in (10, 50, 100, 500):
        top_k = set(retrieved_ids[:k])
        if not ground_truth:
            recall = 0.0
        else:
            recall = len(top_k & ground_truth) / len(ground_truth)
        recall_at_k[f"recall@{k}"] = round(recall, 4)

    return {
        "available": True,
        "index_size": int(index.ntotal),
        "reference_top_n": len(ground_truth),
        "ground_truth_in_index": len(ground_truth & faiss_ids),
        **recall_at_k,
    }


# ---------- 2. ranking quality --------------------------------------------


def ranking_quality() -> dict:
    """Correlation between our ranker and a proxy acceptance signal."""
    from scipy.stats import pearsonr, spearmanr

    from backend.ml.feature_engineering.profile_builder import (
        build_candidate_profiles,
        build_candidate_feature_documents,
        calculate_growth_score,
        calculate_stability_score,
        calculate_learning_velocity,
        calculate_skill_credibility,
    )
    from backend.ml.ranking.ranking_engine import rank_candidates_from_dicts
    from backend.ml.recruitability.recruitability_service import get_recruitability_predictor
    from backend.ml.retrieval.retrieval_service import (
        load_index,
        retrieve_by_embedding,
    )

    with (EMBEDDINGS_DIR / "job_embeddings.pkl").open("rb") as handle:
        job_embedding = pickle_load(handle)["primary"]

    index, ids = load_index(
        str(INDICES_DIR / "candidate_index.faiss"),
        str(INDICES_DIR / "candidate_ids.npy"),
    )
    retrieved = retrieve_by_embedding(job_embedding, top_k=200, index=index, candidate_ids=ids)

    full = {c.get("candidate_id"): c for c in _load_full_candidates()}
    profiles = {p["candidate_id"]: p for p in build_candidate_profiles(_load_full_candidates())}

    recruiter = get_recruitability_predictor()

    ranking_inputs = []
    response_rates: list[float] = []
    candidate_ids_in_order: list[str] = []

    for cid, semantic_score in retrieved:
        raw = full.get(cid)
        profile = profiles.get(cid)
        if not raw or not profile:
            continue
        candidate_payload = dict(profile)
        candidate_payload["candidate_id"] = cid
        try:
            recruitability = recruiter.predict(candidate_payload)
        except Exception:
            recruitability = 0.5
        rr = raw.get("redrob_signals", {}).get("recruiter_response_rate")
        if rr is None:
            continue
        ranking_inputs.append({
            "candidate_id": cid,
            "semantic_score": float(semantic_score),
            "growth_score": float(calculate_growth_score(candidate_payload)),
            "stability_score": float(calculate_stability_score(candidate_payload)),
            "learning_velocity": float(calculate_learning_velocity(candidate_payload)),
            "recruitability_prediction": float(recruitability),
            "skill_credibility": float(calculate_skill_credibility(candidate_payload)),
        })
        response_rates.append(float(rr))
        candidate_ids_in_order.append(cid)

    if len(ranking_inputs) < 5:
        return {"available": False, "reason": "Too few candidates with response_rate signal"}

    ranked = rank_candidates_from_dicts(ranking_inputs, top_k=None)
    rank_scores = np.array([r["candidate_rank_score"] for r in ranked])
    rr_array = np.array(response_rates)

    # Re-align response_rates with the ranker's output order (the ranker sorts by score desc).
    id_to_rr = dict(zip(candidate_ids_in_order, response_rates))
    aligned_rr = np.array([id_to_rr[r["candidate_id"]] for r in ranked])
    aligned_score = np.array([r["candidate_rank_score"] for r in ranked])

    pearson_r, pearson_p = pearsonr(aligned_score, aligned_rr)
    spearman_r, spearman_p = spearmanr(aligned_score, aligned_rr)

    return {
        "available": True,
        "n_candidates": len(aligned_score),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 6),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 6),
    }


def recruitability_metrics() -> dict | None:
    """Read training-time metrics from disk, or None if not yet produced."""
    if not RECRUITABILITY_METRICS_PATH.exists():
        return None
    with RECRUITABILITY_METRICS_PATH.open("r") as handle:
        return json.load(handle)


def pickle_load(handle):
    import pickle
    return pickle.load(handle)


def write_report(retrieval: dict, ranking: dict, recruitability: dict | None) -> None:
    METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# TalentOS — Evaluation Metrics Report\n")
    lines.append("Generated by `backend/scripts/run_evaluation.py`.\n")

    lines.append("## 1. Retrieval quality\n")
    if not retrieval.get("available"):
        lines.append(f"_Not available: {retrieval.get('reason', 'unknown')}_\n")
    else:
        lines.append(f"- FAISS index size: **{retrieval['index_size']:,}** candidate vectors")
        lines.append(f"- Reference top-N (from `sample_submission.csv`): **{retrieval['reference_top_n']}**")
        lines.append(f"- Reference IDs present in the index: **{retrieval['ground_truth_in_index']}**\n")
        lines.append("| K | Recall@K |")
        lines.append("|---|----------|")
        for k in (10, 50, 100, 500):
            key = f"recall@{k}"
            lines.append(f"| {k} | {retrieval[key]:.4f} |")
        lines.append("")

    lines.append("## 2. Ranking quality\n")
    if not ranking.get("available"):
        lines.append(f"_Not available: {ranking.get('reason', 'unknown')}_\n")
    else:
        lines.append(f"- Candidates ranked: **{ranking['n_candidates']}**")
        lines.append(f"- Pearson r (rank_score vs. recruiter_response_rate): **{ranking['pearson_r']}** (p = {ranking['pearson_p']})")
        lines.append(f"- Spearman ρ (rank_score vs. recruiter_response_rate): **{ranking['spearman_r']}** (p = {ranking['spearman_p']})\n")
        lines.append("Interpretation: positive correlation means the ranker favours candidates with")
        lines.append("higher recruiter response rates — a proxy for being open to opportunities.\n")

    lines.append("## 3. Recruitability model (training-time metrics)\n")
    if recruitability is None:
        lines.append("_Training metrics not found at `backend/ml/recruitability/recruitability_metrics.json`._\n")
    else:
        lines.append(f"- MAE: **{recruitability.get('mae', 0):.4f}**")
        lines.append(f"- RMSE: **{recruitability.get('rmse', 0):.4f}**")
        lines.append(f"- R²: **{recruitability.get('r2', 0):.4f}**\n")
        lines.append("Feature importances (split count):\n")
        lines.append("| Feature | Importance |")
        lines.append("|---------|------------|")
        for feature, importance in sorted(
            recruitability.get("feature_importance", {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        ):
            lines.append(f"| {feature} | {int(importance)} |")
        lines.append("")
        if recruitability.get("r2", 0) < 0.1:
            lines.append("> ⚠️ R² is low. This is the honest ceiling for an aggregate-feature-only")
            lines.append("> LightGBM model on a behavioural target with limited within-class variance.")
            lines.append("> Treat the recruitability component as a soft prior; the dominant signal")
            lines.append("> is the BGE semantic-similarity layer. See `evaluation_report.md` for")
            lines.append("> the target-comparison sweep that justifies the choice of target.\n")

    lines.append("## Methodology\n")
    lines.append("- **Retrieval**: BAAI/bge-large-en-v1.5 embeddings, IndexFlatIP, 1024-d normalized vectors.")
    lines.append("- **Ranking**: weighted sum of six components (see `ranking_engine.DEFAULT_WEIGHTS`).")
    lines.append("- **Acceptance proxy**: `recruiter_response_rate` from `redrob_signals`.")
    lines.append("- **Reference ranking**: `backend/data/raw/sample_submission.csv` (top-100 reference).\n")

    lines.append("## 4. Ablation: recruitability as a soft prior\n")
    lines.append("The recruitability component is intentionally weighted at 0.25–0.30 of the")
    lines.append("final rank score, with semantic similarity (BGE) carrying the dominant weight.")
    lines.append("Ablation table — to be populated by re-running the ranking pipeline with the")
    lines.append("recruitability weight zeroed out:\n")
    lines.append("| Configuration                       | Pearson r | Spearman ρ |")
    lines.append("|-------------------------------------|-----------|------------|")
    lines.append("| Full ranking (recruitability = 0.25)| _tbd_     | _tbd_      |")
    lines.append("| Ranking without recruitability      | _tbd_     | _tbd_      |")
    lines.append("| Ranking without semantic (BGE)      | _tbd_     | _tbd_      |")
    lines.append("")
    lines.append("Populate this table by calling `rank_candidates_from_dicts` from")
    lines.append("`backend.ml.ranking.ranking_engine` with custom `RankingWeights` (see")
    lines.append("`docs/architecture.png` for the component weights).\n")

    METRICS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {METRICS_REPORT_PATH}")


# ---------- main -----------------------------------------------------------


def main() -> None:
    print("=== TalentOS evaluation ===\n")

    print("[1/3] Retrieval quality...")
    try:
        retrieval = retrieval_quality()
    except FileNotFoundError as exc:
        retrieval = {"available": False, "reason": str(exc)}
    if retrieval.get("available"):
        print(f"  recall@10={retrieval['recall@10']}, recall@50={retrieval['recall@50']}, "
              f"recall@100={retrieval['recall@100']}, recall@500={retrieval['recall@500']}")
    else:
        print(f"  skipped: {retrieval.get('reason')}")

    print("\n[2/3] Ranking quality...")
    try:
        ranking = ranking_quality()
    except FileNotFoundError as exc:
        ranking = {"available": False, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 — degrade gracefully so the report still ships
        ranking = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if ranking.get("available"):
        print(f"  pearson={ranking['pearson_r']}, spearman={ranking['spearman_r']} (n={ranking['n_candidates']})")
    else:
        print(f"  skipped: {ranking.get('reason')}")

    print("\n[3/3] Recruitability model metrics (training-time)...")
    recruitability = recruitability_metrics()
    if recruitability is not None:
        print(f"  MAE={recruitability['mae']:.4f}, RMSE={recruitability['rmse']:.4f}, R²={recruitability['r2']:.4f}")
    else:
        print("  no recruitability_metrics.json found")

    print("\nWriting report...")
    write_report(retrieval, ranking, recruitability)


if __name__ == "__main__":
    main()
