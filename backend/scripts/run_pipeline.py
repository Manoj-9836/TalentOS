"""Single-command TalentOS pipeline.

Chains the build and ranking steps and writes ranked_candidates.csv. Each
step delegates to the existing module; this file is orchestration only.

Usage:
    python -m backend.scripts.run_pipeline [--sample] [--top-k 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _step(label: str, fn) -> Any:
    """Run a pipeline step with a header and elapsed-time footer."""
    print(f"\n=== {label} ===")
    t0 = time.time()
    result = fn()
    print(f"--- {label} done in {time.time() - t0:.1f}s ---")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TalentOS pipeline end-to-end.")
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "backend" / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "backend" / "data" / "embeddings"))
    parser.add_argument("--indices-dir", default=str(ROOT_DIR / "backend" / "indices"))
    parser.add_argument("--csv-output", default=str(ROOT_DIR / "backend" / "output" / "ranked_candidates.csv"))
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Limit candidates (0 = all)")
    parser.add_argument("--sample", action="store_true", help="Use sample_candidates.json (50 rows)")
    parser.add_argument("--top-k", type=int, default=100, help="Rows in the output CSV")
    parser.add_argument("--retrieve-k", type=int, default=2000, help="FAISS retrieval depth before ranking")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Reuse existing JSONL + FAISS index; skip steps 1 and 2",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Reuse existing FAISS index; skip step 2",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    indices_dir = Path(args.indices_dir)
    csv_output = Path(args.csv_output)

    jsonl_path = output_dir / "candidate_embeddings.jsonl"
    faiss_path = indices_dir / "candidate_index.faiss"
    ids_path = indices_dir / "candidate_ids.npy"
    job_embeddings_path = output_dir / "job_embeddings.pkl"

    print("TalentOS pipeline starting")
    print(f"  raw dir      : {raw_dir}")
    print(f"  embeddings   : {output_dir}")
    print(f"  indices      : {indices_dir}")
    print(f"  csv output   : {csv_output}")

    def step1_generate_embeddings():
        from backend.ml.embeddings.generate_embeddings import (
            _load_candidates,
            _write_jsonl,
            MODEL_NAME,
        )
        from backend.ml.embeddings.embedding_service import get_model
        from backend.ml.embeddings.build_candidate_text import build_candidate_text

        candidates = _load_candidates(raw_dir, sample=args.sample)
        if args.limit > 0:
            candidates = candidates[: args.limit]
        print(f"Loaded {len(candidates)} candidates")

        model = get_model(args.model)
        texts = [build_candidate_text(c) for c in candidates]
        ids = [c.get("candidate_id") or "" for c in candidates]
        if any(not cid for cid in ids):
            raise ValueError("All candidates must have a candidate_id.")

        import numpy as np
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        records = [
            {
                "candidate_id": cid,
                "embedding": vec.tolist(),
                "embedding_model": args.model,
                "embedding_dim": int(vec.shape[0]),
            }
            for cid, vec in zip(ids, embeddings)
        ]
        _write_jsonl(records, jsonl_path)
        print(f"Wrote {len(records)} embedding records -> {jsonl_path}")

    def step2_build_index():
        from backend.ml.retrieval.build_index import (
            build_faiss_index,
            load_embeddings_from_jsonl,
            save_index,
        )
        embeddings, candidate_ids = load_embeddings_from_jsonl(jsonl_path)
        index = build_faiss_index(embeddings)
        save_index(index, candidate_ids, str(faiss_path), str(ids_path))

    def step3_encode_job():
        # Reuse the existing scripts/build_embeddings.py logic. Run it as a
        # subprocess so we don't duplicate the .docx parsing here.
        import subprocess
        cmd = [
            sys.executable,
            "-m",
            "backend.scripts.build_embeddings",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--model",
            args.model,
        ]
        if args.sample:
            cmd.append("--sample")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Job embedding step failed with code {result.returncode}")
        if not job_embeddings_path.exists():
            raise FileNotFoundError(f"Expected {job_embeddings_path} after job embedding step")

    def step6_rank_and_write():
        import pickle
        import pandas as pd
        from backend.app.database.data_loader import load_candidates_full
        from backend.ml.feature_engineering.profile_builder import (
            build_candidate_profiles,
            calculate_growth_score,
            calculate_stability_score,
            calculate_learning_velocity,
            calculate_skill_credibility,
        )
        from backend.ml.recruitability.recruitability_service import get_recruitability_predictor
        from backend.ml.ranking.ranking_engine import rank_candidates_from_dicts
        from backend.ml.retrieval.retrieval_service import (
            load_index,
            retrieve_by_embedding,
        )
        from backend.scripts.generate_submission import (
            _build_reasoning,
            _ai_core_skill_count,
        )

        with job_embeddings_path.open("rb") as handle:
            job_embedding = pickle.load(handle)["primary"]
        print(f"Loaded job embedding (dim={job_embedding.shape[0]})")

        index, faiss_ids = load_index(str(faiss_path), str(ids_path))
        retrieved = retrieve_by_embedding(
            job_embedding, top_k=args.retrieve_k, index=index, candidate_ids=faiss_ids
        )
        print(f"FAISS retrieved {len(retrieved)} candidates")

        full = load_candidates_full(raw_dir)
        full_by_id = {c.get("candidate_id"): c for c in full}
        profiles = build_candidate_profiles(full)
        profiles_by_id = {p.get("candidate_id"): p for p in profiles}
        recruiter = get_recruitability_predictor()

        semantic_scores = {cid: score for cid, score in retrieved}
        ranking_inputs = []
        for cid, _ in retrieved:
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
            ranking_inputs.append({
                "candidate_id": cid,
                "semantic_score": float(semantic_scores.get(cid, 0.0)),
                "growth_score": float(calculate_growth_score(payload)),
                "stability_score": float(calculate_stability_score(payload)),
                "learning_velocity": float(calculate_learning_velocity(payload)),
                "recruitability_prediction": float(recruitability),
                "skill_credibility": float(calculate_skill_credibility(payload)),
            })
        print(f"Assembled features for {len(ranking_inputs)} candidates")

        ranked = rank_candidates_from_dicts(ranking_inputs, top_k=args.top_k)
        print(f"Ranked top {len(ranked)} candidates")

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

        csv_output.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
        df.to_csv(csv_output, index=False)
        print(f"Wrote {len(df)} rows -> {csv_output}")
        print("\nTop 5:")
        print(df.head().to_string(index=False))

    if not args.skip_embeddings:
        _step("Step 1/6: Generate candidate embeddings", step1_generate_embeddings)
        _step("Step 2/6: Build FAISS index", step2_build_index)
    elif not args.skip_index:
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"--skip-embeddings requires existing JSONL at {jsonl_path}"
            )
        _step("Step 2/6: Build FAISS index", step2_build_index)
    else:
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"--skip-index requires existing FAISS index at {faiss_path}"
            )
        print("Skipping steps 1-2 (reusing existing JSONL + FAISS index)")

    _step("Step 3/6: Encode job description", step3_encode_job)
    _step("Steps 4-6/6: Retrieve, rank, and write CSV", step6_rank_and_write)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
