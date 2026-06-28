"""Target-comparison harness for the recruitability model.

Trains the same 5 features against three candidate targets and reports
held-out MAE / RMSE / R^2. Used to pick the most learnable target.

This is a research script, not part of the production pipeline.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from backend.ml.feature_engineering.profile_builder import (
    calculate_growth_score,
    calculate_stability_score,
    calculate_learning_velocity,
    calculate_skill_credibility,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"
METRICS_OUTPUT_PATH = Path(__file__).parent / "target_comparison_metrics.json"

LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def load_raw_candidates(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    raw_path = Path(raw_dir)
    path = raw_path / "candidates.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates dataset: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_feature_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the same 5 engineered features used by train_recruitability.py."""
    rows = []
    for candidate in candidates:
        candidate_profile = {
            "candidate_id": candidate.get("candidate_id"),
            "profile": candidate.get("profile", {}),
            "experience": candidate.get("profile", {}).get("years_of_experience"),
            "skills": candidate.get("skills", []),
            "education": candidate.get("education", []),
            "career_history": candidate.get("career_history", []),
            "signals": candidate.get("redrob_signals", {}),
        }
        signals = candidate.get("redrob_signals", {}) or {}
        rows.append({
            "growth_score": calculate_growth_score(candidate_profile),
            "stability_score": calculate_stability_score(candidate_profile),
            "learning_velocity": calculate_learning_velocity(candidate_profile),
            "skill_credibility": calculate_skill_credibility(candidate_profile),
            "github_activity": float(signals.get("github_activity_score", 0) or 0),
        })
    return pd.DataFrame(rows)


def _signal_series(candidates: list[dict[str, Any]], key: str) -> pd.Series:
    return pd.Series(
        [
            float(((c.get("redrob_signals") or {}).get(key, 0) or 0))
            for c in candidates
        ],
        name=key,
    )


def train_and_score(
    X: pd.DataFrame, y: pd.Series, label: str
) -> tuple[lgb.LGBMRegressor, dict[str, Any]]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    y_pred = model.predict(X_val)
    metrics = {
        "label": label,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "r2": float(r2_score(y_val, y_pred)),
        "feature_importance": dict(
            zip(X.columns.tolist(), model.feature_importances_.tolist())
        ),
    }
    print(
        f"{label:30s} n={metrics['n_train']:>6d}/{metrics['n_val']:>5d}  "
        f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R^2={metrics['r2']:.4f}"
    )
    return model, metrics


def main() -> None:
    print("Loading candidates...")
    candidates = load_raw_candidates()
    print(f"Total candidates: {len(candidates)}")

    X = build_feature_frame(candidates)

    # offer_acceptance_rate: must filter the -1 sentinel.
    y_offer = _signal_series(candidates, "offer_acceptance_rate")
    offer_mask = y_offer >= 0.0
    print(
        f"offer_acceptance_rate: filtered {int((~offer_mask).sum())} sentinel rows; "
        f"training on {int(offer_mask.sum())} clean rows."
    )
    model_offer, m_offer = train_and_score(
        X.loc[offer_mask].reset_index(drop=True),
        y_offer.loc[offer_mask].reset_index(drop=True),
        label="offer_acceptance_rate (clean)",
    )

    # recruiter_response_rate and interview_completion_rate are clean [0, 1] signals.
    model_resp, m_resp = train_and_score(
        X, _signal_series(candidates, "recruiter_response_rate"), label="recruiter_response_rate"
    )
    model_intv, m_intv = train_and_score(
        X, _signal_series(candidates, "interview_completion_rate"), label="interview_completion_rate"
    )

    summary = {
        "offer_acceptance_rate_clean": m_offer,
        "recruiter_response_rate": m_resp,
        "interview_completion_rate": m_intv,
        "recommendation": (
            "interview_completion_rate"
            if m_intv["r2"] >= m_resp["r2"]
            else "recruiter_response_rate"
        ),
        "notes": (
            "All three models use the same 5 engineered features and the same "
            "train/val split (random_state=42, test_size=0.2). The two behavioural "
            "targets are trained on the full 100k cohort; offer_acceptance_rate is "
            "trained on the 40,446 rows with a recorded offer history (the remaining "
            "59,554 are -1 sentinels)."
        ),
    }
    with METRICS_OUTPUT_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {METRICS_OUTPUT_PATH}")
    print(f"Recommended target: {summary['recommendation']}")


if __name__ == "__main__":
    main()
