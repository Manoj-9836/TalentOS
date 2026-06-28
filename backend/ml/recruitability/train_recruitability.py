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
    build_candidate_profiles,
    build_candidate_feature_documents,
    calculate_growth_score,
    calculate_stability_score,
    calculate_learning_velocity,
    calculate_skill_credibility,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"
MODEL_OUTPUT_PATH = Path(__file__).parent / "recruitability_model.pkl"
METRICS_OUTPUT_PATH = Path(__file__).parent / "recruitability_metrics.json"

# Behavioural target. offer_acceptance_rate carries a hidden -1 sentinel for
# ~60% of rows; interview_completion_rate is clean [0, 1]. See
# compare_targets.py for the head-to-head and evaluation_report.md for context.
TARGET_SIGNAL_KEY = "interview_completion_rate"


def load_raw_candidates(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    raw_path = Path(raw_dir)
    path = raw_path / "candidates.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates dataset: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_features_and_target(candidates: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
    feature_rows = []
    target_values = []

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

        growth_score = calculate_growth_score(candidate_profile)
        stability_score = calculate_stability_score(candidate_profile)
        learning_velocity = calculate_learning_velocity(candidate_profile)
        skill_credibility = calculate_skill_credibility(candidate_profile)

        signals = candidate.get("redrob_signals", {}) or {}
        github_activity = float(signals.get("github_activity_score", 0) or 0)

        target = float(signals.get(TARGET_SIGNAL_KEY, 0) or 0)

        feature_rows.append({
            "growth_score": growth_score,
            "stability_score": stability_score,
            "learning_velocity": learning_velocity,
            "github_activity": github_activity,
            "skill_credibility": skill_credibility,
        })
        target_values.append(target)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(target_values, name=TARGET_SIGNAL_KEY)

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
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

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "feature_importance": dict(zip(X.columns.tolist(), model.feature_importances_.tolist())),
    }

    with METRICS_OUTPUT_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    return model


def save_model(model: lgb.LGBMRegressor, path: Path = MODEL_OUTPUT_PATH) -> None:
    with path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main() -> None:
    print("Loading candidates...")
    candidates = load_raw_candidates()

    print(f"Total candidates loaded: {len(candidates)}")

    print(f"Extracting features and target ({TARGET_SIGNAL_KEY})...")
    X, y = extract_features_and_target(candidates)
    print(f"Dataset shape: {X.shape}")
    print(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

    if len(y) < 100:
        raise RuntimeError("Too few rows to train a meaningful model.")

    print("Training LightGBM Regressor...")
    model = train_model(X, y)

    print("Saving model...")
    save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()