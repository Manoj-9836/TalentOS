from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.feature_engineering.profile_builder import (
    calculate_growth_score,
    calculate_stability_score,
    calculate_learning_velocity,
    calculate_skill_credibility,
)

MODEL_PATH = Path(__file__).parent / "recruitability_model.pkl"


class RecruitabilityPredictor:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run train_recruitability.py first.")
        with self.model_path.open("rb") as f:
            self.model = pickle.load(f)

    def predict(self, candidate: dict[str, Any]) -> float:
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

        features = pd.DataFrame([{
            "growth_score": growth_score,
            "stability_score": stability_score,
            "learning_velocity": learning_velocity,
            "github_activity": github_activity,
            "skill_credibility": skill_credibility,
        }])

        prediction = self.model.predict(features)[0]
        return float(np.clip(prediction, 0.0, 1.0))

    def predict_batch(self, candidates: list[dict[str, Any]]) -> list[float]:
        feature_rows = []
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

            feature_rows.append({
                "growth_score": growth_score,
                "stability_score": stability_score,
                "learning_velocity": learning_velocity,
                "github_activity": github_activity,
                "skill_credibility": skill_credibility,
            })

        features = pd.DataFrame(feature_rows)
        predictions = self.model.predict(features)
        return [float(np.clip(p, 0.0, 1.0)) for p in predictions]


def get_recruitability_predictor() -> RecruitabilityPredictor:
    return RecruitabilityPredictor()