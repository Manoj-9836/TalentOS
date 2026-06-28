from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import shap

from backend.ml.ranking.ranking_engine import (
    CandidateRankingInput,
    CandidateRankingOutput,
    DEFAULT_WEIGHTS,
    compute_rank_score,
)


@dataclass
class ExplainabilityResult:
    candidate_id: str
    candidate_rank_score: float
    positive_factors: list[str]
    negative_factors: list[str]
    feature_contributions: dict[str, float]


COMPONENT_LABELS = {
    "semantic": "Semantic Match",
    "recruitability": "Recruitability",
    "growth": "Growth",
    "learning_velocity": "Learning Velocity",
    "skill_credibility": "Skill Credibility",
    "stability": "Stability",
}

POSITIVE_THRESHOLD = 0.02
NEGATIVE_THRESHOLD = -0.02


def _create_ranking_model(weights: RankingWeights = DEFAULT_WEIGHTS):
    def predict(features: np.ndarray) -> np.ndarray:
        results = []
        for row in features:
            candidate = CandidateRankingInput(
                candidate_id="",
                semantic_score=row[0],
                growth_score=row[1],
                stability_score=row[2],
                learning_velocity=row[3],
                recruitability_prediction=row[4],
                skill_credibility=row[5],
            )
            result = compute_rank_score(candidate, weights)
            results.append(result.candidate_rank_score)
        return np.array(results)

    return predict


def explain_candidate(
    candidate: CandidateRankingInput,
    weights: RankingWeights = DEFAULT_WEIGHTS,
    background_samples: np.ndarray | None = None,
) -> ExplainabilityResult:
    model_fn = _create_ranking_model(weights)

    feature_values = np.array([[
        candidate.semantic_score,
        candidate.growth_score / 100.0,
        candidate.stability_score / 100.0,
        candidate.learning_velocity / 100.0,
        candidate.recruitability_prediction,
        candidate.skill_credibility / 100.0,
    ]])

    if background_samples is None:
        background_samples = np.random.rand(100, 6)
        background_samples[:, 0] = np.random.uniform(0.3, 0.9, 100)
        background_samples[:, 1] = np.random.uniform(0.4, 0.9, 100)
        background_samples[:, 2] = np.random.uniform(0.3, 0.9, 100)
        background_samples[:, 3] = np.random.uniform(0.4, 0.9, 100)
        background_samples[:, 4] = np.random.uniform(0.2, 0.8, 100)
        background_samples[:, 5] = np.random.uniform(0.4, 0.9, 100)

    explainer = shap.Explainer(model_fn, background_samples)
    shap_values = explainer(feature_values)

    contributions = {}
    for i, label in enumerate(["semantic", "growth", "stability", "learning_velocity", "recruitability", "skill_credibility"]):
        contributions[label] = float(shap_values.values[0, i])

    positive_factors = []
    negative_factors = []

    for component, value in contributions.items():
        label = COMPONENT_LABELS[component]
        if value > POSITIVE_THRESHOLD:
            positive_factors.append(f"+ Strong {label}")
        elif value < NEGATIVE_THRESHOLD:
            negative_factors.append(f"- Low {label}")

    if not positive_factors and not negative_factors:
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        for component, value in sorted_contrib[:3]:
            label = COMPONENT_LABELS[component]
            positive_factors.append(f"+ {label}")
        for component, value in sorted_contrib[-2:]:
            label = COMPONENT_LABELS[component]
            negative_factors.append(f"- {label}")

    return ExplainabilityResult(
        candidate_id=candidate.candidate_id,
        candidate_rank_score=compute_rank_score(candidate, weights).candidate_rank_score,
        positive_factors=positive_factors,
        negative_factors=negative_factors,
        feature_contributions=contributions,
    )


def explain_candidates(
    candidates: list[CandidateRankingInput],
    weights: RankingWeights = DEFAULT_WEIGHTS,
    background_samples: np.ndarray | None = None,
) -> list[ExplainabilityResult]:
    return [explain_candidate(c, weights, background_samples) for c in candidates]


def explain_candidates_from_dicts(
    candidates: list[dict[str, Any]],
    weights: RankingWeights = DEFAULT_WEIGHTS,
) -> list[dict[str, Any]]:
    inputs = [
        CandidateRankingInput(
            candidate_id=c["candidate_id"],
            semantic_score=c["semantic_score"],
            growth_score=c["growth_score"],
            stability_score=c["stability_score"],
            learning_velocity=c["learning_velocity"],
            recruitability_prediction=c["recruitability_prediction"],
            skill_credibility=c["skill_credibility"],
        )
        for c in candidates
    ]
    results = explain_candidates(inputs, weights)
    return [
        {
            "candidate_id": r.candidate_id,
            "candidate_rank_score": r.candidate_rank_score,
            "positive_factors": r.positive_factors,
            "negative_factors": r.negative_factors,
            "feature_contributions": r.feature_contributions,
        }
        for r in results
    ]


def format_explanation(result: ExplainabilityResult) -> str:
    lines = [f"Candidate {result.candidate_id} - Rank Score: {result.candidate_rank_score:.3f}"]
    lines.append("")
    lines.append("Positive Factors:")
    for factor in result.positive_factors:
        lines.append(f"  {factor}")
    lines.append("")
    lines.append("Negative Factors:")
    for factor in result.negative_factors:
        lines.append(f"  {factor}")
    return "\n".join(lines)