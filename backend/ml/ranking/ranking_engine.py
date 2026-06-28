from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RankingWeights:
    semantic_score: float = 0.35
    recruitability_score: float = 0.25
    growth_score: float = 0.15
    learning_velocity: float = 0.10
    skill_credibility: float = 0.10
    stability_score: float = 0.05

    def validate(self) -> None:
        total = (
            self.semantic_score
            + self.recruitability_score
            + self.growth_score
            + self.learning_velocity
            + self.skill_credibility
            + self.stability_score
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class CandidateRankingInput:
    candidate_id: str
    semantic_score: float
    growth_score: float
    stability_score: float
    learning_velocity: float
    recruitability_prediction: float
    skill_credibility: float


@dataclass
class CandidateRankingOutput:
    candidate_id: str
    candidate_rank_score: float
    semantic_score: float
    growth_score: float
    stability_score: float
    learning_velocity: float
    recruitability_prediction: float
    skill_credibility: float
    component_scores: dict[str, float]


DEFAULT_WEIGHTS = RankingWeights()


def compute_rank_score(
    candidate: CandidateRankingInput,
    weights: RankingWeights = DEFAULT_WEIGHTS,
) -> CandidateRankingOutput:
    weights.validate()

    semantic_component = weights.semantic_score * candidate.semantic_score
    recruitability_component = weights.recruitability_score * candidate.recruitability_prediction
    growth_component = weights.growth_score * (candidate.growth_score / 100.0)
    learning_component = weights.learning_velocity * (candidate.learning_velocity / 100.0)
    skill_component = weights.skill_credibility * (candidate.skill_credibility / 100.0)
    stability_component = weights.stability_score * (candidate.stability_score / 100.0)

    final_score = (
        semantic_component
        + recruitability_component
        + growth_component
        + learning_component
        + skill_component
        + stability_component
    )

    return CandidateRankingOutput(
        candidate_id=candidate.candidate_id,
        candidate_rank_score=final_score,
        semantic_score=candidate.semantic_score,
        growth_score=candidate.growth_score,
        stability_score=candidate.stability_score,
        learning_velocity=candidate.learning_velocity,
        recruitability_prediction=candidate.recruitability_prediction,
        skill_credibility=candidate.skill_credibility,
        component_scores={
            "semantic": semantic_component,
            "recruitability": recruitability_component,
            "growth": growth_component,
            "learning_velocity": learning_component,
            "skill_credibility": skill_component,
            "stability": stability_component,
        },
    )


def rank_candidates(
    candidates: list[CandidateRankingInput],
    weights: RankingWeights = DEFAULT_WEIGHTS,
    top_k: int | None = None,
) -> list[CandidateRankingOutput]:
    ranked = [compute_rank_score(c, weights) for c in candidates]
    ranked.sort(key=lambda x: x.candidate_rank_score, reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def rank_candidates_from_dicts(
    candidates: list[dict[str, Any]],
    weights: RankingWeights = DEFAULT_WEIGHTS,
    top_k: int | None = None,
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
    outputs = rank_candidates(inputs, weights, top_k)
    return [
        {
            "candidate_id": o.candidate_id,
            "candidate_rank_score": o.candidate_rank_score,
            "semantic_score": o.semantic_score,
            "growth_score": o.growth_score,
            "stability_score": o.stability_score,
            "learning_velocity": o.learning_velocity,
            "recruitability_prediction": o.recruitability_prediction,
            "skill_credibility": o.skill_credibility,
            "component_scores": o.component_scores,
        }
        for o in outputs
    ]