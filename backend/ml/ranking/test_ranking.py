from __future__ import annotations

import pytest

from backend.ml.ranking.ranking_engine import (
    CandidateRankingInput,
    RankingWeights,
    compute_rank_score,
    rank_candidates,
    rank_candidates_from_dicts,
)


def test_compute_rank_score_basic() -> None:
    candidate = CandidateRankingInput(
        candidate_id="CAND_001",
        semantic_score=0.85,
        growth_score=80.0,
        stability_score=70.0,
        learning_velocity=75.0,
        recruitability_prediction=0.65,
        skill_credibility=85.0,
    )

    result = compute_rank_score(candidate)

    assert result.candidate_id == "CAND_001"
    assert 0.0 <= result.candidate_rank_score <= 1.0
    assert isinstance(result.component_scores, dict)
    assert len(result.component_scores) == 6


def test_rank_candidates_sorting() -> None:
    candidates = [
        CandidateRankingInput(
            candidate_id="CAND_001",
            semantic_score=0.90,
            growth_score=80.0,
            stability_score=70.0,
            learning_velocity=75.0,
            recruitability_prediction=0.80,
            skill_credibility=85.0,
        ),
        CandidateRankingInput(
            candidate_id="CAND_002",
            semantic_score=0.50,
            growth_score=60.0,
            stability_score=50.0,
            learning_velocity=60.0,
            recruitability_prediction=0.30,
            skill_credibility=55.0,
        ),
        CandidateRankingInput(
            candidate_id="CAND_003",
            semantic_score=0.70,
            growth_score=70.0,
            stability_score=60.0,
            learning_velocity=65.0,
            recruitability_prediction=0.55,
            skill_credibility=70.0,
        ),
    ]

    ranked = rank_candidates(candidates)

    assert len(ranked) == 3
    assert ranked[0].candidate_id == "CAND_001"
    assert ranked[1].candidate_id == "CAND_003"
    assert ranked[2].candidate_id == "CAND_002"
    assert ranked[0].candidate_rank_score > ranked[1].candidate_rank_score
    assert ranked[1].candidate_rank_score > ranked[2].candidate_rank_score


def test_rank_candidates_top_k() -> None:
    candidates = [
        CandidateRankingInput(
            candidate_id=f"CAND_{i:03d}",
            semantic_score=0.9 - i * 0.1,
            growth_score=80.0,
            stability_score=70.0,
            learning_velocity=75.0,
            recruitability_prediction=0.8,
            skill_credibility=85.0,
        )
        for i in range(5)
    ]

    top3 = rank_candidates(candidates, top_k=3)

    assert len(top3) == 3
    assert top3[0].candidate_id == "CAND_000"
    assert top3[2].candidate_id == "CAND_002"


def test_rank_candidates_from_dicts() -> None:
    candidates = [
        {
            "candidate_id": "CAND_001",
            "semantic_score": 0.85,
            "growth_score": 80.0,
            "stability_score": 70.0,
            "learning_velocity": 75.0,
            "recruitability_prediction": 0.65,
            "skill_credibility": 85.0,
        },
        {
            "candidate_id": "CAND_002",
            "semantic_score": 0.60,
            "growth_score": 60.0,
            "stability_score": 50.0,
            "learning_velocity": 60.0,
            "recruitability_prediction": 0.40,
            "skill_credibility": 60.0,
        },
    ]

    ranked = rank_candidates_from_dicts(candidates)

    assert len(ranked) == 2
    assert ranked[0]["candidate_id"] == "CAND_001"
    assert "candidate_rank_score" in ranked[0]
    assert "component_scores" in ranked[0]


def test_custom_weights() -> None:
    custom_weights = RankingWeights(
        semantic_score=0.50,
        recruitability_score=0.20,
        growth_score=0.10,
        learning_velocity=0.10,
        skill_credibility=0.05,
        stability_score=0.05,
    )

    candidate = CandidateRankingInput(
        candidate_id="CAND_001",
        semantic_score=0.80,
        growth_score=80.0,
        stability_score=70.0,
        learning_velocity=75.0,
        recruitability_prediction=0.60,
        skill_credibility=85.0,
    )

    result = compute_rank_score(candidate, custom_weights)

    assert result.candidate_rank_score > 0.0


def test_weights_validation() -> None:
    with pytest.raises(ValueError):
        weights = RankingWeights(
            semantic_score=0.50,
            recruitability_score=0.50,
            growth_score=0.10,
            learning_velocity=0.10,
            skill_credibility=0.10,
            stability_score=0.10,
        )
        weights.validate()


def test_weights_sum_to_one() -> None:
    weights = RankingWeights(
        semantic_score=0.40,
        recruitability_score=0.30,
        growth_score=0.10,
        learning_velocity=0.10,
        skill_credibility=0.05,
        stability_score=0.05,
    )
    weights.validate()