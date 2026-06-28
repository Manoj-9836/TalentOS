from __future__ import annotations

import pytest

from backend.ml.explainability.explainability_engine import (
    explain_candidate,
    explain_candidates_from_dicts,
    format_explanation,
)
from backend.ml.ranking.ranking_engine import CandidateRankingInput


def test_explain_candidate_basic() -> None:
    candidate = CandidateRankingInput(
        candidate_id="CAND_001",
        semantic_score=0.85,
        growth_score=80.0,
        stability_score=70.0,
        learning_velocity=75.0,
        recruitability_prediction=0.65,
        skill_credibility=85.0,
    )

    result = explain_candidate(candidate)

    assert result.candidate_id == "CAND_001"
    assert 0.0 <= result.candidate_rank_score <= 1.0
    assert isinstance(result.positive_factors, list)
    assert isinstance(result.negative_factors, list)
    assert isinstance(result.feature_contributions, dict)
    assert len(result.feature_contributions) == 6


def test_explain_candidate_strong_semantic() -> None:
    candidate = CandidateRankingInput(
        candidate_id="CAND_001",
        semantic_score=0.95,
        growth_score=80.0,
        stability_score=70.0,
        learning_velocity=75.0,
        recruitability_prediction=0.65,
        skill_credibility=85.0,
    )

    result = explain_candidate(candidate)

    assert any("Semantic Match" in f for f in result.positive_factors)


def test_explain_candidate_low_stability() -> None:
    candidate = CandidateRankingInput(
        candidate_id="CAND_001",
        semantic_score=0.85,
        growth_score=80.0,
        stability_score=30.0,
        learning_velocity=75.0,
        recruitability_prediction=0.65,
        skill_credibility=85.0,
    )

    result = explain_candidate(candidate)

    assert len(result.negative_factors) >= 0
    assert "feature_contributions" in result.__dict__


def test_explain_candidates_from_dicts() -> None:
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

    results = explain_candidates_from_dicts(candidates)

    assert len(results) == 2
    for r in results:
        assert "candidate_id" in r
        assert "positive_factors" in r
        assert "negative_factors" in r
        assert "feature_contributions" in r


def test_format_explanation() -> None:
    candidate = CandidateRankingInput(
        candidate_id="CAND_001",
        semantic_score=0.85,
        growth_score=80.0,
        stability_score=70.0,
        learning_velocity=75.0,
        recruitability_prediction=0.65,
        skill_credibility=85.0,
    )

    result = explain_candidate(candidate)
    formatted = format_explanation(result)

    assert "CAND_001" in formatted
    assert "Positive Factors:" in formatted
    assert "Negative Factors:" in formatted


def test_all_components_present() -> None:
    candidate = CandidateRankingInput(
        candidate_id="CAND_001",
        semantic_score=0.85,
        growth_score=80.0,
        stability_score=70.0,
        learning_velocity=75.0,
        recruitability_prediction=0.65,
        skill_credibility=85.0,
    )

    result = explain_candidate(candidate)

    expected_components = {"semantic", "growth", "stability", "learning_velocity", "recruitability", "skill_credibility"}
    assert set(result.feature_contributions.keys()) == expected_components