from __future__ import annotations

import pytest

from backend.ml.recruitability.recruitability_service import RecruitabilityPredictor
from backend.ml.feature_engineering.profile_builder import build_candidate_profiles


@pytest.fixture(scope="module")
def predictor() -> RecruitabilityPredictor:
    return RecruitabilityPredictor()


@pytest.fixture(scope="module")
def sample_candidates() -> list[dict]:
    return build_candidate_profiles()[:5]


def test_predictor_loads(predictor: RecruitabilityPredictor) -> None:
    assert predictor.model is not None


def test_predict_single(predictor: RecruitabilityPredictor, sample_candidates: list[dict]) -> None:
    candidate = sample_candidates[0]
    score = predictor.predict(candidate)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_batch(predictor: RecruitabilityPredictor, sample_candidates: list[dict]) -> None:
    scores = predictor.predict_batch(sample_candidates)
    assert len(scores) == len(sample_candidates)
    for score in scores:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_predict_consistency(predictor: RecruitabilityPredictor, sample_candidates: list[dict]) -> None:
    candidate = sample_candidates[0]
    score1 = predictor.predict(candidate)
    score2 = predictor.predict(candidate)
    assert abs(score1 - score2) < 1e-6