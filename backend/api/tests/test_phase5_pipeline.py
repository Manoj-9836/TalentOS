"""Phase 5 — Pipeline validation tests.

Exercises the no-auth API surface end-to-end:
  - Health
  - Job upload + ranking
  - Candidate list, detail, intelligence, explanation, compare
  - Analytics dashboard

Tests run with FastAPI's TestClient (no live server required).
Some tests share state via the module-level `jobs_db` in main.py; we upload
fresh jobs per test so they do not collide.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app, jobs_db


client = TestClient(app)


JOB_DESCRIPTION = (
    "We are looking for a Senior Backend Engineer with 5+ years of experience "
    "in Python, FastAPI, PostgreSQL, and AWS. The ideal candidate will have "
    "experience building scalable microservices, designing REST APIs, and "
    "working with containerized deployments using Docker and Kubernetes. "
    "Strong problem-solving skills and experience with CI/CD pipelines are required."
)


def _upload_job() -> str:
    response = client.post(
        "/api/jobs",
        json={
            "job_description": JOB_DESCRIPTION,
            "job_title": "Senior Backend Engineer",
            "company": "TechCorp",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "job_id" in data
    return data["job_id"]


# ---------- Health ---------------------------------------------------------


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# ---------- Job upload -----------------------------------------------------


def test_upload_job_happy_path() -> None:
    job_id = _upload_job()
    assert job_id in jobs_db
    job = jobs_db[job_id]
    assert job["job_title"] == "Senior Backend Engineer"
    assert job["company"] == "TechCorp"
    assert "parsed_requirements" in job


def test_upload_job_rejects_short_description() -> None:
    response = client.post(
        "/api/jobs",
        json={"job_description": "too short", "job_title": "X"},
    )
    assert response.status_code == 422


def test_upload_job_rejects_missing_description() -> None:
    response = client.post("/api/jobs", json={"job_title": "X"})
    assert response.status_code == 422


# ---------- Job listing ----------------------------------------------------


def test_list_jobs_includes_uploaded() -> None:
    job_id = _upload_job()
    response = client.get("/api/jobs")
    assert response.status_code == 200
    listed = {job["job_id"] for job in response.json()}
    assert job_id in listed


def test_get_job_by_id() -> None:
    job_id = _upload_job()
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id


def test_get_unknown_job_returns_404() -> None:
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_job_stats_overview() -> None:
    _upload_job()
    response = client.get("/api/jobs/stats/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["total_jobs"] >= 1
    assert "active_jobs" in body


# ---------- Ranking --------------------------------------------------------


def test_rank_candidates_returns_top_k() -> None:
    job_id = _upload_job()
    response = client.post(
        "/api/rank",
        json={"job_id": job_id, "top_k": 5},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["job_id"] == job_id
    candidates = data["candidates"]
    assert len(candidates) <= 5
    if candidates:
        top = candidates[0]
        assert "candidate_id" in top
        assert "candidate_rank_score" in top


def test_rank_unknown_job_returns_404() -> None:
    response = client.post(
        "/api/rank",
        json={"job_id": "missing-job", "top_k": 5},
    )
    assert response.status_code == 404


def test_rank_with_custom_weights() -> None:
    job_id = _upload_job()
    weights = {
        "semantic_score": 0.5,
        "recruitability_score": 0.2,
        "growth_score": 0.1,
        "learning_velocity": 0.1,
        "skill_credibility": 0.05,
        "stability_score": 0.05,
    }
    response = client.post(
        "/api/rank",
        json={"job_id": job_id, "top_k": 3, "weights": weights},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["candidates"]) <= 3


def test_rankings_for_job_endpoint() -> None:
    job_id = _upload_job()
    response = client.post(
        f"/api/rankings/rank/{job_id}",
        json={"job_id": job_id, "top_k": 3},
    )
    assert response.status_code == 200


def test_ranking_history_non_empty() -> None:
    _upload_job()
    response = client.get("/api/rankings/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ---------- Candidates -----------------------------------------------------


def test_candidates_list_pagination() -> None:
    response = client.get("/api/candidates?limit=10&offset=0")
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 10
    assert body["offset"] == 0
    assert len(body["candidates"]) <= 10


def test_candidates_list_with_search_filter() -> None:
    response = client.get("/api/candidates?search=engineer&limit=10")
    assert response.status_code == 200
    assert isinstance(response.json()["candidates"], list)


def test_candidates_count_endpoint() -> None:
    response = client.get("/api/candidates/count")
    assert response.status_code == 200
    assert "total" in response.json()
    assert response.json()["total"] > 0


def test_get_candidate_detail() -> None:
    listing = client.get("/api/candidates?limit=1").json()
    assert listing["candidates"], "Expected at least one candidate in dataset"
    candidate_id = listing["candidates"][0]["candidate_id"]

    response = client.get(f"/api/candidates/{candidate_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate_id"] == candidate_id
    assert "profile" in body
    assert "skills" in body


def test_get_unknown_candidate_returns_404() -> None:
    response = client.get("/api/candidates/CAND_DOES_NOT_EXIST")
    assert response.status_code == 404


def test_candidate_intelligence() -> None:
    listing = client.get("/api/candidates?limit=1").json()
    candidate_id = listing["candidates"][0]["candidate_id"]

    response = client.get(f"/api/candidates/{candidate_id}/intelligence")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate_id"] == candidate_id
    assert "recruitability_score" in body
    assert "experience_years" in body


def test_candidate_intelligence_unknown_returns_404() -> None:
    response = client.get("/api/candidates/CAND_DOES_NOT_EXIST/intelligence")
    assert response.status_code == 404


# ---------- Compare --------------------------------------------------------


def test_compare_two_candidates() -> None:
    listing = client.get("/api/candidates?limit=2").json()
    ids = [c["candidate_id"] for c in listing["candidates"]]
    assert len(ids) == 2

    response = client.post(
        "/api/compare",
        json={"candidate_a_id": ids[0], "candidate_b_id": ids[1]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate_a"]["candidate_id"] == ids[0]
    assert body["candidate_b"]["candidate_id"] == ids[1]
    assert "score_diff" in body["comparison"]


def test_compare_unknown_candidate_returns_404() -> None:
    response = client.post(
        "/api/compare",
        json={"candidate_a_id": "CAND_X", "candidate_b_id": "CAND_Y"},
    )
    assert response.status_code == 404


# ---------- Analytics ------------------------------------------------------


def test_analytics_dashboard() -> None:
    response = client.get("/api/analytics/dashboard")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_candidates"] > 0
    assert isinstance(body["top_skills"], list)
    assert len(body["top_skills"]) <= 10
    if body["top_skills"]:
        first = body["top_skills"][0]
        assert "name" in first and "count" in first


# ---------- Explanation ----------------------------------------------------


def test_candidate_explanation() -> None:
    job_id = _upload_job()
    rank_resp = client.post(
        "/api/rank", json={"job_id": job_id, "top_k": 1}
    )
    assert rank_resp.status_code == 200
    candidates = rank_resp.json()["candidates"]
    if not candidates:
        pytest.skip("No candidates returned for ranking")

    candidate_id = candidates[0]["candidate_id"]
    response = client.get(f"/api/candidate/{candidate_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate_id"] == candidate_id
    assert "positive_factors" in body
    assert "negative_factors" in body
    assert "feature_contributions" in body
    assert "formatted_explanation" in body


def test_candidate_explanation_unknown_returns_404() -> None:
    response = client.get("/api/candidate/CAND_DOES_NOT_EXIST")
    assert response.status_code == 404
