from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_upload_job() -> None:
    job_description = """
    We are looking for a Senior Backend Engineer with 5+ years of experience in Python, 
    FastAPI, PostgreSQL, and AWS. The ideal candidate will have experience building 
    scalable microservices, designing REST APIs, and working with containerized deployments 
    using Docker and Kubernetes. Strong problem-solving skills and experience with 
    CI/CD pipelines are required.
    """
    
    response = client.post("/api/jobs", json={
        "job_description": job_description,
        "job_title": "Senior Backend Engineer",
        "company": "TechCorp"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["job_title"] == "Senior Backend Engineer"
    assert data["company"] == "TechCorp"
    assert "parsed_requirements" in data
    return data["job_id"]


def test_rank_candidates() -> None:
    job_id = test_upload_job()
    
    response = client.post("/api/rank", json={
        "job_id": job_id,
        "top_k": 5
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert "candidates" in data
    assert len(data["candidates"]) <= 5
    assert data["total_candidates"] <= 5
    
    if data["candidates"]:
        candidate = data["candidates"][0]
        assert "candidate_id" in candidate
        assert "candidate_rank_score" in candidate
        assert "component_scores" in candidate


def test_rank_candidates_invalid_job() -> None:
    response = client.post("/api/rank", json={
        "job_id": "invalid_job_id",
        "top_k": 5
    })
    
    assert response.status_code == 404


def test_candidate_explanation() -> None:
    job_id = test_upload_job()
    
    rank_response = client.post("/api/rank", json={
        "job_id": job_id,
        "top_k": 1
    })
    
    assert rank_response.status_code == 200
    candidates = rank_response.json()["candidates"]
    
    if candidates:
        candidate_id = candidates[0]["candidate_id"]
        
        response = client.get(f"/api/candidate/{candidate_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["candidate_id"] == candidate_id
        assert "candidate_rank_score" in data
        assert "positive_factors" in data
        assert "negative_factors" in data
        assert "feature_contributions" in data
        assert "formatted_explanation" in data


def test_candidate_explanation_not_found() -> None:
    response = client.get("/api/candidate/INVALID_ID")
    assert response.status_code == 404