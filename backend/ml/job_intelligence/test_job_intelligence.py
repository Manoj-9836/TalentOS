from __future__ import annotations

import pytest
from job_intelligence import (
    parse_job_description,
    extract_skills,
    extract_experience_level,
    extract_industry,
    extract_role_type,
    generate_job_profile,
    generate_job_profile_from_text,
    generate_job_embedding_text,
    build_job_feature_document,
)


SAMPLE_JOB = {
    "job_id": "JOB_001",
    "title": "Senior Machine Learning Engineer",
    "company": "TechCorp AI",
    "industry": "AI/ML",
    "location": "San Francisco, CA",
    "company_size": "501-1000",
    "description": """
We are looking for a Senior Machine Learning Engineer with 5+ years of experience 
in building production ML systems. You will work with Python, PyTorch, TensorFlow, 
Kubernetes, and AWS. Experience with LLMs, RAG, and MLOps is required.
Strong background in deep learning, NLP, and computer vision preferred.
You will design and deploy scalable ML pipelines using Airflow, Kafka, and Docker.
Experience with feature stores, model registry, and experiment tracking (MLflow).
""",
}


def test_extract_skills():
    skills = extract_skills(SAMPLE_JOB["description"])
    assert "python" in skills
    assert "pytorch" in skills
    assert "tensorflow" in skills
    assert "kubernetes" in skills
    assert "aws" in skills
    assert "llm" in skills or "llms" in skills
    assert "rag" in skills
    assert "mlops" in skills
    assert "deep learning" in skills
    assert "nlp" in skills
    assert "computer vision" in skills
    assert "airflow" in skills
    assert "kafka" in skills
    assert "docker" in skills
    assert "mlflow" in skills


def test_extract_experience_level():
    exp = extract_experience_level(SAMPLE_JOB["description"])
    assert exp >= 5
    assert exp <= 10


def test_extract_industry():
    industry = extract_industry(SAMPLE_JOB["description"])
    assert industry in ["ai/ml", "technology"]


def test_extract_role_type():
    role_type = extract_role_type(SAMPLE_JOB["description"])
    assert role_type == "engineering"


def test_parse_job_description():
    parsed = parse_job_description(SAMPLE_JOB)
    assert "skills" in parsed
    assert "experience" in parsed
    assert "industry" in parsed
    assert "role_type" in parsed
    assert parsed["title"] == SAMPLE_JOB["title"]
    assert parsed["company"] == SAMPLE_JOB["company"]
    assert len(parsed["skills"]) > 0
    assert parsed["experience"] >= 5


def test_generate_job_profile():
    profile = generate_job_profile(SAMPLE_JOB)
    assert "skills" in profile
    assert "experience" in profile
    assert "industry" in profile
    assert "role_type" in profile
    assert isinstance(profile["skills"], list)
    assert isinstance(profile["experience"], int)
    assert isinstance(profile["industry"], str)
    assert isinstance(profile["role_type"], str)


def test_generate_job_profile_from_text():
    text = "Senior Python Developer with 7 years experience in Django, AWS, PostgreSQL"
    profile = generate_job_profile_from_text(text)
    assert "python" in profile["skills"]
    assert "django" in profile["skills"]
    assert "aws" in profile["skills"]
    assert "postgresql" in profile["skills"]
    assert profile["experience"] >= 7


def test_generate_job_embedding_text():
    profile = generate_job_profile(SAMPLE_JOB)
    embedding_text = generate_job_embedding_text({**SAMPLE_JOB, **profile})
    assert "Job Title:" in embedding_text
    assert "Company:" in embedding_text
    assert "Industry:" in embedding_text
    assert "Role Type:" in embedding_text
    assert "Required Experience:" in embedding_text
    assert "Required Skills:" in embedding_text
    assert "Description:" in embedding_text


def test_build_job_feature_document():
    doc = build_job_feature_document(SAMPLE_JOB)
    assert doc["job_id"] == "JOB_001"
    assert doc["title"] == "Senior Machine Learning Engineer"
    assert doc["company"] == "TechCorp AI"
    assert "industry" in doc
    assert "role_type" in doc
    assert "experience_required" in doc
    assert "skills_required" in doc
    assert "skill_count" in doc
    assert doc["skill_count"] == len(doc["skills_required"])


def test_minimal_job():
    minimal_job = {
        "title": "Software Engineer",
        "description": "We need a software engineer with 3 years experience in Java and Spring Boot.",
    }
    profile = generate_job_profile(minimal_job)
    assert "java" in profile["skills"]
    assert "spring boot" in profile["skills"]
    assert profile["experience"] >= 3
    assert profile["role_type"] == "engineering"


def test_manager_role():
    manager_job = {
        "title": "Engineering Manager",
        "description": "Lead a team of 10 engineers. 8+ years experience required. Managing people and projects.",
    }
    profile = generate_job_profile(manager_job)
    assert profile["experience"] >= 8
    assert profile["role_type"] == "management"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])