from __future__ import annotations

from typing import Any
from .job_parser import parse_job_description


def generate_job_profile(job_data: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_job_description(job_data)
    
    return {
        "skills": parsed["skills"],
        "experience": parsed["experience"],
        "industry": parsed["industry"],
        "role_type": parsed["role_type"],
    }


def generate_job_profile_from_text(text: str) -> dict[str, Any]:
    job_data = {"description": text}
    return generate_job_profile(job_data)


def generate_job_embedding_text(job_profile: dict[str, Any]) -> str:
    parts = []
    
    if job_profile.get("title"):
        parts.append(f"Job Title: {job_profile['title']}")
    
    if job_profile.get("company"):
        parts.append(f"Company: {job_profile['company']}")
    
    if job_profile.get("industry"):
        parts.append(f"Industry: {job_profile['industry']}")
    
    if job_profile.get("role_type"):
        parts.append(f"Role Type: {job_profile['role_type']}")
    
    if job_profile.get("experience"):
        parts.append(f"Required Experience: {job_profile['experience']} years")
    
    if job_profile.get("skills"):
        skills_str = ", ".join(job_profile["skills"])
        parts.append(f"Required Skills: {skills_str}")
    
    if job_profile.get("location"):
        parts.append(f"Location: {job_profile['location']}")
    
    if job_profile.get("company_size"):
        parts.append(f"Company Size: {job_profile['company_size']}")
    
    if job_profile.get("description"):
        parts.append(f"Description: {job_profile['description']}")
    
    return "\n".join(parts)


def build_job_feature_document(job_data: dict[str, Any]) -> dict[str, Any]:
    profile = generate_job_profile(job_data)
    
    return {
        "job_id": job_data.get("job_id"),
        "title": job_data.get("title"),
        "company": job_data.get("company"),
        "industry": profile["industry"],
        "role_type": profile["role_type"],
        "experience_required": profile["experience"],
        "skills_required": profile["skills"],
        "skill_count": len(profile["skills"]),
        "location": job_data.get("location"),
        "company_size": job_data.get("company_size"),
        "description": job_data.get("description"),
        "source": job_data.get("source", "unknown"),
    }