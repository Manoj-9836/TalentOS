from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ml.feature_engineering.profile_builder import build_candidate_feature_documents, build_candidate_profiles
from backend.ml.job_intelligence.job_parser import parse_job_description
from backend.ml.job_intelligence.job_profile_generator import generate_job_profile
from backend.ml.ranking.ranking_engine import rank_candidates_from_dicts, RankingWeights
from backend.ml.recruitability.recruitability_service import get_recruitability_predictor
from backend.ml.retrieval.retrieval_service import retrieve_top_candidates
from backend.ml.explainability.explainability_engine import explain_candidates_from_dicts, format_explanation
from backend.app.database.data_loader import (
    load_candidates as load_raw_candidates,
    load_candidates_full,
)


class JobUploadRequest(BaseModel):
    job_description: str = Field(..., min_length=50, description="Full job description text")
    job_title: str | None = Field(None, description="Optional job title")
    company: str | None = Field(None, description="Optional company name")


class JobUploadResponse(BaseModel):
    job_id: str
    job_title: str
    company: str | None
    parsed_requirements: dict[str, Any]
    message: str


class RankRequest(BaseModel):
    job_id: str
    top_k: int = Field(20, ge=1, le=100, description="Number of top candidates to return")
    weights: dict[str, float] | None = Field(None, description="Custom ranking weights")


class RankResponse(BaseModel):
    job_id: str
    candidates: list[dict[str, Any]]
    total_candidates: int


class CandidateResponse(BaseModel):
    candidate_id: str
    profile: dict[str, Any]
    skills: list[dict[str, Any]]
    career_history: list[dict[str, Any]] = []
    education: list[dict[str, Any]] = []
    certifications: list[dict[str, Any]] = []
    languages: list[dict[str, Any]] = []
    redrob_signals: dict[str, Any] = {}
    experience: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    overall_score: float | None = None
    rank: int | None = None


class PaginatedCandidatesResponse(BaseModel):
    candidates: list[CandidateResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class JobResponse(BaseModel):
    job_id: str
    job_title: str
    company: str | None
    job_description: str
    parsed_requirements: dict[str, Any]
    created_at: str


class AnalyticsDashboardResponse(BaseModel):
    total_candidates: int
    total_jobs: int
    total_rankings: int
    avg_score: float
    top_skills: list[dict[str, Any]]
    recent_activity: list[dict[str, Any]]


class CompareRequest(BaseModel):
    candidate_a_id: str
    candidate_b_id: str


class CompareResponse(BaseModel):
    candidate_a: CandidateResponse
    candidate_b: CandidateResponse
    comparison: dict[str, Any]


class CandidateExplanationResponse(BaseModel):
    candidate_id: str
    candidate_rank_score: float
    positive_factors: list[str]
    negative_factors: list[str]
    feature_contributions: dict[str, float]
    formatted_explanation: str


jobs_db: dict[str, dict[str, Any]] = {}
candidates_cache: list[dict[str, Any]] | None = None
retrieval_index = None


def get_candidates() -> list[dict[str, Any]]:
    global candidates_cache
    if candidates_cache is None:
        profiles = build_candidate_profiles()
        feature_docs = build_candidate_feature_documents(profiles)
        for profile, features in zip(profiles, feature_docs):
            features["profile"] = profile.get("profile", {})
            features["skills"] = profile.get("skills", [])
            features["career_history"] = profile.get("career_history", [])
        candidates_cache = feature_docs
    return candidates_cache


def get_candidates_fast(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    raw_candidates = load_raw_candidates()
    return raw_candidates[offset:offset + limit]


def get_candidates_count() -> int:
    raw_candidates = load_raw_candidates()
    return len(raw_candidates)


def _filter_candidates(
    candidates: list[dict[str, Any]],
    search: str | None = None,
    skill: str | None = None,
    industry: str | None = None,
    location: str | None = None,
    min_experience: float | None = None,
) -> list[dict[str, Any]]:
    filtered = candidates

    if search:
        search_lower = search.lower()
        words = search_lower.split()
        filtered = [
            c for c in filtered
            if any(
                word in (c.get("profile", {}).get("anonymized_name", "") or "").lower()
                or word in (c.get("profile", {}).get("current_title", "") or "").lower()
                or word in (c.get("profile", {}).get("headline", "") or "").lower()
                or any(word in (s or "").lower() for s in c.get("skills", []))
                for word in words
            )
        ]

    if skill:
        skill_lower = skill.lower()
        filtered = [
            c for c in filtered
            if any(skill_lower == (s or "").lower() for s in c.get("skills", []))
        ]

    if industry:
        industry_lower = industry.lower()
        filtered = [
            c for c in filtered
            if (c.get("profile", {}).get("current_industry", "") or "").lower() == industry_lower
        ]

    if location:
        loc_lower = location.lower()
        filtered = [
            c for c in filtered
            if loc_lower in (c.get("profile", {}).get("location", "") or "").lower()
            or loc_lower in (c.get("profile", {}).get("country", "") or "").lower()
        ]

    if min_experience is not None:
        filtered = [
            c for c in filtered
            if (c.get("profile", {}).get("years_of_experience") or 0) >= min_experience
        ]

    return filtered


def _serialize_candidate(c: dict[str, Any]) -> CandidateResponse:
    return CandidateResponse(
        candidate_id=c["candidate_id"],
        profile=c.get("profile", {}),
        skills=[{"name": s} for s in c.get("skills", [])],
        career_history=c.get("career_history", []),
        overall_score=None,
        rank=None,
    )


def _candidate_detail(candidate_id: str) -> CandidateResponse:
    """Return the full detail payload for one candidate, drawn from the
    untransformed record (career_history, education, certifications, languages,
    redrob_signals). Skills are normalized to {name, proficiency, endorsements,
    duration_months}; career_history is also exposed as `experience` for the
    frontend detail page."""
    full = load_candidates_full()
    raw = next((c for c in full if c.get("candidate_id") == candidate_id), None)
    if not raw:
        raise HTTPException(status_code=404, detail="Candidate not found")

    profile = raw.get("profile", {}) or {}
    skills_raw = raw.get("skills", []) or []
    skills = [
        {
            "name": s.get("name"),
            "proficiency": s.get("proficiency"),
            "endorsements": int(s.get("endorsements", 0) or 0),
            "duration_months": int(s.get("duration_months", 0) or 0),
        }
        for s in skills_raw
        if s.get("name")
    ]

    career_history = raw.get("career_history", []) or []
    experience = [
        {
            "title": r.get("title"),
            "company": r.get("company"),
            "location": profile.get("location"),
            "industry": r.get("industry"),
            "company_size": r.get("company_size"),
            "start_date": r.get("start_date"),
            "end_date": r.get("end_date"),
            "duration_months": r.get("duration_months"),
            "current": bool(r.get("is_current")),
            "description": r.get("description"),
        }
        for r in career_history
    ]

    education = [
        {
            "degree": e.get("degree"),
            "field_of_study": e.get("field_of_study"),
            "school": e.get("institution"),
            "institution": e.get("institution"),
            "start_year": e.get("start_year"),
            "end_year": e.get("end_year"),
            "year": e.get("end_year"),
            "grade": e.get("grade"),
            "tier": e.get("tier"),
        }
        for e in (raw.get("education", []) or [])
    ]

    certifications = raw.get("certifications", []) or []
    languages = raw.get("languages", []) or []

    try:
        recruiter = get_recruitability_predictor()
        payload = {
            **raw,
            "skills": skills_raw,
            "career_history": career_history,
        }
        prediction = recruiter.predict(payload)
        scores = {
            "recruitability_score": prediction.get("score", 0.5),
            "factors": prediction.get("factors", []),
        }
    except Exception:
        scores = {}

    return CandidateResponse(
        candidate_id=candidate_id,
        profile=profile,
        skills=skills,
        career_history=career_history,
        education=education,
        certifications=certifications,
        languages=languages,
        redrob_signals=raw.get("redrob_signals", {}) or {},
        experience=experience,
        scores=scores,
        overall_score=scores.get("recruitability_score"),
        rank=None,
    )


def get_retrieval_index():
    global retrieval_index
    if retrieval_index is None:
        candidates = get_candidates()
        retrieval_index = retrieve_top_candidates
    return retrieval_index


def simple_keyword_search(candidates: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored = []
    for c in candidates:
        text_parts = []
        if c.get("profile"):
            text_parts.append(c["profile"].get("headline", ""))
            text_parts.append(c["profile"].get("summary", ""))
        for skill in c.get("skills", []):
            text_parts.append(skill.get("name", ""))
        for role in c.get("career_history", []):
            text_parts.append(role.get("title", ""))
            text_parts.append(role.get("description", ""))
        
        full_text = " ".join(text_parts).lower()
        score = sum(1 for word in query_words if word in full_text)
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="TalentOS API",
    description="AI-powered candidate ranking and recruitment platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/jobs", response_model=JobUploadResponse)
async def upload_job(request: JobUploadRequest) -> JobUploadResponse:
    parsed = parse_job_description({"description": request.job_description, "title": request.job_title or "", "company": request.company or ""})
    job_profile = generate_job_profile(parsed)

    import uuid
    job_id = str(uuid.uuid4())[:8]

    jobs_db[job_id] = {
        "job_id": job_id,
        "job_title": request.job_title or parsed.get("title", "Unknown Position"),
        "company": request.company,
        "job_description": request.job_description,
        "parsed_requirements": parsed,
        "job_profile": job_profile,
    }

    return JobUploadResponse(
        job_id=job_id,
        job_title=jobs_db[job_id]["job_title"],
        company=request.company,
        parsed_requirements=parsed,
        message="Job uploaded and parsed successfully",
    )


@app.post("/api/rank", response_model=RankResponse)
async def rank_candidates_endpoint(request: RankRequest) -> RankResponse:
    if request.job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[request.job_id]
    job_profile = job["job_profile"]

    candidates = get_candidates()
    job_query = job.get("job_description", "") or job_profile.get("description", "")
    try:
        retrieved_ids = retrieve_top_candidates(job_query, top_k=request.top_k * 3)
        retrieved = [c for c in candidates if c["candidate_id"] in [rid for rid, _ in retrieved_ids]]
        for c, (_, score) in zip(retrieved, retrieved_ids):
            c["semantic_score"] = score
    except FileNotFoundError:
        retrieved = simple_keyword_search(candidates, job_query, request.top_k * 3)
        for c in retrieved:
            c["semantic_score"] = 0.5

    recruiter = get_recruitability_predictor()
    for c in retrieved:
        c["recruitability_prediction"] = recruiter.predict(c)

    if request.weights:
        weights = RankingWeights(**request.weights)
        ranked = rank_candidates_from_dicts(retrieved, weights=weights, top_k=request.top_k)
    else:
        ranked = rank_candidates_from_dicts(retrieved, top_k=request.top_k)

    return RankResponse(
        job_id=request.job_id,
        candidates=ranked,
        total_candidates=len(ranked),
    )


@app.get("/api/candidate/{candidate_id}", response_model=CandidateExplanationResponse)
async def get_candidate_explanation(candidate_id: str) -> CandidateExplanationResponse:
    candidates = get_candidates()
    candidate = next((c for c in candidates if c["candidate_id"] == candidate_id), None)

    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    recruiter = get_recruitability_predictor()
    candidate["recruitability_prediction"] = recruiter.predict(candidate)

    explanations = explain_candidates_from_dicts([candidate])
    exp = explanations[0]

    return CandidateExplanationResponse(
        candidate_id=exp["candidate_id"],
        candidate_rank_score=exp["candidate_rank_score"],
        positive_factors=exp["positive_factors"],
        negative_factors=exp["negative_factors"],
        feature_contributions=exp["feature_contributions"],
        formatted_explanation=format_explanation(type("obj", (object,), exp)()),
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/jobs", response_model=list[JobResponse])
async def get_jobs() -> list[JobResponse]:
    jobs = []
    for job_id, job in jobs_db.items():
        jobs.append(JobResponse(
            job_id=job_id,
            job_title=job["job_title"],
            company=job.get("company"),
            job_description=job["job_description"],
            parsed_requirements=job["parsed_requirements"],
            created_at=datetime.utcnow().isoformat(),
        ))
    return jobs


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs_db[job_id]
    return JobResponse(
        job_id=job_id,
        job_title=job["job_title"],
        company=job.get("company"),
        job_description=job["job_description"],
        parsed_requirements=job["parsed_requirements"],
        created_at=datetime.utcnow().isoformat(),
    )


@app.get("/api/jobs/stats/overview")
async def get_job_stats() -> dict[str, Any]:
    total_jobs = len(jobs_db)
    total_rankings = sum(1 for _ in jobs_db)
    return {
        "total_jobs": total_jobs,
        "total_rankings": total_rankings,
        "active_jobs": total_jobs,
    }


@app.get("/api/candidates", response_model=PaginatedCandidatesResponse)
async def get_candidates_list(
    limit: int = 20,
    offset: int = 0,
    search: str | None = None,
    skill: str | None = None,
    industry: str | None = None,
    location: str | None = None,
    min_experience: float | None = None,
    sort_by: str | None = None,
) -> PaginatedCandidatesResponse:
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    all_candidates = get_candidates_fast(limit=100000, offset=0)
    filtered = _filter_candidates(
        all_candidates,
        search=search,
        skill=skill,
        industry=industry,
        location=location,
        min_experience=min_experience,
    )

    if sort_by == "experience_desc":
        filtered = sorted(
            filtered,
            key=lambda c: c.get("profile", {}).get("years_of_experience") or 0,
            reverse=True,
        )
    elif sort_by == "name_asc":
        filtered = sorted(
            filtered,
            key=lambda c: (c.get("profile", {}).get("anonymized_name") or "").lower(),
        )

    total = len(filtered)
    page = filtered[offset:offset + limit]
    return PaginatedCandidatesResponse(
        candidates=[_serialize_candidate(c) for c in page],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + limit < total,
    )


@app.get("/api/candidates/count")
async def get_candidates_count_endpoint(
    search: str | None = None,
    skill: str | None = None,
    industry: str | None = None,
    location: str | None = None,
    min_experience: float | None = None,
) -> dict[str, int]:
    all_candidates = get_candidates_fast(limit=100000, offset=0)
    filtered = _filter_candidates(
        all_candidates,
        search=search,
        skill=skill,
        industry=industry,
        location=location,
        min_experience=min_experience,
    )
    return {"total": len(filtered)}


@app.get("/api/candidates/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: str) -> CandidateResponse:
    return _candidate_detail(candidate_id)


@app.get("/api/candidates/{candidate_id}/intelligence")
async def get_candidate_intelligence(candidate_id: str) -> dict[str, Any]:
    detail = _candidate_detail(candidate_id).model_dump()
    return {
        "candidate_id": candidate_id,
        "recruitability_score": (detail.get("scores") or {}).get("recruitability_score", 0.5),
        "factors": (detail.get("scores") or {}).get("factors", []),
        "skills_match": detail.get("skills", []),
        "experience_years": detail.get("profile", {}).get("years_of_experience") or 0,
        "experience": detail.get("experience", []),
        "education": detail.get("education", []),
        "profile": detail.get("profile", {}),
    }


@app.post("/api/rankings/rank/{job_id}", response_model=RankResponse)
async def rank_candidates_for_job(job_id: str, request: RankRequest) -> RankResponse:
    request.job_id = job_id
    return await rank_candidates_endpoint(request)


@app.get("/api/rankings/job/{job_id}", response_model=RankResponse)
async def get_job_rankings(job_id: str) -> RankResponse:
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return RankResponse(
        job_id=job_id,
        candidates=[],
        total_candidates=0,
    )


@app.get("/api/rankings/history")
async def get_ranking_history() -> list[dict[str, Any]]:
    history = []
    for job_id, job in jobs_db.items():
        history.append({
            "job_id": job_id,
            "job_title": job["job_title"],
            "company": job.get("company"),
            "ranked_at": datetime.utcnow().isoformat(),
            "total_candidates": 0,
        })
    return history


@app.get("/api/analytics/dashboard", response_model=AnalyticsDashboardResponse)
async def get_analytics_dashboard() -> AnalyticsDashboardResponse:
    candidates = get_candidates()
    skill_counts = {}
    for c in candidates:
        for skill in c.get("skills", []):
            name = skill.get("name", "")
            if name:
                skill_counts[name] = skill_counts.get(name, 0) + 1
    
    top_skills = [
        {"name": k, "count": v} 
        for k, v in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    return AnalyticsDashboardResponse(
        total_candidates=len(candidates),
        total_jobs=len(jobs_db),
        total_rankings=len(jobs_db),
        avg_score=0.65,
        top_skills=top_skills,
        recent_activity=[
            {"action": "Job created", "job_title": j["job_title"], "timestamp": datetime.utcnow().isoformat()}
            for j in list(jobs_db.values())[-5:]
        ],
    )


@app.post("/api/compare", response_model=CompareResponse)
async def compare_candidates(request: CompareRequest) -> CompareResponse:
    candidates = get_candidates()
    candidate_a = next((c for c in candidates if c["candidate_id"] == request.candidate_a_id), None)
    candidate_b = next((c for c in candidates if c["candidate_id"] == request.candidate_b_id), None)
    
    if not candidate_a or not candidate_b:
        raise HTTPException(status_code=404, detail="One or both candidates not found")
    
    recruiter = get_recruitability_predictor()
    pred_a = recruiter.predict(candidate_a)
    pred_b = recruiter.predict(candidate_b)

    skills_a_raw = candidate_a.get("skills", []) or []
    skills_b_raw = candidate_b.get("skills", []) or []

    return CompareResponse(
        candidate_a=CandidateResponse(
            candidate_id=candidate_a["candidate_id"],
            profile=candidate_a.get("profile", {}),
            skills=[s if isinstance(s, dict) else {"name": s} for s in skills_a_raw],
            career_history=candidate_a.get("career_history", []),
            overall_score=pred_a,
        ),
        candidate_b=CandidateResponse(
            candidate_id=candidate_b["candidate_id"],
            profile=candidate_b.get("profile", {}),
            skills=[s if isinstance(s, dict) else {"name": s} for s in skills_b_raw],
            career_history=candidate_b.get("career_history", []),
            overall_score=pred_b,
        ),
        comparison={
            "score_diff": pred_a - pred_b,
            "skills_a": [s.get("name") if isinstance(s, dict) else s for s in skills_a_raw],
            "skills_b": [s.get("name") if isinstance(s, dict) else s for s in skills_b_raw],
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)