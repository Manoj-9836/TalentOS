from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"

_candidates_cache: dict[str, list[dict[str, Any]]] = {}
_candidates_full_cache: dict[str, list[dict[str, Any]]] = {}
_experience_cache: dict[str, list[dict[str, Any]]] = {}
_skills_cache: dict[str, list[dict[str, Any]]] = {}
_embeddings_cache: dict[str, list[dict[str, Any]]] = {}
_jobs_cache: dict[str, list[dict[str, Any]]] = {}


def _cache_key(raw_dir: Path | str) -> str:
    return str(Path(raw_dir).resolve())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_candidates(raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / "candidates.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates dataset: {path}")
    return _read_jsonl(path)


def _build_candidate_docs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for candidate in candidates:
        profile = candidate.get("profile", {}) or {}
        redrob_signals = candidate.get("redrob_signals", {}) or {}
        skills = candidate.get("skills", []) or []

        docs.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "profile": {
                    "anonymized_name": profile.get("anonymized_name"),
                    "headline": profile.get("headline"),
                    "summary": profile.get("summary"),
                    "location": profile.get("location"),
                    "country": profile.get("country"),
                    "years_of_experience": profile.get("years_of_experience"),
                    "current_title": profile.get("current_title"),
                    "current_company": profile.get("current_company"),
                    "current_company_size": profile.get("current_company_size"),
                    "current_industry": profile.get("current_industry"),
                },
                "skills": [skill.get("name") for skill in skills if skill.get("name")],
                "redrob_signals": redrob_signals,
                "skill_count": len(skills),
                "career_history_count": len(candidate.get("career_history", []) or []),
                "education_count": len(candidate.get("education", []) or []),
                "source_file": "candidates.jsonl",
            }
        )
    return docs


def load_candidates(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    key = _cache_key(raw_dir)
    cached = _candidates_cache.get(key)
    if cached is None:
        candidates = _load_candidates(Path(raw_dir))
        cached = _build_candidate_docs(candidates)
        _candidates_cache[key] = cached
    return cached


def load_candidates_full(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    """Return the untransformed candidate records (with career_history, education,
    certifications, languages, redrob_signals). Cached per raw_dir."""
    key = _cache_key(raw_dir)
    cached = _candidates_full_cache.get(key)
    if cached is None:
        cached = _load_candidates(Path(raw_dir))
        _candidates_full_cache[key] = cached
    return cached


def clear_candidates_cache(raw_dir: Path | str | None = None) -> None:
    """Clear cached data. If raw_dir is given, clear only that entry; else clear all."""
    if raw_dir is None:
        _candidates_cache.clear()
        _candidates_full_cache.clear()
        _experience_cache.clear()
        _skills_cache.clear()
        _embeddings_cache.clear()
        _jobs_cache.clear()
        return
    key = _cache_key(raw_dir)
    _candidates_cache.pop(key, None)
    _candidates_full_cache.pop(key, None)
    _experience_cache.pop(key, None)
    _skills_cache.pop(key, None)
    _embeddings_cache.pop(key, None)
    _jobs_cache.pop(key, None)


def load_experience(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    key = _cache_key(raw_dir)
    cached = _experience_cache.get(key)
    if cached is None:
        candidates = _load_candidates(Path(raw_dir))
        experience_docs: list[dict[str, Any]] = []

        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            for index, record in enumerate(candidate.get("career_history", []) or [], start=1):
                experience_docs.append(
                    {
                        "experience_id": f"{candidate_id}_EXP_{index:02d}",
                        "candidate_id": candidate_id,
                        "company": record.get("company"),
                        "title": record.get("title"),
                        "start_date": record.get("start_date"),
                        "end_date": record.get("end_date"),
                        "duration_months": record.get("duration_months"),
                        "is_current": record.get("is_current"),
                        "industry": record.get("industry"),
                        "company_size": record.get("company_size"),
                        "description": record.get("description"),
                        "source_file": "candidates.jsonl",
                    }
                )

        cached = experience_docs
        _experience_cache[key] = cached
    return cached


def load_skills(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    key = _cache_key(raw_dir)
    cached = _skills_cache.get(key)
    if cached is None:
        candidates = _load_candidates(Path(raw_dir))
        skill_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "skill_name": None,
                "candidate_count": 0,
                "total_endorsements": 0,
                "total_duration_months": 0,
                "proficiency_counts": Counter(),
            }
        )

        for candidate in candidates:
            seen_skills: set[str] = set()
            for skill in candidate.get("skills", []) or []:
                skill_name = skill.get("name")
                if not skill_name:
                    continue
                bucket = skill_stats[skill_name]
                bucket["skill_name"] = skill_name
                bucket["total_endorsements"] += int(skill.get("endorsements", 0) or 0)
                bucket["total_duration_months"] += int(skill.get("duration_months", 0) or 0)
                bucket["proficiency_counts"][skill.get("proficiency", "unknown")] += 1
                if skill_name not in seen_skills:
                    bucket["candidate_count"] += 1
                    seen_skills.add(skill_name)

        skill_docs: list[dict[str, Any]] = []
        for skill_name, bucket in sorted(skill_stats.items()):
            candidate_count = bucket["candidate_count"]
            skill_docs.append(
                {
                    "skill_name": skill_name,
                    "candidate_count": candidate_count,
                    "total_endorsements": bucket["total_endorsements"],
                    "average_endorsements": round(bucket["total_endorsements"] / candidate_count, 2)
                    if candidate_count
                    else 0,
                    "average_duration_months": round(bucket["total_duration_months"] / candidate_count, 2)
                    if candidate_count
                    else 0,
                    "proficiency_counts": dict(bucket["proficiency_counts"]),
                    "source_file": "candidates.jsonl",
                }
            )

        cached = skill_docs
        _skills_cache[key] = cached
    return cached


def _normalize_salary_range(signals: dict[str, Any]) -> tuple[float | None, float | None]:
    salary = signals.get("expected_salary_range_inr_lpa", {}) or {}
    return salary.get("min"), salary.get("max")


def _skill_embedding_features(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    skills = candidate.get("skills", []) or []

    proficiencies = [skill.get("proficiency") for skill in skills if skill.get("proficiency")]
    endorsements = [int(skill.get("endorsements", 0) or 0) for skill in skills]
    durations = [int(skill.get("duration_months", 0) or 0) for skill in skills]

    salary_min, salary_max = _normalize_salary_range(signals)

    return {
        "candidate_id": candidate.get("candidate_id"),
        "embedding": [
            float(profile.get("years_of_experience", 0) or 0),
            float(signals.get("profile_completeness_score", 0) or 0),
            float(signals.get("recruiter_response_rate", 0) or 0),
            float(signals.get("avg_response_time_hours", 0) or 0),
            float(signals.get("profile_views_received_30d", 0) or 0),
            float(signals.get("applications_submitted_30d", 0) or 0),
            float(signals.get("connection_count", 0) or 0),
            float(signals.get("endorsements_received", 0) or 0),
            float(signals.get("notice_period_days", 0) or 0),
            float(salary_min or 0),
            float(salary_max or 0),
            float(signals.get("github_activity_score", 0) or 0),
            float(signals.get("search_appearance_30d", 0) or 0),
            float(signals.get("saved_by_recruiters_30d", 0) or 0),
            float(signals.get("interview_completion_rate", 0) or 0),
            float(signals.get("offer_acceptance_rate", 0) or 0),
        ],
        "skill_count": len(skills),
        "unique_skill_count": len({skill.get("name") for skill in skills if skill.get("name")}),
        "avg_skill_endorsements": round(sum(endorsements) / len(endorsements), 2) if endorsements else 0,
        "avg_skill_duration_months": round(sum(durations) / len(durations), 2) if durations else 0,
        "proficiency_breakdown": dict(Counter(proficiencies)),
        "source_file": "candidates.jsonl",
    }


def load_candidate_embeddings(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    key = _cache_key(raw_dir)
    cached = _embeddings_cache.get(key)
    if cached is None:
        candidates = _load_candidates(Path(raw_dir))
        docs: list[dict[str, Any]] = []
        for candidate in candidates:
            doc = _skill_embedding_features(candidate)
            doc["embedding_model"] = "structured_feature_vector_v1"
            docs.append(doc)
        cached = docs
        _embeddings_cache[key] = cached
    return cached


def load_jobs(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    key = _cache_key(raw_dir)
    cached = _jobs_cache.get(key)
    if cached is None:
        candidates = _load_candidates(Path(raw_dir))
        job_map: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}

        for candidate in candidates:
            profile = candidate.get("profile", {}) or {}
            title = profile.get("current_title")
            company = profile.get("current_company")
            industry = profile.get("current_industry")
            company_size = profile.get("current_company_size")
            location = profile.get("location")
            map_key = (title, company, industry)

            entry = job_map.setdefault(
                map_key,
                {
                    "job_id": f"JOB_{len(job_map) + 1:06d}",
                    "title": title,
                    "company": company,
                    "industry": industry,
                    "company_size": company_size,
                    "location": location,
                    "candidate_count": 0,
                    "source_file": "candidates.jsonl",
                },
            )
            entry["candidate_count"] += 1

        cached = list(job_map.values())
        _jobs_cache[key] = cached
    return cached


def build_rankings(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    raw_path = Path(raw_dir)
    submission_path = raw_path / "sample_submission.csv"
    if not submission_path.exists():
        raise FileNotFoundError(f"Missing ranking file: {submission_path}")

    rankings_df = pd.read_csv(submission_path)
    docs: list[dict[str, Any]] = []
    for index, row in rankings_df.iterrows():
        docs.append(
            {
                "ranking_id": f"RANK_{index + 1:06d}",
                "candidate_id": row.get("candidate_id"),
                "rank": int(row.get("rank", 0) or 0),
                "score": float(row.get("score", 0) or 0),
                "reasoning": row.get("reasoning"),
                "source_file": "sample_submission.csv",
            }
        )

    return docs
