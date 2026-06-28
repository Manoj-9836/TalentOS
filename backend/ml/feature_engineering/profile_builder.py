from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_raw_candidates(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    raw_path = Path(raw_dir)
    path = raw_path / "candidates.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates dataset: {path}")
    return _read_jsonl(path)


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
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
    }


def build_candidate_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    career_history = candidate.get("career_history", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile": _compact_profile(profile),
        "experience": profile.get("years_of_experience"),
        "skills": skills,
        "education": education,
        "career_history": career_history,
        "signals": signals,
        "skill_count": len(skills),
        "education_count": len(education),
        "career_history_count": len(career_history),
        "source_file": "candidates.jsonl",
    }


def build_candidate_profiles(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    return [build_candidate_profile(candidate) for candidate in load_raw_candidates(raw_dir)]


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _career_metrics(candidate_profile: dict[str, Any]) -> dict[str, Any]:
    history = candidate_profile.get("career_history", []) or []
    if not history:
        return {
            "promotion_count": 0,
            "industry_changes": 0,
            "average_tenure_months": 0.0,
            "short_stint_count": 0,
            "current_tenure_months": 0.0,
        }

    promotion_count = max(len(history) - 1, 0)
    industry_changes = 0
    short_stint_count = 0
    total_months = 0
    current_tenure_months = 0
    previous_industry = None

    for index, role in enumerate(history):
        duration = int(role.get("duration_months", 0) or 0)
        total_months += duration
        if duration < 12:
            short_stint_count += 1
        if role.get("is_current"):
            current_tenure_months = duration
        if previous_industry is not None and role.get("industry") != previous_industry:
            industry_changes += 1
        previous_industry = role.get("industry")

    return {
        "promotion_count": promotion_count,
        "industry_changes": industry_changes,
        "average_tenure_months": round(total_months / len(history), 2),
        "short_stint_count": short_stint_count,
        "current_tenure_months": current_tenure_months,
    }


def calculate_growth_score(candidate_profile: dict[str, Any]) -> int:
    years = float(candidate_profile.get("experience") or candidate_profile.get("profile", {}).get("years_of_experience") or 0)
    skills = candidate_profile.get("skills", []) or []
    metrics = _career_metrics(candidate_profile)

    years_component = min(years / 15.0, 1.0) * 35
    promotion_component = min(metrics["promotion_count"] / 6.0, 1.0) * 35
    skill_breadth_component = min(len(skills) / 20.0, 1.0) * 30
    return _clamp_score(years_component + promotion_component + skill_breadth_component)


def calculate_stability_score(candidate_profile: dict[str, Any]) -> int:
    metrics = _career_metrics(candidate_profile)
    current_tenure_component = min(metrics["current_tenure_months"] / 36.0, 1.0) * 40
    average_tenure_component = min(metrics["average_tenure_months"] / 24.0, 1.0) * 35
    penalty_component = min(metrics["short_stint_count"] * 8 + metrics["industry_changes"] * 5, 40)
    return _clamp_score(current_tenure_component + average_tenure_component + 25 - penalty_component)


def calculate_learning_velocity(candidate_profile: dict[str, Any]) -> int:
    skills = candidate_profile.get("skills", []) or []
    if not skills:
        return 0

    skill_count = len(skills)
    durations = [int(skill.get("duration_months", 0) or 0) for skill in skills]
    endorsements = [int(skill.get("endorsements", 0) or 0) for skill in skills]
    proficiency_counts = Counter(skill.get("proficiency", "unknown") for skill in skills)

    breadth_component = min(skill_count / 20.0, 1.0) * 35
    recency_component = max(0.0, 1.0 - min(sum(durations) / len(durations), 60.0) / 60.0) * 25
    endorsement_component = min(sum(endorsements) / max(skill_count, 1) / 25.0, 1.0) * 20
    advanced_share = (proficiency_counts.get("advanced", 0) + proficiency_counts.get("expert", 0)) / skill_count
    proficiency_component = min(advanced_share, 1.0) * 20
    return _clamp_score(breadth_component + recency_component + endorsement_component + proficiency_component)


def calculate_skill_credibility(candidate_profile: dict[str, Any]) -> int:
    skills = candidate_profile.get("skills", []) or []
    signals = candidate_profile.get("signals", {}) or {}
    if not skills:
        return 0

    proficiency_weights = {
        "beginner": 0.25,
        "intermediate": 0.55,
        "advanced": 0.8,
        "expert": 1.0,
    }

    weighted_skill_score = 0.0
    total_endorsements = 0
    total_duration = 0
    for skill in skills:
        prof = skill.get("proficiency", "beginner")
        weight = proficiency_weights.get(prof, 0.4)
        weighted_skill_score += weight * 100
        total_endorsements += int(skill.get("endorsements", 0) or 0)
        total_duration += int(skill.get("duration_months", 0) or 0)

    assessment_scores = signals.get("skill_assessment_scores", {}) or {}
    assessment_component = 0.0
    if assessment_scores:
        assessment_component = sum(float(value) for value in assessment_scores.values()) / len(assessment_scores)

    endorsement_component = min(total_endorsements / max(len(skills), 1) / 30.0, 1.0) * 25
    duration_component = min(total_duration / max(len(skills), 1) / 36.0, 1.0) * 15
    weighted_component = (weighted_skill_score / len(skills)) * 0.35
    assessment_component = min(assessment_component, 100.0) * 0.25

    return _clamp_score(weighted_component + assessment_component + endorsement_component + duration_component)


def calculate_recruitability_score(candidate_profile: dict[str, Any]) -> int:
    signals = candidate_profile.get("signals", {}) or {}
    growth_score = calculate_growth_score(candidate_profile)
    stability_score = calculate_stability_score(candidate_profile)
    learning_velocity = calculate_learning_velocity(candidate_profile)
    skill_credibility = calculate_skill_credibility(candidate_profile)

    profile_completeness = float(signals.get("profile_completeness_score", 0) or 0)
    recruiter_response_rate = float(signals.get("recruiter_response_rate", 0) or 0) * 100
    offer_acceptance_rate = float(signals.get("offer_acceptance_rate", 0) or 0)
    if offer_acceptance_rate < 0:
        offer_acceptance_rate = 0
    else:
        offer_acceptance_rate *= 100
    interview_completion_rate = float(signals.get("interview_completion_rate", 0) or 0) * 100
    search_visibility = min(float(signals.get("search_appearance_30d", 0) or 0) / 250.0, 1.0) * 100
    recruiter_saves = min(float(signals.get("saved_by_recruiters_30d", 0) or 0) / 25.0, 1.0) * 100

    score = (
        profile_completeness * 0.08
        + recruiter_response_rate * 0.15
        + offer_acceptance_rate * 0.1
        + interview_completion_rate * 0.1
        + search_visibility * 0.07
        + recruiter_saves * 0.05
        + growth_score * 0.13
        + stability_score * 0.12
        + learning_velocity * 0.1
        + skill_credibility * 0.2
    )
    return _clamp_score(score)


def build_candidate_feature_document(candidate_profile: dict[str, Any]) -> dict[str, Any]:
    career_metrics = _career_metrics(candidate_profile)
    growth_score = calculate_growth_score(candidate_profile)
    stability_score = calculate_stability_score(candidate_profile)
    learning_velocity = calculate_learning_velocity(candidate_profile)
    skill_credibility = calculate_skill_credibility(candidate_profile)
    recruitability_score = calculate_recruitability_score(candidate_profile)

    return {
        "candidate_id": candidate_profile.get("candidate_id"),
        "growth_score": growth_score,
        "stability_score": stability_score,
        "learning_velocity": learning_velocity,
        "skill_credibility": skill_credibility,
        "recruitability_score": recruitability_score,
        **career_metrics,
        "skill_count": len(candidate_profile.get("skills", []) or []),
        "education_count": len(candidate_profile.get("education", []) or []),
        "experience": candidate_profile.get("experience"),
        "source_file": candidate_profile.get("source_file", "candidates.jsonl"),
    }


def build_candidate_feature_documents(candidate_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_candidate_feature_document(profile) for profile in candidate_profiles]
