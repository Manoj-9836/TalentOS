from __future__ import annotations

from typing import Any, Dict, List


def build_profile(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Merge candidate record into a normalized profile document.

    Output shape:
    {
      "candidate_id": ...,
      "skills": [...],
      "experience": <years_of_experience or derived int>,
      "education": [...],
      "career_history": [...],
      "signals": {...}
    }
    """
    profile = candidate.get("profile", {}) or {}

    years = profile.get("years_of_experience")
    try:
        years_of_experience = int(float(years)) if years is not None else None
    except Exception:
        years_of_experience = None

    skills = candidate.get("skills") or []
    # if skills are objects, extract names
    skills_list: List[str] = []
    for s in skills:
        if isinstance(s, str):
            skills_list.append(s)
        elif isinstance(s, dict):
            name = s.get("name") or s.get("skill")
            if name:
                skills_list.append(name)

    education = candidate.get("education") or []
    career_history = candidate.get("career_history") or []
    signals = candidate.get("redrob_signals") or {}

    return {
        "candidate_id": candidate.get("candidate_id"),
        "skills": skills_list,
        "experience": years_of_experience,
        "education": education,
        "career_history": career_history,
        "signals": signals,
        "skill_count": len(skills_list),
        "career_history_count": len(career_history),
    }


def _scale(value: float, src_min: float, src_max: float) -> int:
    """Scale a value linearly to 0-100 and clamp."""
    if value is None:
        return 0
    try:
        v = float(value)
    except Exception:
        return 0
    if src_max == src_min:
        return 0
    frac = (v - src_min) / (src_max - src_min)
    return max(0, min(100, int(round(frac * 100))))


def calculate_growth_score(profile: Dict[str, Any]) -> int:
    # growth based on skill count, years, and profile_views
    skill_count = profile.get("skill_count", 0) or 0
    years = profile.get("experience") or 0
    views = profile.get("signals", {}).get("profile_views_received_30d", 0) or 0
    score = skill_count * 3 + min(years, 20) * 2 + (views / 50)
    return _scale(score, 0, 200)


def calculate_stability_score(profile: Dict[str, Any]) -> int:
    # stability approximated by average job duration if available
    ch = profile.get("career_history", []) or []
    durations = []
    for job in ch:
        d = job.get("duration_months") if isinstance(job, dict) else None
        try:
            if d is not None:
                durations.append(float(d))
        except Exception:
            continue
    if durations:
        avg_months = sum(durations) / len(durations)
    else:
        # fallback: use years_of_experience as proxy
        avg_months = (profile.get("experience") or 0) * 12 / max(1, (profile.get("career_history_count") or 1))
    # map avg_months (0..120+) to 0..100
    return _scale(avg_months, 0, 120)


def calculate_learning_velocity(profile: Dict[str, Any]) -> int:
    # distinct skills per year
    years = profile.get("experience") or 0
    unique_skills = len(set(profile.get("skills", [])))
    per_year = unique_skills / max(1.0, float(years) if years else 1.0)
    # reasonable range 0..10 skills/year
    return _scale(per_year, 0, 10)


def calculate_skill_credibility(profile: Dict[str, Any]) -> int:
    # proxy using recruiter response rate and github activity
    signals = profile.get("signals", {}) or {}
    rr = signals.get("recruiter_response_rate")
    gh = signals.get("github_activity_score")
    try:
        rr_v = float(rr) if rr is not None else 0
    except Exception:
        rr_v = 0
    try:
        gh_v = float(gh) if gh is not None else 0
    except Exception:
        gh_v = 0
    score = rr_v * 60 + gh_v * 40
    return _scale(score, 0, 100)


def calculate_recruitability_score(profile: Dict[str, Any]) -> int:
    # composite of growth, stability and credibility
    g = calculate_growth_score(profile)
    s = calculate_stability_score(profile)
    c = calculate_skill_credibility(profile)
    # weighted sum
    composite = 0.5 * g + 0.2 * s + 0.3 * c
    return int(round(max(0, min(100, composite))))


def build_profile_and_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    profile = build_profile(candidate)
    features = {
        "candidate_id": profile.get("candidate_id"),
        "growth_score": calculate_growth_score(profile),
        "stability_score": calculate_stability_score(profile),
        "learning_velocity": calculate_learning_velocity(profile),
        "skill_credibility": calculate_skill_credibility(profile),
        "recruitability_score": calculate_recruitability_score(profile),
        "source": "profile_builder_v1",
    }
    return {"profile": profile, "features": features}
