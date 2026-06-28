from __future__ import annotations

from typing import Any, Dict, List


def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """Convert a candidate profile into a structured text representation for embedding."""
    parts: List[str] = []

    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    career_history = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    name = profile.get("anonymized_name")
    if name:
        parts.append(f"Name: {name}")

    headline = profile.get("headline")
    if headline:
        parts.append(f"Headline: {headline}")

    summary = profile.get("summary")
    if summary:
        parts.append(f"Summary: {summary}")

    location = profile.get("location")
    country = profile.get("country")
    if location or country:
        loc_parts = [p for p in [location, country] if p]
        parts.append(f"Location: {', '.join(loc_parts)}")

    years_exp = profile.get("years_of_experience")
    if years_exp is not None:
        parts.append(f"Experience: {years_exp} years")

    current_title = profile.get("current_title")
    current_company = profile.get("current_company")
    if current_title or current_company:
        title_parts = [p for p in [current_title, current_company] if p]
        parts.append(f"Current Role: {' at '.join(title_parts)}")

    current_industry = profile.get("current_industry")
    if current_industry:
        parts.append(f"Industry: {current_industry}")

    if skills:
        skill_names = []
        for s in skills:
            if isinstance(s, str):
                skill_names.append(s)
            elif isinstance(s, dict):
                name = s.get("name") or s.get("skill")
                if name:
                    skill_names.append(name)
        if skill_names:
            parts.append(f"Skills: {', '.join(skill_names)}")

    if career_history:
        exp_lines = []
        for job in career_history:
            if isinstance(job, dict):
                company = job.get("company")
                title = job.get("title")
                duration = job.get("duration_months")
                desc = job.get("description")
                is_current = job.get("is_current")
                job_parts = []
                if company:
                    job_parts.append(f"Company: {company}")
                if title:
                    job_parts.append(f"Title: {title}")
                if duration:
                    job_parts.append(f"Duration: {duration} months")
                if is_current:
                    job_parts.append("Current: Yes")
                if desc:
                    job_parts.append(f"Description: {desc[:200]}")
                exp_lines.append(" | ".join(job_parts))
        if exp_lines:
            parts.append("Career History: " + " ; ".join(exp_lines))

    if education:
        edu_lines = []
        for edu in education:
            if isinstance(edu, dict):
                inst = edu.get("institution")
                degree = edu.get("degree")
                field = edu.get("field_of_study")
                tier = edu.get("tier")
                edu_parts = []
                if inst:
                    edu_parts.append(f"Institution: {inst}")
                if degree:
                    edu_parts.append(f"Degree: {degree}")
                if field:
                    edu_parts.append(f"Field: {field}")
                if tier:
                    edu_parts.append(f"Tier: {tier}")
                edu_lines.append(" | ".join(edu_parts))
        if edu_lines:
            parts.append("Education: " + " ; ".join(edu_lines))

    recruiter_rr = signals.get("recruiter_response_rate")
    interview_cr = signals.get("interview_completion_rate")
    offer_ar = signals.get("offer_acceptance_rate")
    profile_views = signals.get("profile_views_received_30d")
    github_score = signals.get("github_activity_score")
    open_to_work = signals.get("open_to_work_flag")

    signal_parts = []
    if recruiter_rr is not None and recruiter_rr >= 0:
        signal_parts.append(f"Recruiter Response Rate: {recruiter_rr:.2f}")
    if interview_cr is not None and interview_cr >= 0:
        signal_parts.append(f"Interview Completion Rate: {interview_cr:.2f}")
    if offer_ar is not None and offer_ar >= 0:
        signal_parts.append(f"Offer Acceptance Rate: {offer_ar:.2f}")
    if profile_views is not None:
        signal_parts.append(f"Profile Views (30d): {profile_views}")
    if github_score is not None and github_score >= 0:
        signal_parts.append(f"GitHub Activity: {github_score}")
    if open_to_work is not None:
        signal_parts.append(f"Open to Work: {open_to_work}")

    if signal_parts:
        parts.append("Signals: " + " | ".join(signal_parts))

    return "\n".join(parts)


def build_job_text(job: Dict[str, Any]) -> str:
    """Convert a job description into a structured text representation for embedding."""
    parts: List[str] = []

    title = job.get("title")
    company = job.get("company")
    industry = job.get("industry")
    location = job.get("location")
    company_size = job.get("company_size")
    description = job.get("description")

    if title:
        parts.append(f"Job Title: {title}")
    if company:
        parts.append(f"Company: {company}")
    if industry:
        parts.append(f"Industry: {industry}")
    if location:
        parts.append(f"Location: {location}")
    if company_size:
        parts.append(f"Company Size: {company_size}")
    if description:
        parts.append(f"Description: {description}")

    return "\n".join(parts)