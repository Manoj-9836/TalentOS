"""
Full Ranking Engine - End-to-End Candidate Ranking System for Hackathon.

This engine combines:
1. Semantic matching using BGE embeddings (Fitzpatrick et al. scoring)
2. Recruitability prediction (recruiter_response_rate based)
3. Aggregate scores (growth, stability, learning velocity, skill credibility)

Scoring weights:
- Semantic match: 35%
- Recruitability prediction: 30%
- Aggregate scores: 15% (growth 5%, stability 5%, learning velocity 2.5%, skill credibility 2.5%)

Total: 100%
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ml.embeddings.embedding_service import (
    encode_text,
    get_model,
    cosine_similarity,
)
from ml.feature_engineering.profile_builder import (
    calculate_growth_score,
    calculate_stability_score,
    calculate_learning_velocity,
    calculate_skill_credibility,
)
from ml.recruitability.recruitability_service import RecruitabilityPredictor, get_recruitability_predictor


class FullRankingEngine:
    """
    Complete ranking engine combining semantic matching and recruitability prediction.

    Designed for the Redrob Intelligent Candidate Discovery & Ranking Challenge.
    """

    def __init__(
        self,
        recruitability_predictor: RecruitabilityPredictor | None = None,
        model_name: str = "BAAI/bge-large-en-v1.5",
    ):
        self.predictor = recruitability_predictor or get_recruitability_predictor()
        self.model = get_model(model_name)
        self.embedding_cache: dict[str, Any] = {}

    def build_candidate_text(self, candidate: dict[str, Any]) -> str:
        """Build text representation of candidate for embedding."""
        profile = candidate.get("profile", {})
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        summary = profile.get("summary", "")
        headline = profile.get("headline", "")
        current_title = profile.get("current_title", "")

        skill_names = ", ".join([s.get("name", "") for s in skills[:20]])
        recent_exp = career[0] if career else {}
        current_role = f"{recent_exp.get('title', '')} at {recent_exp.get('company', '')}"

        return f"{headline}. {summary} Current: {current_role}. Skills: {skill_names}."

    def build_job_text(self, job: dict[str, Any]) -> str:
        """Build text representation of job description for embedding."""
        title = job.get("title", "")
        description = job.get("description", "")
        company = job.get("company", "")
        industry = job.get("industry", "")
        location = job.get("location", "")

        return f"{title} at {company} ({industry}). Location: {location}. {description}"

    def compute_semantic_similarity(
        self,
        candidate: dict[str, Any],
        job_description: dict[str, Any],
    ) -> float:
        """Compute cosine similarity between candidate and job embeddings."""
        candidate_id = candidate.get("candidate_id", "")
        job_id = job_description.get("job_id", "unknown")

        # Check cache
        cache_key = (candidate_id, job_id)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Encode texts
        candidate_text = self.build_candidate_text(candidate)
        job_text = self.build_job_text(job_description)

        candidate_emb = encode_text(candidate_text, self.model)
        job_emb = encode_text(job_text, self.model)

        similarity = cosine_similarity(candidate_emb, job_emb)

        # Cache for reuse
        self.embedding_cache[cache_key] = similarity

        return similarity

    def compute_recruitability_score(self, candidate: dict[str, Any]) -> float:
        """Compute recruitability score (0-1) from the v1 LightGBM model."""
        return float(self.predictor.predict(candidate))

    def compute_aggregate_scores(self, candidate: dict[str, Any]) -> dict[str, float]:
        """Compute all aggregate scores."""
        profile = {
            "profile": candidate.get("profile", {}),
            "experience": candidate.get("profile", {}).get("years_of_experience"),
            "skills": candidate.get("skills", []),
            "career_history": candidate.get("career_history", []),
        }

        return {
            "growth_score": calculate_growth_score(profile),
            "stability_score": calculate_stability_score(profile),
            "learning_velocity": calculate_learning_velocity(profile),
            "skill_credibility": calculate_skill_credibility(profile),
        }

    def rank_candidate(
        self,
        candidate: dict[str, Any],
        job_description: dict[str, Any],
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Rank a single candidate against a job description.

        Returns complete ranking info with scores and reasoning.
        """
        if weights is None:
            weights = {
                "semantic": 0.35,
                "recruitability": 0.30,
                "aggregate": 0.15,
                "growth": 0.05,
                "stability": 0.05,
                "learning": 0.025,
                "skill": 0.025,
            }

        # Compute scores
        semantic_score = self.compute_semantic_similarity(candidate, job_description)
        recruitability_score = self.compute_recruitability_score(candidate)
        agg_scores = self.compute_aggregate_scores(candidate)

        # Weighted combination
        final_score = (
            semantic_score * weights["semantic"] +
            recruitability_score * weights["recruitability"] +
            (
                agg_scores["growth_score"] * weights["growth"] +
                agg_scores["stability_score"] * weights["stability"] +
                agg_scores["learning_velocity"] * weights["learning"] +
                agg_scores["skill_credibility"] * weights["skill"]
            ) / 100.0 * weights["aggregate"]
        )

        # Build reasoning
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        name = profile.get("anonymized_name", "Candidate")
        title = profile.get("current_title", "Professional")
        experience = profile.get("years_of_experience", 0)
        skills = candidate.get("skills", [])
        ai_keywords = ["nlp", "machine learning", "deep learning", "ai", "ml", "llm", "python", "tensorflow", "pytorch", "scikit-learn", "nlp"]
        ai_skills = [s["name"] for s in skills if any(kw in s["name"].lower() for kw in ai_keywords)]
        ai_count = len(ai_skills)

        reasoning_parts = [
            f"{name} ({experience:.1f} yrs)",
            f"{title}",
            f"{ai_count} AI skills",
            f"recruitability: {recruitability_score:.2f}",
            f"semantic: {semantic_score:.2f}",
        ]

        return {
            "candidate_id": candidate.get("candidate_id"),
            "rank_score": round(final_score, 4),
            "semantic_score": round(semantic_score, 4),
            "recruitability_score": round(recruitability_score, 4),
            "growth_score": agg_scores["growth_score"],
            "stability_score": agg_scores["stability_score"],
            "learning_velocity": agg_scores["learning_velocity"],
            "skill_credibility": agg_scores["skill_credibility"],
            "reasoning": ". ".join(reasoning_parts),
            "profile": profile,
            "signals": signals,
        }

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        job_description: dict[str, Any],
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Rank multiple candidates against a job description."""
        results = [self.rank_candidate(c, job_description) for c in candidates]
        results.sort(key=lambda x: x["rank_score"], reverse=True)
        return results[:top_k]

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.embedding_cache.clear()


def create_submission(
    ranked_candidates: list[dict[str, Any]],
    output_path: Path | str = None,
) -> list[dict[str, Any]]:
    """
    Create submission file from ranked candidates.

    Output format:
    - candidate_id: string
    - rank: integer (1-indexed)
    - score: float (0-1)
    - reasoning: string
    """
    submission = []
    for i, cand in enumerate(ranked_candidates, 1):
        submission.append({
            "candidate_id": cand["candidate_id"],
            "rank": i,
            "score": round(cand["rank_score"], 4),
            "reasoning": cand["reasoning"],
        })

    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump(submission, f, indent=2)
        print(f"Saved submission to {output_path}")

    return submission


def get_full_ranking_engine() -> FullRankingEngine:
    """Get or create singleton ranking engine."""
    return FullRankingEngine()


if __name__ == "__main__":
    # Test the engine
    print("Testing FullRankingEngine...")

    engine = get_full_ranking_engine()

    # Load some test candidates
    import json
    with open("backend/data/raw/candidates.jsonl") as f:
        candidates = [json.loads(line) for line in f][:10]

    # Create a test job
    test_job = {
        "job_id": "JOB_001",
        "title": "AI Engineer",
        "description": "Looking for AI/ML engineers with Python, TensorFlow, and NLP experience.",
        "company": "TechCorp",
        "industry": "Technology",
        "location": "Remote",
    }

    # Rank them
    results = engine.rank_candidates(candidates, test_job, top_k=5)

    print("\nTop 5 Ranked Candidates:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['candidate_id']} - Score: {r['rank_score']:.4f}")
        print(f"   Reasoning: {r['reasoning']}")

    print("\nEngine test complete!")
