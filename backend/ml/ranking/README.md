# Phase 9 — Ranking Engine

The deliverable for this phase is `ranking_engine.py` in this directory.

## What it does

Combines six per-candidate signals into a single `candidate_rank_score`:

| Component            | Default weight | Source                                                  |
|----------------------|----------------|---------------------------------------------------------|
| semantic_score       | 0.35           | BGE cosine similarity (Phase 6 + Phase 7)               |
| recruitability_score | 0.25           | LightGBM regressor (Phase 8)                            |
| growth_score         | 0.15           | `profile_builder.calculate_growth_score`                |
| learning_velocity    | 0.10           | `profile_builder.calculate_learning_velocity`           |
| skill_credibility    | 0.10           | `profile_builder.calculate_skill_credibility`           |
| stability_score      | 0.05           | `profile_builder.calculate_stability_score`             |

Weights are configurable via `RankingWeights(...)`; they must sum to 1.0.

## API

```python
from backend.ml.ranking.ranking_engine import (
    rank_candidates_from_dicts,
    rank_candidates,
    compute_rank_score,
    RankingWeights,
    DEFAULT_WEIGHTS,
)

# From dict inputs (as produced by the rest of the pipeline):
ranked = rank_candidates_from_dicts(candidates, top_k=20)

# With custom weights:
custom = RankingWeights(semantic_score=0.5, recruitability_score=0.2,
                        growth_score=0.1, learning_velocity=0.1,
                        skill_credibility=0.05, stability_score=0.05)
ranked = rank_candidates_from_dicts(candidates, weights=custom, top_k=20)
```

## Output schema

Each ranked candidate is a dict:

```python
{
    "candidate_id": str,
    "candidate_rank_score": float,
    "semantic_score": float,
    "recruitability_prediction": float,
    "growth_score": float,
    "learning_velocity": float,
    "skill_credibility": float,
    "stability_score": float,
    "component_scores": dict,  # per-feature weighted contributions
}
```
