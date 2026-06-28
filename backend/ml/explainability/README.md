# Phase 10 — Explainability Layer

The deliverable for this phase is `explainability_engine.py` in this directory.

## What it does

Produces a SHAP-based explanation for each ranked candidate, returning:

- `positive_factors` — human-readable strings (e.g. `"+ Strong Semantic Match"`)
- `negative_factors` — same shape, the weak points
- `feature_contributions` — raw SHAP values per feature
- `formatted_explanation` — multi-line prose block ready for display
- `candidate_rank_score` — copied from the ranking engine for convenience

## API

```python
from backend.ml.explainability.explainability_engine import (
    explain_candidates_from_dicts,
    explain_candidate,
    format_explanation,
    ExplainabilityResult,
)

explanations = explain_candidates_from_dicts(candidates)  # list[dict]
for exp in explanations[:3]:
    print(format_explanation(exp))
```

## Output example

```
Candidate CAND_0004989 - Rank Score: 0.945

Positive Factors:
  + Strong Semantic Match
  + Strong Growth
  + Strong Recruitability

Negative Factors:
  - Low Stability
```

## How SHAP is invoked

The engine wraps `shap.Explainer` around a thin pure-numpy "model" that mirrors the weighted-sum scoring from `ranking_engine`. Because the ranker is closed-form, SHAP can attribute the final score exactly to each component.

A synthetic background sample is generated for SHAP when no real background distribution is provided — see `explainability_engine.explain_candidate`.
