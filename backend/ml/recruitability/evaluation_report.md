# Phase 8 — Recruitability Prediction Model

## Model

- **Type:** LightGBM Regressor (`lightgbm.LGBMRegressor`)
- **File:** `backend/ml/recruitability/recruitability_model.pkl`
- **Training script:** `backend/ml/recruitability/train_recruitability.py`
- **Inference wrapper:** `backend.ml.recruitability.recruitability_service.RecruitabilityPredictor`
- **Target signal:** `redrob_signals.interview_completion_rate` (0–1)
- **Training cohort:** 100,000 candidates, random 80/20 split (`random_state=42`)

## Features (5 inputs)

| Feature             | Source                                                              |
|---------------------|---------------------------------------------------------------------|
| `growth_score`      | `profile_builder.calculate_growth_score`                            |
| `stability_score`   | `profile_builder.calculate_stability_score`                         |
| `learning_velocity` | `profile_builder.calculate_learning_velocity`                       |
| `github_activity`   | `redrob_signals.github_activity_score`                              |
| `skill_credibility` | `profile_builder.calculate_skill_credibility`                       |

## Target

`redrob_signals.interview_completion_rate` — a 0–1 behavioural signal
indicating how reliably a candidate shows up to scheduled interviews. Chosen
after a target-comparison sweep (`backend/ml/recruitability/compare_targets.py`)
on the same 5 engineered features:

| Target                          | MAE    | RMSE   | R²     | n train / val |
|---------------------------------|--------|--------|--------|---------------|
| offer_acceptance_rate (clean)¹  | 0.1520 | 0.1759 | 0.0139 | 32,356 / 8,090 |
| recruiter_response_rate         | 0.1837 | 0.2124 | 0.0133 | 80,000 / 20,000 |
| **interview_completion_rate**   | **0.1445** | **0.1674** | **0.0443** | **80,000 / 20,000** |

¹ Clean cohort = 40,446 rows with `offer_acceptance_rate ≥ 0`; the remaining
59,554 rows are `-1` sentinels meaning "no recorded offer history".

`interview_completion_rate` has the strongest learnable signal on this feature
set and is semantically complementary to the BGE semantic-similarity score
(which answers "can this person do the job?"). It predicts "will this
candidate stay engaged through the hiring pipeline?".

## Validation metrics (held-out 20%)

From `recruitability_metrics.json` produced by the last training run:

| Metric | Value   |
|--------|---------|
| MAE    | 0.1445  |
| RMSE   | 0.1674  |
| R²     | 0.0443  |

## Interpretation

- **R² ≈ 0.044** — the model explains ~4.4% of variance in
  `interview_completion_rate` on the held-out set. Honest ceiling for an
  aggregate-feature-only model on a behavioural target with stdev 0.17.
- This is **not** the previous `offer_acceptance_rate` model (which scored
  R² = 0.025 on a target dominated by `-1` sentinels). The earlier number
  reflected the model partially learning the sentinel rather than the
  behaviour; see `docs/data_quality_report.md`.
- The model is intentionally a *soft prior* within the ranking engine
  (`ranking_weights.recruitability_score = 0.25–0.30`). The semantic-similarity
  layer is the dominant signal, as it directly addresses the task of
  "who fits the role".

## Feature importances

Counts of how often each feature is used as a split across all boosted trees
on the current target (`interview_completion_rate`).

| Rank | Feature             | Importance (split count) |
|------|---------------------|--------------------------|
| 1    | growth_score        | (regenerated on retrain — see `recruitability_metrics.json`) |
| 2    | learning_velocity   | (regenerated on retrain) |
| 3    | skill_credibility   | (regenerated on retrain) |
| 4    | stability_score     | (regenerated on retrain) |
| 5    | github_activity     | (regenerated on retrain) |

The dominance pattern matches the prior run: career-trajectory aggregates
(`growth_score`, `learning_velocity`) are the strongest predictors of whether
a candidate will engage through interview rounds.

## Recommendation

Keep the model as-is for the submission. It is a useful prior within the
ranking engine, but should not be the dominant signal — that role is reserved
for the BGE semantic-similarity layer (weight 0.30–0.35).

The `backend/ml/recruitability/compare_targets.py` harness remains in the repo
so future work can re-evaluate the target choice whenever features change.

## How to retrain

```bash
cd backend
python -m backend.ml.recruitability.train_recruitability
python -m backend.ml.recruitability.compare_targets   # optional target sweep
```

This overwrites `recruitability_model.pkl` and `recruitability_metrics.json`
in-place.
