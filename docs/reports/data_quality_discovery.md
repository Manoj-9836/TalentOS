# Data Quality Discovery — TalentOS

> PPT-ready one-pager. Source material for the "real data science" slide.

## TL;DR

Mid-project, two data-quality findings forced us to redesign our recruitability
model rather than ship a high-R² number built on bad data. We made the call
to publish honest metrics (R² = 0.044 on a clean target) rather than
inflated metrics (R² = 0.999 from a target leak).

---

## Finding 1 — Hidden missing-data sentinel

The `redrob_signals.offer_acceptance_rate` field uses `-1.0` to encode
"no recorded offer history" for **59,554 of 100,000 candidates** (59.6%).
The data dictionary did not call this out.

| Value of `offer_acceptance_rate` | Candidates | % of dataset |
|----------------------------------|------------|--------------|
| -1.0 (sentinel)                  | 59,554     | 59.55%       |
| 0.0                              | 0          | 0.00%        |
| 0.0 < x ≤ 1.0 (real rates)       | 40,446     | 40.45%       |

### Why it matters

Our initial v1 model trained on all 100,000 rows with this field as the
target. R² ≈ 0.025. Diagnosis: the model was learning to detect the
sentinel ("does this candidate have any recorded offers?") rather than
predicting offer behaviour. The aggregate-quality features do not encode
the sentinel cleanly, so the model had little to learn from.

### What we did

- Documented the sentinel in `docs/data_quality_report.md`.
- Compared targets with `backend/ml/recruitability/compare_targets.py`.
- Selected `interview_completion_rate` as the primary target (clean [0,1],
  no sentinels, semantically complementary to the BGE semantic-similarity
  layer).
- Retrained v1 on the cleaned target.

---

## Finding 2 — Target leakage in an expanded experiment

While experimenting with a wider feature set (`recruitability_v2`), we
discovered that **the target column had been included in the feature
matrix**. The model fit the identity; R² ≈ 0.9998 across every target
we tried.

This was not a real signal. It was a string of the form:

> "predict `offer_acceptance_rate` from `offer_acceptance_rate`"

### Why it matters

If we had shipped that number, anyone scanning the repo with even
casual ML literacy would have flagged the leakage. The credibility cost
was higher than the upside of a flashy metric.

### What we did

- Deleted `train_recruitability_v2.py`, `recruitability_service_v2.py`,
  `recruitability_model_v2.pkl`, `recruitability_metrics_v2.json`.
- Re-pointed the production ranking engine to the v1 service.
- Kept the `compare_targets.py` harness as a documented research tool.

---

## Honest numbers, before / after

| Model variant                  | Target                          | R²     | MAE    |
|--------------------------------|---------------------------------|--------|--------|
| v1 (raw, sentinels in target)  | offer_acceptance_rate           | 0.025  | 0.696  |
| v2 (target leaked into feats)  | recruiter_response_rate         | 0.9998 | 0.0015 |
| **v1 (clean target, retrained)** | **interview_completion_rate** | **0.044** | **0.145** |

The honest v1 model explains ~4.4% of variance in interview completion
behaviour from aggregate features alone. That ceiling is consistent with
the narrow distribution of the target (mean 0.62, std 0.17) and the
five-feature aggregate view we use.

---

## What this tells judges

1. We read the data dictionary before training.
2. We caught a hidden sentinel that wasn't documented.
3. We caught a target leak in our own expanded experiment.
4. We chose to publish the lower, honest number rather than the
   flashy, broken one.
5. We kept the recruitability signal as a soft prior inside the
   ranking engine — because semantic similarity (BGE) is the dominant
   signal for "does this candidate fit the role?".

---

## Where the recruitability model sits in the system

```
Rank score = 0.35  · semantic_score          (BGE-large-en-v1.5)
           + 0.25  · recruitability_score    (LightGBM, interview_completion_rate)
           + 0.15  · aggregate_quality       (growth + stability + learning + skill)
           + 0.25  · floor weight on engagement signals
```

The semantic layer carries the most weight because it directly answers
"can this person do the job?". The recruitability model contributes the
"will they engage with the hiring process?" prior — independent,
complementary, and explicitly bounded.

---

## Artefacts

| File                                                   | Purpose                                  |
|--------------------------------------------------------|------------------------------------------|
| `backend/ml/recruitability/train_recruitability.py`    | v1 training (clean target)               |
| `backend/ml/recruitability/compare_targets.py`         | Target-comparison harness                |
| `backend/ml/recruitability/recruitability_metrics.json`| Held-out validation metrics              |
| `backend/ml/recruitability/evaluation_report.md`       | Model card                               |
| `docs/data_quality_report.md`                          | Updated DQ report with sentinel finding  |
| `docs/reports/metrics_report.md`                       | Pipeline metrics report                  |
| `docs/feature_inventory.md`                            | Modelling-input guidance                 |
