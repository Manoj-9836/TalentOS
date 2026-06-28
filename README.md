<p align="center">
  <img src="https://raw.githubusercontent.com/Manoj-9836/TalentOS/refs/heads/main/Architecture/Architecture.png" alt="TalentOS Architecture" width="100%">
</p>

<h1 align="center">TalentOS – AI Recruiting Intelligence System</h1>

<p align="center">
<b>Intelligent Matching • Predictive Hiring • Explainable AI</b>
<br>
<i>Moving beyond keyword matching to Recruitability Intelligence.</i>
</p>

---

## Overview

TalentOS is an AI-powered recruiting intelligence platform designed to rank candidates based on **semantic understanding, career intelligence, and behavioral hiring signals** instead of traditional keyword matching.

Traditional Applicant Tracking Systems (ATS) rely heavily on keyword filters, often overlooking highly qualified candidates whose resumes may not exactly match the job description.

TalentOS addresses this problem using a multi-stage AI pipeline that combines:

- Semantic Matching using **BGE Embeddings**
- Efficient Candidate Retrieval using **FAISS**
- Candidate Intelligence through custom engineered features
- Recruitability Prediction using **LightGBM**
- Explainable AI using **SHAP**

The result is a transparent and scalable candidate ranking system capable of processing over **100,000 candidate profiles**.

---

## Key Features

- Semantic Job–Candidate Matching (BGE Large v1.5)
- Vector Search using FAISS
- Career Intelligence Scoring
- Learning Velocity Analysis
- Stability Analysis
- Skill Credibility Analysis
- Recruitability Prediction
- Explainable AI Rankings (SHAP)
- End-to-End Candidate Ranking Pipeline
- Submission CSV Generator

---

## System Architecture

```
Job Description
        │
        ▼
Content Processing
        │
        ▼
Semantic Embeddings (BGE)
        │
        ▼
FAISS Retrieval
        │
        ▼
Top-K Candidate Selection
        │
        ▼
Feature Engineering

 • Growth Score
 • Stability Score
 • Learning Velocity
 • Skill Credibility

        │
        ▼
Recruitability Prediction (LightGBM)
        │
        ▼
Ranking Engine
        │
        ▼
Explainability Engine (SHAP)
        │
        ▼
Ranked Candidates CSV
```

### Detailed Pipeline Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  candidates.jsonl│     │ job_description │     │  redrob_signals │
│   (100k rows)   │     │     (.docx)     │     │  (behavioural)  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BGE Embedding Layer                          │
│  build_candidate_text() → SentenceTransformer.encode()          │
│  → 1024-d normalized vectors → FAISS IndexFlatIP                │
└────────┬────────────────────────────────────────────┬───────────┘
         │                                             │
         ▼                                             ▼
┌─────────────────┐                         ┌─────────────────────┐
│  Retrieval      │                         │  Recruitability     │
│  (FAISS Top-K)  │                         │  (LightGBM)         │
│  Top-2000       │                         │  Target: interview  │
│  Cosine Sim     │                         │  completion_rate    │
└────────┬────────┘                         └──────────┬──────────┘
         │                                             │
         └──────────────┬──────────────────────────────┘
                         ▼
              ┌──────────────────────────┐
              │   Ranking Engine         │
              │  Weighted Linear Sum     │
              │  6 Components            │
              └───────────┬──────────────┘
                          ▼
              ┌──────────────────────────┐
              │   ranked_candidates.csv  │
              │   (Submission Format)    │
              └──────────────────────────┘
```

---

## Repository Structure

```
TalentOS/

backend/
│
├── api/
├── data/
├── indices/
├── ml/
│   ├── embeddings/
│   ├── retrieval/
│   ├── feature_engineering/
│   ├── recruitability/
│   ├── ranking/
│   └── explainability/
│
├── output/
├── scripts/
│
docs/
│
README.md
```

---

## Technology Stack

### Core ML/AI Stack

| Component | Technology | Version / Purpose |
|---|---|---|
| Embedding Model | BAAI/bge-large-en-v1.5 | 1024-dim normalized embeddings (SOTA for retrieval) |
| Vector Index | FAISS (IndexFlatIP) | Exact inner-product search, 100k vectors in memory |
| Ranking Model | LightGBM Regressor | 200 estimators, early stopping, 5 engineered features |
| Feature Engineering | Custom Python (`profile_builder.py`) | 6 engineered scores: growth, stability, learning_velocity, skill_credibility, recruitability |
| Embedding Generation | SentenceTransformers | Batch size 32, normalized embeddings |
| API Framework | FastAPI + Uvicorn | REST API with Swagger docs |
| Data Processing | pandas, numpy, pyarrow | 100k candidate JSONL processing |

### Infrastructure

| Layer | Technology |
|---|---|
| Data Storage | JSONL (`candidates.jsonl`), FAISS index (`.faiss` + `.npy`), pickle (embeddings) |
| Pipeline Orchestration | Python scripts with argparse orchestration |
| Evaluation | scipy.stats (Pearson/Spearman), sklearn.metrics |
| Explainability | SHAP (`backend/ml/ranking/explainer.py`) |

### Summary Stack

| Category | Technology |
|---|---|
| Backend | FastAPI |
| Embedding Model | BGE Large v1.5 |
| Vector Search | FAISS |
| ML Model | LightGBM |
| Explainability | SHAP |
| Language | Python |
| Data Format | JSONL, CSV, Pickle |

---

## AI Pipeline

### Phase 1 – Data Understanding

**Input Sources**

- Job Description
- Candidate Profiles
- Resume Information
- Redrob Platform Signals

#### Raw Data Schema (100k candidates, `candidates.jsonl`)

| Category | Fields |
|---|---|
| Profile | `candidate_id`, `anonymized_name`, `headline`, `summary`, `location`, `country`, `years_of_experience`, `current_title`, `current_company`, `current_company_size`, `current_industry` |
| Skills | Array of `{name, endorsements, duration_months, proficiency}` |
| Career History | Array of `{company, title, start_date, end_date, duration_months, is_current, industry, company_size, description}` |
| Education | Array of `{institution, degree, field_of_study, tier}` |
| RedRob Signals | `recruiter_response_rate`, `interview_completion_rate`, `offer_acceptance_rate` (-1 sentinel), `profile_views_received_30d`, `search_appearance_30d`, `saved_by_recruiters_30d`, `github_activity_score`, `connection_count`, `endorsements_received`, `notice_period_days`, `expected_salary_range_inr_lpa`, `avg_response_time_hours`, `applications_submitted_30d`, `interview_completion_rate`, `offer_acceptance_rate` |

#### Data Quality Findings (Critical for Judges)

| Issue | Impact | Remediation |
|---|---|---|
| `offer_acceptance_rate` = -1 in 59.6% rows | Sentinel for "no offer history" | Excluded from training; used `interview_completion_rate` instead |
| `skill_assessment_scores` 99.7% missing | Useless for modeling | Excluded from features |
| 0 duplicate `candidate_ids` | Clean identity | No dedup needed |
| All rates bounded [0,1] except `offer_acceptance_rate` | Clean behavioral signals | Validated before training |

---

### Phase 2 – Feature Engineering

Each candidate is transformed into structured intelligence features that capture candidate quality beyond simple keyword matching.

#### Feature Engineering (`profile_builder.py`)

| Feature | Formula | Range | Interpretation |
|---|---|---|---|
| `growth_score` | `skill_count×3 + min(years,20)×2 + views/50` | 0-100 | Career momentum |
| `stability_score` | `avg_job_duration_months` scaled 0-120 → 0-100 | 0-100 | Tenure stability |
| `learning_velocity` | `unique_skills / max(1, years_exp)` scaled 0-10 → 0-100 | 0-100 | Skill acquisition rate |
| `skill_credibility` | `recruiter_response_rate×60 + github_activity×40` scaled 0-100 | 0-100 | External validation |
| `recruitability_score` | `0.5×growth + 0.2×stability + 0.3×credibility` | 0-100 | Composite prior |

---

### Phase 3 – Semantic Matching

**Model**

```
BAAI/bge-large-en-v1.5
```

Every candidate profile is converted into a **1024-dimensional embedding**. The incoming Job Description is embedded into the same semantic space. Similarity is computed using cosine similarity.

#### Retrieval System Parameters

| Parameter | Value |
|---|---|
| Embedding Model | BAAI/bge-large-en-v1.5 (1024-dim) |
| Index Type | FAISS IndexFlatIP (exact inner product) |
| Vector Normalization | L2-normalized (cosine = inner product) |
| Index Size | 100,000 vectors |
| Retrieval Depth (K) | 2,000 candidates (pre-rank) |
| Query | Job description text embedded via same BGE model |
| Latency | ~50ms for top-2000 on CPU |

**Candidate Text Representation (`build_candidate_text.py`)**

Structured text includes: Name, Headline, Summary, Location, Experience (years), Current Role, Industry, Skills (all), Career History (company, title, duration, description), Education (institution, degree, field, tier), RedRob Signals (response rates, github activity, profile views, open_to_work flag).

---

### Phase 4 – Candidate Retrieval

**Vector Store**

```
FAISS IndexFlatIP
```

Instead of comparing against every candidate, TalentOS retrieves only the **Top-K semantically relevant candidates**, dramatically reducing computation while maintaining retrieval quality.

---

### Phase 5 – Recruitability Prediction

**Model**

```
LightGBM Regressor
```

The recruitability model predicts **Interview Completion Rate**, which represents a candidate's likelihood of remaining engaged throughout the hiring process.

#### Training Configuration

```python
LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric="rmse"
)
```

#### Why `interview_completion_rate`? (Target Selection — Critical Decision)

During data exploration we discovered:

- Nearly **60%** of `offer_acceptance_rate` values were sentinel values (`-1`) representing missing history.
- Training directly on this target introduced significant noise.
- `interview_completion_rate` provided a cleaner behavioral signal with no missing sentinels.
- Semantically complementary to BGE: BGE answers *"can they do the job?"* — recruitability answers *"will they stay engaged?"*

Rather than predicting hiring success directly, Recruitability serves as a **behavioral prior** that complements semantic matching.

| Target | MAE | RMSE | R² | Train/Val | Note |
|---|---|---|---|---|---|
| `offer_acceptance_rate` (clean) | 0.1520 | 0.1759 | 0.0139 | 32k/8k | 60% sentinel -1 |
| `recruiter_response_rate` | 0.1837 | 0.2124 | 0.0133 | 80k/20k | Noisy |
| **`interview_completion_rate`** | **0.1445** | **0.1674** | **0.0443** | 80k/20k | Clean [0,1], no sentinels |

#### Training Metrics (Held-out 20%)

| Metric | Value |
|---|---|
| MAE | 0.1445 |
| RMSE | 0.1674 |
| R² | 0.0443 |

#### Feature Importance (Split Count)

| Rank | Feature | Importance |
|---|---|---|
| 1 | growth_score | ~574 |
| 2 | github_activity | ~550 |
| 3 | learning_velocity | ~520 |
| 4 | stability_score | ~518 |
| 5 | skill_credibility | ~500 |

---

### Phase 6 – Ranking Engine

The final ranking combines multiple intelligence signals:

```
Semantic Match
Career Intelligence
Recruitability
Skill Credibility
Growth Score
Learning Velocity
Stability Score
```

#### Weight Configuration (`RankingWeights` dataclass)

| Component | Weight | Source | Normalization |
|---|---|---|---|
| `semantic_score` | 0.35 | BGE cosine similarity | [0,1] |
| `recruitability_score` | 0.25 | LightGBM prediction | [0,1] |
| `growth_score` | 0.15 | profile_builder | [0,100] → /100 |
| `learning_velocity` | 0.10 | profile_builder | [0,100] → /100 |
| `skill_credibility` | 0.10 | profile_builder | [0,100] → /100 |
| `stability_score` | 0.05 | profile_builder | [0,100] → /100 |

#### Final Score Formula

```
final_score = 0.35×semantic + 0.25×recruitability + 0.15×(growth/100)
            + 0.10×(learning/100) + 0.10×(credibility/100) + 0.05×(stability/100)
```

**Design Rationale**

- Semantic similarity (0.35) = Dominant signal for role fit
- Recruitability (0.25) = Soft prior for pipeline engagement
- Growth/Learning/Credibility (0.35 total) = Career trajectory signals
- Stability (0.05) = Minor tenure signal

Each candidate receives a final ranking score.

---

### Phase 7 – Explainability

TalentOS provides transparent rankings using **SHAP** (`backend/ml/ranking/explainer.py`).

Example explanation:

```
Candidate Rank #1

Positive Factors
✔ Strong Semantic Match
✔ High Growth Score
✔ Excellent Learning Velocity

Negative Factors
✖ Moderate Stability
```

This enables recruiters to understand *why* a candidate was recommended.

---

## Evaluation Metrics

TalentOS evaluates each stage independently.

### Retrieval Metrics — Recall@K

| K | Recall@K | Meaning |
|---|---|---|
| 10 | TBD* | Top-10 contains reference candidate |
| 50 | TBD* | Top-50 contains reference candidate |
| 100 | TBD* | Top-100 contains reference candidate |
| 500 | TBD* | Top-500 contains reference candidate |

*Run `python -m backend.scripts.run_evaluation` after building the index to populate.*

### Ranking Quality — Correlation with Proxy Signal

| Metric | Value | Interpretation |
|---|---|---|
| Pearson r | TBD | Linear correlation between rank_score and recruiter_response_rate |
| Spearman ρ | TBD* | Rank correlation (more robust) |
| N candidates | TBD* | Sample size for correlation |

### Recruitability Model Metrics (Training Time)

| Metric | Value | Context |
|---|---|---|
| MAE | 0.1445 | Mean absolute error on interview_completion_rate |
| RMSE | 0.1674 | Root mean squared error |
| R² | 0.0443 | Explains 4.4% variance — honest ceiling for aggregate features |

### Regression Metrics (for the Recruitability Prediction model)

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Correlation Metrics (to validate ranking quality)

- Pearson Correlation
- Spearman Correlation

These measure how well predicted recruitability aligns with observed behavioral hiring signals.

---

## Why These Metrics?

| Metric | Rationale |
|---|---|
| **Recall@K** | The retrieval stage is responsible for selecting the most relevant candidates before ranking. A higher Recall@K ensures relevant candidates are not discarded early. |
| **R²** | Measures how much variance in the behavioral target is explained by the model. |
| **MAE** | Provides the average prediction error in understandable units. |
| **RMSE** | Penalizes larger prediction errors more heavily than MAE. Useful for identifying unstable models. |
| **Pearson Correlation** | Measures linear agreement between predicted and actual behavioral signals. |
| **Spearman Correlation** | Evaluates ranking consistency rather than exact prediction values, making it particularly relevant for candidate ranking. |

---

## Ablation Study Design (For Presentation)

| Configuration | Pearson r | Spearman ρ | Purpose |
|---|---|---|---|
| Full ranking (recruitability=0.25) | TBD | TBD | Baseline |
| Ranking without recruitability | TBD | TBD | Ablation: Is recruitability adding value? |
| Ranking without semantic (BGE) | TBD | TBD | Ablation: Dominant signal check |

Run via: `rank_candidates_from_dicts()` with custom `RankingWeights`.

---

## API & Deployment (FastAPI)

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs` | Upload JD, get `job_id` |
| `POST /api/rank` | Run full pipeline for `job_id` → top-K ranked candidates |
| `GET /api/candidates` | Paginated candidate search with filters |
| `GET /api/candidates/{id}` | Full candidate profile + signals |
| `GET /api/candidate/{id}` | SHAP explanation for candidate |
| `POST /api/compare` | Side-by-side candidate comparison |
| `GET /api/analytics/dashboard` | Aggregate stats |

---

## Installation

### Create Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate (Windows)

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

## Running TalentOS

### Step 1 – Generate Candidate Embeddings

Smoke Test

```bash
python -m backend.ml.embeddings.generate_embeddings --sample
```

Full Dataset

```bash
python -m backend.ml.embeddings.generate_embeddings
```

Output

```
backend/data/embeddings/candidate_embeddings.jsonl
```

### Step 2 – Build FAISS Index

```bash
python -m backend.ml.retrieval.build_index
```

Output

```
backend/indices/candidate_index.faiss
backend/indices/candidate_ids.npy
```

### Step 3 – Encode Job Description

```bash
python -m backend.scripts.build_embeddings
```

Output

```
backend/data/embeddings/job_embeddings.pkl
```

### Step 4 – Run Complete Pipeline

```bash
python -m backend.scripts.run_pipeline
```

Skip rebuilding embeddings

```bash
python -m backend.scripts.run_pipeline --skip-embeddings --skip-index
```

Custom output

```bash
python -m backend.scripts.run_pipeline --output backend/output/ranked_candidates.csv --top-k 100
```

Output

```
backend/output/ranked_candidates.csv
```

### Step 5 – Run Evaluation

```bash
python -m backend.scripts.run_evaluation
```

Output

```
docs/reports/metrics_report.md
```

---

## Quick Demo (CPU)

```bash
python -m backend.ml.embeddings.generate_embeddings --sample

python -m backend.ml.retrieval.build_index

python -m backend.scripts.build_embeddings --sample

python -m backend.scripts.run_pipeline --sample --top-k 20
```

Generates a ranked CSV in under 5 minutes using the sample dataset.

---

## Pipeline Execution (One-Command Reference)

**First run (full build ~1.5h CPU / 10min GPU)**

```bash
pip install -r backend/requirements.txt
python -m backend.ml.embeddings.generate_embeddings    # Step 1: Embed 100k candidates
python -m backend.ml.retrieval.build_index              # Step 2: FAISS index
python -m backend.scripts.build_embeddings              # Step 3: Job embedding
python -m backend.scripts.run_pipeline                  # Step 4-6: Rank → CSV
```

**Subsequent runs (reuse artifacts ~30 sec)**

```bash
python -m backend.scripts.run_pipeline --skip-embeddings --skip-index
```

**Evaluation**

```bash
python -m backend.scripts.run_evaluation
```

### Outputs

| File | Path |
|---|---|
| Submission file (candidate_id, rank, score, reasoning) | `backend/output/ranked_candidates.csv` |
| Evaluation metrics (auto-generated) | `docs/reports/metrics_report.md` |
| Trained LightGBM model | `backend/ml/recruitability/recruitability_model.pkl` |
| FAISS index | `backend/indices/candidate_index.faiss` |
| Job description embedding | `backend/data/embeddings/job_embeddings.pkl` |

---

## Submission Format Compliance

`ranked_candidates.csv` matches the required schema:

```
candidate_id,rank,score,reasoning
CAND_000123,1,0.8472,"Strong match: 8/10 core skills (Python, FastAPI, PostgreSQL, AWS). 7 years experience. High growth trajectory (growth_score=82). Recruiter response rate 0.73."
```

---

## Key Innovations to Highlight

| Innovation | Why It Matters |
|---|---|
| BGE + FAISS Exact Search | No ANN approximation — exact cosine similarity at 100k scale |
| Target Selection Discipline | Rejected `offer_acceptance_rate` (59.6% sentinel) for clean `interview_completion_rate` |
| Soft Prior Design | Recruitability weighted 0.25, not dominant — semantic similarity leads |
| Structured Candidate Text | Rich text representation (skills, history, signals) vs naive concatenation |
| SHAP Explainability | Per-candidate feature contributions in API |
| Full Reproducible Pipeline | Single command from raw data to submission CSV |
| Honest Metrics Reporting | R²=0.044 reported transparently with interpretation |

---

## Metrics Summary for Judges (One Slide)

| Metric Category | Key Numbers |
|---|---|
| Dataset Scale | 100,000 candidates, ~95 raw fields each |
| Embedding Dim | 1024 (BGE-large) |
| Retrieval | FAISS IndexFlatIP, Top-2000 exact search |
| Ranking Components | 6 weighted signals |
| Recruitability Model | LightGBM, 5 features, R²=0.044 (honest) |
| Target Signal | interview_completion_rate (clean [0,1]) |
| Pipeline | 6 steps, single-command execution |
| API | FastAPI, 8 endpoints, Swagger docs |
| Submission | CSV with rank, score, human-readable reasoning |

---

## How the Model Ranks Candidates (End-to-End)

1. **Job Description** → BGE embedding (1024-d)
2. **FAISS Search** → Top-2000 candidates by cosine similarity
3. For each retrieved candidate:
   - Load full profile + signals
   - Compute 4 profile scores (growth, stability, learning_velocity, skill_credibility)
   - Predict recruitability via LightGBM
   - Get semantic_score from FAISS
4. **Ranking Engine** → Weighted sum of 6 components
5. **Sort** by final_score → Top-K → Generate reasoning string
6. **Output** → `ranked_candidates.csv` (submission format)

---

## Common Issues

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| FAISS index not found | Run `python -m backend.ml.retrieval.build_index` |
| `job_embeddings.pkl` missing | Run `python -m backend.scripts.build_embeddings` |
| Slow embedding generation | Use GPU or run with `--sample` |

---

## Future Improvements

- Cross-Encoder Re-ranking
- Online Learning from Recruiter Feedback
- Multi-Language Resume Support
- Dynamic Ranking Weight Optimization
- Bias & Fairness Monitoring

---

## Authors

**TalentOS**
AI Recruiting Intelligence System

Built for the **Data & AI Challenge Hackathon 2026**.

*"Because great talent shouldn't be missed simply because the right keyword wasn't on a resume."*
