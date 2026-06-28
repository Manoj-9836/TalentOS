# TalentOS Pipeline API

The TalentOS pipeline API exposes the candidate ranking pipeline over HTTP for validation and integration. As of Phase 5, **all endpoints are unauthenticated** — there is no `/api/auth/*` surface.

- **Base URL:** `http://localhost:8000` (default)
- **OpenAPI spec:** `GET /openapi.json`
- **Interactive docs:** `GET /docs` (Swagger UI), `GET /redoc`
- **Content type:** `application/json` for all request/response bodies

---

## Sections

1. [Health](#health)
2. [Jobs](#jobs)
3. [Ranking](#ranking)
4. [Candidates](#candidates)
5. [Analytics](#analytics)
6. [Compare](#compare)

---

## Health

### `GET /health`

Liveness probe.

**Auth:** none
**Status codes:** `200`

**Response**

```json
{ "status": "healthy" }
```

**cURL**

```bash
curl http://localhost:8000/health
```

---

## Jobs

### `POST /api/jobs`

Parse a job description and register it in the in-memory job store. Returns a `job_id` used by the ranking endpoints.

**Auth:** none
**Status codes:** `200`, `422` (validation error)

**Request body**

```ts
{
  job_description: string;       // required, min 50 chars
  job_title?: string;            // optional
  company?: string;              // optional
}
```

**Response**

```ts
{
  job_id: string;
  job_title: string;
  company: string | null;
  parsed_requirements: object;   // structured requirements from job_parser
  message: string;
}
```

**cURL**

```bash
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "We are looking for a Senior Backend Engineer with 5+ years of experience in Python, FastAPI, PostgreSQL, and AWS...",
    "job_title": "Senior Backend Engineer",
    "company": "TechCorp"
  }'
```

---

### `GET /api/jobs`

List all jobs uploaded in this server session.

**Auth:** none
**Status codes:** `200`

**Response:** array of `JobResponse`.

---

### `GET /api/jobs/{job_id}`

Fetch a single job by id.

**Auth:** none
**Status codes:** `200`, `404`

---

### `GET /api/jobs/stats/overview`

Aggregate job stats.

**Auth:** none
**Status codes:** `200`

**Response**

```json
{
  "total_jobs": 3,
  "total_rankings": 3,
  "active_jobs": 3
}
```

---

## Ranking

### `POST /api/rank`

Run the full retrieval → ranking pipeline against an uploaded job.

**Auth:** none
**Status codes:** `200`, `404` (job not found)

**Request body**

```ts
{
  job_id: string;
  top_k?: number;                // default 20, range 1–100
  weights?: {                    // optional override of default weights
    semantic_score?: number;
    recruitability_score?: number;
    growth_score?: number;
    learning_velocity?: number;
    skill_credibility?: number;
    stability_score?: number;
  };
}
```

Default weights (must sum to 1.0):

| Component            | Weight |
|----------------------|--------|
| semantic_score       | 0.35   |
| recruitability_score | 0.25   |
| growth_score         | 0.15   |
| learning_velocity    | 0.10   |
| skill_credibility    | 0.10   |
| stability_score      | 0.05   |

**Response**

```ts
{
  job_id: string;
  total_candidates: number;
  candidates: Array<{
    candidate_id: string;
    candidate_rank_score: number;
    semantic_score: number;
    recruitability_prediction: number;
    growth_score: number;
    learning_velocity: number;
    skill_credibility: number;
    stability_score: number;
    component_scores: object;
  }>;
}
```

**cURL**

```bash
curl -X POST http://localhost:8000/api/rank \
  -H "Content-Type: application/json" \
  -d '{"job_id": "<job_id>", "top_k": 10}'
```

---

### `POST /api/rankings/rank/{job_id}`

Convenience wrapper for `POST /api/rank` where `job_id` is in the URL.

**Auth:** none
**Status codes:** `200`, `404`

---

### `GET /api/rankings/job/{job_id}`

Retrieve the most recent ranking result for a job (in-memory only; cleared on server restart).

**Auth:** none
**Status codes:** `200`, `404`

---

### `GET /api/rankings/history`

List jobs that have been ranked in this session.

**Auth:** none
**Status codes:** `200`

---

## Candidates

### `GET /api/candidates`

Paginated list of candidates with optional filters.

**Auth:** none
**Status codes:** `200`

**Query parameters**

| Param            | Type   | Default | Notes                                          |
|------------------|--------|---------|------------------------------------------------|
| limit            | int    | 20      | Clamped to 1–100                               |
| offset           | int    | 0       |                                                |
| search           | string | —       | Matches name, title, headline, or skills       |
| skill            | string | —       | Exact skill match                              |
| industry         | string | —       | Exact industry match                           |
| location         | string | —       | Substring match on `location` or `country`     |
| min_experience   | float  | —       | Minimum `years_of_experience`                  |
| sort_by          | enum   | —       | `experience_desc` or `name_asc`                |

**Response**

```ts
{
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  candidates: Array<{
    candidate_id: string;
    profile: object;
    skills: Array<{name: string}>;
    career_history: object[];
  }>;
}
```

---

### `GET /api/candidates/count`

Count of candidates matching the same filter set as the list endpoint.

**Auth:** none
**Status codes:** `200`

---

### `GET /api/candidates/{candidate_id}`

Full detail for one candidate, including normalized skills, career history, education, certifications, languages, redrob signals, and a computed recruitability score.

**Auth:** none
**Status codes:** `200`, `404`

---

### `GET /api/candidates/{candidate_id}/intelligence`

Intelligence view of a candidate: predicted recruitability, contributing factors, skill match, experience breakdown.

**Auth:** none
**Status codes:** `200`, `404`

---

### `GET /api/candidate/{candidate_id}`

SHAP-based explanation for a candidate — the same input that drives the `/api/rank` ranking decision. Returns positive/negative factors, per-feature contributions, and a formatted prose explanation.

**Auth:** none
**Status codes:** `200`, `404`

**Response**

```ts
{
  candidate_id: string;
  candidate_rank_score: number;
  positive_factors: string[];
  negative_factors: string[];
  feature_contributions: Record<string, number>;
  formatted_explanation: string;
}
```

---

## Analytics

### `GET /api/analytics/dashboard`

Aggregate metrics: candidate count, job count, ranking count, average score, top 10 skills, recent activity.

**Auth:** none
**Status codes:** `200`

**Response**

```ts
{
  total_candidates: number;
  total_jobs: number;
  total_rankings: number;
  avg_score: number;
  top_skills: Array<{name: string; count: number}>;
  recent_activity: Array<{action: string; job_title: string; timestamp: string}>;
}
```

---

## Compare

### `POST /api/compare`

Compare two candidates side-by-side.

**Auth:** none
**Status codes:** `200`, `404`

**Request body**

```ts
{
  candidate_a_id: string;
  candidate_b_id: string;
}
```

**Response**

```ts
{
  candidate_a: CandidateResponse;
  candidate_b: CandidateResponse;
  comparison: {
    score_diff: number;            // pred(a) - pred(b)
    skills_a: string[];
    skills_b: string[];
  };
}
```

**cURL**

```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"candidate_a_id": "CAND_0004989", "candidate_b_id": "CAND_0001195"}'
```

---

## Error format

All errors follow FastAPI's standard shape:

```json
{ "detail": "Job not found" }
```

Validation errors return `422` with a `detail` array of field-level errors.

---

## Running locally

```bash
cd backend
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper:

```bash
python backend/api/main.py
```
