# Postman Collection

The Postman v2.1 collection for this API lives at:

```
backend/api/postman_collection.json
```

## Importing

1. Open Postman → **Import** → **File** → select `backend/api/postman_collection.json`.
2. The collection "TalentOS Pipeline API" will appear in the sidebar.
3. The variable `base_url` defaults to `http://localhost:8000` — change it if your server runs elsewhere.

## Suggested run order

1. **Health → Health Check** — confirm the server is up.
2. **Jobs → Upload Job** — auto-saves `job_id` into the collection variable.
3. **Ranking → Rank Candidates** — uses the saved `job_id`.
4. **Candidates → List Candidates** — auto-saves the first `candidate_id`.
5. **Compare → Compare Two Candidates** — uses the saved `candidate_id` for both A and B.

## Auth

There is no auth. The collection has no `auth` folder.
