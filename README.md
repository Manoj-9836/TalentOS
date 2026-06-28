Commands to run TalentOS                                                                                                                                  
                                                                                                                                                            
  There are three layers. Run them in order from the project root F:\Gitprojects\TalentOS.                                                                  
   
  0. One-time environment setup                                                                                                                             
                                                                                                                                                          
  # Optional but recommended: virtual env
  python -m venv .venv
  .venv\Scripts\activate

  # Install dependencies
  pip install -r backend/requirements.txt

  If sentence-transformers is missing (it was in your earlier session), this fixes it.

  1. Generate candidate embeddings

  Pick one — --sample is the 50-row smoke test (~2 min on CPU), no flag is the full 100k (~1.5h on CPU, ~10 min on GPU):

  # Fast smoke test
  python -m backend.ml.embeddings.generate_embeddings --sample

  # Full dataset (when ready for the real submission)
  python -m backend.ml.embeddings.generate_embeddings

  Output: backend/data/embeddings/candidate_embeddings.jsonl

  2. Build the FAISS index

  python -m backend.ml.retrieval.build_index

  Output: backend/indices/candidate_index.faiss + candidate_ids.npy

  3. Encode the job description

  python -m backend.scripts.build_embeddings

  Output: backend/data/embeddings/job_embeddings.pkl

  4. Run the full pipeline and write the ranked CSV

  This is the single command that does it all — embeddings + index + JD + ranking + CSV in one go. Use it after you've done steps 1–3 once.

  # Or do it all in one shot (does steps 1-4 internally)
  python -m backend.scripts.run_pipeline

  # Skip re-embedding and re-indexing if you've already built them
  python -m backend.scripts.run_pipeline --skip-embeddings --skip-index

  # Custom output
  python -m backend.scripts.run_pipeline --output backend/output/ranked_candidates.csv --top-k 100

  Output: backend/output/ranked_candidates.csv

  5. Run evaluation (optional but recommended)

  python -m backend.scripts.run_evaluation

  Output: docs/reports/metrics_report.md (Refreshes R² numbers and Recall@K)

  6. View results

  # The ranked output
  type backend\output\ranked_candidates.csv

  # Or open in Excel / pandas
  python -c "import pandas as pd; df = pd.read_csv('backend/output/ranked_candidates.csv'); print(df.head(20))"

  ---
  Full happy-path sequence (copy-paste)

  # One time
  pip install -r backend/requirements.txt

  # Build artifacts (run once, then reuse)
  python -m backend.ml.embeddings.generate_embeddings
  python -m backend.ml.retrieval.build_index
  python -m backend.scripts.build_embeddings

  # Generate the submission CSV
  python -m backend.scripts.run_pipeline

  # (Optional) refresh the evaluation report
  python -m backend.scripts.run_evaluation

  After the first build, only the last two commands are needed for subsequent runs.

  Common gotchas

  ┌────────────────────────────────────────────┬───────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │                  Symptom                   │                     Cause                     │                         Fix                          │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ ModuleNotFoundError: sentence_transformers │ Dependency missing                            │ pip install sentence-transformers                    │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ MongoDB URI not provided                   │ You're running the old generate_embeddings.py │ Use the refactored one in backend/ml/embeddings/     │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ FAISS index not found                      │ Skipped step 2                                │ Run python -m backend.ml.retrieval.build_index       │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ job_embeddings.pkl not found               │ Skipped step 3                                │ Run python -m backend.scripts.build_embeddings       │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Encoding the full 100k takes forever       │ CPU only                                      │ Use a GPU machine, or submit with --sample (50 rows) │
  └────────────────────────────────────────────┴───────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

  If you're on a machine without GPU and want a quick demo

  python -m backend.ml.embeddings.generate_embeddings --sample
  python -m backend.ml.retrieval.build_index
  python -m backend.scripts.build_embeddings --sample
  python -m backend.scripts.run_pipeline --sample --top-k 20

  This produces a 20-row ranked CSV in under 5 minutes using the 50-row sample dataset. Good for demoing the architecture without waiting for the full 100k
  encoding.