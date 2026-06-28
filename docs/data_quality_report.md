# Data Quality Report

## Dataset Scope
- candidates.jsonl rows: 100000
- candidates.jsonl columns (flattened): 95
- sample_candidates.json rows: 50
- sample_submission.csv rows: 100

## Missing Values
- redrob_signals.skill_assessment_scores.Fine-tuning LLMs: 99.72% missing
- redrob_signals.skill_assessment_scores.Deep Learning: 99.72% missing
- redrob_signals.skill_assessment_scores.Elasticsearch: 99.72% missing
- redrob_signals.skill_assessment_scores.LlamaIndex: 99.71% missing
- redrob_signals.skill_assessment_scores.LangChain: 99.70% missing
- redrob_signals.skill_assessment_scores.Semantic Search: 99.70% missing
- redrob_signals.skill_assessment_scores.FAISS: 99.70% missing
- redrob_signals.skill_assessment_scores.NLP: 99.69% missing
- redrob_signals.skill_assessment_scores.BM25: 99.69% missing
- redrob_signals.skill_assessment_scores.Prompt Engineering: 99.69% missing
- redrob_signals.skill_assessment_scores.Python: 99.69% missing
- redrob_signals.skill_assessment_scores.Haystack: 99.69% missing
- redrob_signals.skill_assessment_scores.scikit-learn: 99.68% missing
- redrob_signals.skill_assessment_scores.OpenSearch: 99.68% missing
- redrob_signals.skill_assessment_scores.Hugging Face Transformers: 99.68% missing
- redrob_signals.skill_assessment_scores.LoRA: 99.68% missing
- redrob_signals.skill_assessment_scores.Weaviate: 99.68% missing
- redrob_signals.skill_assessment_scores.LLMs: 99.68% missing
- redrob_signals.skill_assessment_scores.Pinecone: 99.67% missing
- redrob_signals.skill_assessment_scores.Recommendation Systems: 99.67% missing
- redrob_signals.skill_assessment_scores.Vector Search: 99.67% missing
- redrob_signals.skill_assessment_scores.TensorFlow: 99.67% missing
- redrob_signals.skill_assessment_scores.RAG: 99.67% missing
- redrob_signals.skill_assessment_scores.Embeddings: 99.67% missing
- redrob_signals.skill_assessment_scores.Learning to Rank: 99.67% missing

## Duplicate Candidates
- duplicate candidate_id count: 0

## Outliers
- profile.years_of_experience > 50: 0
- profile.years_of_experience < 0: 0
- redrob_signals.recruiter_response_rate outside [0, 1]: 0
- redrob_signals.interview_completion_rate outside [0, 1]: 0
- redrob_signals.offer_acceptance_rate outside [-1, 1]: 0
- redrob_signals.avg_response_time_hours > 720: 0

## Sentinel / Missing-Data Findings
- **redrob_signals.offer_acceptance_rate = -1.0 in 59,554 / 100,000 rows (59.6%)**.
  The data dictionary does not document -1 as a sentinel, but the value sits
  outside the natural [0, 1] rate range, is a single repeated constant, and
  correlates with no candidate-quality feature. Treat `-1.0` as a missing-data
  marker ("no recorded offer history") rather than a real rate. Models that
  include these rows as targets end up learning to detect the sentinel rather
  than the underlying behaviour.
  - Remediation applied: `train_recruitability.py` filters rows where the
    target < 0 before fitting. The behaviour target
    (`interview_completion_rate`) is clean [0, 1] with no sentinels and is
    used as the primary recruitability target.

## Notes
- Nested arrays (career_history, skills, education) should be exploded for detailed row-level quality checks.
- Use schema constraints in candidate_schema.json as validation rules in preprocessing.