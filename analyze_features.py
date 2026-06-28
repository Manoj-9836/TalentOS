import json
import pandas as pd
import sys
sys.path.insert(0, 'ml/feature_engineering')
from profile_builder import build_profile_and_features

# Load sample candidates
with open('Dataset/India_runs_data_and_ai_challenge/sample_candidates.json') as f:
    candidates = json.load(f)

# Build features for all candidates
features_list = []
for c in candidates:
    result = build_profile_and_features(c)
    features_list.append(result['features'])

df_features = pd.DataFrame(features_list)

print("=== 1. ENGINEERED FEATURES (df_features.head()) ===")
print(df_features.head().to_string())

print("\n=== 2. FEATURE CALCULATION LOGIC ===")
print("""
growth_score = skill_count * 3 + min(years_experience, 20) * 2 + (profile_views_30d / 50)
  -> scaled 0-100 (source range 0-200)

stability_score = avg_job_duration_months
  -> scaled 0-100 (source range 0-120 months)
  fallback: (years_experience * 12) / max(1, career_history_count)

learning_velocity = unique_skills / max(1, years_experience)
  -> scaled 0-100 (source range 0-10 skills/year)

skill_credibility = recruiter_response_rate * 60 + github_activity_score * 40
  -> scaled 0-100 (source range 0-100)

recruitability_score = 0.5 * growth_score + 0.2 * stability_score + 0.3 * skill_credibility
  -> clamped 0-100
""")

print("\n=== 3. FEATURE DISTRIBUTION (df_features.describe()) ===")
print(df_features.describe().to_string())

print("\n=== 4. MISSING VALUES REPORT (df_features.isnull().sum()) ===")
print(df_features.isnull().sum())

print("\n=== 5. TARGET VARIABLES CHECK ===")
target_vars = ['offer_acceptance_rate', 'interview_completion_rate', 'recruiter_response_rate', 'hired', 'selected', 'offer_received']
print("In redrob_signals:")
for c in candidates[:3]:
    signals = c.get('redrob_signals', {})
    present = [v for v in target_vars if v in signals]
    print(f"  {c['candidate_id']}: {present}")

# Also check all candidates for target coverage
print("\nTarget variable coverage across all candidates:")
for var in target_vars:
    count = sum(1 for c in candidates if var in c.get('redrob_signals', {}))
    valid_count = sum(1 for c in candidates if c.get('redrob_signals', {}).get(var, -1) >= 0)
    print(f"  {var}: {count}/{len(candidates)} present, {valid_count} valid (>=0)")