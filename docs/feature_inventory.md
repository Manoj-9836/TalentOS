# Feature Inventory

## Candidate Features
- skills: Present
- profile.years_of_experience: Present
- profile.headline: Present
- profile.summary: Present
- education: Present
- profile.current_title: Present
- profile.current_industry: Present
- profile.location: Present
- profile.country: Present

## Career Features
- career_history: Present
- career_history.duration_months: Derived
- career_history.title: Derived
- career_history.industry: Derived
- profile.years_of_experience: Present

## Platform Features
- redrob_signals.recruiter_response_rate: Present
- redrob_signals.profile_views_received_30d: Present
- redrob_signals.offer_acceptance_rate: Present (note: -1 sentinel, see data_quality_report.md)
- redrob_signals.search_appearance_30d: Present
- redrob_signals.saved_by_recruiters_30d: Present
- redrob_signals.interview_completion_rate: Present
- redrob_signals.connection_count: Present
- redrob_signals.github_activity_score: Present

## Recommended Modeling Inputs
- profile.years_of_experience
- profile.current_title
- profile.current_industry
- redrob_signals.recruiter_response_rate
- redrob_signals.interview_completion_rate (primary behavioural target)
- redrob_signals.profile_views_received_30d
- redrob_signals.search_appearance_30d

## Modelling Caveats
- Do **not** use `offer_acceptance_rate` directly as a target without filtering
  rows where it equals `-1`. 59,554 of 100,000 rows carry this missing-data
  sentinel; using them as targets biases the model towards detecting the
  sentinel rather than predicting offer behaviour. See
  `docs/data_quality_report.md` for the data-quality finding.