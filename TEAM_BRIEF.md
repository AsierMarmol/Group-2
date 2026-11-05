# Team brief
## Project and scientific contract
Business case: README.md section 1; README PR requires Gabriel approval and dev merge before setup submission.
Use case: hypothetical fund's Market Risk Management assesses risk using scores; no automated trades.
Notebook: notebooks/crypto_volatility_ml.ipynb, finalized standalone reference; original source files unchanged.
Target: high_volatility | Problem type: classification | ID: symbol + candle opening UTC time.
Feature groups: numeric: 1/5/15/30 returns, 5/15/30 RV, 59-return benchmark, range/activity ratios, cyclical UTC hour/weekday -> scale for logistic only.
Feature groups: symbol -> one-hot encoding; unsupported symbols are rejected.
Excluded columns: future_rv, high_volatility and timestamps (future leakage/identifiers).
Model: histogram gradient boosting, 100 iterations, 31 leaves, learning_rate .1, min_samples_leaf100, seed42, early_stoppingFalse.
Metric: average_precision; exactly BTC/ETH/SOL equal mean; validation >=1.20 x same-row past-60-candle RV benchmark.
Split: train Apr2025–Mar2026; validation Apr–Jun2026; final Jul–Sep2026; UTC decision-time boundaries, label-end purging; no shuffle/stratification.
Validation rules: finite positive OHLC, UTC minute grid, finite nonnegative volume, integer nonnegative trades, valid candle duration; exact duplicates removable, conflicts rejected.
README follow-ups: reviewed ML/data evidence -> Asier; business benefit, calibration, live limits and independent confirmation remain TBD.
## Interfaces and settings
features.py signature: get_feature_preprocessor(numeric_columns,categorical_columns,scale=True); configured windows in main.SETTINGS.
Agreed extra parameters: optional candle roles/rules and loader file-schema settings; optional evaluate group_column/expected_groups; chronological split_settings/time_column/horizon/frequency.
utils.py helpers: load_csv, save_csv, save_model, load_model; private compatibility helpers support implementation tests.
Environment: Python3.12; environment.yml and requirements-verified.txt define the tested packages; use the same interpreter for notebooks/tests/pipeline.
Decisions: 60 candles contain 59 returns; y strictly exceeds per-asset eligible-training linear RV90; labels never enter X; risk scores uncalibrated.
Decisions: inference never silently drops rows; raw training-candle cleanup explicitly opts into dropping invalid rows before labels exist.
Decisions: runtime settings live only in main.SETTINGS; YAML/CI/logging extensions deferred; candidate gate pass is not release approval.
Decisions: final period has prior analytical exposure; time-block rather than independent-row uncertainty is needed if added later.
## Ownership and review
Owners (team): utils.py -> Gabriel; reviewer Asier; branch gabrielstec/feature.
Owners (team): load_data.py/clean_data.py -> Gabriel; reviewer Maria Alejandra; branch gabrielstec/feature.
Owners (team): validate.py -> Christian; reviewer Selin; branch christian/feature.
Owners (team): target.py/features.py/candle_contract.py -> Maria Alejandra with Selin for inference handoff; reviewer Gabriel; branch ale/feature.
Owners (team): train.py/evaluate.py/infer.py -> Selin; reviewer Christian; branch selin/features.
Owners (team): main.py/setup -> Asier; reviewers Gabriel(setup)/Selin(main); branch asier/feature for now.
Owners (team): notebooks -> Selin; reviewer Asier; branch selin/features.
Decisions: README PR -> team setup -> utils -> load/clean -> validate -> target/features -> model -> main -> notebooks; each waits for prior dev merge.
Decisions: config/logger/active CI uploads are deferred to Session10+; release uses reviewed dev->main PR; no direct dev/main commits.
Decisions: reuse confirmed member branches with separate PR milestones; exception to the teacher's branch-per-module cleanup workflow.
