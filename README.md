# Crypto Volatility Early-Warning System

> The fund should operationalize volatility prediction with reproducible pipelines, automated tests and monitoring to support timely risk assessments beyond reactive alerts.

## 1. Business case

### Context

**Client:** Hypothetical cryptocurrency investment fund, Market Risk Management. **Assumed maturity:** data/tools/people **medium**, MLOps processes **low**, reflecting available public data and technical staff but no established model releases. Client ownership: **TBD**.

### Problem and baseline

**Assumed current process:** automated price/volume alerts followed by manual investigation. Delays are **unmeasured**. **Business KPI:** median seconds to complete a usable risk assessment. **Baseline: TBD**, using approximately 15–20 timed historical scenarios and an assessment-quality check. Student results are not actual fund performance.

### Goal and success metrics

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business | Median assessment time (seconds) | **TBD** | Improvement **TBD**; quality safeguard **TBD** |
| ML | `average_precision` (equal-weight mean across BTC, ETH, SOL) | **0.408660**, measured on April–June 2026; past-60-candle realized-volatility benchmark | **≥ 1.20 × benchmark** on chronological validation data |
| Operational | API latency, data freshness, cost per prediction | **TBD** | Limits **TBD** |

The **20% ML gate** was pre-agreed. The integrated local pipeline run on 10 November 2026 reproduced validation macro AP **0.514824**, above the required **0.490392** (1.20 × benchmark). This is a validation-only pass; the model remains a candidate pending review. Per-asset and historical final-period results must be included in the model evaluation handoff. The final period has prior analytical exposure. Failed candidates are not promoted.

### Solution

For BTCUSDT, ETHUSDT and SOLUSDT, a classifier uses **60 completed one-minute candles** to score unusually high volatility in the **next 15 minutes**. Calibration is **TBD**, so outputs remain risk scores. A **proposed** Binance REST poller triggers minute-by-minute predictions; FastAPI `POST /predict` also supports requests on demand. Historical replay supports testing.

Analysts combine scores with portfolio exposure and leverage to consider hedging, reduced positions or paused trades. **No automated trades or directional forecasts.** Withhold stale/invalid scores; fall back to conventional monitoring and manual assessment.

**MLOps stages:** GitHub scaffold → tested one-command pipeline → CI/CD, Docker and live prediction API. The local one-command pipeline is implemented and verified. Notebook integration, team approval and the later deployment stage remain pending.

### Data

**Source:** [Binance Spot one-minute archives](https://data.binance.vision/) for training/testing; Binance Spot API for new completed candles. One row = one asset-minute. Verified local history: **April 2025–September 2026**, **2,367,360 valid candles** across 54 files. Local checksums identify copies but do not establish publisher authenticity.

**Label:** Next-15-minute realized volatility exceeds the **asset-specific training-data 90th percentile**. Use time-ordered splits and identical past-only features in training and inference. Feature/label timing and chronological boundaries are specified in `TEAM_BRIEF.md` and `SETTINGS`. Data Engineering owns acquisition, storage and validation; ML Engineering owns features, labels and quality. No personal data is expected.

### Value, cost and risks

**Expected benefit:** faster, more consistent risk assessment, **not demonstrated**. Pilot resources: part-time existing staff roles, student/university computers and open-source tools; no paid hosting planned. Hours, duration and actual costs: **TBD**. No 24/7 uptime or reduction in portfolio losses is promised; expansion requires validation.

| Risk | Mitigation |
| --- | --- |
| Missing or inconsistent candles | Check schema, UTC timestamps, gaps and duplicates |
| Data leakage or weak predictions | Chronological tests, train-only thresholds and fixed quality gate |
| Market drift or misleading scores | Track inputs/outcomes, review calibration and retrain under control |
| Feed/API or release failure | Withhold stale scores; manual fallback; preserve approved artifacts |

### Open gaps

- **Measure:** business baseline/improvement, calibration, latency, freshness and costs. The integrated local pipeline reproduced the dataset and benchmark/model measurements; independent future confirmation remains open.
- **Define:** business target/rubric, optional alert policy, service limits and rollback. Feature/label timing, chronological splitting and model reload were verified in the integrated local pipeline.
- **Confirm:** team acceptance of the tested notebook and modules, client owners, student hours, target hardware and deployment. Team/module contracts are in `TEAM_BRIEF.md`.

## 2. Stage-2 setup and execution

**Integration status:** The merged modules and local main-pipeline implementation run end to end.
The main-pipeline changes await Selin's review and merge; notebooks remain a separate milestone.
No model is automatically approved by a successful run or quality-gate pass.

Runtime settings belong in `SETTINGS` at the top of `src/main.py`. `config.yaml` remains
the original repository placeholder for the later course stage. No active CI, API, polling or deployment is included.

Open the repository root in VS Code. In **Terminal > New Terminal**:

```powershell
conda env create -f environment.yml
conda activate crypto-volatility-mlops
python -m pytest -vv
python -m flake8 src tests
python -m src.main
```

If a fresh Miniconda installation reports `CondaToSNonInteractiveError` for default channels, run
`conda config --show-sources` and inspect the active channel configuration. The tested installation uses:

```yaml
channels:
  - conda-forge
default_channels: []
```

Update only the channel entries in the relevant configuration, preserving unrelated settings, then retry creation.
The project environment also declares `nodefaults`. No acceptance of unused default-channel terms is required.
Windows environment creation was verified with Python 3.12.16; other operating systems remain untested.

Select the same environment with **>Python: Select Interpreter**. If the environment already exists, update it only
after environment.yml changes: `conda env update -f environment.yml --prune`.

Put the 54 original monthly CSVs in `data/raw/BTCUSDT/`, `ETHUSDT/` and `SOLUSDT/` (April 2025–September 2026).
Alternatively, set `$env:ML_RAW_DIR` to your existing raw-data folder before running; do not copy large datasets unnecessarily.
Data must be supplied separately and are not committed to Git. In a new terminal reactivate the environment.

The pipeline loads and validates candles, builds past-only features/future labels, splits chronologically, fits training-only
cutoffs/preprocessing/model, evaluates validation against the 60-candle RV benchmark and records the historical final period.
The latter has prior analytical exposure and is not an untouched holdout. No optimization is performed by the pipeline.

Outputs: `data/processed/clean.csv`, candidate bundles under `models/candidates/` or `models/failed_candidates/`,
and validation/final metrics, diagnostic predictions and the pre-test selection lock under `reports/`.
The fixed 20% validation gate does not automatically approve a model; an accepted artifact is never overwritten.
The teacher-compatible `probability` output is an uncalibrated risk score; binary predictions are diagnostics, not an agreed alert policy.
Raw-candle historical replay is available through `src.infer.score_candles` with an explicit timezone-aware `as_of`.
Load joblib files only from trusted sources.

Run both notebooks with their kernel working directory at the repository root. The standalone notebook is the frozen
research reference; the second notebook runs the modular command and checks reload/parity. New experiments belong in
a separate notebook. Record the actual environment and execution results during integration.

### Local integration verification — 10 November 2026

Verified on Windows with Python 3.12.16 in the team Conda environment:

- 105 tests passed, zero failures/errors/skips; full source/test Flake8 passed.
- Diagnostic statement/branch coverage: 78% overall, 89% for `src/main.py`. No course minimum is assumed.
- All 54 CSVs loaded: 2,367,360 candles; no large dataset copy was made.
- Eligible rows: training 1,576,578; validation 393,078; historical final period 397,398.
- Validation macro AP: model 0.514824, benchmark 0.408660, required threshold 0.490392; gate passed.
- Historical final-period macro AP: model 0.414614, benchmark 0.300345. This period has prior analytical exposure.
- Fresh-process reload reproduced a stored prediction with training disabled; historical microsecond and REST
  millisecond opening times produced identical scores. The saved artifact checksum matched the metrics report.

The model remains `candidate_pending_review`. The risk scores are uncalibrated, the business benefit is unmeasured,
and neither live readiness nor independent future performance is established. Notebook execution is pending.
Follow-up tests should strengthen raw-input rejection coverage and directly exercise the public utils/preprocessor
interfaces. Full run details are generated in `reports/metrics.json`; data and model outputs stay outside Git.

## 3. Team integration

All five members work from the approved `TEAM_BRIEF.md` and the teacher's kickoff guidance.
Each owner submits only their assigned milestone and relevant folder documentation.

| Member | Responsibilities | Working branch |
| --- | --- | --- |
| Asier | Shared setup, environment, main pipeline, integration checks and release coordination | `asier/feature` |
| Gabriel | Shared file helpers, data loading and cleaning | `gabrielstec/feature` |
| Christian | Data validation | `christian/feature` |
| Maria Alejandra | Targets, features and candle contract | `ale/feature` |
| Selin | Training, evaluation, inference and notebooks | `selin/features` |

Integration order: business README → shared setup → utils → loading/cleaning → validation →
targets/features → model modules → main pipeline → notebooks → release.
Gabriel reviews the initial README and setup; other designated reviewers are listed in `TEAM_BRIEF.md`.
Verify prerequisite merges before submitting the next milestone. Independent preparation may proceed while waiting.

Changes reach `dev` through reviewed pull requests. Before each milestone, safely synchronize the member branch
with `dev`, preserve existing work, and check that the PR contains only that milestone. Reusing member branches
is an explicit exception to the teacher's branch-per-module cleanup workflow.

Human reviewers approve and merge; do not bypass checks or commit directly to `dev` or `main`.
After all required modules and notebooks are integrated, verify the complete tests, lint, pipeline,
validation quality gate and saved-model inference. Resolve pending dependency tests before release.
The final release is a separately reviewed `dev` → `main` PR.

Calibration, business assessment-time benefit, live freshness limits and independent future confirmation remain unresolved.
