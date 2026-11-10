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

For BTCUSDT, ETHUSDT and SOLUSDT, a classifier uses **60 completed one-minute candles** to score unusually high volatility in the **next 15 minutes**. Calibration is **TBD**, so outputs remain risk scores. This deliverable runs a local batch pipeline and historical replay. A Binance REST poller and FastAPI `POST /predict` are proposed for later stages; neither is implemented here.

Analysts combine scores with portfolio exposure and leverage to consider hedging, reduced positions or paused trades. **No automated trades or directional forecasts.** Withhold stale/invalid scores; fall back to conventional monitoring and manual assessment.

**MLOps stages:** GitHub scaffold → tested one-command pipeline → CI/CD, Docker and live prediction API. The local one-command pipeline is implemented and verified. Both notebooks are integrated and locally verified. Final correction review, release approval and later deployment stages remain pending.

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
- **Confirm:** final correction/release approval, client owners, student hours, target hardware and deployment. Team/module contracts are in `TEAM_BRIEF.md`.

## 2. Setup and local execution

**Integration status:** The main pipeline and both notebook milestones are merged into `dev` through PRs #10 and #11.
The final local review identified and fixed a missing-symbol input-validation issue and added cross-module regression
checks. These final corrections still require review and merge into `dev` before the final `dev` → `main` release.
The source-code release covers the first deliverable; it does not approve a model for live use or implement later stages.

Runtime settings belong in `SETTINGS` at the top of `src/main.py`. `config.yaml` remains
the original repository placeholder for the later course stage. No active CI, API, polling or deployment is included.

Open the repository root in VS Code. In **Terminal > New Terminal**:

```powershell
conda env create -f environment.yml
conda activate crypto-volatility-mlops
python -m pytest -vv
python -m flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests
python -m pytest --cov=src --cov-branch --cov-report=term-missing
python -m pylint src
python -m radon cc -s -a src
python -m black --check src tests
python -m isort --check-only src tests
python -m detect_secrets scan
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
Windows environment creation was verified with Python 3.12.16. Selin recorded a macOS notebook run; other platform installation checks were not repeated in this final review.

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

Run `notebooks/crypto_volatility_ml.ipynb` first and then `notebooks/volatility_model.ipynb` from clean kernels,
with their working directory at the repository root and `ML_RAW_DIR` set before launching Jupyter when needed.
Use the team Python 3.12 environment for both. The standalone reference retains an old Python 3.14/requirements.txt
setup note; the instructions here and in `notebooks/README.md` supersede that historical note.
The standalone notebook is the frozen
research reference; the second notebook runs the modular command and checks reload/parity. New experiments belong in
a separate notebook. Record the actual environment and execution results during integration.

### Local integration verification — 10 November 2026

Verified on Windows with Python 3.12.16 in the team Conda environment:

- Current suite and exact statement/branch coverage are recorded in the grading verification below.
- Full source/test Flake8 and dependency checks passed; coverage is measured separately for statements and branches.
- All 54 CSVs loaded: 2,367,360 candles; no large dataset copy was made.
- Eligible rows: training 1,576,578; validation 393,078; historical final period 397,398.
- Validation macro AP: model 0.514824, benchmark 0.408660, required threshold 0.490392; gate passed.
- Historical final-period macro AP: model 0.414614, benchmark 0.300345. This period has prior analytical exposure.
- Fresh-process reload reproduced a stored prediction with training disabled; historical microsecond and REST
  millisecond opening times produced identical scores. The saved artifact checksum matched the metrics report.

The model remains `candidate_pending_review`. The risk scores are uncalibrated, the business benefit is unmeasured,
and neither live readiness nor independent future performance is established. Both notebooks passed from clean
kernels on the full dataset. Standalone/modular metrics, cutoffs and row counts agreed within `1e-12` on Windows;
nine fresh-process raw windows matched stored predictions and all seven malformed notebook demonstrations were rejected.
The final automated checks also cover public CSV/model/preprocessor interfaces, frozen preprocessing at inference,
invalid raw windows/values/symbols/timestamps, candidate approval guards and target/future-column exclusion.
Full run details are generated in `reports/metrics.json`; data and model outputs stay outside Git.

Selin's recorded macOS scores differ slightly from the verified Windows scores while both pass the gate.
The exact cause was not isolated; do not require cross-platform bit-for-bit model scores. See `notebooks/README.md`.
The course LICENSE placeholder is unchanged; no open-source license or new reuse permissions are claimed.
Business benefit, calibration, client ownership, service limits and licensing decisions remain explicit open items.
They do not represent implemented later-stage functionality.

### Grading-tool verification — 10 November 2026

- 150 tests passed, zero failures/errors/skips. The suite includes real synthetic end-to-end `main()` execution.
- Statement coverage: **99.87%** (750/751); branch outcomes: **99.29%** (280/282), measured separately.
  This exceeds the supplied slides' 90% branch gate and 95% full-coverage criterion. No exclusions were added.
- Exact isolated Flake8 command: no violations. Black and isort checks passed with consistent 88-column layout.
- Pylint 4.1.3: **9.75/10**, above the slides' 9.5 full-mark criterion. Remaining warnings concern preserved public
  contract names/argument counts, compatibility arguments, local-variable counts and a docstring length; none are suppressed.
- Radon 6.0.1: average **A (4.35)**; no function exceeds complexity 10.
- detect-secrets 1.5.0 reports one high-entropy string in the frozen notebook's embedded PNG. Inspection confirms
  image data, not a credential. The notebook and finding are preserved; an empty scanner result is not claimed.
- Both frozen notebooks reran from clean kernels after the refactors; source cells, metadata and committed outputs
  remain unchanged. The full pipeline reproduced the validation/final AP above, and all three assets passed a fresh
  artifact reload with fitting disabled, source/artifact checksum validation and microsecond/millisecond score parity.

The remaining uncovered paths are the default raw-directory selection and the CLI guard in `main.py`;
the real CLI was separately exercised on the full dataset. Local tool versions and a public rubric do not establish
the outcome of the teacher's unavailable private grader. Manual mentoring, reused branches, prior final-period exposure
and unresolved business/service decisions remain disclosed. YAML migration and logging extensions are later-deliverable
work, as confirmed for the current scope. No deferred infrastructure or model optimisation was added.

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
