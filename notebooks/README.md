# Notebook roles

- `crypto_volatility_ml.ipynb`: the frozen standalone ML reference. It loads the 54 raw files itself, compares
  three predeclared model families on validation, locks the selection, records the final period and verifies
  raw-candle reload. Its cells are unchanged from the agreed reference; only outputs are refreshed.
- `volatility_model.ipynb`: the modular workflow. It runs `python -m src.main`, summarises data, per-asset/macro AP
  and the validation gate, reloads the saved candidate in a fresh process (no training) for raw-candle inference,
  demonstrates input rejection and compares its results with the standalone outputs when they exist.

Keep the reference unchanged. New experiments belong in a separate notebook;
the production pipeline ports final choices and does not tune on the final evaluation period.

## Running

Use the team kernel (`crypto-volatility-mlops`, Python 3.12) with its working directory at the repository root;
do not modify `sys.path`. Set `ML_RAW_DIR` to an existing raw-data root before starting Jupyter if the CSVs are not
in `data/raw/`. `ML_PROJECT_DIR` may select the standalone notebook's project root.
Run the standalone notebook first if the modular comparison is wanted; otherwise that section reports "unavailable".
The standalone setup note (pip/`requirements.txt`, Python 3.14) predates the team environment; follow this README.

Generated files stay out of Git: the modular notebook writes `data/processed/`, `models/` and `reports/`;
the standalone notebook writes `results/metrics/` and `results/model/`. Load joblib files only from trusted runs.
Committed outputs replace local folders with `<repo>` and `<ML_RAW_DIR>`.

## Recorded execution — 10 November 2026

macOS (Apple Silicon), Python 3.12.16, team Conda environment; both notebooks executed from clean kernels with the
repository root as working directory and the 54 CSVs read through `ML_RAW_DIR` (no copy).

| Quantity | Standalone | Modular |
| --- | --- | --- |
| Validation macro AP (model / benchmark) | 0.515145 / 0.408660 | 0.515145 / 0.408660 |
| Required (1.20 × benchmark); gate | 0.490392; passed | 0.490392; passed |
| Final-period macro AP (model / benchmark), prior exposure | 0.414280 / 0.300345 | 0.414280 / 0.300345 |
| Eligible rows train / validation / final | 1,576,578 / 393,078 / 397,398 | same |

Per-asset AP, row counts and training cutoffs agree to within 1e-12, with identical raw-file hashes, model
parameters, predictor order, label rule and split boundaries. Fresh-process reload matched batch predictions for
9 windows with identical microsecond and millisecond results; 7 malformed requests were rejected.

The Windows run recorded in the main README gave validation macro AP 0.514824 and final 0.414614 with the same
data, cutoffs and validation benchmark. The exact cause of the observed cross-platform/build score difference was not isolated. Both pass the gate.
The candidate remains `candidate_pending_review`; scores are uncalibrated and the final period is not a holdout.

`docs/figures/final_diagnostics_v2.png`: final-period precision–recall and reliability curves from the standalone run.

## Final Windows verification — 10 November 2026

Both notebooks were rerun from clean Python 3.12.16 kernels with the repository root as working directory and the
full 54-file dataset supplied by ML_RAW_DIR. Notebook source cells and the frozen reference were preserved.
Standalone/modular per-asset and macro results, cutoffs, rows, input hashes and declared settings agreed within 1e-12.
Validation macro AP was 0.514824 (benchmark 0.408660; required 0.490392); final-period macro AP was 0.414614
(benchmark 0.300345, prior-exposure limitation applies). Nine raw reload windows matched batch predictions with
microsecond/millisecond parity; all seven malformed demonstration requests were rejected.
The final corrections add automated public-interface/input-rejection checks and reject missing asset values within
a raw window. The notebook demos are a small sample, not the entire automated regression suite.
Committed outputs and the diagnostic figure record Selin's macOS run; they are not relabelled as Windows outputs.
Source-code release review remains separate from model approval, calibration and later deployment work.

Both frozen notebooks were re-executed from clean kernels after the grading refactors on 10 November 2026.
Public interfaces, scientific SETTINGS and notebook contents were preserved; standalone/modular parity checks passed.
