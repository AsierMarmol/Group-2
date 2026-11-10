# Automated verification

Activate the team Python 3.12 environment and run from the repository root:

```powershell
python -m pytest -vv
python -m flake8 src tests
```

Module owners add their tests first, record an actual expected failure, and then implement their module.
Use synthetic data and pytest's disposable temporary folders, never production datasets or outputs.
The shared conftest.py and fixture_settings.json provide synthetic candle fixtures; dependencies import lazily.
The JSON and generated temporary YAML are test fixtures, not runtime configuration. Runtime values remain in main.SETTINGS.

pytest.ini uses .pytest_cache/pytest-tmp. Do not store project data there: pytest clears its dedicated temporary child.
The suite now covers the merged modules and main integration; an empty collection is never a passing suite.
Run relevant unit tests, available integrations and lint at each milestone; record actual counts and pending dependencies.
Do not claim skipped tests passed. Final integration requires zero pending dependency skips.
Coverage is diagnostic, without an assumed minimum. CI and custom logging are deferred.

## Main-pipeline checks

`tests/test_main.py` covers chronological decision-time splits, boundary-label purging, aligned X/y,
training-only thresholds, gate equality and invalid evidence, portable settings and inventory checks.
The synthetic end-to-end entrypoint test patches SETTINGS, verifies selection is recorded before final evaluation,
preserves an accepted-artifact sentinel, and checks candidate metadata, per-asset macro AP and prediction identifiers.
A fresh Python process verifies raw-candle reload without training and parity between microsecond/millisecond inputs.
Temporary tests may deliberately require a stronger gate to exercise failure; the default 1.20 gate is never lowered.

Run `python -m pytest -vv tests/test_main.py`, then full tests and `python -m flake8 src tests`.
Optional diagnostic: `python -m pytest --cov=src --cov-branch --cov-report=term-missing`.
Fresh-process raw-input paths are not fully reflected in the parent process's coverage report.
## Final cross-module checks

`tests/test_release_contracts.py` exercises the public CSV/model persistence, unfitted recipe, training and inference
interfaces together. It checks index preservation, unchanged preprocessing during inference, unknown categories,
target/future-column exclusion, raw timestamp parity, malformed windows/values/symbols/timestamps and approval/freshness guards.
The missing-symbol regression was observed failing before the validation fix; no test history is fabricated.
Only in-memory test flags exercise an approved artifact; no test promotes or saves an approved model.

Final local suite: 109 passed, zero failures/errors/skips; diagnostic coverage 90% overall and 91% for main.py.
No blanket coverage threshold is imposed and untested branches are not claimed as verified.
