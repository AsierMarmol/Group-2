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
No module tests are included in the setup milestone. An empty collection does not establish a passing test suite.
Run relevant unit tests, available integrations and lint at each milestone; record actual counts and pending dependencies.
Do not claim skipped tests passed. Final integration requires zero pending dependency skips.
Coverage is diagnostic, without an assumed minimum. CI and custom logging are deferred.
