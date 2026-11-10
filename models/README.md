# Model artifacts

The pipeline writes candidate bundles to candidates/model.joblib when the validation gate passes,
or failed_candidates/model.joblib when it fails. Neither route overwrites an accepted model.joblib.
Rerunning the same route replaces that route's experimental candidate; this is not a release registry.

Bundles contain fitted preprocessing/model, predictor order, feature definitions, training-only per-asset cutoffs,
settings, uncalibrated status and provenance. The version identifies the recipe/data/source fingerprint;
reports/metrics.json records the serialized artifact's SHA-256 separately.
Passing the gate leaves approval_status as candidate_pending_review; failure gives failed_quality_gate.
Only trusted serialized artifacts should be loaded. Generated models are ignored by Git.

Local verification on 10 November 2026 reloaded the real-data candidate in a fresh process with fitting disabled
and matched stored predictions using historical and REST timestamp units. Release approval remains pending.

To inspect a trusted artifact, call `bundle = utils.load_model(path)` after `from src import utils`.
The fitted sklearn pipeline is `bundle["model"]`; metadata is `bundle["metadata"]`.
For raw candles use `infer.score_candles` with the schema in `data/inference/README.md`;
do not pass raw OHLCV columns directly to the feature-based pipeline.
