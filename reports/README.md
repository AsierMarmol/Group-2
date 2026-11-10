# Evaluation outputs

The pipeline generates metrics.json, predictions.csv and selection_lock.json here.
The metrics report includes validation/final-period per-asset AP, equal-weight macro AP, same-row RV benchmark,
the fixed relative gate, label cutoffs, split row counts, candidate path/status, source/data hashes and versions.
selection_lock.json records the selected settings/predictors and validation gate before final-period diagnostics;
it is a per-run audit record, not an immutable release approval or experiment-tracking service.

predictions.csv begins with symbol and UTC open_time, followed by prediction and probability.
probability is an uncalibrated risk score; the estimator-default prediction is not an approved warning policy.
The final period has prior analytical exposure and must not guide tuning or be called an untouched holdout.
Generated reports are ignored by Git. Keep actual run measurements and limitations in the main README.
