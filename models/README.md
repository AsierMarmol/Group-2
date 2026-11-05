# Model artifacts

The integrated pipeline writes candidate bundles under candidates/ or failed_candidates/.
Bundles contain fitted preprocessing/model, feature definitions, training-only target cutoffs, settings and metadata.
A quality-gate pass does not approve release. Do not automatically overwrite an accepted model.joblib.
Verify loading and predictions without retraining during model integration. Load only trusted serialized files.
Generated models are ignored by Git; this README may be tracked.
