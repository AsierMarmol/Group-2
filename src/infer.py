"""Infer functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"


def _run_feature_inference(model, x_new: pd.DataFrame) -> pd.DataFrame:
    """Predict the class and the high-volatility probability for each new row.

    Args:
        model: Fitted pipeline with ``predict`` and ``predict_proba``.
        x_new: New rows with the feature columns used in training.

    Returns:
        ``prediction`` and ``probability`` columns, indexed like ``x_new``.

    Raises:
        TypeError: ``x_new`` is not a DataFrame, or ``model`` cannot predict probabilities.
        ValueError: ``x_new`` has no rows.
    """
    if not isinstance(x_new, pd.DataFrame):
        raise TypeError(f"x_new must be a DataFrame, got {type(x_new).__name__}")
    if x_new.empty:
        raise ValueError("No rows to predict")
    if not all((callable(getattr(model, name, None)) for name in ("predict", "predict_proba"))):
        raise TypeError(f"Model of type {type(model).__name__} needs predict and predict_proba")
    predictions = pd.DataFrame(
        {PREDICTION_COLUMN: model.predict(x_new).astype(int), PROBABILITY_COLUMN: model.predict_proba(x_new)[:, 1]},
        index=x_new.index,
    )
    logger.info("Scored %d rows; %d flagged as high volatility", len(predictions), predictions[PREDICTION_COLUMN].sum())
    return predictions


def score_candles(
    candles: pd.DataFrame, bundle_path: Path | str, as_of: pd.Timestamp | str, unit: str = "auto"
) -> dict:
    """Inputs: candles, bundle_path, as_of, unit.
    Outputs: Versioned risk_score for an exact completed historical window.
    Why: Keep one tested pipeline responsibility."""
    from src import candle_contract

    return candle_contract.predict_candles(candles, bundle_path, mode="replay", as_of=as_of, unit=unit)


def run_inference(model: object, X_infer: pd.DataFrame, include_proba: bool = False) -> pd.DataFrame:
    """Inputs: model, X_infer, include_proba.
    Outputs: Same-index prediction and optional probability (uncalibrated risk score).
    Why: inference never trains; classes are diagnostics, not approved alerts."""
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Model must support predict_proba")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected class mapping")
    out = _run_feature_inference(model, X_infer)
    return out if include_proba else out[[PREDICTION_COLUMN]]
