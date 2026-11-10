"""Evaluate functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def _evaluation_arrays(
    y_true: pd.Series, scores: pd.Series, groups: pd.Series
) -> tuple:
    """Check alignment, lengths and score finiteness before calculating any metric."""
    supplied = [x for x in [y_true, scores, groups] if isinstance(x, pd.Series)]
    if supplied and any(not supplied[0].index.equals(x.index) for x in supplied[1:]):
        raise ValueError("Evaluation indices do not align")
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    assets = np.asarray(groups).astype(str)
    if labels.size == 0:
        raise ValueError("No rows to evaluate")
    if not labels.size == values.size == assets.size:
        raise ValueError(
            f"Lengths differ: {labels.size} labels, {values.size} scores, {assets.size} groups"
        )
    if not np.isfinite(values).all():
        raise ValueError("Scores contain missing values")
    return labels, values, assets


def evaluate_scores(
    y_true: pd.Series, scores: pd.Series, groups: pd.Series
) -> dict[str, float]:
    """Inputs: y_true, scores, groups.
    Outputs: Equal-weight mean over supplied groups plus their individual AP; one-class groups raise.
    Why: Score any risk score (model probability or benchmark) against true labels."""
    labels, values, assets = _evaluation_arrays(y_true, scores, groups)
    per_asset = {}
    for asset in sorted(set(assets)):
        mask = assets == asset
        if np.unique(labels[mask]).size < 2:
            raise ValueError(f"No asset metric: {asset} needs both classes")
        per_asset[asset] = (
            float(average_precision_score(labels[mask], values[mask])),
            float(roc_auc_score(labels[mask], values[mask])),
        )
    metrics = {
        "average_precision": float(np.mean([pair[0] for pair in per_asset.values()])),
        "roc_auc": float(np.mean([pair[1] for pair in per_asset.values()])),
        "positive_rate": float(labels.mean()),
        "n_rows": int(labels.size),
    }
    metrics.update(
        {f"average_precision_{asset}": pair[0] for asset, pair in per_asset.items()}
    )
    return metrics


def _evaluate_group_model(
    model, x_eval: pd.DataFrame, y_eval: pd.Series, group_column: str
) -> dict[str, float]:
    """Score a fitted pipeline's probabilities on unseen rows.

    Args:
        model: Fitted pipeline with ``predict_proba``.
        x_eval: Inputs of the evaluation split (must include ``group_column``).
        y_eval: True labels of the evaluation split.
        group_column: Column that identifies the asset, for the macro average.

    Returns:
        The metrics of :func:`evaluate_scores` plus ``brier_score`` (calibration, lower is better).

    Raises:
        TypeError: ``model`` has no ``predict_proba`` method.
        ValueError: ``group_column`` is missing from ``x_eval``.
    """
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError(
            f"Model of type {type(model).__name__} has no predict_proba method"
        )
    if group_column not in x_eval.columns:
        raise ValueError(
            f"Group column '{group_column}' is missing from the evaluation data"
        )
    probabilities = model.predict_proba(x_eval)[:, 1]
    metrics = evaluate_scores(y_eval, probabilities, x_eval[group_column])
    metrics["brier_score"] = float(brier_score_loss(y_eval, probabilities))
    logger.info(
        "Evaluated %d rows: average_precision %.4f, roc_auc %.4f",
        metrics["n_rows"],
        metrics["average_precision"],
        metrics["roc_auc"],
    )
    return metrics


def evaluate_model(
    model: object,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    problem_type: str,
    group_column: str | None = None,
    expected_groups: list[str] | None = None,
) -> dict[str, float]:
    """Inputs: model, X_eval, y_eval, problem_type, group_column, expected_groups.
    Outputs: Macro/per-asset classification metrics; unsupported or non-evaluable data raise.
    Why: Keep one tested pipeline responsibility."""
    if problem_type != "classification":
        raise ValueError("This project supports classification")
    if isinstance(y_eval, pd.Series) and (not X_eval.index.equals(y_eval.index)):
        raise ValueError("Evaluation indices do not align")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected class mapping")
    if group_column is None:
        groups = pd.Series("all", index=X_eval.index)
    else:
        groups = X_eval[group_column]
    if expected_groups is not None and set(groups) != set(expected_groups):
        raise ValueError("Evaluation requires all configured assets")
    score = model.predict_proba(X_eval)[:, 1]
    metrics = evaluate_scores(y_eval, score, groups)
    metrics["brier_score"] = float(brier_score_loss(y_eval, score))
    return metrics
