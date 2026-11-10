"""Train functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import logging

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
MODEL_REGISTRY = {"hist_gradient_boosting": HistGradientBoostingClassifier, "logistic_regression": LogisticRegression}


def build_estimator(training: dict) -> object:
    """Inputs: training.
    Outputs: the annotated return value or a validation error.
    Why: Create the unfitted classifier named in SETTINGS."""
    model_type = training["model_type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'; choose one of {sorted(MODEL_REGISTRY)}")
    params = dict(training.get("params") or {})
    params["random_state"] = training["random_state"]
    return MODEL_REGISTRY[model_type](**params)


def _check_training_data(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Fail fast on empty, misaligned or single-class training data."""
    if not isinstance(x_train, pd.DataFrame):
        raise TypeError(f"x_train must be a DataFrame, got {type(x_train).__name__}")
    if x_train.empty:
        raise ValueError("x_train has no rows")
    if len(x_train) != len(y_train):
        raise ValueError(f"x_train has {len(x_train)} rows but y_train has {len(y_train)}")
    if pd.Series(y_train).nunique() < 2:
        raise ValueError("y_train needs both classes (0 and 1) to train a classifier")


def _train_configured_model(
    x_train: pd.DataFrame, y_train: pd.Series, recipe: ColumnTransformer, training: dict
) -> Pipeline:
    """Fit the feature recipe and the classifier together on the training split.

    Args:
        x_train: Training inputs (extra columns are dropped by the recipe).
        y_train: Training labels, 0 or 1.
        recipe: Unfitted ColumnTransformer from features.py (cloned, never modified).
        training: The ``training`` section of config.yaml.

    Returns:
        The fitted pipeline with steps ``features`` and ``model``.

    Raises:
        TypeError: ``x_train`` is not a DataFrame.
        ValueError: Empty or misaligned data, a single class, or an unknown model type.
    """
    return train_model(
        x_train,
        y_train,
        recipe,
        "classification",
        model_params={"model_type": training["model_type"], **training.get("params", {})},
        random_state=training["random_state"],
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
    model_params: dict | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Inputs: X_train, y_train, preprocessor, problem_type, model_params, random_state.
    Outputs: Fitted Pipeline with preprocess and model steps.
    Why: Keep one tested pipeline responsibility."""
    if problem_type != "classification":
        raise ValueError("This project supports classification")
    _check_training_data(X_train, y_train)
    if isinstance(y_train, pd.Series) and (not X_train.index.equals(y_train.index)):
        raise ValueError("Training indices do not align")
    if set(y_train) != {0, 1}:
        raise ValueError("Training labels must be 0 and 1")
    settings = dict(model_params or {})
    model_type = settings.pop("model_type", "logistic_regression")
    model = build_estimator({"model_type": model_type, "params": settings, "random_state": random_state})
    fitted = Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
    fitted.fit(X_train, y_train)
    return fitted
