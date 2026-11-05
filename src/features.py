"""Build past-only candle predictors and an unfitted preprocessing recipe.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import candle_contract


def add_lookback_features(
    candles: pd.DataFrame, columns: dict, windows: dict, frequency: str, feature_settings: dict | None = None
) -> pd.DataFrame:
    """Inputs: candles, columns, windows, frequency, feature_settings.
    Outputs: Canonical predictors alongside input columns for complete per-asset windows.
    Why: compute each asset independently and reject windows crossing gaps."""
    if candles.empty:
        raise ValueError("No candles")
    short, long = (int(windows["short"]), int(windows["long"]))
    if not 2 <= short < long:
        raise ValueError("Windows must satisfy 2 <= short < long")
    needed = [
        columns[x] for x in ["symbol_col", "time_col", "close_col", "high_col", "low_col", "volume_col", "trades_col"]
    ]
    if any((c not in candles for c in needed)):
        raise ValueError("Missing columns for features")
    frames = []
    for _, g in candles.groupby(columns["symbol_col"], sort=True):
        g = g.sort_values(columns["time_col"]).reset_index(drop=True)
        d = g.rename(
            columns={
                columns["symbol_col"]: "symbol",
                columns["time_col"]: "open_time",
                columns["close_col"]: "close",
                columns["high_col"]: "high",
                columns["low_col"]: "low",
                columns["volume_col"]: "volume",
                columns["trades_col"]: "trades",
            }
        )
        f, valid = candle_contract.past_features(d, input_candles=long, frequency=frequency, **feature_settings or {})
        frames.append(pd.concat([g, f], axis=1).loc[valid])
    return pd.concat(frames, ignore_index=True)


def get_feature_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str], scale: bool = True
) -> ColumnTransformer:
    """Inputs: numeric_columns, categorical_columns, scale.
    Outputs: Unfitted ColumnTransformer; train.py fits it on training rows.
    Why: train.py fits scaling/encoding exclusively on training data."""
    if not numeric_columns and (not categorical_columns):
        raise ValueError("At least one feature column is required")
    if set(numeric_columns) & set(categorical_columns):
        raise ValueError("Feature groups overlap")
    parts = []
    if numeric_columns:
        parts.append(("numeric", StandardScaler() if scale else "passthrough", numeric_columns))
    if categorical_columns:
        parts.append(("symbol", OneHotEncoder(handle_unknown="error", sparse_output=False), categorical_columns))
    return ColumnTransformer(parts, remainder="drop")


# Recommendations for later: assess calibration with chronological fitting/evaluation.
# Review feature and score drift on new periods before proposing controlled retraining.
