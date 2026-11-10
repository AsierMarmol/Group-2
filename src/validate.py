"""Validate functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fail_if_any(mask: pd.Series, message: str) -> None:
    """Raise a ValueError naming ``message`` when any row is flagged."""
    count = int(mask.sum())
    if count:
        raise ValueError(f"{count} row(s) {message}")


def _check_schema(df: pd.DataFrame, required: list[str], time_column: str) -> None:
    """Check the data is a non-empty DataFrame with the required columns and datetimes."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("No rows to validate")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        raise TypeError(
            f"Column '{time_column}' must hold datetimes, got {df[time_column].dtype}"
        )


def _check_values(df: pd.DataFrame, columns: dict, rules: dict) -> None:
    """Check missing values, price and volume ranges, candle shape and symbols."""
    _fail_if_any(
        df[[columns["symbol_col"], columns["time_col"], *rules["value_columns"]]]
        .isna()
        .any(axis=1),
        "have missing values",
    )
    _fail_if_any(
        (df[rules["positive_columns"]] <= 0).any(axis=1), "have a non-positive price"
    )
    _fail_if_any(
        (df[rules["non_negative_columns"]] < 0).any(axis=1),
        "have a negative volume or trade count",
    )
    body = df[[columns["open_col"], columns["close_col"]]]
    outside = (df[columns["high_col"]] < body.max(axis=1)) | (
        df[columns["low_col"]] > body.min(axis=1)
    )
    _fail_if_any(outside, "have an open or close outside the low-high range")
    _fail_if_any(
        ~df[columns["symbol_col"]].isin(rules["allowed_symbols"]),
        f"have a symbol outside {rules['allowed_symbols']}",
    )


def _check_time_order(df: pd.DataFrame, columns: dict, frequency: str) -> None:
    """Check candles are unique, sorted within each symbol and aligned to whole minutes."""
    symbol, time = (columns["symbol_col"], columns["time_col"])
    _fail_if_any(
        df.duplicated(subset=[symbol, time]),
        "are duplicate candles (same symbol and time)",
    )
    steps = df.groupby(symbol, sort=False)[time].diff()
    _fail_if_any(steps <= pd.Timedelta(0), "are out of time order within their symbol")
    _fail_if_any(
        df[time] != df[time].dt.floor(frequency),
        f"are not aligned to {frequency} boundaries",
    )


def _validate_candles(
    df: pd.DataFrame, columns: dict, rules: dict, target_column: str | None = None
) -> bool:
    """Check clean candles against every rule in SETTINGS; never modify them.

    Args:
        df: Clean candles from clean_data.py.
        columns: The ``columns`` section of SETTINGS (role -> column name).
        rules: The ``rules`` section of SETTINGS.
        target_column: Target to check against ``rules['target_values']``, if present.

    Returns:
        True when the data passes every check.

    Raises:
        TypeError: ``df`` is not a DataFrame or the time column does not hold datetimes.
        ValueError: A column is missing or a rule is broken; the message gives the row count.
    """
    required = [columns["symbol_col"], columns["time_col"], *rules["value_columns"]]
    if target_column:
        required.append(target_column)
    _check_schema(df, required, columns["time_col"])
    if str(df[columns["time_col"]].dt.tz) != "UTC":
        raise ValueError("Opening times must be timezone-aware UTC")
    _fail_if_any(
        np.isinf(df[rules["value_columns"]]).any(axis=1), "have nonfinite values"
    )
    _fail_if_any(df[columns["trades_col"]].mod(1).ne(0), "have fractional trade counts")
    _check_values(df, columns, rules)
    _check_time_order(df, columns, rules["frequency"])
    if target_column:
        _fail_if_any(
            ~df[target_column].isin(rules["target_values"]),
            f"have a target outside {rules['target_values']}",
        )
    logger.info(
        "Validation passed: %d rows, %d symbols",
        len(df),
        df[columns["symbol_col"]].nunique(),
    )
    return True


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    target_column: str | None = None,
    columns: dict | None = None,
    rules: dict | None = None,
) -> bool:
    """Inputs: df, required_columns, target_column, columns, rules.
    Outputs: True or a validation exception; input is never changed.
    Why: Keep one tested pipeline responsibility."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a DataFrame")
    if df.empty:
        raise ValueError("No rows to validate")
    needed = list(required_columns) + ([target_column] if target_column else [])
    if any((c not in df for c in needed)):
        raise ValueError("Missing required columns")
    if df[needed].isna().any().any():
        raise ValueError("Missing required values")
    if columns is not None and rules is not None:
        return _validate_candles(df, columns, rules, target_column)
    return True
