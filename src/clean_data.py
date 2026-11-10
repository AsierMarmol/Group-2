"""Clean data functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
MICROSECONDS_PER_MILLISECOND = 1000


def to_utc_datetime(values: pd.Series, microseconds_above: float) -> pd.Series:
    """Inputs: values, microseconds_above.
    Outputs: UTC Series; unreadable values become NaT and are rejected or excluded by mode.
    Why: Convert epoch timestamps to UTC datetimes."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True)
    numbers = pd.to_numeric(values, errors="coerce")
    is_micro = numbers > microseconds_above
    milliseconds = numbers.where(~is_micro, numbers / MICROSECONDS_PER_MILLISECOND)
    return pd.to_datetime(milliseconds, unit="ms", utc=True)


def _drop_rows(df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    """Drop the rows flagged by ``mask`` and log how many went."""
    count = int(mask.sum())
    if count:
        logger.warning("Dropped %d row(s) with %s", count, reason)
    return df.loc[~mask].copy() if count else df


def _inconsistent_candles(df: pd.DataFrame, columns: dict) -> pd.Series:
    """Flag candles whose open or close lies outside their own low-high range."""
    body = df[[columns["open_col"], columns["close_col"]]]
    return (df[columns["high_col"]] < body.max(axis=1)) | (df[columns["low_col"]] > body.min(axis=1))


def _log_missing_minutes(df: pd.DataFrame, columns: dict, frequency: str) -> None:
    """Log, per symbol, how many minutes are missing between its first and last candle."""
    step = pd.Timedelta(frequency)
    for symbol, times in df.groupby(columns["symbol_col"])[columns["time_col"]]:
        expected = int((times.max() - times.min()) / step) + 1
        missing = expected - times.size
        if missing:
            logger.warning("%s: %d missing minute(s) between %s and %s", symbol, missing, times.min(), times.max())


def _select_columns(df_raw: pd.DataFrame, columns: dict, rules: dict, target_column: str | None) -> list[str]:
    """Return the columns to keep, raising a clear error when any is missing."""
    keep = [columns["symbol_col"], columns["time_col"], *rules["value_columns"]]
    if target_column:
        keep.append(target_column)
    missing = [column for column in keep if column not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns for cleaning: {missing}")
    return keep


def _clean_candles(df_raw: pd.DataFrame, columns: dict, rules: dict, target_column: str | None = None) -> pd.DataFrame:
    """Return a clean copy of the raw candles; the input is never modified.

    Args:
        df_raw: Raw candles from load_data.py.
        columns: The ``columns`` section of SETTINGS (role -> column name).
        rules: The ``rules`` section of SETTINGS (value, positive and non-negative columns).
        target_column: Target to keep, if the data already has one; rows without it are dropped.

    Returns:
        Clean candles sorted by symbol and time, with a fresh index.

    Raises:
        ValueError: A required column (or ``target_column``) is missing.
    """
    duration_column = rules.get("close_time_column")
    if duration_column and duration_column in df_raw:
        opening = pd.to_numeric(df_raw[columns["time_col"]], errors="raise")
        closing = pd.to_numeric(df_raw[duration_column], errors="raise")
        multiplier = np.where(opening > rules["microseconds_above"], 1000000, 1000)
        expected = pd.Timedelta(rules["frequency"]).total_seconds() * multiplier - 1
        if not np.isfinite(closing).all() or not (closing - opening).eq(expected).all():
            raise ValueError("Invalid raw candle duration")
    keep = _select_columns(df_raw, columns, rules, target_column)
    symbol, time = (columns["symbol_col"], columns["time_col"])
    df = df_raw.loc[:, keep].copy()
    df[time] = to_utc_datetime(df[time], rules["microseconds_above"])
    for column in rules["value_columns"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = _drop_rows(df, df[keep].isna().any(axis=1), "missing or unreadable values")
    df[symbol] = df[symbol].astype(str).str.strip().str.upper()
    df = _drop_rows(df, (df[rules["positive_columns"]] <= 0).any(axis=1), "non-positive prices")
    df = _drop_rows(df, (df[rules["non_negative_columns"]] < 0).any(axis=1), "negative volumes or trade counts")
    df = _drop_rows(df, _inconsistent_candles(df, columns), "open or close outside the low-high range")
    unique = df.drop_duplicates()
    if unique.duplicated(subset=[symbol, time], keep=False).any():
        raise ValueError("Conflicting duplicate candles")
    df = _drop_rows(df, ~np.isfinite(df[rules["value_columns"]]).all(axis=1), "nonfinite values")
    df = _drop_rows(df, df[columns["trades_col"]].mod(1).ne(0), "fractional trade counts")
    df = _drop_rows(df, df.duplicated(subset=[symbol, time], keep="first"), "duplicate candles")
    df = df.sort_values([symbol, time]).reset_index(drop=True)
    _log_missing_minutes(df, columns, rules["frequency"])
    logger.info("Cleaned candles: kept %d of %d rows", len(df), len(df_raw))
    return df


def clean_dataframe(
    df_raw: pd.DataFrame,
    target_column: str | None = None,
    columns: dict | None = None,
    rules: dict | None = None,
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """Inputs: df_raw, target_column, columns, rules, drop_invalid.
    Outputs: New DataFrame; malformed inference input raises rather than silently losing rows.
    Why: Keep one tested pipeline responsibility."""
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("Expected a DataFrame")
    if columns is not None and rules is not None:
        out = _clean_candles(df_raw, columns, rules, target_column)
        if target_column is None and (not drop_invalid) and (len(out) != len(df_raw)):
            raise ValueError("Invalid inference input: cleaning would drop rows")
        return out
    out = df_raw.copy()
    if target_column is not None:
        if target_column not in out:
            raise ValueError("Missing target column")
        return out.dropna(subset=[target_column]).drop_duplicates().reset_index(drop=True)
    return out
