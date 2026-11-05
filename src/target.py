"""Target functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _future_volatility_for_symbol(group: pd.DataFrame, columns: dict, horizon: int, frequency: str) -> pd.Series:
    """Realized volatility of the next ``horizon`` one-minute returns of one asset."""
    time = columns["time_col"]
    prices = group.set_index(time)[columns["close_col"]].sort_index()
    full_index = pd.date_range(prices.index.min(), prices.index.max(), freq=frequency, name=time)
    squared_returns = np.log(prices.reindex(full_index)).diff().pow(2)
    future_sum = squared_returns.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    future = np.sqrt(future_sum).reindex(prices.index)
    return pd.Series(future.to_numpy(), index=group.sort_values(time).index)


def add_future_volatility(candles: pd.DataFrame, columns: dict, horizon: int, frequency: str) -> pd.DataFrame:
    """Inputs: candles, columns, horizon, frequency.
    Outputs: Copied candles with future RV used only for label construction.
    Why: Add each row's realized volatility over the next ``horizon`` minutes."""
    if candles.empty:
        raise ValueError("No candles to label")
    missing = [
        columns[role] for role in ("symbol_col", "time_col", "close_col") if columns[role] not in candles.columns
    ]
    if missing:
        raise ValueError(f"Missing columns for the target: {missing}")
    if int(horizon) < 1:
        raise ValueError(f"Horizon must be at least 1 minute, got {horizon}")
    out = candles.copy()
    out[columns["future_volatility_col"]] = pd.concat(
        [
            _future_volatility_for_symbol(group, columns, int(horizon), frequency)
            for _, group in candles.groupby(columns["symbol_col"], sort=False)
        ]
    )
    return out


def fit_volatility_cutoffs(train: pd.DataFrame, columns: dict, quantile: float) -> dict[str, float]:
    """Inputs: train, columns, quantile.
    Outputs: Per-asset quantiles of eligible training outcomes only.
    Why: Learn each asset's high-volatility cutoff from the TRAINING split only."""
    if not 0 < float(quantile) < 1:
        raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")
    labelled = train.dropna(subset=[columns["future_volatility_col"]])
    if labelled.empty:
        raise ValueError("No training rows with a known future volatility")
    cutoffs = labelled.groupby(columns["symbol_col"])[columns["future_volatility_col"]].quantile(float(quantile))
    logger.info("Volatility cutoffs (training only, quantile %.2f): %s", quantile, cutoffs.round(6).to_dict())
    return {str(symbol): float(value) for symbol, value in cutoffs.items()}


def assign_target(df: pd.DataFrame, columns: dict, cutoffs: dict[str, float]) -> pd.DataFrame:
    """Inputs: df, columns, cutoffs.
    Outputs: Labelled rows using frozen cutoffs; incomplete outcomes excluded.
    Why: Label rows 1 when their future volatility is above their asset's cutoff, else 0."""
    unknown = sorted(set(df[columns["symbol_col"]]) - set(cutoffs))
    if unknown:
        raise ValueError(f"No volatility cutoff for symbols: {unknown}")
    labelled = df.dropna(subset=[columns["future_volatility_col"]]).copy()
    dropped = len(df) - len(labelled)
    if dropped:
        logger.info("Dropped %d row(s) without a known future (cannot be labelled)", dropped)
    cutoff = labelled[columns["symbol_col"]].map(cutoffs)
    labelled[columns["target_col"]] = (labelled[columns["future_volatility_col"]] > cutoff).astype(int)
    return labelled.reset_index(drop=True)
