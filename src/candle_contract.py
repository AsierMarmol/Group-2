"""Candle contract functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _numeric_open(values: pd.Series, unit: str, microseconds_above: float) -> pd.Series:
    """Validate integer Binance timestamps and resolve their declared or inferred unit."""
    if not np.isfinite(values).all() or not values.mod(1).eq(0).all():
        raise ValueError("Invalid numeric opening timestamp")
    if unit == "auto":
        median = values.abs().median()
        unit = (
            "us"
            if median >= microseconds_above
            else "ms" if median >= 100000000000.0 else None
        )
    if unit not in ("us", "ms"):
        raise ValueError("Timestamp unit must be us or ms")
    return pd.to_datetime(values, unit=unit, utc=True, errors="raise")


def _require_symbol(
    frame: pd.DataFrame, symbol: str | None, allowed_symbols: list[str] | None
) -> None:
    """Attach an explicit asset when supplied, then require one nonmissing supported asset."""
    if symbol is not None:
        if "symbol" in frame and (
            frame.symbol.isna().any() or not frame.symbol.eq(symbol).all()
        ):
            raise ValueError("Symbol mismatch")
        frame["symbol"] = symbol
    if (
        "symbol" not in frame
        or frame.symbol.isna().any()
        or frame.symbol.nunique() != 1
    ):
        raise ValueError("Require one supported symbol")
    if allowed_symbols is not None and frame.symbol.iloc[0] not in allowed_symbols:
        raise ValueError("Require one supported symbol")


def utc_open(
    values: pd.Series, unit: str = "auto", microseconds_above: float = 100000000000000.0
) -> pd.Series:
    """Inputs: values, unit, microseconds_above.
    Outputs: Minute-aligned timezone-aware UTC Series, or a timestamp validation error.
    Why: Convert Binance opening timestamps; historical=us, REST default=ms."""
    s = pd.Series(values).copy()
    if pd.api.types.is_numeric_dtype(s):
        out = _numeric_open(s, unit, microseconds_above)
    else:
        if any((pd.Timestamp(x).tzinfo is None for x in s)):
            raise ValueError("Opening timestamps must declare UTC or another timezone")
        out = pd.to_datetime(s, utc=True, errors="raise")
    if out.isna().any() or not out.eq(out.dt.floor("min")).all():
        raise ValueError("Opening times must be complete UTC minute boundaries")
    return out.astype("datetime64[us, UTC]")


def normalize_candles(
    frame: pd.DataFrame,
    symbol: str | None = None,
    unit: str = "auto",
    allowed_symbols: list[str] | None = None,
    microseconds_above: float = 100000000000000.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inputs: frame, symbol, unit, allowed_symbols, microseconds_above.
    Outputs: Single-asset UTC frame and validity mask; order/duplicate/schema errors raise.
    Why: One pipeline responsibility."""
    d = frame.copy().reset_index(drop=True)
    required = ["open_time", "open", "high", "low", "close", "volume", "trades"]
    if any((c not in d for c in required)):
        raise ValueError(
            "Missing candle columns: " + str([c for c in required if c not in d])
        )
    if len(d) == 0:
        raise ValueError("Empty candle input")
    d["open_time"] = utc_open(
        d.open_time, unit=unit, microseconds_above=microseconds_above
    ).to_numpy()
    d["open_time"] = pd.to_datetime(d.open_time, utc=True)
    _require_symbol(d, symbol, allowed_symbols)
    if d.open_time.duplicated().any() or not d.open_time.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered opening times")
    for c in ["open", "high", "low", "close", "volume", "trades"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    valid = (
        np.isfinite(d[["open", "high", "low", "close", "volume", "trades"]]).all(axis=1)
        & d[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & d[["volume", "trades"]].ge(0).all(axis=1)
        & d.trades.mod(1).eq(0)
        & d.low.le(d[["open", "close"]].min(axis=1))
        & d.high.ge(d[["open", "close"]].max(axis=1))
    )
    return (d, valid)


def past_features(
    d: pd.DataFrame,
    input_candles: int = 60,
    return_windows: tuple = (1, 5, 15, 30),
    rv_windows: tuple = (5, 15, 30),
    activity_short: int = 5,
    frequency: str = "1min",
) -> tuple[pd.DataFrame, pd.Series]:
    """Inputs: d, input_candles, return_windows, rv_windows, activity_short, frequency.
    Outputs: Past-only numeric predictors and a consecutive-window validity mask.
    Why: One pipeline responsibility."""
    if d.symbol.isna().any() or d.symbol.nunique() != 1:
        raise ValueError("Features require one asset")
    f = pd.DataFrame(index=d.index)
    logc = np.log(d.close)
    r = logc.diff()
    for k in return_windows:
        f[f"return_{k}"] = logc - logc.shift(k)
    for k in rv_windows:
        f[f"rv_{k}"] = np.sqrt(r.pow(2).rolling(k, min_periods=k).sum())
    f["rv_60c_59r"] = np.sqrt(
        r.pow(2).rolling(input_candles - 1, min_periods=input_candles - 1).sum()
    )
    f["range_mean_60"] = (
        ((d.high - d.low) / d.close)
        .rolling(input_candles, min_periods=input_candles)
        .mean()
    )
    for c, name in [("volume", "volume_ratio_5_60"), ("trades", "trades_ratio_5_60")]:
        denom = (
            d[c]
            .rolling(input_candles, min_periods=input_candles)
            .mean()
            .replace(0, np.nan)
        )
        f[name] = (
            d[c].rolling(activity_short, min_periods=activity_short).mean() / denom
        )
    decision = d.open_time + pd.Timedelta(frequency)
    for label, values, period in [
        ("hour", decision.dt.hour, 24),
        ("weekday", decision.dt.day_of_week, 7),
    ]:
        f[label + "_sin"] = np.sin(2 * np.pi * values / period)
        f[label + "_cos"] = np.cos(2 * np.pi * values / period)
    consecutive = (
        d.open_time.diff()
        .eq(pd.Timedelta(frequency))
        .rolling(input_candles - 1, min_periods=input_candles - 1)
        .sum()
        .eq(input_candles - 1)
    )
    return (f, consecutive & np.isfinite(f).all(axis=1))


def _completed_decision(
    frame: pd.DataFrame, mode: str, as_of: str | pd.Timestamp | None, frequency: str
) -> tuple:
    """Resolve explicit replay/live time and reject an input candle that is not complete."""
    if mode not in ("live", "replay"):
        raise ValueError("mode must be live or replay")
    if mode == "replay" and as_of is None:
        raise ValueError("Replay requires explicit as_of")
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone aware")
    decision = frame.open_time.iloc[-1] + pd.Timedelta(frequency)
    if now < decision:
        raise ValueError("Last candle is not completed")
    return decision, now


def _require_live_approval(
    metadata: dict, now: pd.Timestamp, decision: pd.Timestamp, max_age: float | None
) -> None:
    """Enforce artifact approval and an explicit freshness policy for the optional live guard."""
    if metadata["approval_status"] != "approved":
        raise ValueError("Candidate is not approved for live use")
    if max_age is None or not np.isfinite(max_age) or max_age < 0:
        raise ValueError("Live use requires a nonnegative freshness limit")
    if (now - decision).total_seconds() > max_age:
        raise ValueError("Stale candle window")


def _raw_prediction_row(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Build the final eligible past-only row using the saved training recipe."""
    predictors, eligible = past_features(
        frame,
        input_candles=settings["features"]["windows"]["long"],
        frequency=settings["rules"]["frequency"],
        **settings["features"]["parameters"],
    )
    if not eligible.iloc[-1]:
        raise ValueError("Invalid feature window")
    row = predictors.iloc[[-1]].copy()
    row[settings["columns"]["symbol_col"]] = frame.symbol.iloc[0]
    return row


def _risk_score(model: object, row: pd.DataFrame) -> float:
    """Require the saved binary class order and a finite probability-shaped risk score."""
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected class mapping")
    score = float(model.predict_proba(row)[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model score")
    return score


def predict_candles(
    candles: pd.DataFrame,
    bundle_or_path: dict | str | Path,
    *,
    mode: str = "replay",
    as_of: str | pd.Timestamp | None = None,
    max_age_seconds: float | None = None,
    unit: str = "auto",
    symbol: str | None = None,
) -> dict:
    """Inputs: candles, bundle_or_path.
    Outputs: Versioned risk_score after window, completion, asset and freshness checks.
    Why: Keep one tested pipeline responsibility."""
    bundle = (
        joblib.load(bundle_or_path)
        if isinstance(bundle_or_path, (str, Path))
        else bundle_or_path
    )
    settings = bundle["metadata"]["settings"]
    frequency = settings["rules"]["frequency"]
    count = settings["features"]["windows"]["long"]
    d, valid = normalize_candles(
        candles,
        symbol=symbol,
        unit=unit,
        allowed_symbols=settings["rules"]["allowed_symbols"],
        microseconds_above=settings["rules"]["microseconds_above"],
    )
    if (
        len(d) != count
        or not valid.all()
        or (not d.open_time.diff().iloc[1:].eq(pd.Timedelta(frequency)).all())
    ):
        raise ValueError("Require exactly the configured consecutive valid candles")
    decision, now = _completed_decision(d, mode, as_of, frequency)
    if mode == "live":
        _require_live_approval(bundle["metadata"], now, decision, max_age_seconds)
    row = _raw_prediction_row(d, settings)
    score = _risk_score(bundle["model"], row[bundle["metadata"]["predictor_order"]])
    return {
        "symbol": d.symbol.iloc[0],
        "risk_score": score,
        "decision_time": decision.isoformat(),
        "model_version": bundle["metadata"]["model_version"],
        "calibration_status": "uncalibrated",
        "approval_status": bundle["metadata"]["approval_status"],
        "mode": mode,
        "status": "ok",
        "horizon_minutes": settings["target"]["horizon"],
    }
