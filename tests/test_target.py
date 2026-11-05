"""
test_target.py  [Tests for src/target.py]
=========================================
Future volatility has known values and is NaN when the future is incomplete; cutoffs come
from the training rows only; labels follow the cutoffs; rows without a future are dropped.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import add_lookback_features
from src.target import add_future_volatility, assign_target, fit_volatility_cutoffs
from tests.conftest import make_clean_candles

HORIZON = 15
FREQUENCY = "1min"


def test_unit_future_volatility_has_known_values(columns):
    """Unit test: future volatility has known values."""
    out = add_future_volatility(make_clean_candles(100), columns, HORIZON, FREQUENCY)
    assert out["future_rv"].iloc[0] == pytest.approx(0.01 * np.sqrt(15))
    assert out["future_rv"].iloc[-HORIZON:].isna().all()
    assert out["future_rv"].iloc[:-HORIZON].notna().all()


def test_unit_future_volatility_is_missing_before_a_gap(columns):
    """Unit test: future volatility is missing before a gap."""
    candles = make_clean_candles(100).drop(index=50).reset_index(drop=True)
    out = add_future_volatility(candles, columns, HORIZON, FREQUENCY)
    assert out["future_rv"].iloc[35:50].isna().all()  # their next 15 minutes include the gap
    assert out["future_rv"].iloc[34] == pytest.approx(0.01 * np.sqrt(15))


def test_unit_future_volatility_keeps_input_unchanged(columns):
    """Unit test: future volatility keeps input unchanged."""
    candles = make_clean_candles(40)
    before = candles.copy()
    add_future_volatility(candles, columns, HORIZON, FREQUENCY)
    pd.testing.assert_frame_equal(candles, before)


def test_unit_future_volatility_rejects_bad_input(columns):
    """Unit test: future volatility rejects bad input."""
    with pytest.raises(ValueError, match="No candles"):
        add_future_volatility(make_clean_candles(5).iloc[0:0], columns, HORIZON, FREQUENCY)
    with pytest.raises(ValueError, match="Missing columns for the target"):
        add_future_volatility(make_clean_candles(5).drop(columns=["close"]), columns, HORIZON, FREQUENCY)
    with pytest.raises(ValueError, match="at least 1 minute"):
        add_future_volatility(make_clean_candles(5), columns, 0, FREQUENCY)


def _rows(symbol, values):
    return pd.DataFrame({"symbol": symbol, "future_rv": values})


def test_unit_cutoffs_are_the_training_quantile_per_asset(columns):
    """Unit test: cutoffs are the training quantile per asset."""
    train = pd.concat([_rows("BTCUSDT", np.arange(1, 11)), _rows("ETHUSDT", np.arange(11, 21))])
    cutoffs = fit_volatility_cutoffs(train, columns, 0.9)
    assert cutoffs == {"BTCUSDT": pytest.approx(9.1), "ETHUSDT": pytest.approx(19.1)}


def test_unit_cutoffs_reject_bad_quantile_or_no_future(columns):
    """Unit test: cutoffs reject bad quantile or no future."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        fit_volatility_cutoffs(_rows("BTCUSDT", [1.0]), columns, 1.0)
    with pytest.raises(ValueError, match="No training rows"):
        fit_volatility_cutoffs(_rows("BTCUSDT", [np.nan]), columns, 0.9)


def test_unit_assign_target_labels_above_the_cutoff_and_drops_unknown_futures(columns):
    """Unit test: assign_target labels above the cutoff and drops unknown futures."""
    rows = _rows("BTCUSDT", [1.0, 5.0, np.nan, 9.0])
    labelled = assign_target(rows, columns, {"BTCUSDT": 4.0})
    assert labelled["high_volatility"].tolist() == [0, 1, 1]
    assert labelled.index.tolist() == [0, 1, 2]
    assert len(rows) == 4  # input unchanged


def test_unit_assign_target_rejects_symbols_without_a_cutoff(columns):
    """Unit test: assign_target rejects symbols without a cutoff."""
    with pytest.raises(ValueError, match="No volatility cutoff"):
        assign_target(_rows("SOLUSDT", [1.0]), columns, {"BTCUSDT": 4.0})


def test_integration_cutoffs_ignore_rows_outside_training(clean_df, columns):
    """Integration test: cutoffs ignore rows outside training."""
    labelled = add_future_volatility(clean_df, columns, HORIZON, FREQUENCY)
    table = add_lookback_features(labelled, columns, {"short": 15, "long": 60}, FREQUENCY)
    train = table[table["open_time"] < "2026-01-02"]
    cutoffs = fit_volatility_cutoffs(train, columns, 0.9)
    changed = table.copy()
    changed.loc[changed["open_time"] >= "2026-01-02", "future_rv"] *= 100
    assert fit_volatility_cutoffs(changed[changed["open_time"] < "2026-01-02"], columns, 0.9) == cutoffs
    positive_rate = assign_target(train, columns, cutoffs)["high_volatility"].mean()
    assert positive_rate == pytest.approx(0.1, abs=0.01)
