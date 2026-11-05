"""Canonical features and leakage checks."""

import numpy as np
import pandas as pd
import pytest
from src import candle_contract, clean_data, features
from tests.conftest import make_clean_candles


def test_60_candles_and_future_isolation():
    d = make_clean_candles(200).rename(columns={"trade_count": "trades"})
    f, _ = candle_contract.past_features(d)
    assert f.loc[150, "return_30"] == pytest.approx(0.30)
    assert f.loc[150, "rv_60c_59r"] == pytest.approx(0.01 * np.sqrt(59))
    w, ok = candle_contract.past_features(d.iloc[91:151].reset_index(drop=True))
    assert ok.iloc[-1]
    np.testing.assert_allclose(w.iloc[-1], f.loc[150], atol=1e-12)
    d.loc[151:, "close"] *= 2
    changed, _ = candle_contract.past_features(d)
    np.testing.assert_allclose(changed.loc[150], f.loc[150], atol=1e-12)


def test_gap_and_assets():
    d = make_clean_candles(200).rename(columns={"trade_count": "trades"})
    _, ok = candle_contract.past_features(d.drop(index=100).reset_index(drop=True))
    assert not ok.iloc[100:159].any()
    with pytest.raises(ValueError):
        candle_contract.past_features(pd.concat([d, d.assign(symbol="ETHUSDT")]))


def test_builder_does_not_mutate(clean_df, columns, cfg):
    before = clean_df.copy()
    out = features.add_lookback_features(
        clean_df, columns, cfg["features"]["windows"], "1min", cfg["features"]["parameters"]
    )
    assert np.isfinite(out[cfg["features"]["numeric"]]).all().all()
    pd.testing.assert_frame_equal(before, clean_df)


def test_conflicting_duplicate(raw_df, columns, rules):
    extra = raw_df.iloc[[0]].copy()
    extra["volume"] += 1
    with pytest.raises(ValueError, match="Conflicting"):
        clean_data.clean_dataframe(pd.concat([raw_df, extra]), columns=columns, rules=rules, drop_invalid=True)
