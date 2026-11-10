"""Cross-module release checks using public interfaces and disposable synthetic data."""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tests.conftest import make_clean_candles


def test_unit_raw_contract_rejects_missing_asset_inside_window():
    from src import candle_contract

    window = make_clean_candles(60).rename(columns={"trade_count": "trades"})
    window.loc[20, "symbol"] = None
    with pytest.raises(ValueError, match="symbol"):
        candle_contract.normalize_candles(
            window, allowed_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        )
    with pytest.raises(ValueError, match="[Ss]ymbol"):
        candle_contract.normalize_candles(window, symbol="BTCUSDT")
    with pytest.raises(ValueError, match="asset"):
        candle_contract.past_features(window)


def test_integration_public_io_preprocessor_training_and_inference(tmp_path):
    from src import features, infer, train, utils

    frame = pd.DataFrame(
        {
            "return_1": np.linspace(-1, 1, 90),
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"] * 30,
        }
    )
    before = frame.copy(deep=True)
    csv_path = tmp_path / "csv" / "features.csv"
    assert utils.save_csv(df=frame, filepath=csv_path) is None
    pd.testing.assert_frame_equal(utils.load_csv(filepath=csv_path), frame)
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(FileNotFoundError):
        utils.load_csv(filepath=tmp_path / "missing.csv")
    recipe = features.get_feature_preprocessor(["return_1"], ["symbol"], scale=True)
    with pytest.raises(NotFittedError):
        check_is_fitted(recipe)
    labels = frame.return_1.gt(0).astype(int)
    model = train.train_model(
        frame,
        labels,
        recipe,
        "classification",
        model_params={"model_type": "logistic_regression"},
        random_state=42,
    )
    with pytest.raises(NotFittedError):
        check_is_fitted(
            recipe
        )  # Training must clone rather than mutate the unfitted recipe.
    expected_mean = frame.return_1.mean()
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert scaler.mean_[0] == pytest.approx(expected_mean)
    artifact = tmp_path / "model" / "pipeline.joblib"
    assert utils.save_model(model=model, filepath=artifact) is None
    loaded = utils.load_model(filepath=artifact)
    rows = frame.iloc[10:15].copy()
    original = infer.run_inference(model, rows, include_proba=True)
    restored = infer.run_inference(loaded, rows, include_proba=True)
    pd.testing.assert_frame_equal(original, restored)
    assert restored.index.equals(rows.index)
    assert restored.columns.tolist() == ["prediction", "probability"]
    assert infer.run_inference(loaded, rows).columns.tolist() == ["prediction"]
    infer.run_inference(loaded, rows.assign(return_1=1_000_000), include_proba=True)
    assert loaded.named_steps["preprocess"].named_transformers_["numeric"].mean_[
        0
    ] == pytest.approx(expected_mean)
    with pytest.raises(ValueError, match="unknown"):
        infer.run_inference(loaded, rows.assign(symbol="DOGEUSDT"))
    for groups in [([], []), (["same"], ["same"])]:
        with pytest.raises(ValueError):
            features.get_feature_preprocessor(*groups)


def test_unit_release_predictors_exclude_future_labels_and_timestamps():
    from src import main

    for role in ["future_volatility_col", "target_col", "time_col"]:
        settings = copy.deepcopy(main.SETTINGS)
        settings["features"]["numeric"].append(settings["columns"][role])
        with pytest.raises(ValueError, match="exclude"):
            main._input_columns(settings)


def test_integration_raw_reload_accepts_units_and_rejects_invalid_requests(cfg):
    from src import candle_contract, infer, main, utils

    report = main.run_pipeline(cfg)
    bundle = utils.load_model(report["artifact"])
    window = make_clean_candles(60).rename(columns={"trade_count": "trades"})
    before = window.copy(deep=True)
    decision = window.open_time.iloc[-1] + pd.Timedelta(minutes=1)
    original = infer.score_candles(window, report["artifact"], decision)
    for unit, divisor in [("us", 1000), ("ms", 1_000_000)]:
        encoded = window.copy()
        encoded["open_time"] = encoded.open_time.map(lambda time: time.value // divisor)
        assert (
            infer.score_candles(encoded, report["artifact"], decision, unit=unit)
            == original
        )
    fractional = window.copy()
    fractional["trades"] = fractional.trades.astype(float)
    fractional.loc[20, "trades"] = 1.5
    missing_symbol = window.copy()
    missing_symbol.loc[20, "symbol"] = None
    gap = window.copy()
    gap.loc[30:, "open_time"] += pd.Timedelta(minutes=1)
    cases = {
        "empty": window.iloc[:0],
        "short": window.iloc[:-1],
        "duplicate": pd.concat([window.iloc[:1], window.iloc[:-1]], ignore_index=True),
        "unordered": window.iloc[::-1],
        "gap": gap,
        "negative close": window.assign(close=-1.0),
        "infinite volume": window.assign(volume=np.inf),
        "zero denominator": window.assign(volume=0.0),
        "fractional trades": fractional,
        "unsupported symbol": window.assign(symbol="DOGEUSDT"),
        "missing symbol": missing_symbol,
        "missing field": window.drop(columns="close"),
        "naive timestamps": window.assign(
            open_time=window.open_time.dt.tz_localize(None)
        ),
        "off-minute": window.assign(
            open_time=window.open_time + pd.Timedelta(seconds=1)
        ),
    }
    for name, frame in cases.items():
        with pytest.raises(ValueError) as caught:
            infer.score_candles(frame, report["artifact"], decision)
        assert str(caught.value), name
    with pytest.raises(ValueError, match="completed"):
        infer.score_candles(
            window, report["artifact"], decision - pd.Timedelta(nanoseconds=1)
        )
    with pytest.raises(ValueError, match="timezone"):
        infer.score_candles(window, report["artifact"], decision.tz_localize(None))
    with pytest.raises(ValueError, match="explicit"):
        candle_contract.predict_candles(window, bundle, mode="replay")
    with pytest.raises(ValueError, match="not approved"):
        candle_contract.predict_candles(
            window, bundle, mode="live", as_of=decision, max_age_seconds=60
        )
    # An in-memory test flag exercises policy checks; it is never written to an artifact.
    approved = {
        "model": bundle["model"],
        "metadata": {**bundle["metadata"], "approval_status": "approved"},
    }
    for limit in [None, -1, float("inf")]:
        with pytest.raises(ValueError, match="freshness"):
            candle_contract.predict_candles(
                window, approved, mode="live", as_of=decision, max_age_seconds=limit
            )
    with pytest.raises(ValueError, match="Stale"):
        candle_contract.predict_candles(
            window,
            approved,
            mode="live",
            as_of=decision + pd.Timedelta(minutes=2),
            max_age_seconds=60,
        )
    assert bundle["metadata"]["approval_status"] != "approved"
    pd.testing.assert_frame_equal(window, before)
