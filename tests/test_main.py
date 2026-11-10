"""Chronological orchestration, frozen thresholds and reproducible inference."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import main, utils
from tests.conftest import make_clean_candles


def _split_frame():
    times = pd.date_range("2026-01-01", periods=4320, freq="min", tz="UTC")
    return pd.DataFrame({"open_time": times, "original_row": np.arange(len(times))})


def _split_settings():
    return {"train_end": "2026-01-01 23:59", "validation_end": "2026-01-02 23:59"}


def test_unit_chronological_split_purges_crossing_labels_and_preserves_alignment():
    frame = _split_frame()
    before = frame.copy(deep=True)
    labels = frame.original_row.copy()
    parts = main.split_data(
        frame,
        labels,
        0.2,
        0.2,
        42,
        False,
        split_settings=_split_settings(),
        time_column="open_time",
        horizon=15,
        frequency="1min",
    )
    train, validation, test, y_train, y_validation, y_test = parts
    assert train.open_time.max() == pd.Timestamp("2026-01-01 23:44", tz="UTC")
    assert validation.open_time.min() == pd.Timestamp("2026-01-01 23:59", tz="UTC")
    assert validation.open_time.max() == pd.Timestamp("2026-01-02 23:44", tz="UTC")
    assert test.open_time.min() == pd.Timestamp("2026-01-02 23:59", tz="UTC")
    for x, y in zip(parts[:3], parts[3:]):
        assert x.index.equals(y.index)
        assert x.original_row.tolist() == y.tolist()
    assert set(train.original_row).isdisjoint(validation.original_row)
    assert set(validation.original_row).isdisjoint(test.original_row)
    assert len(frame) - sum(map(len, parts[:3])) == 28
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("case", ["stratify", "missing_dates", "misaligned", "naive", "bad_horizon"])
def test_unit_split_rejects_unsafe_settings(case):
    frame = _split_frame()
    y = frame.original_row.copy()
    kwargs = {"split_settings": _split_settings(), "time_column": "open_time", "horizon": 15}
    if case == "missing_dates":
        kwargs["split_settings"] = None
    elif case == "misaligned":
        y.index = y.index + 1
    elif case == "naive":
        frame.open_time = frame.open_time.dt.tz_localize(None)
    elif case == "bad_horizon":
        kwargs["horizon"] = 0
    with pytest.raises((ValueError, TypeError)):
        main.split_data(frame, y, 0.2, 0.2, 42, case == "stratify", **kwargs)


def test_unit_gate_uses_relative_improvement_and_accepts_equality():
    settings = {"metric": "average_precision", "min_ratio_vs_benchmark": 1.2}
    result = main.check_quality_gate({"average_precision": 0.36}, {"average_precision": 0.3}, settings)
    assert result["threshold"] == pytest.approx(0.36)
    assert result["passed"] is True
    assert not main.check_quality_gate({"average_precision": 0.35}, {"average_precision": 0.3}, settings)["passed"]


@pytest.mark.parametrize("case", ["nan_model", "infinite_benchmark", "test_gate", "wrong_direction", "bad_ratio"])
def test_unit_gate_rejects_invalid_evidence(case):
    gate = {"metric": "average_precision", "min_ratio_vs_benchmark": 1.2}
    scores, benchmark = {"average_precision": 0.5}, {"average_precision": 0.3}
    if case == "nan_model":
        scores["average_precision"] = float("nan")
    elif case == "infinite_benchmark":
        benchmark["average_precision"] = float("inf")
    elif case == "test_gate":
        gate["split"] = "test"
    elif case == "wrong_direction":
        gate["direction"] = "lower_is_better"
    else:
        gate["min_ratio_vs_benchmark"] = 0
    with pytest.raises(ValueError):
        main.check_quality_gate(scores, benchmark, gate)


def test_unit_settings_are_portable_and_not_mutated(tmp_path):
    before = copy.deepcopy(main.SETTINGS)
    resolved = main.resolved_settings(raw_dir=tmp_path / "external", output_root=tmp_path / "outputs")
    assert all(path.is_absolute() for path in resolved["paths"].values())
    assert resolved["paths"]["raw_dir"] == tmp_path / "external"
    resolved["features"]["numeric"].append("changed")
    assert main.SETTINGS == before
    assert main.SETTINGS["target"] == {"horizon": 15, "quantile": 0.9}
    assert main.SETTINGS["quality_gate"]["min_ratio_vs_benchmark"] == 1.2


def test_unit_training_cutoffs_exclude_later_and_purged_rows(cfg):
    frames = []
    for asset in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        frame = make_clean_candles(4320, symbol=asset)
        frame["future_rv"] = np.arange(len(frame)) % 10 + 1.0
        frame.loc[frame.open_time >= "2026-01-01 23:45", "future_rv"] = 1_000_000.0
        frames.append(frame)
    settings = copy.deepcopy(main.SETTINGS)
    settings.update(cfg)
    splits, cutoffs = main._labelled_splits(pd.concat(frames, ignore_index=True), settings)
    eligible = np.arange(1425) % 10 + 1.0
    assert cutoffs == {
        asset: pytest.approx(np.quantile(eligible, 0.9)) for asset in settings["rules"]["allowed_symbols"]
    }
    assert len(splits["train"]) == 1425 * 3


def test_unit_inventory_rejects_missing_asset_before_training(cfg):
    settings = copy.deepcopy(main.SETTINGS)
    settings.update(cfg)
    settings["rules"]["allowed_symbols"].append("MISSING")
    with pytest.raises(ValueError, match="inventory"):
        main.run_pipeline(settings)


def test_integration_entrypoint_saves_failed_candidate_without_overwriting_accepted(cfg, monkeypatch):
    settings = copy.deepcopy(main.SETTINGS)
    settings.update(cfg)
    settings["quality_gate"]["min_ratio_vs_benchmark"] = 1000.0
    sentinel = settings["paths"]["model"]
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_bytes(b"accepted-artifact-must-survive")
    calls = []
    evaluate_model = main.evaluate.evaluate_model

    def recorded_evaluation(*args, **kwargs):
        lock = settings["paths"]["metrics"].with_name("selection_lock.json")
        if calls:
            assert lock.exists(), "Selection must be locked before final-period evaluation"
            assert json.loads(lock.read_text())["quality_gate"]["passed"] is False
        calls.append(True)
        return evaluate_model(*args, **kwargs)

    monkeypatch.setattr(main.evaluate, "evaluate_model", recorded_evaluation)
    monkeypatch.setattr(main, "SETTINGS", settings)
    monkeypatch.setenv("ML_RAW_DIR", str(settings["paths"]["raw_dir"]))
    assert main.main() is None
    assert len(calls) == 2
    assert sentinel.read_bytes() == b"accepted-artifact-must-survive"
    report = json.loads(settings["paths"]["metrics"].read_text())
    assert report["quality_gate"]["passed"] is False
    assert "failed_candidates" in report["artifact"]
    assert report["approval_status"] == "failed_quality_gate"
    bundle = utils.load_model(Path(report["artifact"]))
    assert set(bundle["metadata"]["label_cutoffs"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert "prior" in bundle["metadata"]["test_status"]
    assert "future_rv" not in bundle["metadata"]["predictor_order"]
    assert "high_volatility" not in bundle["metadata"]["predictor_order"]
    assert report["validation"]["average_precision"] == pytest.approx(
        np.mean([report["validation"][f"average_precision_{asset}"] for asset in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]])
    )
    predictions = pd.read_csv(settings["paths"]["predictions"])
    assert len(predictions) == report["rows"]["test"]
    assert predictions.columns.tolist() == ["symbol", "open_time", "prediction", "probability"]
    clean = pd.read_csv(settings["paths"]["clean_data"], parse_dates=["open_time"])
    end = pd.to_datetime(predictions.loc[predictions.symbol == "BTCUSDT", "open_time"], utc=True).iloc[-1]
    window = clean.loc[(clean.symbol == "BTCUSDT") & (clean.open_time <= end)].tail(60)
    window = window.rename(columns={"trade_count": "trades"})
    columns = ["symbol", "open_time", "open", "high", "low", "close", "volume", "trades"]
    us, ms = window[columns].copy(), window[columns].copy()
    us["open_time"] = us.open_time.map(lambda t: t.value // 1000)
    ms["open_time"] = ms.open_time.map(lambda t: t.value // 1_000_000)
    payload = settings["paths"]["metrics"].with_name("raw-reload-check.json")
    payload.write_text(json.dumps({"us": us.to_dict("records"), "ms": ms.to_dict("records")}))
    script = (
        "import json,sys,pandas as pd; from src import infer; "
        "p=json.load(open(sys.argv[2])); "
        "a=infer.score_candles(pd.DataFrame(p['us']),sys.argv[1],sys.argv[3],unit='us'); "
        "b=infer.score_candles(pd.DataFrame(p['ms']),sys.argv[1],sys.argv[3],unit='ms'); "
        "assert a==b; print(json.dumps(a))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, report["artifact"], str(payload), (end + pd.Timedelta(minutes=1)).isoformat()],
        cwd=Path(main.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    score = json.loads(result.stdout)["risk_score"]
    expected = predictions.loc[
        (predictions.symbol == "BTCUSDT") & (pd.to_datetime(predictions.open_time, utc=True) == end)
    ]
    assert score == pytest.approx(expected.probability.iloc[0], abs=1e-12)
