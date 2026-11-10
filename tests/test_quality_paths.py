"""Public error paths and inference policy boundaries, using synthetic inputs only."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import (
    candle_contract,
    clean_data,
    evaluate,
    features,
    infer,
    load_data,
    main,
    train,
    validate,
)
from tests.conftest import make_clean_candles, make_raw_candles


@pytest.fixture
def classifier():
    """A real, small fitted classifier for public-interface checks."""
    frame = pd.DataFrame({"value": [-2.0, -1.0, 1.0, 2.0], "symbol": ["BTCUSDT"] * 4})
    labels = pd.Series([0, 0, 1, 1])
    recipe = features.get_feature_preprocessor(["value"], ["symbol"])
    model = train.train_model(frame, labels, recipe, "classification")
    return model, frame, labels


@pytest.mark.parametrize(
    "values,unit",
    [
        ([np.nan], "auto"),
        ([1.5], "ms"),
        ([1_700_000_000], "auto"),
        ([1_700_000_000_000], "seconds"),
    ],
)
def test_unit_timestamp_rejects_noninteger_missing_or_unsupported_unit(values, unit):
    with pytest.raises(ValueError, match="timestamp|unit"):
        candle_contract.utc_open(pd.Series(values), unit=unit)


def test_unit_symbol_override_can_supply_missing_column_but_cannot_change_asset():
    frame = make_clean_candles(60).rename(columns={"trade_count": "trades"})
    normalized, valid = candle_contract.normalize_candles(
        frame.drop(columns="symbol"), symbol="BTCUSDT"
    )
    assert normalized.symbol.eq("BTCUSDT").all() and valid.all()
    with pytest.raises(ValueError, match="Symbol mismatch"):
        candle_contract.normalize_candles(frame, symbol="ETHUSDT")
    with pytest.raises(ValueError, match="supported symbol"):
        candle_contract.normalize_candles(frame.drop(columns="symbol"))


@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_unit_preprocessor_supports_a_single_declared_group(kind):
    numeric, categorical = (["value"], []) if kind == "numeric" else ([], ["symbol"])
    recipe = features.get_feature_preprocessor(numeric, categorical)
    # The only production fitting remains in train.py; tests may inspect an unfitted recipe.
    assert [name for name, _, _ in recipe.transformers] == [
        kind if kind == "numeric" else "symbol"
    ]


@pytest.mark.parametrize("case", ["empty", "windows", "columns"])
def test_unit_feature_builder_rejects_invalid_schema_and_windows(case, columns):
    frame = make_clean_candles(60)
    windows = {"short": 15, "long": 60}
    if case == "empty":
        frame = frame.iloc[:0]
    elif case == "windows":
        windows["short"] = 60
    else:
        frame = frame.drop(columns="close")
    with pytest.raises(ValueError, match="No candles|Windows|Missing columns"):
        features.add_lookback_features(frame, columns, windows, "1min")


@pytest.mark.parametrize("case", ["problem", "indices", "labels"])
def test_unit_public_training_rejects_unsupported_problem_or_labels(case, classifier):
    _, frame, labels = classifier
    recipe = features.get_feature_preprocessor(["value"], ["symbol"])
    problem = "regression" if case == "problem" else "classification"
    if case == "indices":
        labels.index = labels.index + 1
    elif case == "labels":
        labels = labels.replace({1: 2})
    with pytest.raises(ValueError, match="classification|indices|labels"):
        train.train_model(frame, labels, recipe, problem)


@pytest.mark.parametrize("case", ["problem", "indices", "classes", "groups"])
def test_unit_public_evaluation_rejects_incompatible_contract(case, classifier):
    model, frame, labels = classifier
    problem = "regression" if case == "problem" else "classification"
    if case == "indices":
        labels.index = labels.index + 1
    elif case == "classes":
        model = SimpleNamespace(classes_=[1, 0])
    expected = ["BTCUSDT", "ETHUSDT"] if case == "groups" else None
    with pytest.raises(
        ValueError, match="classification|indices|class mapping|configured assets"
    ):
        evaluate.evaluate_model(model, frame, labels, problem, "symbol", expected)


def test_unit_public_evaluation_without_group_column_and_index_mismatch(classifier):
    model, frame, labels = classifier
    result = evaluate.evaluate_model(model, frame, labels, "classification")
    assert result["average_precision_all"] == 1.0
    with pytest.raises(ValueError, match="indices"):
        evaluate.evaluate_scores(
            labels, pd.Series([0.1, 0.2, 0.8, 0.9], index=[4, 5, 6, 7]), frame.symbol
        )


def test_unit_public_inference_rejects_missing_probability_method_and_class_order(
    classifier,
):
    _, frame, _ = classifier
    with pytest.raises(TypeError, match="predict_proba"):
        infer.run_inference(object(), frame)
    with pytest.raises(ValueError, match="class mapping"):
        infer.run_inference(
            SimpleNamespace(classes_=[1, 0], predict_proba=lambda _: None), frame
        )


def test_unit_loader_rejects_ambiguous_symbol_configuration(tmp_path, cfg):
    with pytest.raises(ValueError, match="symbol column is required"):
        load_data.load_raw_data(tmp_path, cfg["data"])
    with pytest.raises(ValueError, match="clashes"):
        load_data.load_raw_data(tmp_path, cfg["data"], "open_time")


def test_unit_generic_cleaning_preserves_input_and_removes_only_missing_targets_and_duplicates():
    frame = pd.DataFrame({"value": [1, 1, 3], "label": [0.0, 0.0, np.nan]})
    before = frame.copy(deep=True)
    pd.testing.assert_frame_equal(clean_data.clean_dataframe(frame), frame)
    result = clean_data.clean_dataframe(frame, target_column="label")
    assert result.to_dict("list") == {"value": [1], "label": [0.0]}
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="Missing target"):
        clean_data.clean_dataframe(frame, target_column="absent")
    with pytest.raises(TypeError, match="DataFrame"):
        clean_data.clean_dataframe([])


def test_unit_cleaning_rejects_invalid_candle_duration(columns, rules):
    frame = make_raw_candles(3, 42).assign(symbol="BTCUSDT")
    frame.loc[1, "close_time"] -= 1
    with pytest.raises(ValueError, match="duration"):
        clean_data.clean_dataframe(frame, columns=columns, rules=rules)


@pytest.mark.parametrize(
    "case", ["type", "empty", "column", "null", "timezone", "infinite", "trades"]
)
def test_unit_public_validation_rejects_invalid_rows(case, columns, rules):
    frame = make_clean_candles(3)
    if case == "type":
        frame = []
    elif case == "empty":
        frame = frame.iloc[:0]
    elif case == "column":
        frame = frame.drop(columns="close")
    elif case == "null":
        frame.loc[0, "close"] = np.nan
    elif case == "timezone":
        frame.open_time = frame.open_time.dt.tz_convert("Europe/Madrid")
    elif case == "infinite":
        frame.loc[0, "volume"] = np.inf
    else:
        frame["trade_count"] = 1.5
    with pytest.raises((ValueError, TypeError)):
        validate.validate_dataframe(frame, ["close"], columns=columns, rules=rules)


def test_unit_generic_validation_needs_no_candle_settings():
    assert (
        validate.validate_dataframe(pd.DataFrame({"label": [0, 1]}), [], "label")
        is True
    )


@pytest.mark.parametrize(
    "case", ["frequency", "reversed", "empty", "reserved", "duplicates"]
)
def test_unit_chronological_split_rejects_invalid_boundaries_or_identifiers(case):
    frame = pd.DataFrame(
        {"open_time": pd.date_range("2026-01-01", periods=4320, freq="min", tz="UTC")}
    )
    boundaries = {"train_end": "2026-01-01 23:59", "validation_end": "2026-01-02 23:59"}
    frequency = "0min" if case == "frequency" else "1min"
    if case == "reversed":
        boundaries["train_end"] = "2026-01-03"
    elif case == "empty":
        frame = frame.iloc[:10]
    elif case == "reserved":
        frame["__split_target__"] = 0
    elif case == "duplicates":
        frame.index = [0] * len(frame)
    labels = pd.Series(0, index=frame.index)
    with pytest.raises(ValueError):
        main.split_data(
            frame,
            labels,
            0.2,
            0.2,
            42,
            False,
            boundaries,
            "open_time",
            frequency=frequency,
        )


def test_unit_timezone_aware_boundaries_preserve_chronology():
    frame = pd.DataFrame(
        {"open_time": pd.date_range("2026-01-01", periods=4320, freq="min", tz="UTC")}
    )
    boundaries = {
        "train_end": "2026-01-02 00:59+01:00",
        "validation_end": "2026-01-03 00:59+01:00",
    }
    result = main.split_by_time(frame, "open_time", boundaries, 15, "1min")
    assert result["train"].open_time.max() == pd.Timestamp("2026-01-01 23:44", tz="UTC")


def test_unit_gate_rejects_a_different_metric():
    with pytest.raises(ValueError, match="average_precision"):
        main.check_quality_gate(
            {"accuracy": 0.9},
            {"accuracy": 0.5},
            {"metric": "accuracy", "min_ratio_vs_benchmark": 1.2},
        )


def test_integration_raw_policy_checks_class_mapping_scores_and_freshness(cfg):
    """Policy checks use a real saved model; only invalid model outputs are substituted."""
    report = main.run_pipeline(cfg)
    from src import utils

    bundle = utils.load_model(report["artifact"])
    window = make_clean_candles(60).rename(columns={"trade_count": "trades"})
    decision = window.open_time.iloc[-1] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="mode"):
        candle_contract.predict_candles(window, bundle, mode="invalid", as_of=decision)
    for classes, score, message in [
        ([1, 0], 0.5, "class mapping"),
        ([0, 1], np.nan, "model score"),
        ([0, 1], 1.1, "model score"),
    ]:
        altered = {
            **bundle,
            "model": SimpleNamespace(
                classes_=classes,
                predict_proba=lambda _, score=score: np.array([[1 - score, score]]),
            ),
        }
        with pytest.raises(ValueError, match=message):
            candle_contract.predict_candles(window, altered, as_of=decision)
    approved = {
        **bundle,
        "metadata": {**bundle["metadata"], "approval_status": "approved"},
    }
    result = candle_contract.predict_candles(
        window, approved, mode="live", as_of=decision, max_age_seconds=0
    )
    assert result["status"] == "ok" and 0 <= result["risk_score"] <= 1
    # Current-time selection is tested locally; no API polling or external service is involved.
    current = window.copy()
    current["open_time"] = pd.date_range(
        pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=60),
        periods=60,
        freq="min",
    )
    assert (
        candle_contract.predict_candles(
            current, approved, mode="live", max_age_seconds=120
        )["status"]
        == "ok"
    )
    assert bundle["metadata"]["approval_status"] != "approved"


def test_integration_pipeline_rejects_out_of_scope_dates_and_missing_evaluation_assets(
    cfg,
):
    settings = copy.deepcopy(cfg)
    # The file inventory remains January, but some loaded candles precede this start date.
    settings["scope"]["start"] = "2026-01-02"
    with pytest.raises(ValueError, match="outside approved date scope"):
        main.run_pipeline(settings)
    settings["scope"] = None
    table = main._prepare_table(settings)
    splits, _ = main._labelled_splits(table, settings)
    invalid_gate = copy.deepcopy(settings)
    invalid_gate["quality_gate"]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        main._train_and_evaluate(splits, invalid_gate)
    splits["validation"] = splits["validation"].loc[
        splits["validation"].symbol != "SOLUSDT"
    ]
    with pytest.raises(ValueError, match="all three"):
        main._train_and_evaluate(splits, settings)


def test_unit_candle_cleaning_never_silently_discards_inference_rows(columns, rules):
    frame = make_raw_candles(4, 42).assign(symbol="BTCUSDT")
    frame.loc[0, "volume"] = -1
    with pytest.raises(ValueError, match="cleaning would drop rows"):
        clean_data.clean_dataframe(frame, columns=columns, rules=rules)


def test_unit_generic_loader_reads_a_single_csv_without_candle_settings(tmp_path):
    frame = pd.DataFrame({"value": [1, 2], "label": [0, 1]})
    source = tmp_path / "sample.csv"
    frame.to_csv(source, index=False)
    pd.testing.assert_frame_equal(load_data.load_raw_data(source), frame)
