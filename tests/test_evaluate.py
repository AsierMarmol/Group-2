"""
test_evaluate.py  [Tests for src/evaluate.py]
=============================================
Metrics are plain numbers with known values, macro-averaged per asset; single-class assets
are skipped; an object without predict_proba, missing scores or misaligned inputs are rejected.
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluate import _evaluate_group_model as evaluate_model
from src.evaluate import evaluate_scores
from src.features import get_feature_preprocessor
from src.train import _train_configured_model as train_model


def test_unit_scores_have_known_values():
    """Unit test: scores have known values."""
    metrics = evaluate_scores([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], ["BTCUSDT"] * 4)
    assert metrics["average_precision"] == pytest.approx(0.8333, abs=1e-4)  # scikit-learn docs example
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["positive_rate"] == pytest.approx(0.5)
    assert metrics["n_rows"] == 4
    assert metrics["average_precision_BTCUSDT"] == metrics["average_precision"]


def test_unit_average_precision_is_the_mean_over_assets():
    """Unit test: average precision is the mean over assets."""
    y = [0, 1, 0, 1, 0, 0, 1, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8]
    groups = ["BTCUSDT"] * 4 + ["ETHUSDT"] * 4
    metrics = evaluate_scores(y, scores, groups)
    per_asset = [
        metrics["average_precision_BTCUSDT"],
        metrics["average_precision_ETHUSDT"],
    ]
    assert metrics["average_precision"] == pytest.approx(np.mean(per_asset))
    assert metrics["average_precision_BTCUSDT"] == pytest.approx(1.0)


def test_unit_single_class_assets_are_skipped_with_a_warning(caplog):
    """Unit test: single class assets are skipped with a warning."""
    with pytest.raises(ValueError, match="needs both classes"):
        evaluate_scores(
            [0, 1, 0, 0],
            [0.1, 0.9, 0.5, 0.6],
            ["BTCUSDT", "BTCUSDT", "SOLUSDT", "SOLUSDT"],
        )


def test_unit_scores_reject_bad_inputs():
    """Unit test: scores reject bad inputs."""
    with pytest.raises(ValueError, match="No rows"):
        evaluate_scores([], [], [])
    with pytest.raises(ValueError, match="Lengths differ"):
        evaluate_scores([0, 1], [0.1], ["BTCUSDT", "BTCUSDT"])
    with pytest.raises(ValueError, match="missing values"):
        evaluate_scores([0, 1], [0.1, np.nan], ["BTCUSDT", "BTCUSDT"])
    with pytest.raises(ValueError, match="needs both classes"):
        evaluate_scores([0, 0], [0.1, 0.2], ["BTCUSDT", "BTCUSDT"])


def test_unit_evaluate_model_rejects_objects_without_predict_proba():
    """Unit test: evaluate_model rejects objects without predict_proba."""
    x = pd.DataFrame({"a": [1.0], "symbol": ["BTCUSDT"]})
    with pytest.raises(TypeError, match="no predict_proba"):
        evaluate_model(object(), x, [1], "symbol")


def test_integration_evaluate_a_trained_pipeline():
    """Integration test: evaluate a trained pipeline."""
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=400), "symbol": np.repeat(["BTCUSDT", "ETHUSDT"], 200)})
    y = pd.Series((x["a"] > 1).astype(int))
    model = train_model(
        x,
        y,
        get_feature_preprocessor(["a"], ["symbol"]),
        {"model_type": "logistic_regression", "random_state": 0, "params": {}},
    )
    metrics = evaluate_model(model, x, y, "symbol")
    assert 0.9 < metrics["average_precision"] <= 1.0
    assert 0 <= metrics["brier_score"] < 0.1
    with pytest.raises(ValueError, match="Group column 'asset' is missing"):
        evaluate_model(model, x, y, "asset")
