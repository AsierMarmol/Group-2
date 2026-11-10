"""
test_train.py  [Tests for src/train.py]
=======================================
Training returns a fitted pipeline that predicts; the same data and seed give the same model;
the recipe passed in is never fitted; empty, misaligned or single-class data raise clear errors.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from src.features import add_lookback_features, get_feature_preprocessor
from src.target import add_future_volatility, assign_target, fit_volatility_cutoffs
from src.train import _train_configured_model as train_model
from src.train import build_estimator

LOGISTIC = {"model_type": "logistic_regression", "random_state": 0, "params": {"max_iter": 200}}
BOOSTING = {"model_type": "hist_gradient_boosting", "random_state": 0, "params": {"max_iter": 10}}


def _toy(rows=300, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "symbol": rng.choice(["BTCUSDT", "ETHUSDT"], size=rows),
        }
    )
    y = pd.Series((x["a"] + rng.normal(0, 0.5, rows) > 1).astype(int))
    return (x, y)


def _recipe():
    return get_feature_preprocessor(["a", "b"], ["symbol"])


def test_unit_build_estimator_uses_type_params_and_seed():
    """Unit test: build_estimator uses type params and seed."""
    model = build_estimator(LOGISTIC)
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 200 and model.random_state == 0
    assert isinstance(build_estimator(BOOSTING), HistGradientBoostingClassifier)


def test_unit_build_estimator_accepts_missing_params():
    """Unit test: build_estimator accepts missing params."""
    model = build_estimator({"model_type": "logistic_regression", "random_state": 1, "params": None})
    assert model.random_state == 1


def test_unit_build_estimator_rejects_unknown_types():
    """Unit test: build_estimator rejects unknown types."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_estimator({"model_type": "magic_forest", "random_state": 0, "params": {}})


def test_unit_train_returns_a_fitted_pipeline_that_predicts():
    """Unit test: train returns a fitted pipeline that predicts."""
    x, y = _toy()
    model = train_model(x, y, _recipe(), LOGISTIC)
    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["preprocess", "model"]
    probabilities = model.predict_proba(x)[:, 1]
    assert probabilities.min() >= 0 and probabilities.max() <= 1


def test_unit_same_data_and_seed_give_the_same_model():
    """Unit test: same data and seed give the same model."""
    x, y = _toy()
    first = train_model(x, y, _recipe(), BOOSTING).predict_proba(x)
    second = train_model(x, y, _recipe(), BOOSTING).predict_proba(x)
    np.testing.assert_allclose(first, second)


def test_unit_train_never_fits_the_recipe_passed_in():
    """Unit test: train never fits the recipe passed in."""
    x, y = _toy()
    recipe = _recipe()
    train_model(x, y, recipe, LOGISTIC)
    with pytest.raises(NotFittedError):
        check_is_fitted(recipe)


def test_unit_train_rejects_bad_training_data():
    """Unit test: train rejects bad training data."""
    x, y = _toy()
    with pytest.raises(TypeError, match="must be a DataFrame"):
        train_model(x.to_numpy(), y, _recipe(), LOGISTIC)
    with pytest.raises(ValueError, match="no rows"):
        train_model(x.iloc[0:0], y.iloc[0:0], _recipe(), LOGISTIC)
    with pytest.raises(ValueError, match="rows but y_train has"):
        train_model(x, y.iloc[:-1], _recipe(), LOGISTIC)
    with pytest.raises(ValueError, match="both classes"):
        train_model(x, pd.Series(0, index=x.index), _recipe(), LOGISTIC)


def test_integration_train_on_pipeline_features(clean_df, columns):
    """Integration test: train on pipeline features."""
    table = add_lookback_features(
        add_future_volatility(clean_df, columns, 15, "1min"), columns, {"short": 15, "long": 60}, "1min"
    )
    labelled = assign_target(table, columns, fit_volatility_cutoffs(table, columns, 0.9))
    numeric = ["return_1", "rv_15", "rv_60c_59r", "range_mean_60", "volume_ratio_5_60"]
    model = train_model(labelled, labelled["high_volatility"], get_feature_preprocessor(numeric, ["symbol"]), BOOSTING)
    assert model.predict_proba(labelled).shape == (len(labelled), 2)
