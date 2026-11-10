"""
test_infer.py  [Tests for src/infer.py]
=======================================
Output has the fixed columns and the input index; probabilities lie in [0, 1];
wrong input types, empty input or a model that cannot predict raise a clear error.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import get_feature_preprocessor
from src.infer import PREDICTION_COLUMN, PROBABILITY_COLUMN
from src.infer import _run_feature_inference as run_inference
from src.train import _train_configured_model as train_model
from src.utils import _save_model_returning_path as save_model
from src.utils import load_model

LOGISTIC = {"model_type": "logistic_regression", "random_state": 0, "params": {}}


@pytest.fixture(name="model_and_rows")
def fixture_model_and_rows():
    """A small fitted logistic pipeline and the rows it was trained on."""
    rng = np.random.default_rng(2)
    x = pd.DataFrame({"a": rng.normal(size=200), "symbol": np.repeat(["BTCUSDT", "SOLUSDT"], 100)})
    y = pd.Series((x["a"] > 0.8).astype(int))
    model = train_model(x, y, get_feature_preprocessor(["a"], ["symbol"]), LOGISTIC)
    return (model, x)


def test_unit_output_has_fixed_columns_and_input_index(model_and_rows):
    """Unit test: output has fixed columns and input index."""
    model, x = model_and_rows
    new_rows = x.iloc[10:20].set_index(pd.Index(range(100, 110)))
    out = run_inference(model, new_rows)
    assert list(out.columns) == [PREDICTION_COLUMN, PROBABILITY_COLUMN] == ["prediction", "probability"]
    assert out.index.equals(new_rows.index)
    assert out[PROBABILITY_COLUMN].between(0, 1).all()
    assert set(out[PREDICTION_COLUMN]) <= {0, 1}


def test_unit_infer_rejects_bad_inputs(model_and_rows):
    """Unit test: infer rejects bad inputs."""
    model, x = model_and_rows
    with pytest.raises(TypeError, match="must be a DataFrame"):
        run_inference(model, x.to_numpy())
    with pytest.raises(ValueError, match="No rows"):
        run_inference(model, x.iloc[0:0])
    with pytest.raises(TypeError, match="needs predict and predict_proba"):
        run_inference(object(), x)


def test_integration_saved_model_gives_the_same_predictions(model_and_rows, tmp_path):
    """Integration test: saved model gives the same predictions."""
    model, x = model_and_rows
    loaded = load_model(save_model(model, tmp_path / "model.joblib"))
    pd.testing.assert_frame_equal(run_inference(loaded, x), run_inference(model, x))
