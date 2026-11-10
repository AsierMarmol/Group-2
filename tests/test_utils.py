"""
test_utils.py  [Tests for src/utils.py]
=======================================
CSV, JSON and model files round-trip; missing folders are created; bad paths raise clear errors.
"""

import hashlib
import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils import _save_model_returning_path as save_model
from src.utils import (
    check_input_file,
    ensure_parent_dir,
    file_sha256,
    load_model,
    read_csv,
    write_csv,
    write_json,
)


def test_unit_ensure_parent_dir_creates_missing_folders(tmp_path):
    """Unit test: ensure_parent_dir creates missing folders."""
    target = tmp_path / "a" / "b" / "file.csv"
    assert ensure_parent_dir(target) == target
    assert target.parent.is_dir()


def test_unit_check_input_file_accepts_a_real_file(tmp_path):
    """Unit test: check_input_file accepts a real file."""
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    assert check_input_file(path) == path


def test_unit_check_input_file_rejects_missing_path(tmp_path):
    """Unit test: check_input_file rejects missing path."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        check_input_file(tmp_path / "missing.csv")


def test_unit_check_input_file_rejects_a_folder(tmp_path):
    """Unit test: check_input_file rejects a folder."""
    with pytest.raises(IsADirectoryError, match="found a folder"):
        check_input_file(tmp_path)


def test_unit_check_input_file_rejects_an_empty_file(tmp_path):
    """Unit test: check_input_file rejects an empty file."""
    path = tmp_path / "empty.csv"
    path.touch()
    with pytest.raises(ValueError, match="File is empty"):
        check_input_file(path)


def test_unit_csv_round_trip_keeps_known_values(tmp_path):
    """Unit test: csv round trip keeps known values."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = write_csv(df, tmp_path / "new" / "out.csv")
    pd.testing.assert_frame_equal(read_csv(path), df)


def test_unit_write_csv_rejects_non_dataframes(tmp_path):
    """Unit test: write_csv rejects non dataframes."""
    with pytest.raises(TypeError, match="expects a DataFrame"):
        write_csv([1, 2], tmp_path / "out.csv")


def test_unit_write_json_writes_readable_json(tmp_path):
    """Unit test: write_json writes readable json."""
    path = write_json(
        {"score": 0.5, "when": pd.Timestamp("2026-01-01")}, tmp_path / "r" / "m.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "score": 0.5,
        "when": "2026-01-01 00:00:00",
    }


def test_unit_write_json_rejects_non_dicts(tmp_path):
    """Unit test: write_json rejects non dicts."""
    with pytest.raises(TypeError, match="expects a dict"):
        write_json([1], tmp_path / "m.json")


def test_unit_file_sha256_matches_hashlib_on_small_and_multi_chunk_files(tmp_path):
    """Unit test: file_sha256 matches hashlib on small and multi chunk files."""
    small = tmp_path / "small.txt"
    small.write_bytes(b"abc")
    big = tmp_path / "big.bin"
    big.write_bytes(b"0123456789" * 300000)
    for path in (small, big):
        assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(file_sha256(small)) == 64


def test_unit_file_sha256_rejects_missing_files(tmp_path):
    """Unit test: file_sha256 rejects missing files."""
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.csv")


def test_unit_load_model_rejects_missing_file(tmp_path):
    """Unit test: load_model rejects missing file."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "model.joblib")


def test_integration_saved_model_predicts_like_the_original(tmp_path):
    """Integration test: saved model predicts like the original."""
    x = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]})
    y = [0, 0, 1, 1]
    model = LogisticRegression().fit(x, y)
    loaded = load_model(save_model(model, tmp_path / "models" / "model.joblib"))
    assert list(loaded.predict(x)) == list(model.predict(x))
