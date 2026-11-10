"""Utils functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_parent_dir(path: Path) -> Path:
    """Inputs: path.
    Outputs: the annotated return value or a validation error.
    Why: Create the parent folder of ``path`` when it is missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check_input_file(path: Path) -> Path:
    """Inputs: path.
    Outputs: the annotated return value or a validation error.
    Why: Check that ``path`` is an existing, non-empty file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file but found a folder: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")
    return path


def read_csv(path: Path, **read_options: Any) -> pd.DataFrame:
    """Inputs: path.
    Outputs: DataFrame using explicitly supplied pandas read options.
    Why: Read a CSV file after checking that it exists and is not empty."""
    path = check_input_file(path)
    df = pd.read_csv(path, **read_options)
    logger.debug("Read %d rows from %s", len(df), path)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Inputs: df, path.
    Outputs: Destination Path after writing CSV without an index.
    Why: Write a DataFrame to CSV without its index, creating folders as needed."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"write_csv expects a DataFrame, got {type(df).__name__}")
    path = ensure_parent_dir(path)
    df.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def write_json(data: dict, path: Path) -> Path:
    """Inputs: data, path.
    Outputs: Destination Path after writing a readable report.
    Why: Write a dictionary as readable JSON, creating folders as needed."""
    if not isinstance(data, dict):
        raise TypeError(f"write_json expects a dict, got {type(data).__name__}")
    path = ensure_parent_dir(path)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _save_model_returning_path(model: Any, path: Path) -> Path:
    """Save a fitted model with joblib, creating folders as needed.

    Args:
        model: Fitted scikit-learn pipeline (or any picklable object).
        path: Destination file, usually ``models/model.joblib``.

    Returns:
        The destination path.
    """
    path = ensure_parent_dir(path)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def file_sha256(path: Path) -> str:
    """Inputs: path.
    Outputs: the annotated return value or a validation error.
    Why: Return the SHA-256 checksum of a file, read in chunks so large files fit in memory.
    """
    path = check_input_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(filepath: Path) -> Any:
    """Inputs: filepath.
    Outputs: Loaded trusted joblib artifact; missing/empty files raise.
    Why: Load a model saved by :func:`save_model`."""
    path = check_input_file(filepath)
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def load_csv(filepath: Path) -> pd.DataFrame:
    """Inputs: filepath.
    Outputs: DataFrame from a checked, existing nonempty CSV.
    Why: Keep one tested pipeline responsibility."""
    return read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Inputs: df, filepath.
    Outputs: None; writes CSV without an index and creates parent folders.
    Why: Keep one tested pipeline responsibility."""
    write_csv(df, filepath)


def save_model(model: Any, filepath: Path) -> None:
    """Inputs: model, filepath.
    Outputs: None; writes a trusted joblib artifact and creates parent folders.
    Why: Keep one tested pipeline responsibility."""
    _save_model_returning_path(model, filepath)
