"""Load data functions for the volatility research pipeline.
Inputs: explicit function parameters and resolved settings.
Outputs: validated return values or saved artifacts.
Used by: orchestrator and adjacent module tests.
Why: separate responsibilities and preserve forecasting chronology."""

import logging
import re
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)


def list_raw_files(raw_dir: Path, file_pattern: str) -> list[Path]:
    """Inputs: raw_dir, file_pattern.
    Outputs: Sorted source Paths or a clear missing-input error.
    Why: List the raw files to load, in a stable order."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Expected a folder of raw files: {raw_dir}")
    files = sorted((path for path in raw_dir.glob(file_pattern) if path.is_file()))
    if not files:
        raise ValueError(f"No files matching '{file_pattern}' in {raw_dir}")
    return files


def symbol_from_filename(path: Path, symbol_regex: str) -> str:
    """Inputs: path, symbol_regex.
    Outputs: the annotated return value or a validation error.
    Why: Read the trading pair from a raw file name."""
    name = Path(path).name
    match = re.match(symbol_regex, name)
    if match is None:
        raise ValueError(
            f"Cannot read a symbol from file name '{name}' with pattern '{symbol_regex}'"
        )
    return match.group(1)


def has_header_row(path: Path) -> bool:
    """Inputs: path.
    Outputs: the annotated return value or a validation error.
    Why: Tell whether a CSV starts with a header line instead of numbers."""
    with Path(path).open(encoding="utf-8") as handle:
        first_field = handle.readline().split(",", 1)[0].strip()
    try:
        float(first_field)
    except ValueError:
        return True
    return False


def read_candle_file(path: Path, column_names: list[str]) -> pd.DataFrame:
    """Inputs: path, column_names.
    Outputs: Candle DataFrame with declared columns, or a source-format error.
    Why: Read one raw candle file and name its columns."""
    path = utils.check_input_file(path)
    skip = 1 if has_header_row(path) else 0
    try:
        df = utils.read_csv(path, header=None, skiprows=skip)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"No candle rows in {path}") from error
    if df.shape[1] != len(column_names):
        raise ValueError(
            f"{path.name} has {df.shape[1]} columns, expected {len(column_names)}: {column_names}"
        )
    df.columns = list(column_names)
    return df


def load_raw_data(
    raw_data_path: Path,
    data_settings: dict | None = None,
    symbol_column: str | None = None,
) -> pd.DataFrame:
    """Inputs: raw_data_path, data_settings, symbol_column.
    Outputs: Raw DataFrame; source files are never changed.
    Why: Read every raw candle file into one DataFrame with a symbol column."""
    raw_dir = Path(raw_data_path)
    if data_settings is None:
        return utils.read_csv(raw_dir)
    if symbol_column is None:
        raise ValueError("A symbol column is required for candle-folder loading")
    column_names = list(data_settings["raw_columns"])
    if symbol_column in column_names:
        raise ValueError(
            f"Symbol column '{symbol_column}' clashes with a raw column name"
        )
    frames = []
    for path in list_raw_files(raw_dir, data_settings["file_pattern"]):
        frame = read_candle_file(path, column_names)
        frame[symbol_column] = symbol_from_filename(path, data_settings["symbol_regex"])
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d files in %s", len(df), len(frames), raw_dir)
    return df
