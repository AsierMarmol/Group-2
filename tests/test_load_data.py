"""
test_load_data.py  [Tests for src/load_data.py]
===============================================
Valid files load with the expected rows and columns; a missing folder, an empty file,
a wrong column count or an unreadable file name raise a clear error; raw files stay unchanged.
"""

import pandas as pd
import pytest

from src.clean_data import _clean_candles as clean_dataframe
from src.load_data import (
    has_header_row,
    list_raw_files,
    load_raw_data,
    read_candle_file,
    symbol_from_filename,
)
from tests.conftest import RAW_COLUMNS, make_raw_candles

SYMBOL_REGEX = "^([A-Z0-9]+)-1m-"


def test_unit_list_raw_files_is_sorted_and_filtered(tmp_path):
    """Unit test: list_raw_files is sorted and filtered."""
    for name in ("SOLUSDT-1m-2026-01.csv", "BTCUSDT-1m-2026-01.csv", "notes.txt"):
        (tmp_path / name).write_text("1\n", encoding="utf-8")
    files = list_raw_files(tmp_path, "*-1m-*.csv")
    assert [path.name for path in files] == [
        "BTCUSDT-1m-2026-01.csv",
        "SOLUSDT-1m-2026-01.csv",
    ]


def test_unit_list_raw_files_finds_per_symbol_subfolders(tmp_path):
    """Unit test: list_raw_files finds per symbol subfolders."""
    for symbol in ("BTCUSDT", "ETHUSDT"):
        folder = tmp_path / symbol
        folder.mkdir()
        (folder / f"{symbol}-1m-2026-01.csv").write_text("1\n", encoding="utf-8")
    files = list_raw_files(tmp_path, "**/*-1m-*.csv")
    assert [path.name for path in files] == [
        "BTCUSDT-1m-2026-01.csv",
        "ETHUSDT-1m-2026-01.csv",
    ]


def test_unit_list_raw_files_rejects_missing_folder(tmp_path):
    """Unit test: list_raw_files rejects missing folder."""
    with pytest.raises(FileNotFoundError, match="Raw data folder not found"):
        list_raw_files(tmp_path / "missing", "*.csv")


def test_unit_list_raw_files_rejects_a_file_path(tmp_path):
    """Unit test: list_raw_files rejects a file path."""
    path = tmp_path / "file.csv"
    path.write_text("1\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        list_raw_files(path, "*.csv")


def test_unit_list_raw_files_rejects_folder_without_matches(tmp_path):
    """Unit test: list_raw_files rejects folder without matches."""
    with pytest.raises(ValueError, match="No files matching"):
        list_raw_files(tmp_path, "*-1m-*.csv")


def test_unit_symbol_from_filename_reads_the_pair():
    """Unit test: symbol_from_filename reads the pair."""
    assert symbol_from_filename("ETHUSDT-1m-2025-04.csv", SYMBOL_REGEX) == "ETHUSDT"


def test_unit_symbol_from_filename_rejects_unexpected_names():
    """Unit test: symbol_from_filename rejects unexpected names."""
    with pytest.raises(ValueError, match="Cannot read a symbol"):
        symbol_from_filename("ethusdt_april.csv", SYMBOL_REGEX)


def test_unit_has_header_row_detects_both_layouts(tmp_path):
    """Unit test: has_header_row detects both layouts."""
    with_header = tmp_path / "a.csv"
    with_header.write_text("open_time,open\n1,2\n", encoding="utf-8")
    without_header = tmp_path / "b.csv"
    without_header.write_text("1743465600000000,2\n", encoding="utf-8")
    assert has_header_row(with_header) is True
    assert has_header_row(without_header) is False


def test_unit_read_candle_file_names_columns_in_file_order(tmp_path):
    """Unit test: read_candle_file names columns in file order."""
    path = tmp_path / "BTCUSDT-1m-2026-01.csv"
    make_raw_candles(5, seed=0).to_csv(path, header=False, index=False)
    df = read_candle_file(path, RAW_COLUMNS)
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 5


def test_unit_read_candle_file_skips_a_header_line(tmp_path):
    """Unit test: read_candle_file skips a header line."""
    path = tmp_path / "BTCUSDT-1m-2026-01.csv"
    make_raw_candles(5, seed=0).to_csv(path, header=True, index=False)
    assert len(read_candle_file(path, RAW_COLUMNS)) == 5


def test_unit_read_candle_file_rejects_wrong_column_count(tmp_path):
    """Unit test: read_candle_file rejects wrong column count."""
    path = tmp_path / "BTCUSDT-1m-2026-01.csv"
    make_raw_candles(3, seed=0).iloc[:, :5].to_csv(path, header=False, index=False)
    with pytest.raises(ValueError, match="has 5 columns, expected 12"):
        read_candle_file(path, RAW_COLUMNS)


def test_unit_read_candle_file_rejects_header_without_rows(tmp_path):
    """Unit test: read_candle_file rejects header without rows."""
    path = tmp_path / "BTCUSDT-1m-2026-01.csv"
    path.write_text(",".join(RAW_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No candle rows"):
        read_candle_file(path, RAW_COLUMNS)


def test_unit_read_candle_file_rejects_an_empty_file(tmp_path):
    """Unit test: read_candle_file rejects an empty file."""
    path = tmp_path / "BTCUSDT-1m-2026-01.csv"
    path.touch()
    with pytest.raises(ValueError, match="File is empty"):
        read_candle_file(path, RAW_COLUMNS)


def test_unit_load_raw_data_returns_every_row_with_a_symbol(raw_dir, cfg):
    """Unit test: load_raw_data returns every row with a symbol."""
    df = load_raw_data(raw_dir, cfg["data"], "symbol")
    assert len(df) == 3 * 3 * 24 * 60
    assert list(df.columns) == [*RAW_COLUMNS, "symbol"]
    assert sorted(df["symbol"].unique()) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_unit_load_raw_data_leaves_raw_files_unchanged(raw_dir, cfg):
    """Unit test: load_raw_data leaves raw files unchanged."""
    before = {path.name: path.read_bytes() for path in raw_dir.iterdir()}
    load_raw_data(raw_dir, cfg["data"], "symbol")
    assert {path.name: path.read_bytes() for path in raw_dir.iterdir()} == before


def test_unit_load_raw_data_rejects_a_clashing_symbol_column(raw_dir, cfg):
    """Unit test: load_raw_data rejects a clashing symbol column."""
    with pytest.raises(ValueError, match="clashes with a raw column"):
        load_raw_data(raw_dir, cfg["data"], "open")


def test_integration_loaded_data_feeds_clean_data(raw_dir, cfg):
    """Integration test: loaded data feeds clean_data."""
    raw = load_raw_data(raw_dir, cfg["data"], cfg["columns"]["symbol_col"])
    clean = clean_dataframe(raw, cfg["columns"], cfg["rules"])
    assert len(clean) == len(raw)
    assert isinstance(clean["open_time"].dtype, pd.DatetimeTZDtype)
