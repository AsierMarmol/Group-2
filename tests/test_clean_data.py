"""
test_clean_data.py  [Tests for src/clean_data.py]
=================================================
Timestamps become UTC; impossible, missing and duplicate candles are dropped and counted;
the index is reset; a missing column raises a clear error; cleaning twice changes nothing.
"""

import numpy as np
import pandas as pd
import pytest

from src.clean_data import _clean_candles as clean_dataframe
from src.clean_data import to_utc_datetime
from src.load_data import load_raw_data

MICRO_THRESHOLD = 1.0e14


def test_unit_to_utc_datetime_reads_ms_and_us_the_same_way():
    """Unit test: to_utc_datetime reads ms and us the same way."""
    milliseconds = pd.Series([1_743_465_600_000])
    microseconds = pd.Series([1_743_465_600_000_000])
    expected = pd.Timestamp("2025-04-01 00:00:00", tz="UTC")
    assert to_utc_datetime(milliseconds, MICRO_THRESHOLD).iloc[0] == expected
    assert to_utc_datetime(microseconds, MICRO_THRESHOLD).iloc[0] == expected


def test_unit_to_utc_datetime_keeps_datetimes_and_flags_garbage():
    """Unit test: to_utc_datetime keeps datetimes and flags garbage."""
    already = pd.Series(pd.to_datetime(["2025-04-01 00:01:00"], utc=True))
    assert to_utc_datetime(already, MICRO_THRESHOLD).equals(already)
    assert to_utc_datetime(pd.Series(["not a time"]), MICRO_THRESHOLD).isna().all()


def test_unit_clean_keeps_input_unchanged(raw_df, columns, rules):
    """Unit test: clean keeps input unchanged."""
    before = raw_df.copy()
    clean_dataframe(raw_df, columns, rules)
    pd.testing.assert_frame_equal(raw_df, before)


def test_unit_clean_output_is_sorted_with_fresh_index(raw_df, columns, rules):
    """Unit test: clean output is sorted with fresh index."""
    shuffled = raw_df.sample(frac=1.0, random_state=0)
    clean = clean_dataframe(shuffled, columns, rules)
    assert list(clean.index) == list(range(len(clean)))
    assert clean.equals(
        clean.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    )


def test_unit_clean_drops_bad_rows_and_counts_them(raw_df, columns, rules, caplog):
    """Unit test: clean drops bad rows and counts them."""
    bad = raw_df.copy()
    bad.loc[0, "close"] = np.nan
    bad.loc[1, "open"] = -1.0
    bad.loc[2, "volume"] = -5.0
    bad.loc[3, "high"] = bad.loc[3, "low"] / 2
    bad = pd.concat([bad, bad.iloc[[10]]], ignore_index=True)
    with caplog.at_level("WARNING"):
        clean = clean_dataframe(bad, columns, rules)
    assert len(clean) == len(raw_df) - 4
    for reason in (
        "missing",
        "non-positive",
        "negative",
        "outside the low-high",
        "duplicate",
    ):
        assert reason in caplog.text


def test_unit_clean_logs_missing_minutes(raw_df, columns, rules, caplog):
    """Unit test: clean logs missing minutes."""
    with caplog.at_level("WARNING"):
        clean_dataframe(raw_df.drop(index=[100, 101]), columns, rules)
    assert "2 missing minute(s)" in caplog.text


def test_unit_clean_standardises_symbol_text(raw_df, columns, rules):
    """Unit test: clean standardises symbol text."""
    messy = raw_df.copy()
    messy["symbol"] = " " + messy["symbol"].str.lower() + " "
    assert set(clean_dataframe(messy, columns, rules)["symbol"]) == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    }


def test_unit_clean_rejects_missing_columns(raw_df, columns, rules):
    """Unit test: clean rejects missing columns."""
    with pytest.raises(ValueError, match="Missing columns for cleaning"):
        clean_dataframe(raw_df.drop(columns=["close"]), columns, rules)


def test_unit_clean_rejects_a_missing_target_column(raw_df, columns, rules):
    """Unit test: clean rejects a missing target column."""
    with pytest.raises(ValueError, match="high_volatility"):
        clean_dataframe(raw_df, columns, rules, target_column="high_volatility")


def test_unit_clean_drops_rows_without_a_target(raw_df, columns, rules):
    """Unit test: clean drops rows without a target."""
    labelled = raw_df.copy()
    labelled["high_volatility"] = 0
    labelled.loc[:9, "high_volatility"] = np.nan
    clean = clean_dataframe(labelled, columns, rules, target_column="high_volatility")
    assert len(clean) == len(raw_df) - 10
    assert "high_volatility" in clean.columns


def test_unit_cleaning_twice_gives_the_same_result(raw_df, columns, rules):
    """Unit test: cleaning twice gives the same result."""
    once = clean_dataframe(raw_df, columns, rules)
    twice = clean_dataframe(once, columns, rules)
    pd.testing.assert_frame_equal(once, twice)


def test_integration_clean_loaded_files_with_repeated_lines(
    raw_dir, cfg, columns, rules
):
    """Integration test: clean loaded files with repeated lines."""
    first = sorted(raw_dir.iterdir())[0]
    first.write_text(
        first.read_text(encoding="utf-8") * 2, encoding="utf-8"
    )  # every line twice
    raw = load_raw_data(raw_dir, cfg["data"], columns["symbol_col"])
    clean = clean_dataframe(raw, columns, rules)
    assert len(raw) == 4 * 3 * 24 * 60
    assert len(clean) == 3 * 3 * 24 * 60
    assert not clean.duplicated(["symbol", "open_time"]).any()
