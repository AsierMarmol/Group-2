"""
test_validate.py  [Tests for src/validate.py]
=============================================
Valid data passes; every broken rule raises an error that names it; the data is never modified.
"""

import pandas as pd
import pytest
from src.validate import _validate_candles as validate_dataframe


def test_unit_valid_data_passes(clean_df, columns, rules):
    """Unit test: valid data passes."""
    assert validate_dataframe(clean_df, columns, rules) is True


def test_unit_validation_never_modifies_the_data(clean_df, columns, rules):
    """Unit test: validation never modifies the data."""
    before = clean_df.copy()
    validate_dataframe(clean_df, columns, rules)
    pd.testing.assert_frame_equal(clean_df, before)


def test_unit_rejects_non_dataframes(columns, rules):
    """Unit test: rejects non dataframes."""
    with pytest.raises(TypeError, match="Expected a DataFrame"):
        validate_dataframe([1, 2], columns, rules)


def test_unit_rejects_empty_data(clean_df, columns, rules):
    """Unit test: rejects empty data."""
    with pytest.raises(ValueError, match="No rows"):
        validate_dataframe(clean_df.iloc[0:0], columns, rules)


def test_unit_rejects_missing_columns(clean_df, columns, rules):
    """Unit test: rejects missing columns."""
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(clean_df.drop(columns=["volume"]), columns, rules)


def test_unit_rejects_a_time_column_without_datetimes(clean_df, columns, rules):
    """Unit test: rejects a time column without datetimes."""
    df = clean_df.copy()
    df["open_time"] = df["open_time"].astype(str)
    with pytest.raises(TypeError, match="must hold datetimes"):
        validate_dataframe(df, columns, rules)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("close", float("nan")), "missing values"),
        (("open", 0.0), "non-positive price"),
        (("trade_count", -1), "negative volume or trade count"),
        (("symbol", "DOGEUSDT"), "symbol outside"),
    ],
)
def test_unit_rejects_broken_value_rules(clean_df, columns, rules, change, message):
    """Unit test: rejects broken value rules."""
    column, value = change
    df = clean_df.copy()
    df.loc[5, column] = value
    with pytest.raises(ValueError, match=message):
        validate_dataframe(df, columns, rules)


def test_unit_rejects_open_or_close_outside_the_range(clean_df, columns, rules):
    """Unit test: rejects open or close outside the range."""
    df = clean_df.copy()
    df.loc[5, "high"] = df.loc[5, "low"]
    with pytest.raises(ValueError, match="1 row\\(s\\) have an open or close outside"):
        validate_dataframe(df, columns, rules)


def test_unit_rejects_duplicate_candles(clean_df, columns, rules):
    """Unit test: rejects duplicate candles."""
    df = pd.concat([clean_df.iloc[:1], clean_df], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate candles"):
        validate_dataframe(df, columns, rules)


def test_unit_rejects_candles_out_of_time_order(clean_df, columns, rules):
    """Unit test: rejects candles out of time order."""
    df = clean_df.iloc[[1, 0, *range(2, len(clean_df))]].reset_index(drop=True)
    with pytest.raises(ValueError, match="out of time order"):
        validate_dataframe(df, columns, rules)


def test_unit_rejects_times_off_the_minute(clean_df, columns, rules):
    """Unit test: rejects times off the minute."""
    df = clean_df.copy()
    df.loc[5, "open_time"] = df.loc[5, "open_time"] + pd.Timedelta(seconds=7)
    with pytest.raises(ValueError, match="not aligned"):
        validate_dataframe(df, columns, rules)


def test_unit_checks_target_values_when_asked(clean_df, columns, rules):
    """Unit test: checks target values when asked."""
    df = clean_df.copy()
    df["high_volatility"] = 0
    assert validate_dataframe(df, columns, rules, target_column="high_volatility") is True
    df.loc[3, "high_volatility"] = 2
    with pytest.raises(ValueError, match="target outside"):
        validate_dataframe(df, columns, rules, target_column="high_volatility")


def test_unit_rejects_a_missing_target_column(clean_df, columns, rules):
    """Unit test: rejects a missing target column."""
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(clean_df, columns, rules, target_column="high_volatility")


def test_integration_validation_stops_data_that_skipped_cleaning(raw_df, columns, rules):
    """Integration test: validation stops data that skipped cleaning."""
    with pytest.raises(TypeError, match="must hold datetimes"):
        validate_dataframe(raw_df, columns, rules)
