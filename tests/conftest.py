"""
conftest.py  [Shared test setup]
================================
Small, synthetic Binance-style candles and a matching config, so every test runs offline
in seconds. No real data, accounts or secrets are ever used by the tests.
"""

import importlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
MICROSECONDS_PER_MINUTE = 60_000_000


def make_raw_candles(minutes: int, seed: int, start: pd.Timestamp = START) -> pd.DataFrame:
    """One asset's raw Binance klines (microsecond timestamps) with calm and volatile spells."""
    rng = np.random.default_rng(seed)
    spells = np.repeat(rng.random(minutes // 30 + 1) < 0.3, 30)[:minutes]
    sigma = np.where(spells, 0.004, 0.001)
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, sigma)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.gamma(2.0, 50.0, minutes) * (1 + 3 * spells)
    trades = rng.poisson(100 * (1 + 3 * spells))
    open_time = (start.value // 1000) + np.arange(minutes) * MICROSECONDS_PER_MINUTE
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": open_time + MICROSECONDS_PER_MINUTE - 1,
            "quote_volume": volume * close,
            "trade_count": trades,
            "taker_buy_base_volume": volume / 2,
            "taker_buy_quote_volume": volume * close / 2,
            "ignore": 0,
        }
    )[RAW_COLUMNS]


def make_clean_candles(minutes: int, symbol: str = "BTCUSDT", log_step: float = 0.01) -> pd.DataFrame:
    """Clean candles with a known shape: every close is exp(log_step) times the previous one.

    Known values: one-minute log return = log_step, candle range = log_step + log(1.002 / 0.998),
    volume = 10 and trade count = 5 every minute.
    """
    close = 100 * np.exp(log_step * np.arange(minutes))
    open_ = np.concatenate([[close[0] / np.exp(log_step)], close[:-1]])
    return pd.DataFrame(
        {
            "symbol": symbol,
            "open_time": pd.date_range(START, periods=minutes, freq="1min"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 10.0,
            "quote_volume": 10.0 * close,
            "trade_count": 5,
        }
    )


def write_raw_dir(folder: Path, minutes: int, symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT")) -> Path:
    """Write one headerless Binance-style file per symbol into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    for seed, symbol in enumerate(symbols):
        make_raw_candles(minutes, seed).to_csv(folder / f"{symbol}-1m-2026-01.csv", header=False, index=False)
    return folder


def write_test_config(folder: Path, raw_dir: Path) -> Path:
    """Build disposable synthetic-test settings for a three-day sample; runtime SETTINGS remain in main.py."""
    cfg = json.loads((PROJECT_ROOT / "tests/fixture_settings.json").read_text(encoding="utf-8"))
    cfg["paths"] = {
        "raw_dir": str(raw_dir),
        "clean_data": str(folder / "data" / "processed" / "clean.csv"),
        "model": str(folder / "models" / "model.joblib"),
        "predictions": str(folder / "reports" / "predictions.csv"),
        "metrics": str(folder / "reports" / "metrics.json"),
        "log_file": str(folder / "logs" / "pipeline.log"),
    }
    cfg["scope"] = {"start": "2026-01-01", "end_exclusive": "2026-01-04"}
    cfg["split"] = {
        "train_end": "2026-01-01 23:59:00",
        "validation_end": "2026-01-02 23:59:00",
    }
    cfg["training"]["params"] = {
        "max_iter": 20,
        "max_leaf_nodes": 8,
        "early_stopping": False,
    }
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(name="raw_dir")
def fixture_raw_dir(tmp_path: Path) -> Path:
    """Three assets, three days of synthetic minutes each."""
    return write_raw_dir(tmp_path / "raw", minutes=3 * 24 * 60)


@pytest.fixture(name="config_path")
def fixture_config_path(tmp_path: Path, raw_dir: Path) -> Path:
    """A valid config.yaml pointing at the synthetic data in a temporary folder."""
    return write_test_config(tmp_path, raw_dir)


@pytest.fixture(name="cfg")
def fixture_cfg(config_path: Path) -> dict:
    """Loaded settings for the synthetic data."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["paths"] = {k: Path(v) for k, v in cfg["paths"].items()}
    return cfg


@pytest.fixture(name="columns")
def fixture_columns(cfg: dict) -> dict:
    """Column roles from the disposable synthetic-test settings."""
    return cfg["columns"]


@pytest.fixture(name="rules")
def fixture_rules(cfg: dict) -> dict:
    """Data rules from the disposable synthetic-test settings."""
    return cfg["rules"]


@pytest.fixture(name="raw_df")
def fixture_raw_df(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """Synthetic raw candles as load_data returns them."""
    # imported when the fixture runs, so this file works before load_data.py is merged
    load_data = importlib.import_module("src.load_data")
    return load_data.load_raw_data(raw_dir, cfg["data"], cfg["columns"]["symbol_col"])


@pytest.fixture(name="clean_df")
def fixture_clean_df(raw_df: pd.DataFrame, columns: dict, rules: dict) -> pd.DataFrame:
    """Synthetic candles after clean_data."""
    clean_data = importlib.import_module("src.clean_data")  # same reason as raw_df
    return clean_data.clean_dataframe(raw_df, columns=columns, rules=rules, drop_invalid=True)


@pytest.fixture(name="restore_root_logger", autouse=True)
def fixture_restore_root_logger():
    """Undo any logging setup a test makes, so tests never affect each other."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in [h for h in root.handlers if h not in handlers]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)


def pytest_configure(config):
    """Ensure the teacher's repo-local disposable cache parent exists."""
    (Path(__file__).resolve().parents[1] / ".pytest_cache").mkdir(exist_ok=True)
