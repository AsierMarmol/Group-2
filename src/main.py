"""Orchestrate the fixed crypto-volatility research pipeline.
Inputs: project-relative SETTINGS and an optional ML_RAW_DIR override.
Outputs: clean candles, diagnostic scores, metrics and a reproducible candidate bundle.
Used by: python -m src.main and the modular notebook.
Why: keep chronological boundaries, training-only learning and artifact handoffs explicit."""

import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
from pathlib import Path

import pandas as pd

from src import clean_data, evaluate, features, infer, load_data
from src import target as target_module
from src import train, utils, validate

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "paths": {
        "raw_dir": "data/raw",
        "clean_data": "data/processed/clean.csv",
        "model": "models/model.joblib",
        "predictions": "reports/predictions.csv",
        "metrics": "reports/metrics.json",
    },
    "problem": {"problem_type": "classification"},
    "columns": {
        "symbol_col": "symbol",
        "time_col": "open_time",
        "open_col": "open",
        "high_col": "high",
        "low_col": "low",
        "close_col": "close",
        "volume_col": "volume",
        "quote_volume_col": "quote_volume",
        "trades_col": "trade_count",
        "future_volatility_col": "future_rv",
        "target_col": "high_volatility",
    },
    "data": {
        "file_pattern": "**/*-1m-*.csv",
        "symbol_regex": "^([A-Z0-9]+)-1m-",
        "raw_columns": [
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
        ],
    },
    "rules": {
        "frequency": "1min",
        "close_time_column": "close_time",
        "microseconds_above": 100000000000000.0,
        "value_columns": ["open", "high", "low", "close", "volume", "quote_volume", "trade_count"],
        "positive_columns": ["open", "high", "low", "close"],
        "non_negative_columns": ["volume", "quote_volume", "trade_count"],
        "allowed_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "target_values": [0, 1],
    },
    "features": {
        "parameters": {"return_windows": (1, 5, 15, 30), "rv_windows": (5, 15, 30), "activity_short": 5},
        "windows": {"short": 15, "long": 60},
        "numeric": [
            "return_1",
            "return_5",
            "return_15",
            "return_30",
            "rv_5",
            "rv_15",
            "rv_30",
            "rv_60c_59r",
            "range_mean_60",
            "volume_ratio_5_60",
            "trades_ratio_5_60",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
        ],
        "categorical": ["symbol"],
    },
    "target": {"horizon": 15, "quantile": 0.9},
    "split": {"train_end": "2026-03-31 23:59:00", "validation_end": "2026-06-30 23:59:00"},
    "training": {
        "model_type": "hist_gradient_boosting",
        "random_state": 42,
        "params": {
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "learning_rate": 0.1,
            "min_samples_leaf": 100,
            "early_stopping": False,
        },
    },
    "quality_gate": {
        "metric": "average_precision",
        "direction": "higher_is_better",
        "split": "validation",
        "benchmark_column": "rv_60c_59r",
        "min_ratio_vs_benchmark": 1.2,
    },
    "logging": {"level": "INFO"},
    "feature_definitions": {
        "returns": "log(C[t]/C[t-k]), k=1/5/15/30",
        "rv": "sqrt(sum of last k squared log returns), k=5/15/30; benchmark 59 internal returns",
        "range": "mean((high-low)/close) over 60 candles",
        "activity": "5-candle mean / 60-candle mean, volume/trades",
        "calendar": "cyclical UTC decision hour and weekday",
    },
    "scope": {"start": "2025-04-01", "end_exclusive": "2026-10-01", "expected_months": 18},
}


def split_by_time(table: pd.DataFrame, time_column: str, split: dict, horizon: int, frequency: str) -> dict:
    """Inputs: candle table, UTC time column, inclusive split dates, horizon and frequency.
    Outputs: Chronological partitions excluding outcomes that cross their ending boundary.
    Why: Assign by prediction decision time and keep future labels within their own period."""
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("Horizon must be a positive integer")
    step = pd.Timedelta(frequency)
    if step <= pd.Timedelta(0):
        raise ValueError("Frequency must be positive")
    times = table[time_column]
    if not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC" or times.isna().any():
        raise ValueError("Split opening times must be timezone-aware UTC with no missing values")
    train_stop = pd.Timestamp(split["train_end"])
    validation_stop = pd.Timestamp(split["validation_end"])
    train_stop = train_stop.tz_localize("UTC") if train_stop.tzinfo is None else train_stop.tz_convert("UTC")
    validation_stop = (
        validation_stop.tz_localize("UTC") if validation_stop.tzinfo is None else validation_stop.tz_convert("UTC")
    )
    # SETTINGS dates name the final decision minute; the next period starts one minute later.
    train_stop, validation_stop = train_stop + step, validation_stop + step
    if not train_stop < validation_stop:
        raise ValueError("Training boundary must precede validation boundary in SETTINGS['split']")
    decision = times + step
    outcome_available = decision + int(horizon) * step
    parts = {
        "train": table[(decision < train_stop) & (outcome_available <= train_stop)],
        "validation": table[
            (decision >= train_stop) & (decision < validation_stop) & (outcome_available <= validation_stop)
        ],
        "test": table[decision >= validation_stop],
    }
    empty = [name for name, part in parts.items() if part.empty]
    if empty:
        raise ValueError(f"Empty split(s) {empty}: check SETTINGS['split'] and the available dates")
    removed = len(table) - sum(len(part) for part in parts.values())
    logger.info("Chronological split: %s; %d boundary rows purged", {k: len(v) for k, v in parts.items()}, removed)
    # Stable sorting keeps the canonical asset order on equal timestamps.
    return {name: part.sort_values(time_column, kind="stable").reset_index(drop=True) for name, part in parts.items()}


def check_quality_gate(metrics: dict, benchmark: dict, gate: dict) -> dict:
    """Inputs: validation metrics, same-row benchmark metrics and gate settings.
    Outputs: Model AP, benchmark AP, relative threshold and a Boolean gate result.
    Why: A validation gate pass records candidate quality, never automatic release approval."""
    if gate.get("direction", "higher_is_better") != "higher_is_better":
        raise ValueError("This project's AP quality gate requires higher_is_better")
    if gate.get("split", "validation") != "validation":
        raise ValueError("The quality gate must use validation, never the final period")
    ratio = float(gate["min_ratio_vs_benchmark"])
    if not math.isfinite(ratio) or ratio < 1.2:
        raise ValueError("Quality-gate ratio must be finite and at least the agreed 1.20")
    name = gate["metric"]
    if name != "average_precision":
        raise ValueError("The agreed quality metric is average_precision")
    score, reference = float(metrics[name]), float(benchmark[name])
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (score, reference)):
        raise ValueError("Model and benchmark AP must be finite values between zero and one")
    threshold = ratio * reference
    result = {
        "metric": name,
        "model": score,
        "benchmark": reference,
        "min_ratio_vs_benchmark": ratio,
        "threshold": threshold,
        "passed": bool(score >= threshold),
    }
    logger.log(
        logging.INFO if result["passed"] else logging.WARNING,
        "Quality gate %s: %s %.6f vs threshold %.6f (%.2f x benchmark %.6f)",
        "met" if result["passed"] else "NOT met",
        name,
        score,
        threshold,
        ratio,
        reference,
    )
    return result


def _input_columns(cfg: dict) -> list[str]:
    """Select configured predictors while explicitly blocking future outcomes and identifiers."""
    groups = cfg["features"]
    names = list(dict.fromkeys([*groups["numeric"], *groups["categorical"], cfg["columns"]["symbol_col"]]))
    forbidden = {cfg["columns"][key] for key in ("target_col", "future_volatility_col", "time_col")}
    if forbidden.intersection(names):
        raise ValueError("Predictors must exclude targets, future volatility and timestamps")
    return names


def _prepare_table(cfg: dict) -> pd.DataFrame:
    """Load, clean, validate and save candles; add the label basis and look-back features."""
    columns, rules = (cfg["columns"], cfg["rules"])
    scope = cfg.get("scope")
    if scope:
        files = load_data.list_raw_files(cfg["paths"]["raw_dir"], cfg["data"]["file_pattern"])
        months = pd.period_range(scope["start"], pd.Timestamp(scope["end_exclusive"]) - pd.Timedelta(days=1), freq="M")
        expected = {f"{s}-1m-{m}.csv" for s in rules["allowed_symbols"] for m in months}
        if {p.name for p in files} != expected or len(files) != len(expected):
            raise ValueError("Raw inventory differs from the approved asset/month scope")
    raw = load_data.load_raw_data(cfg["paths"]["raw_dir"], cfg["data"], columns["symbol_col"])
    candles = clean_data.clean_dataframe(raw, columns=columns, rules=rules, drop_invalid=True)
    validate.validate_dataframe(candles, list(candles.columns), columns=columns, rules=rules)
    if scope:
        times = candles[columns["time_col"]]
        if not (
            times.ge(pd.Timestamp(scope["start"], tz="UTC")) & times.lt(pd.Timestamp(scope["end_exclusive"], tz="UTC"))
        ).all():
            raise ValueError("Candle outside approved date scope")
    utils.save_csv(candles, cfg["paths"]["clean_data"])
    candles = target_module.add_future_volatility(candles, columns, cfg["target"]["horizon"], rules["frequency"])
    return features.add_lookback_features(
        candles, columns, cfg["features"]["windows"], rules["frequency"], feature_settings=cfg["features"]["parameters"]
    )


def _labelled_splits(table: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    """Split by time, learn the cutoffs on train only, then label every split."""
    columns, target = (cfg["columns"], cfg["target"])
    x_train, x_val, x_test, _, _, _ = split_data(
        table,
        table[columns["future_volatility_col"]],
        test_size=0.0,
        val_size=0.0,
        random_state=cfg["training"]["random_state"],
        stratify=False,
        split_settings=cfg["split"],
        time_column=columns["time_col"],
        horizon=target["horizon"],
        frequency=cfg["rules"]["frequency"],
    )
    splits = {"train": x_train, "validation": x_val, "test": x_test}
    cutoffs = target_module.fit_volatility_cutoffs(splits["train"], columns, target["quantile"])
    labelled = {name: target_module.assign_target(part, columns, cutoffs) for name, part in splits.items()}
    for name, part in labelled.items():
        validate.validate_dataframe(
            part, list(part.columns), target_column=columns["target_col"], columns=columns, rules=cfg["rules"]
        )
        logger.info("%s split: %d rows, positive rate %.3f", name, len(part), part[columns["target_col"]].mean())
    return (labelled, cutoffs)


def _train_and_evaluate(splits: dict, cfg: dict) -> tuple:
    """Fit on train, score validation against the benchmark, apply the gate, audit on test once."""
    inputs, target, group = (_input_columns(cfg), cfg["columns"]["target_col"], cfg["columns"]["symbol_col"])
    gate_settings = cfg["quality_gate"]
    recipe = features.get_feature_preprocessor(
        cfg["features"]["numeric"],
        cfg["features"]["categorical"],
        scale=cfg["training"]["model_type"] == "logistic_regression",
    )
    model = train.train_model(
        splits["train"][inputs],
        splits["train"][target],
        recipe,
        cfg["problem"]["problem_type"],
        model_params={"model_type": cfg["training"]["model_type"], **cfg["training"]["params"]},
        random_state=cfg["training"]["random_state"],
    )
    results = {}
    if gate_settings["split"] != "validation":
        raise ValueError("The quality gate must use validation, never the final period")
    for name in ("validation", "test"):
        if name == "test":
            results["quality_gate"] = check_quality_gate(
                results[gate_settings["split"]], results[f"benchmark_{gate_settings['split']}"], gate_settings
            )
            utils.write_json(
                {
                    "training": cfg["training"],
                    "predictors": inputs,
                    "quality_gate": results["quality_gate"],
                    "test_status": "prior exposure",
                },
                cfg["paths"]["metrics"].parent / "selection_lock.json",
            )
        part = splits[name]
        if set(part[group]) != set(cfg["rules"]["allowed_symbols"]):
            raise ValueError("Release evaluation requires all three configured assets")
        results[name] = evaluate.evaluate_model(
            model,
            part[inputs],
            part[target],
            cfg["problem"]["problem_type"],
            group_column=group,
            expected_groups=cfg["rules"]["allowed_symbols"],
        )
        results[f"benchmark_{name}"] = evaluate.evaluate_scores(
            part[target], part[gate_settings["benchmark_column"]], part[group]
        )
    results["quality_gate"] = check_quality_gate(
        results[gate_settings["split"]], results[f"benchmark_{gate_settings['split']}"], gate_settings
    )
    return (model, results)


def _write_outputs(model, test: pd.DataFrame, results: dict, cfg: dict) -> None:
    """Save the model, the test-period predictions and the run report."""
    paths, columns = (cfg["paths"], cfg["columns"])
    model_path = paths["model"].parent / "candidates" / paths["model"].name
    if not results["quality_gate"]["passed"]:
        model_path = paths["model"].parent / "failed_candidates" / paths["model"].name
    metadata = {
        "approval_status": "candidate_pending_review" if results["quality_gate"]["passed"] else "failed_quality_gate",
        "validation_gate_pass": results["quality_gate"]["passed"],
        "model_version": "research-candidate",
        "predictor_order": _input_columns(cfg),
        "feature_definitions": cfg["feature_definitions"],
        "label_cutoffs": results["volatility_cutoffs"],
        "settings": cfg,
        "calibration_status": "uncalibrated",
        "test_status": "prior analytical exposure; not independent confirmation",
    }
    metadata["evidence"] = results["evidence"]
    identity = {
        "training": cfg["training"],
        "features": cfg["features"],
        "cutoffs": results["volatility_cutoffs"],
        "source": results["evidence"]["source_hashes"],
        "data": results["evidence"]["data_manifest"],
    }
    metadata["model_version"] = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    utils.save_model({"model": model, "metadata": metadata}, model_path)
    results["artifact_sha256"] = utils.file_sha256(model_path)
    results["artifact"] = str(model_path)
    results["approval_status"] = metadata["approval_status"]
    predictions = infer.run_inference(model, test[_input_columns(cfg)], include_proba=True)
    identifiers = test[[columns["symbol_col"], columns["time_col"]]]
    utils.save_csv(pd.concat([identifiers, predictions], axis=1), paths["predictions"])
    utils.write_json(results, paths["metrics"])


def _run_evidence(cfg: dict) -> dict:
    """Record what reproduces the run: input checksums, model settings, split and gate."""
    raw_files = load_data.list_raw_files(cfg["paths"]["raw_dir"], cfg["data"]["file_pattern"])
    return {
        "python": platform.python_version(),
        "package_versions": {
            name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib"]
        },
        "source_hashes": {p.name: utils.file_sha256(p) for p in Path(__file__).resolve().parent.glob("*.py")},
        "settings": cfg,
        "data_manifest": {path.name: utils.file_sha256(path) for path in raw_files},
        "training": cfg["training"],
        "split": cfg["split"],
        "quality_gate": cfg["quality_gate"],
    }


def run_pipeline(cfg: dict) -> dict:
    """Inputs: cfg.
    Outputs: Evaluation report and candidate artifacts; never automatic approval.
    Why: Run every step with already-loaded settings and return the run report."""
    table = _prepare_table(cfg)
    splits, cutoffs = _labelled_splits(table, cfg)
    model, results = _train_and_evaluate(splits, cfg)
    results["volatility_cutoffs"] = cutoffs
    results["rows"] = {name: len(part) for name, part in splits.items()}
    results["evidence"] = _run_evidence(cfg)
    _write_outputs(model, splits["test"], results, cfg)
    return results


def resolved_settings(raw_dir: Path | None = None, output_root: Path | None = None) -> dict:
    """Inputs: raw_dir, output_root.
    Outputs: Copied SETTINGS with paths resolved against the project or requested output root.
    Why: Keep one tested pipeline responsibility."""
    cfg = copy.deepcopy(SETTINGS)
    base = PROJECT_ROOT if output_root is None else Path(output_root).resolve()
    cfg["paths"] = {k: (base / v).resolve() for k, v in cfg["paths"].items()}
    if raw_dir is not None:
        cfg["paths"]["raw_dir"] = Path(raw_dir).resolve()
    return cfg


def main() -> None:
    """Inputs: SETTINGS.
    Outputs: None; executes the fixed pipeline and logs its validation gate outcome.
    Why: Keep one tested pipeline responsibility."""
    logging.basicConfig(level=SETTINGS["logging"]["level"], format="%(levelname)s %(name)s: %(message)s")
    report = run_pipeline(
        resolved_settings(raw_dir=Path(os.environ["ML_RAW_DIR"]) if os.environ.get("ML_RAW_DIR") else None)
    )
    logger.info("Candidate artifact: %s", report["artifact"])
    logger.info("Metrics: %s; predictions: %s", SETTINGS["paths"]["metrics"], SETTINGS["paths"]["predictions"])
    logger.info(
        "Pipeline finished; validation quality gate %s; candidate still requires review",
        report["quality_gate"]["passed"],
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
    split_settings: dict | None = None,
    time_column: str | None = None,
    horizon: int = 15,
    frequency: str = "1min",
) -> tuple:
    """Inputs: aligned X/y, compatibility split arguments and explicit UTC calendar boundaries.
    Outputs: X_train, X_val, X_test, y_train, y_val, y_test; paired rows retain alignment.
    Why: Forecasting uses dates rather than random shares; random_state never shuffles rows."""
    if stratify or split_settings is None or time_column is None:
        raise ValueError("Forecasting requires explicit chronological boundaries, never stratification")
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("Split indices must align and identify unique rows")
    table = X.copy()
    key = "__split_target__"
    if key in table:
        raise ValueError("Reserved split column")
    table[key] = y
    parts = split_by_time(table, time_column, split_settings, horizon, frequency)
    order = ["train", "validation", "test"]
    return tuple(parts[name].drop(columns=key) for name in order) + tuple(parts[name][key] for name in order)


if __name__ == "__main__":
    main()
