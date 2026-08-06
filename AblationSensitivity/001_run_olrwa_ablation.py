"""Run the manuscript SCCM component ablation on representative streams."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Datasets.Synthetic2.Abrupt import ADS05
from Datasets.Synthetic2.Gradual import GDS05
from Hyperparameters import Hyperparameter
from Models.OLR_WA import OLR_WA, OLR_WA_SCCM_DriftQuality
from Utils.ProtocolConfig import EVALUATION_SEEDS, SCCM_SAFE_BAND, SCCM_Z

DATASETS = {
    "ADS05": (ADS05.get_DS05, "abrupt"),
    "GDS05": (GDS05.get_GDS05, "alternating_gradual"),
}


def finite_mean(values) -> float:
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else math.nan


def variants() -> dict[str, dict[str, object]]:
    return {
        "full_sccm": {
            "multiplier": SCCM_Z, "safe_band": SCCM_SAFE_BAND,
            "enable_recalibration": True, "sccm_window_size": 4,
            "used_kpi_window_size": 4,
        },
        "without_recalibration": {
            "multiplier": SCCM_Z, "safe_band": SCCM_SAFE_BAND,
            "enable_recalibration": False, "sccm_window_size": 4,
            "used_kpi_window_size": 4,
        },
        "without_safe_band": {
            "multiplier": SCCM_Z, "safe_band": 0.0,
            "enable_recalibration": True, "sccm_window_size": 4,
            "used_kpi_window_size": 4,
        },
        "rho_0.0228_z2.0": {
            "multiplier": float(norm.ppf(1.0 - 0.02275013194817921)),
            "safe_band": SCCM_SAFE_BAND, "enable_recalibration": True,
            "sccm_window_size": 4, "used_kpi_window_size": 4,
        },
        "rho_0.1587_z1.0": {
            "multiplier": float(norm.ppf(1.0 - 0.15865525393145707)),
            "safe_band": SCCM_SAFE_BAND, "enable_recalibration": True,
            "sccm_window_size": 4, "used_kpi_window_size": 4,
        },
        "kpi_window_10": {
            "multiplier": SCCM_Z, "safe_band": SCCM_SAFE_BAND,
            "enable_recalibration": True, "sccm_window_size": 10,
            "used_kpi_window_size": 10,
        },
        "kpi_window_20": {
            "multiplier": SCCM_Z, "safe_band": SCCM_SAFE_BAND,
            "enable_recalibration": True, "sccm_window_size": 20,
            "used_kpi_window_size": 20,
        },
        "kpi_window_30": {
            "multiplier": SCCM_Z, "safe_band": SCCM_SAFE_BAND,
            "enable_recalibration": True, "sccm_window_size": 30,
            "used_kpi_window_size": 30,
        },
    }


def run_dataset(dataset_name: str, echo: bool = False) -> list[dict[str, object]]:
    getter, drift_type = DATASETS[dataset_name]
    rows: list[dict[str, object]] = []
    for seed in EVALUATION_SEEDS:
        X, y, *_ = getter(seed=seed, return_meta=True)
        train_count = int(0.90 * len(y))
        X_train, y_train = X[:train_count], y[:train_count]
        X_test, y_test = X[train_count:], y[train_count:]
        increment_size = Hyperparameter.olr_wa_increment_size(X_train.shape[1], user_defined_val=10)

        stream = sys.stdout if echo else io.StringIO()
        with contextlib.redirect_stdout(stream):
            _, base_r2, base_mse = OLR_WA.olr_wa(
                X_train, y_train,
                Hyperparameter.olr_wa_w_base,
                Hyperparameter.olr_wa_w_inc,
                Hyperparameter.olr_wa_base_model_size0,
                increment_size, X_test, y_test,
            )
        rows.append({
            "dataset": dataset_name, "drift_type": drift_type, "seed": seed,
            "variant": "base_model", "avg_r2": finite_mean(base_r2),
            "avg_mse": finite_mean(base_mse), "configuration_json": "{}",
        })

        for variant_name, configuration in variants().items():
            stream = sys.stdout if echo else io.StringIO()
            with contextlib.redirect_stdout(stream):
                _, r2_values, mse_values = OLR_WA_SCCM_DriftQuality.olr_wa_sccm(
                    X_train, y_train,
                    Hyperparameter.olr_wa_w_base,
                    Hyperparameter.olr_wa_w_inc,
                    Hyperparameter.olr_wa_base_model_size0,
                    increment_size, X_test, y_test,
                    kpi="R2",
                    max_recalibration_batches=5,
                    **configuration,
                )
            rows.append({
                "dataset": dataset_name, "drift_type": drift_type, "seed": seed,
                "variant": variant_name, "avg_r2": finite_mean(r2_values),
                "avg_mse": finite_mean(mse_values),
                "configuration_json": json.dumps(configuration, sort_keys=True),
            })
        print(f"Completed {dataset_name}, seed={seed}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="ADS05,GDS05")
    parser.add_argument("--echo-model-output", action="store_true")
    args = parser.parse_args()
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    rows: list[dict[str, object]] = []
    for dataset in selected:
        rows.extend(run_dataset(dataset, echo=args.echo_model_output))

    output_dir = PROJECT_ROOT / "results" / "ablation_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "olrwa_ablation_seed_level.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Mean and standard deviation across the five evaluation seeds.
    import pandas as pd
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "drift_type", "variant"], as_index=False).agg(
        avg_r2_mean=("avg_r2", "mean"), avg_r2_std=("avg_r2", "std"),
        avg_mse_mean=("avg_mse", "mean"), avg_mse_std=("avg_mse", "std"),
        seed_count=("seed", "count"),
    )
    summary.to_csv(output_dir / "olrwa_ablation_mean_std.csv", index=False)
    print(f"Ablation results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
