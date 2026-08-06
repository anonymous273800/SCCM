"""Measure standalone learners over the complete five-seed synthetic benchmark."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality2"
for value in (str(PROJECT_ROOT), str(DDQ_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from ddq_common import load_dataset
from resource_metrics import measure_call
from Hyperparameters import Hyperparameter
from Models.OLR_WA import OLR_WA
from Models.PA import PA
from Models.RLS import RLS
from Models.WidrowHoff import WidrowHoff
from Utils.ProtocolConfig import EVALUATION_SEEDS

DATASETS = [f"ADS{i:02d}" for i in range(1, 7)] + [f"IDS{i:02d}" for i in range(1, 7)] + [f"GDS{i:02d}" for i in range(1, 7)]
MODELS = ("OLR-WA", "PA", "RLS", "WidrowHoff")


def call_base(model, X_train, y_train, X_test, y_test):
    if model == "OLR-WA":
        increment = Hyperparameter.olr_wa_increment_size(X_train.shape[1], user_defined_val=10)
        return OLR_WA.olr_wa(
            X_train, y_train, Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc, Hyperparameter.olr_wa_base_model_size0,
            increment, X_test, y_test,
        )
    if model == "PA":
        return PA.pa_generic(X_train, y_train, Hyperparameter.pa_C, Hyperparameter.pa_epsilon, X_test, y_test, report_interval=10)
    if model == "RLS":
        return RLS.rls_generic(X_train, y_train, Hyperparameter.rls_lambda_, Hyperparameter.rls_delta, X_test, y_test, report_interval=10)
    return WidrowHoff.widrow_hoff_generic(X_train, y_train, Hyperparameter.wf_learning_rate, X_test, y_test, report_interval=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echo-model-output", action="store_true")
    args = parser.parse_args()
    rows = []
    for dataset in DATASETS:
        for seed in EVALUATION_SEEDS:
            X, y, _, drift_type = load_dataset(dataset, seed)
            train_count = int(0.90 * len(y))
            X_train, y_train = X[:train_count], y[:train_count]
            X_test, y_test = X[train_count:], y[train_count:]
            for model in MODELS:
                sink = sys.stdout if args.echo_model_output else io.StringIO()
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    _, measurement = measure_call(call_base, model, X_train, y_train, X_test, y_test)
                rows.append({
                    "method": "Standalone learner", "model": model,
                    "dataset": dataset, "drift_type": drift_type, "seed": seed,
                    "processed_samples": len(y),
                    "runtime_seconds": measurement.runtime_seconds,
                    "runtime_per_1000_samples": measurement.runtime_seconds * 1000.0 / len(y),
                    "peak_rss_mb": measurement.peak_rss_mb,
                    "peak_rss_delta_mb": measurement.peak_rss_delta_mb,
                    "adaptations_per_1000": 0.0,
                    "recalibrations_per_1000": 0.0,
                })
                print(f"Completed {model} {dataset} seed={seed}")
    output = PROJECT_ROOT / "ComputationalCost" / "results"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "standalone_resource_seed_level.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} standalone resource rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
