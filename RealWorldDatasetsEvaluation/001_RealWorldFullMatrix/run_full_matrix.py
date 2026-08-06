from __future__ import annotations
import argparse
import gc
import json
import math
import random
import sys
sys.dont_write_bytecode = True
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealWorldDatasetsEvaluation.config import DATASETS, MODELS, METHODS, SEEDS
from RealWorldDatasetsEvaluation.common.data_registry import load_dataset
from RealWorldDatasetsEvaluation.common.instrumentation import Activity, capture_events, count_sccm_calls, measure, per_1000
from RealWorldDatasetsEvaluation.common.io_utils import append_csv, append_jsonl, json_text
from RealWorldDatasetsEvaluation.common.model_registry import build_call
from RealWorldDatasetsEvaluation.common.project import results_dir

FIELDS = [
    "dataset","model","method","seed","status","total_samples","train_samples","stream_samples",
    "processed_samples","n_features","increment_size","report_interval","final_r2","avg_r2","std_r2",
    "min_r2","max_r2","n_r2","avg_mse","std_mse","min_mse","max_mse","n_mse",
    "runtime_seconds","runtime_per_1000_samples","rss_before_mb","peak_rss_mb","peak_rss_delta_mb",
    "memory_measurement_method","detector_detections","adaptation_activations","sccm_adaptations",
    "sccm_recalibrations","total_interventions","interventions_per_1000_samples","detection_indices",
    "configuration_json","error","traceback"
]


def finite_summary(values):
    array = np.asarray(values if values is not None else [], dtype=float).ravel()
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"avg": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {
        "avg": float(np.mean(array)), "std": float(np.std(array, ddof=1)) if len(array)>1 else 0.0,
        "min": float(np.min(array)), "max": float(np.max(array)), "n": int(len(array))
    }


def normalize_result(result):
    if not isinstance(result, (tuple, list)):
        raise TypeError(f"Expected tuple/list result, got {type(result).__name__}")
    if len(result) == 3:
        final_r2, r2_values, mse_values = result
    elif len(result) == 2:
        final_r2, mse_values = result
        r2_values = []
    else:
        raise ValueError(f"Unexpected result length: {len(result)}")
    return float(final_r2), r2_values, mse_values


def parse_csv_list(text, allowed):
    if not text:
        return list(allowed)
    requested = [x.strip() for x in text.split(",") if x.strip()]
    invalid = [x for x in requested if x not in allowed]
    if invalid:
        raise ValueError(f"Invalid values {invalid}; allowed: {allowed}")
    return requested


def existing_success_keys(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    ok = df[df["status"].astype(str).str.lower().eq("complete")]
    return set(zip(ok.dataset.astype(str), ok.model.astype(str), ok.method.astype(str), ok.seed.astype(int)))



def remove_selected_keys(path: Path, selected_keys: set[tuple[str, str, str, int]]) -> int:
    """Remove prior complete/failed rows for keys that will be rerun.

    A timestamped backup is written first. This prevents duplicate rows from
    contaminating aggregation after recovery runs.
    """
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if df.empty:
        return 0
    keys = list(zip(
        df["dataset"].astype(str), df["model"].astype(str),
        df["method"].astype(str), df["seed"].astype(int),
    ))
    keep = np.array([key not in selected_keys for key in keys], dtype=bool)
    removed = int((~keep).sum())
    if removed:
        backup = path.with_name(path.stem + ".before_recovery.csv")
        if not backup.exists():
            df.to_csv(backup, index=False)
        df.loc[keep].to_csv(path, index=False)
    return removed

def run(args) -> int:
    datasets = parse_csv_list(args.datasets, DATASETS)
    models = parse_csv_list(args.models, MODELS)
    methods = parse_csv_list(args.methods, METHODS)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(SEEDS)
    invalid_seeds = [s for s in seeds if s not in SEEDS]
    if invalid_seeds:
        raise ValueError(f"Invalid seeds {invalid_seeds}; allowed: {SEEDS}")

    output = results_dir("raw") / "realworld_seed_level.csv"
    event_path = results_dir("logs") / "method_events.jsonl"
    selected_keys = {
        (dataset, model, method, seed)
        for dataset in datasets for model in models
        for method in methods for seed in seeds
    }
    if args.replace_existing:
        removed = remove_selected_keys(output, selected_keys)
        print(f"Removed {removed} existing rows for the selected recovery matrix.")
    completed = existing_success_keys(output) if args.resume else set()
    total = len(datasets)*len(models)*len(methods)*len(seeds)
    counter = 0

    for dataset in datasets:
        print(f"Loading dataset {dataset}...")
        try:
            data = load_dataset(dataset)
        except Exception as exc:
            print(f"DATASET ERROR {dataset}: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise
            continue

        for model in models:
            for seed in seeds:
                for method in methods:
                    counter += 1
                    key = (dataset, model, method, seed)
                    if key in completed:
                        print(f"[{counter}/{total}] SKIP complete {key}")
                        continue
                    print(f"[{counter}/{total}] {dataset} | {model} | {method} | seed={seed}")
                    random.seed(seed)
                    np.random.seed(seed)
                    activity = Activity()
                    row = {
                        "dataset": dataset, "model": model, "method": method, "seed": seed,
                        "status": "failed", "total_samples": data.total_samples,
                        "train_samples": data.train_samples, "stream_samples": data.stream_samples,
                        "processed_samples": int(data.fit_X.shape[0]), "n_features": data.n_features,
                        "increment_size": data.increment_size, "report_interval": data.report_interval,
                    }
                    try:
                        call, configuration = build_call(model, method, dataset, data)
                        with capture_events(activity, echo=args.echo_model_output):
                            if method == "SCCM":
                                with count_sccm_calls(activity):
                                    result, resources = measure(call)
                            else:
                                result, resources = measure(call)
                        final_r2, r2_values, mse_values = normalize_result(result)
                        r2 = finite_summary(r2_values)
                        mse = finite_summary(mse_values)
                        if method.startswith(("ADWIN-", "KSWIN-")):
                            activity.adaptation_activations = max(
                                activity.adaptation_activations, activity.detector_detections
                            )
                        total_interventions = (
                            activity.sccm_adaptations + activity.sccm_recalibrations
                            if method == "SCCM" else activity.adaptation_activations
                        )
                        processed = int(data.fit_X.shape[0])
                        row.update({
                            "status":"complete", "final_r2":final_r2,
                            "avg_r2":r2["avg"],"std_r2":r2["std"],"min_r2":r2["min"],"max_r2":r2["max"],"n_r2":r2["n"],
                            "avg_mse":mse["avg"],"std_mse":mse["std"],"min_mse":mse["min"],"max_mse":mse["max"],"n_mse":mse["n"],
                            "runtime_seconds":resources.runtime_seconds,
                            "runtime_per_1000_samples":per_1000(resources.runtime_seconds, processed),
                            "rss_before_mb":resources.rss_before_mb,"peak_rss_mb":resources.peak_rss_mb,
                            "peak_rss_delta_mb":resources.peak_rss_delta_mb,
                            "memory_measurement_method":resources.measurement_method,
                            "detector_detections":activity.detector_detections,
                            "adaptation_activations":activity.adaptation_activations,
                            "sccm_adaptations":activity.sccm_adaptations,
                            "sccm_recalibrations":activity.sccm_recalibrations,
                            "total_interventions":total_interventions,
                            "interventions_per_1000_samples":per_1000(total_interventions, processed),
                            "detection_indices":";".join(map(str, activity.detection_indices)),
                            "configuration_json":json_text(configuration), "error":"", "traceback":"",
                        })
                        for line in activity.event_lines:
                            append_jsonl(event_path, {"dataset":dataset,"model":model,"method":method,"seed":seed,"event":line})
                    except Exception as exc:
                        row.update({"error":f"{type(exc).__name__}: {exc}", "traceback":traceback.format_exc()})
                        print(f"  ERROR: {row['error']}")
                        if not args.continue_on_error:
                            append_csv(output, row, FIELDS)
                            raise
                    append_csv(output, row, FIELDS)
                    gc.collect()
    print(f"Raw results: {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the complete real-world 4x8x10x5 matrix.")
    parser.add_argument("--datasets", help="Comma-separated subset")
    parser.add_argument("--models", help="Comma-separated subset")
    parser.add_argument("--methods", help="Comma-separated subset")
    parser.add_argument("--seeds", help="Comma-separated subset")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--echo-model-output", action="store_true")
    parser.add_argument(
        "--replace-existing", action="store_true",
        help="Back up and remove selected keys before rerunning them."
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
