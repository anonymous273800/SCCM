from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality2"
for path in (str(PROJECT_ROOT), str(DDQ_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ddq_common import (  # noqa: E402
    consolidate_alarm_episodes,
    filter_episodes_by_size,
    match_alarm_indices_to_true_drifts,
    match_episodes_to_true_drifts,
    read_csv,
    write_csv,
)
from ddq2_statistics import as_float, descriptive  # noqa: E402

DEFAULT_SEEDS = [0, 1, 42, 123, 7]
BASE_MODELS = ["OLR-WA", "PA", "RLS", "WidrowHoff"]
BASELINES = [
    "ADWIN-RESET", "ADWIN-WINDOW", "ADWIN-SSPT", "ADWIN-OHL",
    "KSWIN-RESET", "KSWIN-WINDOW", "KSWIN-SSPT", "KSWIN-OHL",
]
DATASETS_BY_DRIFT = {
    "abrupt": [f"ADS{i:02d}" for i in range(1, 7)],
    "incremental": [f"IDS{i:02d}" for i in range(1, 7)],
    "gradual": [f"GDS{i:02d}" for i in range(1, 7)],
}
COUNT_METRICS = [
    "true_drifts", "detector_detections", "adaptation_activations",
    "alarm_episodes", "tp", "fp", "fn", "episode_tp", "episode_fp",
    "episode_fn", "intervention_events",
]
SUMMARY_METRICS = COUNT_METRICS + [
    "precision", "recall", "f1", "mean_delay_samples",
    "mean_delay_increments", "episode_precision", "episode_recall",
    "episode_f1", "episode_mean_delay_samples", "episode_mean_delay_increments",
    "runtime_seconds", "runtime_per_1000_samples", "peak_rss_mb",
    "peak_rss_delta_mb", "adaptations_per_1000", "interventions_per_1000",
]


def parse_indices(value: Any) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(item) for item in str(value).split(";") if item.strip()]


def baseline_method(row: dict[str, Any]) -> str:
    return f"{row['detector']}-{row['adaptation']}"


def load_raw_records() -> list[dict[str, Any]]:
    raw_root = DDQ_ROOT / "BaselineResults" / "raw"
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.glob("baseline_activity_*.csv")):
        for row in read_csv(path):
            item = dict(row)
            item["source_file"] = str(path.relative_to(DDQ_ROOT))
            rows.append(item)
    return rows


def evaluate_record(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detections = parse_indices(row.get("detection_indices"))
    true_points = parse_indices(row.get("all_true_drift_points"))
    n_samples = int(float(row.get("full_dataset_samples", 0) or 0))
    tolerance_ratio = float(row.get("tolerance_ratio", 0.05) or 0.05)
    cooldown_factor = float(row.get("cooldown_factor", 2.0) or 2.0)
    min_episode_size = int(float(row.get("min_episode_size", 2) or 1))
    tolerance = int(round(tolerance_ratio * n_samples))
    cooldown = int(round(cooldown_factor * tolerance))
    processing_increment = max(1, int(float(row.get("processing_increment", 1) or 1)))
    method = baseline_method(row)
    full_method = f"{row['model']}-{method}"
    seed = int(float(row.get("seed", 0) or 0))

    raw = match_alarm_indices_to_true_drifts(
        detections,
        true_points,
        tolerance,
        model=full_method,
        dataset=row.get("dataset", ""),
        seed=seed,
        drift_type=row.get("drift_type", ""),
        evaluation_level="raw_detector_primary",
    )

    events = [{"alarm_index": index} for index in detections]
    candidate_episodes = consolidate_alarm_episodes(events, cooldown)
    retained_episodes = filter_episodes_by_size(candidate_episodes, min_episode_size)
    episode = match_episodes_to_true_drifts(
        retained_episodes,
        true_points,
        tolerance,
        model=full_method,
        dataset=row.get("dataset", ""),
        seed=seed,
        drift_type=row.get("drift_type", ""),
    )
    for detail in episode["detail_rows"]:
        detail["evaluation_level"] = "detector_episode_supplemental"

    def delay_values(result: dict[str, Any]) -> tuple[Any, Any]:
        mean_samples = result["mean_delay"]
        mean_increments = (
            mean_samples / processing_increment if mean_samples is not None else ""
        )
        return (mean_samples if mean_samples is not None else "", mean_increments)

    mean_samples, mean_increments = delay_values(raw)
    episode_mean_samples, episode_mean_increments = delay_values(episode)
    processed_samples = int(float(row.get("processed_samples", row.get("monitored_samples", 0)) or 0))

    result = {
        "row_type": "seed_result",
        "method_family": "detector_adaptation_baseline",
        "primary_evaluation": "raw_detector_alarm",
        "supplemental_evaluation": "episode_first_trigger_in_window",
        "base_model": row.get("model", ""),
        "model": full_method,
        "method": method,
        "detector": row.get("detector", ""),
        "adaptation": row.get("adaptation", ""),
        "dataset": row.get("dataset", ""),
        "drift_type": row.get("drift_type", ""),
        "seed": seed,
        "true_drifts": len(true_points),
        "true_drift_points": ";".join(map(str, true_points)),
        "detector_detections": len(detections),
        "detection_indices": ";".join(map(str, detections)),
        "adaptation_activations": int(float(row.get("adaptation_activations", 0) or 0)),
        "recalibration_activations": 0,
        "intervention_events": int(float(row.get("adaptation_activations", 0) or 0)),
        "candidate_episodes": len(candidate_episodes),
        "alarm_episodes": len(retained_episodes),
        "removed_small_episodes": max(0, len(candidate_episodes) - len(retained_episodes)),
        "tolerance": tolerance,
        "tolerance_ratio": tolerance_ratio,
        "cooldown": cooldown,
        "cooldown_factor": cooldown_factor,
        "min_episode_size": min_episode_size,
        "processing_increment": processing_increment,
        "full_dataset_samples": n_samples,
        "monitored_samples": int(float(row.get("monitored_samples", processed_samples) or processed_samples)),
        "processed_samples": processed_samples,
        "tp": raw["tp"],
        "fp": raw["fp"],
        "fn": raw["fn"],
        "precision": raw["precision"],
        "recall": raw["recall"],
        "f1": raw["f1"],
        "mean_delay_samples": mean_samples,
        "mean_delay_increments": mean_increments,
        "mean_delay": mean_samples,
        "mean_delay_batches": mean_increments,
        "delay_sum": raw["delay_sum"],
        "delay_count": raw["delay_count"],
        "episode_tp": episode["tp"],
        "episode_fp": episode["fp"],
        "episode_fn": episode["fn"],
        "episode_precision": episode["precision"],
        "episode_recall": episode["recall"],
        "episode_f1": episode["f1"],
        "episode_mean_delay_samples": episode_mean_samples,
        "episode_mean_delay_increments": episode_mean_increments,
        "episode_delay_sum": episode["delay_sum"],
        "episode_delay_count": episode["delay_count"],
        "runtime_seconds": float(row.get("runtime_seconds", 0) or 0),
        "runtime_per_1000_samples": float(row.get("runtime_per_1000_samples", 0) or 0),
        "rss_before_mb": row.get("rss_before_mb", ""),
        "peak_rss_mb": row.get("peak_rss_mb", ""),
        "peak_rss_delta_mb": row.get("peak_rss_delta_mb", ""),
        "memory_measurement_method": row.get("memory_measurement_method", ""),
        "adaptations_per_1000": float(row.get("adaptations_per_1000", 0) or 0),
        "recalibrations_per_1000": 0.0,
        "interventions_per_1000": float(row.get("interventions_per_1000", 0) or 0),
        "configuration_json": row.get("configuration_json", ""),
        "quality_config_json": row.get("quality_config_json", ""),
        "source_script": row.get("source_script", ""),
        "status": row.get("status", ""),
        "error": row.get("error", ""),
    }
    details = raw["detail_rows"] + episode["detail_rows"]
    return result, details


def add_descriptives(output: dict[str, Any], members: list[dict[str, Any]]) -> None:
    for metric in SUMMARY_METRICS:
        stats = descriptive([as_float(row.get(metric)) for row in members])
        for stat_name, value in stats.items():
            output[f"{metric}_{stat_name}"] = value


def aggregate_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["base_model"], row["method"], row["dataset"], row["drift_type"])].append(row)
    output = []
    for (base_model, method, dataset, drift_type), members in sorted(grouped.items()):
        item = {
            "base_model": base_model,
            "method": method,
            "dataset": dataset,
            "drift_type": drift_type,
            "seed_count": len(members),
            "seeds": ";".join(str(value) for value in sorted({row["seed"] for row in members})),
            "variability_unit": "seed",
        }
        add_descriptives(item, members)
        output.append(item)
    return output


def micro_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def aggregate_drift_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["base_model"], row["method"], row["drift_type"], row["seed"])].append(row)
    output = []
    for (base_model, method, drift_type, seed), members in sorted(grouped.items()):
        tp = sum(int(row["tp"]) for row in members)
        fp = sum(int(row["fp"]) for row in members)
        fn = sum(int(row["fn"]) for row in members)
        precision, recall, f1 = micro_metrics(tp, fp, fn)
        delay_sum = sum(float(row.get("delay_sum", 0) or 0) for row in members)
        delay_count = sum(int(row.get("delay_count", 0) or 0) for row in members)
        increment_delay_sum = sum(
            float(row.get("mean_delay_increments", 0) or 0) * int(row.get("delay_count", 0) or 0)
            for row in members
        )
        item = {
            "base_model": base_model,
            "method": method,
            "drift_type": drift_type,
            "seed": seed,
            "dataset_count": len(members),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "delay_sum": delay_sum,
            "delay_count": delay_count,
            "mean_delay_samples": delay_sum / delay_count if delay_count else "",
            "mean_delay_increments": increment_delay_sum / delay_count if delay_count else "",
            "runtime_seconds": sum(float(row.get("runtime_seconds", 0) or 0) for row in members),
            "processed_samples": sum(int(row.get("processed_samples", 0) or 0) for row in members),
            "adaptation_activations": sum(int(row.get("adaptation_activations", 0) or 0) for row in members),
            "intervention_events": sum(int(row.get("intervention_events", 0) or 0) for row in members),
            "peak_rss_mb": max(float(row.get("peak_rss_mb", 0) or 0) for row in members),
            "peak_rss_delta_mb": max(float(row.get("peak_rss_delta_mb", 0) or 0) for row in members),
        }
        item["runtime_per_1000_samples"] = (
            item["runtime_seconds"] * 1000 / item["processed_samples"]
            if item["processed_samples"] else 0.0
        )
        item["adaptations_per_1000"] = (
            item["adaptation_activations"] * 1000 / item["processed_samples"]
            if item["processed_samples"] else 0.0
        )
        item["interventions_per_1000"] = (
            item["intervention_events"] * 1000 / item["processed_samples"]
            if item["processed_samples"] else 0.0
        )
        output.append(item)
    return output


def paper_variability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["base_model"], row["method"], row["drift_type"])].append(row)
    output = []
    metrics = [
        "precision", "recall", "f1", "fp", "fn", "mean_delay_samples",
        "mean_delay_increments", "runtime_per_1000_samples", "peak_rss_delta_mb",
        "adaptations_per_1000", "interventions_per_1000",
    ]
    for (base_model, method, drift_type), members in sorted(grouped.items()):
        item = {
            "base_model": base_model,
            "method": method,
            "drift_type": drift_type,
            "seed_count": len(members),
            "variability_unit": "seed_after_pooling_six_datasets",
        }
        for metric in metrics:
            stats = descriptive([as_float(row.get(metric)) for row in members])
            for name, value in stats.items():
                item[f"{metric}_{name}"] = value
        output.append(item)
    return output


def build_completeness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed = {
        (row["base_model"], row["method"], row["dataset"], int(row["seed"]))
        for row in rows if row.get("status") == "ok"
    }
    output = []
    for drift_type, datasets in DATASETS_BY_DRIFT.items():
        for base_model in BASE_MODELS:
            for method in BASELINES:
                for dataset in datasets:
                    for seed in DEFAULT_SEEDS:
                        output.append({
                            "base_model": base_model,
                            "method": method,
                            "dataset": dataset,
                            "drift_type": drift_type,
                            "seed": seed,
                            "status": "complete" if (base_model, method, dataset, seed) in observed else "missing",
                        })
    return output


def main() -> None:
    out_dir = DDQ_ROOT / "BaselineResults" / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_records = load_raw_records()
    valid_records = [row for row in raw_records if row.get("status", "ok") == "ok"]
    failed_records = [row for row in raw_records if row.get("status", "ok") != "ok"]
    seed_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for row in valid_records:
        result, details = evaluate_record(row)
        seed_rows.append(result)
        detail_rows.extend(details)

    dataset_rows = aggregate_by_dataset(seed_rows)
    drift_seed_rows = aggregate_drift_seed(seed_rows)
    paper_rows = paper_variability(drift_seed_rows)
    completeness = build_completeness(seed_rows)

    write_csv(seed_rows, out_dir / "baseline_alarm_quality_seed_level.csv")
    write_csv(failed_records, out_dir / "baseline_failed_method_runs.csv")
    write_csv(detail_rows, out_dir / "baseline_alarm_details.csv")
    write_csv(dataset_rows, out_dir / "baseline_alarm_quality_by_dataset_mean_std.csv")
    write_csv(drift_seed_rows, out_dir / "baseline_alarm_quality_by_model_method_drift_seed.csv")
    write_csv(paper_rows, out_dir / "baseline_alarm_quality_for_paper_mean_std.csv")
    write_csv(completeness, out_dir / "baseline_run_completeness.csv")

    complete = sum(row["status"] == "complete" for row in completeness)
    print("Raw baseline method-seed rows:", len(raw_records))
    print("Failed method-seed rows:", len(failed_records))
    print("Evaluated seed rows:", len(seed_rows))
    print("Expected seed rows:", len(completeness))
    print("Complete rows:", complete)
    print("Missing rows:", len(completeness) - complete)
    print("Primary baseline evaluation: raw detector alarms")
    print("Supplemental baseline evaluation: episodes matched by the first trigger only")
    print("Saved results in", out_dir)


if __name__ == "__main__":
    main()
