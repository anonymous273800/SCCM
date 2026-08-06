from __future__ import annotations

import argparse
import ast
import contextlib
import io
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
    get_true_drift_points,
    is_truthy,
    load_dataset,
    match_alarm_indices_to_true_drifts,
    match_episodes_to_true_drifts,
    read_csv,
    select_candidate_events,
    write_csv,
)
from ddq2_statistics import as_float, descriptive  # noqa: E402

DEFAULT_TOLERANCES = [0.025, 0.05, 0.10]
DEFAULT_COOLDOWNS = [1.0, 2.0, 3.0]
DEFAULT_MIN_EPISODES = [1, 2, 3]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_config(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    configs = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    configs.append(ast.literal_eval(node.value))
    if not configs:
        raise ValueError(f"No active CONFIG found in {path}")
    return dict(configs[-1])


def parse_indices(value: Any) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(item) for item in str(value).split(";") if item.strip()]


def result_row(
    *,
    family: str,
    base_model: str,
    method: str,
    dataset: str,
    drift_type: str,
    seed: int,
    evaluation_level: str,
    tolerance_ratio: float,
    cooldown_factor: Any,
    min_episode_size: Any,
    processing_increment: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    mean_samples = result["mean_delay"]
    mean_increments = (
        mean_samples / processing_increment if mean_samples is not None else ""
    )
    return {
        "method_family": family,
        "base_model": base_model,
        "method": method,
        "dataset": dataset,
        "drift_type": drift_type,
        "seed": seed,
        "evaluation_level": evaluation_level,
        "tolerance_ratio": tolerance_ratio,
        "cooldown_factor": cooldown_factor,
        "min_episode_size": min_episode_size,
        "processing_increment": processing_increment,
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "delay_sum": result["delay_sum"],
        "delay_count": result["delay_count"],
        "mean_delay_samples": mean_samples if mean_samples is not None else "",
        "mean_delay_increments": mean_increments,
    }


def load_sccm_sensitivity(
    tolerances: list[float],
    cooldowns: list[float],
    min_sizes: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for script in sorted(DDQ_ROOT.rglob("quality_run.py")):
        config = read_config(script)
        output_dir = script.parent / "quality_outputs"
        stem = config["model"].replace("-", "").replace(" ", "") + "_" + config["dataset"]
        events_path = output_dir / f"{stem}_events.csv"
        summary_path = output_dir / f"{stem}_summary.csv"
        if not summary_path.exists():
            continue
        events = read_csv(events_path)
        summaries = [
            row for row in read_csv(summary_path)
            if row.get("row_type") == "seed_result"
        ]
        for summary in summaries:
            seed = int(float(summary["seed"]))
            seed_events = [
                row for row in events
                if int(float(row.get("seed", -1) or -1)) == seed
            ]
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                X, _, meta, drift_type = load_dataset(config["dataset"], seed)
            true_points = get_true_drift_points(meta, drift_type)
            n_samples = int(X.shape[0])
            processing_increment = max(1, int(float(summary.get("processing_increment", 1) or 1)))
            candidates = select_candidate_events(
                seed_events, config.get("candidate_source", "long_term")
            )
            alarm_indices = [int(row["alarm_index"]) for row in candidates]
            method = summary.get("method", summary.get("model", ""))
            base_model = summary.get("base_model", config["model"])

            for tolerance_ratio in tolerances:
                tolerance = int(round(tolerance_ratio * n_samples))
                raw = match_alarm_indices_to_true_drifts(
                    alarm_indices,
                    true_points,
                    tolerance,
                    model=method,
                    dataset=config["dataset"],
                    seed=seed,
                    drift_type=drift_type,
                    evaluation_level="raw_trigger_supplemental",
                )
                raw_rows.append(result_row(
                    family="SCCM",
                    base_model=base_model,
                    method=method,
                    dataset=config["dataset"],
                    drift_type=drift_type,
                    seed=seed,
                    evaluation_level="raw_trigger_supplemental",
                    tolerance_ratio=tolerance_ratio,
                    cooldown_factor="",
                    min_episode_size="",
                    processing_increment=processing_increment,
                    result=raw,
                ))

                for cooldown_factor in cooldowns:
                    cooldown = int(round(cooldown_factor * tolerance))
                    candidate_episodes = consolidate_alarm_episodes(candidates, cooldown)
                    for min_episode_size in min_sizes:
                        episodes = filter_episodes_by_size(
                            candidate_episodes, min_episode_size
                        )
                        matched = match_episodes_to_true_drifts(
                            episodes,
                            true_points,
                            tolerance,
                            model=method,
                            dataset=config["dataset"],
                            seed=seed,
                            drift_type=drift_type,
                        )
                        episode_rows.append(result_row(
                            family="SCCM",
                            base_model=base_model,
                            method=method,
                            dataset=config["dataset"],
                            drift_type=drift_type,
                            seed=seed,
                            evaluation_level="episode_first_trigger_in_window_primary",
                            tolerance_ratio=tolerance_ratio,
                            cooldown_factor=cooldown_factor,
                            min_episode_size=min_episode_size,
                            processing_increment=processing_increment,
                            result=matched,
                        ))
    return raw_rows, episode_rows


def load_baseline_sensitivity(
    tolerances: list[float],
    cooldowns: list[float],
    min_sizes: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_path = DDQ_ROOT / "BaselineResults" / "aggregated" / "baseline_alarm_quality_seed_level.csv"
    if not seed_path.exists():
        return [], []
    source_rows = read_csv(seed_path)
    raw_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for row in source_rows:
        detections = parse_indices(row.get("detection_indices"))
        true_points = parse_indices(row.get("true_drift_points"))
        n_samples = int(float(row.get("full_dataset_samples", 0) or 0))
        processing_increment = max(1, int(float(row.get("processing_increment", 1) or 1)))
        seed = int(float(row.get("seed", 0) or 0))
        events = [{"alarm_index": index} for index in detections]

        for tolerance_ratio in tolerances:
            tolerance = int(round(tolerance_ratio * n_samples))
            raw = match_alarm_indices_to_true_drifts(
                detections,
                true_points,
                tolerance,
                model=row.get("model", ""),
                dataset=row.get("dataset", ""),
                seed=seed,
                drift_type=row.get("drift_type", ""),
                evaluation_level="raw_detector_primary",
            )
            raw_rows.append(result_row(
                family="detector_adaptation_baseline",
                base_model=row.get("base_model", ""),
                method=row.get("method", ""),
                dataset=row.get("dataset", ""),
                drift_type=row.get("drift_type", ""),
                seed=seed,
                evaluation_level="raw_detector_primary",
                tolerance_ratio=tolerance_ratio,
                cooldown_factor="",
                min_episode_size="",
                processing_increment=processing_increment,
                result=raw,
            ))

            for cooldown_factor in cooldowns:
                cooldown = int(round(cooldown_factor * tolerance))
                candidate_episodes = consolidate_alarm_episodes(events, cooldown)
                for min_episode_size in min_sizes:
                    retained = filter_episodes_by_size(candidate_episodes, min_episode_size)
                    episode = match_episodes_to_true_drifts(
                        retained,
                        true_points,
                        tolerance,
                        model=row.get("model", ""),
                        dataset=row.get("dataset", ""),
                        seed=seed,
                        drift_type=row.get("drift_type", ""),
                    )
                    episode_rows.append(result_row(
                        family="detector_adaptation_baseline",
                        base_model=row.get("base_model", ""),
                        method=row.get("method", ""),
                        dataset=row.get("dataset", ""),
                        drift_type=row.get("drift_type", ""),
                        seed=seed,
                        evaluation_level="detector_episode_supplemental",
                        tolerance_ratio=tolerance_ratio,
                        cooldown_factor=cooldown_factor,
                        min_episode_size=min_episode_size,
                        processing_increment=processing_increment,
                        result=episode,
                    ))
    return raw_rows, episode_rows


def micro(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["method_family"], row["base_model"], row["method"],
            row["drift_type"], row["evaluation_level"],
            row["tolerance_ratio"], row["cooldown_factor"],
            row["min_episode_size"], row["seed"],
        )
        seed_groups[key].append(row)

    seed_rows = []
    for key, members in sorted(seed_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        tp = sum(int(row["tp"]) for row in members)
        fp = sum(int(row["fp"]) for row in members)
        fn = sum(int(row["fn"]) for row in members)
        precision, recall, f1 = micro(tp, fp, fn)
        delay_sum = sum(float(row.get("delay_sum", 0) or 0) for row in members)
        delay_count = sum(int(row.get("delay_count", 0) or 0) for row in members)
        increment_sum = sum(
            float(row.get("mean_delay_increments", 0) or 0) * int(row.get("delay_count", 0) or 0)
            for row in members
        )
        seed_rows.append({
            "method_family": key[0], "base_model": key[1], "method": key[2],
            "drift_type": key[3], "evaluation_level": key[4],
            "tolerance_ratio": key[5], "cooldown_factor": key[6],
            "min_episode_size": key[7], "seed": key[8],
            "dataset_count": len(members), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "delay_sum": delay_sum, "delay_count": delay_count,
            "mean_delay_samples": delay_sum / delay_count if delay_count else "",
            "mean_delay_increments": increment_sum / delay_count if delay_count else "",
        })

    final_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = tuple(row[field] for field in (
            "method_family", "base_model", "method", "drift_type",
            "evaluation_level", "tolerance_ratio", "cooldown_factor",
            "min_episode_size",
        ))
        final_groups[key].append(row)

    output = []
    metrics = ["tp", "fp", "fn", "precision", "recall", "f1", "mean_delay_samples", "mean_delay_increments"]
    for key, members in sorted(final_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        item = {
            "method_family": key[0], "base_model": key[1], "method": key[2],
            "drift_type": key[3], "evaluation_level": key[4],
            "tolerance_ratio": key[5], "cooldown_factor": key[6],
            "min_episode_size": key[7], "seed_count": len(members),
            "variability_unit": "seed_after_pooling_six_datasets",
        }
        for metric in metrics:
            stats = descriptive([as_float(row.get(metric)) for row in members])
            for stat_name, value in stats.items():
                item[f"{metric}_{stat_name}"] = value
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute alarm metrics from saved raw events without rerunning models."
    )
    parser.add_argument("--tolerances", default=",".join(map(str, DEFAULT_TOLERANCES)))
    parser.add_argument("--cooldowns", default=",".join(map(str, DEFAULT_COOLDOWNS)))
    parser.add_argument("--min-episode-sizes", default=",".join(map(str, DEFAULT_MIN_EPISODES)))
    args = parser.parse_args()
    tolerances = parse_float_list(args.tolerances)
    cooldowns = parse_float_list(args.cooldowns)
    min_sizes = parse_int_list(args.min_episode_sizes)

    out_dir = DDQ_ROOT / "SensitivityResults"
    out_dir.mkdir(parents=True, exist_ok=True)

    sccm_raw, sccm_episode = load_sccm_sensitivity(tolerances, cooldowns, min_sizes)
    baseline_raw, baseline_episode = load_baseline_sensitivity(tolerances, cooldowns, min_sizes)

    write_csv(sccm_raw, out_dir / "sccm_raw_trigger_tolerance_sensitivity_seed_level.csv")
    write_csv(sccm_episode, out_dir / "sccm_episode_parameter_sensitivity_seed_level.csv")
    write_csv(baseline_raw, out_dir / "baseline_raw_alarm_tolerance_sensitivity_seed_level.csv")
    write_csv(baseline_episode, out_dir / "baseline_episode_parameter_sensitivity_seed_level.csv")

    all_rows = sccm_raw + sccm_episode + baseline_raw + baseline_episode
    write_csv(summarize_sensitivity(all_rows), out_dir / "parameter_sensitivity_mean_std.csv")
    print("SCCM raw rows:", len(sccm_raw))
    print("SCCM episode rows:", len(sccm_episode))
    print("Baseline raw rows:", len(baseline_raw))
    print("Baseline episode rows:", len(baseline_episode))
    print("Saved sensitivity results in", out_dir)


if __name__ == "__main__":
    main()
