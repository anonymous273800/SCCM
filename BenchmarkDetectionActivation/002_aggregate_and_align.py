from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmark_detection_common import (
    EVALUATION_SEEDS,
    RESULTS_DIRECTORY_NAME,
    find_project_root,
)


RESULT_PREFIX = "benchmark_exact"
RAW_FILE_NAMES = (
    "benchmark_activity_olr_wa.csv",
    "benchmark_activity_pa.csv",
    "benchmark_activity_rls.csv",
    "benchmark_activity_widrowhoff.csv",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_indices(value: Any) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return sorted(int(item) for item in str(value).split(";") if item.strip())


def join_indices(values: Iterable[int]) -> str:
    return ";".join(str(value) for value in values)


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def consolidate_alarm_episodes(detection_indices: list[int], cooldown: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_end = -1

    for detection_index in sorted(detection_indices):
        if current is None or detection_index > current_end:
            current = {
                "episode_alarm_index": detection_index,
                "episode_start_index": detection_index,
                "episode_end_index": detection_index,
                "episode_size": 1,
                "episode_alarm_indices": [detection_index],
            }
            episodes.append(current)
            # This fixed boundary is anchored at the first trigger, exactly as
            # in DriftDetectionQuality.ddq_common.consolidate_alarm_episodes.
            current_end = detection_index + int(cooldown)
        else:
            current["episode_end_index"] = detection_index
            current["episode_size"] += 1
            current["episode_alarm_indices"].append(detection_index)

    return episodes


def filter_episodes_by_size(
    episodes: list[dict[str, Any]], min_episode_size: int
) -> list[dict[str, Any]]:
    if int(min_episode_size) <= 1:
        return list(episodes)
    return [
        episode
        for episode in episodes
        if int(episode.get("episode_size", 1)) >= int(min_episode_size)
    ]


def match_exact_protocol(
    episodes: list[dict[str, Any]],
    true_drift_points: list[int],
    tolerance: int,
    monitored_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    used_episode_ids: set[int] = set()
    matched_episode_ids: set[int] = set()
    delays: list[int] = []
    drift_rows: list[dict[str, Any]] = []
    unmatched_drift_positions: list[int] = []

    a0 = 0
    a1 = 0
    a_gt1 = 0

    for drift_position, drift_point in enumerate(true_drift_points):
        tolerance_start = int(drift_point)
        tolerance_end = int(drift_point) + int(tolerance)
        nearby_episode_ids: list[int] = []
        nearby_episode_starts: list[int] = []
        nearby_episode_alarm_lists: list[str] = []
        matched_episode_id: int | None = None
        matched_alarm_index: int | None = None

        for episode_id, episode in enumerate(episodes):
            alarm_indices = sorted(
                int(value)
                for value in episode.get(
                    "episode_alarm_indices", [episode["episode_alarm_index"]]
                )
            )
            # The episode alarm time is its first trigger. A later trigger cannot
            # convert an episode that began before the drift into a true positive.
            episode_alarm_index = int(episode["episode_alarm_index"])
            if not (tolerance_start <= episode_alarm_index <= tolerance_end):
                continue

            nearby_episode_ids.append(episode_id)
            nearby_episode_starts.append(episode_alarm_index)
            nearby_episode_alarm_lists.append(join_indices(alarm_indices))

            if matched_episode_id is None and episode_id not in used_episode_ids:
                matched_episode_id = episode_id
                matched_alarm_index = episode_alarm_index

        nearby_count = len(nearby_episode_ids)
        if nearby_count == 0:
            a0 += 1
        elif nearby_count == 1:
            a1 += 1
        else:
            a_gt1 += 1

        if matched_episode_id is None or matched_alarm_index is None:
            unmatched_drift_positions.append(drift_position)
            drift_rows.append(
                {
                    "row_type": "true_drift",
                    "true_drift_point": drift_point,
                    "tolerance_start": tolerance_start,
                    "tolerance_end": tolerance_end,
                    "nearby_alarm_episode_count": nearby_count,
                    "nearby_episode_starts": join_indices(nearby_episode_starts),
                    "nearby_episode_alarm_lists": " | ".join(
                        nearby_episode_alarm_lists
                    ),
                    "episode_start": "",
                    "episode_end": "",
                    "episode_size": "",
                    "episode_alarm_indices": "",
                    "matched_alarm_index": "",
                    "delay_samples": "",
                    "status": "FN",
                    "late_alarm_index": "",
                    "late_delay_samples": "",
                }
            )
        else:
            used_episode_ids.add(matched_episode_id)
            matched_episode_ids.add(matched_episode_id)
            delay = int(matched_alarm_index) - int(drift_point)
            delays.append(delay)
            episode = episodes[matched_episode_id]
            alarm_indices = sorted(
                int(value)
                for value in episode.get(
                    "episode_alarm_indices", [episode["episode_alarm_index"]]
                )
            )
            drift_rows.append(
                {
                    "row_type": "true_drift",
                    "true_drift_point": drift_point,
                    "tolerance_start": tolerance_start,
                    "tolerance_end": tolerance_end,
                    "nearby_alarm_episode_count": nearby_count,
                    "nearby_episode_starts": join_indices(nearby_episode_starts),
                    "nearby_episode_alarm_lists": " | ".join(
                        nearby_episode_alarm_lists
                    ),
                    "episode_start": int(episode["episode_start_index"]),
                    "episode_end": int(episode["episode_end_index"]),
                    "episode_size": int(episode.get("episode_size", 1)),
                    "episode_alarm_indices": join_indices(alarm_indices),
                    "matched_alarm_index": int(matched_alarm_index),
                    "delay_samples": delay,
                    "status": "TP",
                    "late_alarm_index": "",
                    "late_delay_samples": "",
                }
            )

    # Supplemental late-response analysis. It does not change the exact SCCM
    # TP/FP/FN values. Only unused retained episodes are considered, and a late
    # response must occur after d+T but before the next true drift.
    unmatched_episode_ids = [
        episode_id for episode_id in range(len(episodes)) if episode_id not in used_episode_ids
    ]
    used_late_episode_ids: set[int] = set()
    late_delays: list[int] = []

    for drift_position in unmatched_drift_positions:
        drift_point = int(true_drift_points[drift_position])
        late_start = drift_point + int(tolerance) + 1
        late_end = (
            int(true_drift_points[drift_position + 1]) - 1
            if drift_position + 1 < len(true_drift_points)
            else int(monitored_samples) - 1
        )
        selected_episode_id: int | None = None
        selected_alarm_index: int | None = None

        for episode_id in unmatched_episode_ids:
            if episode_id in used_late_episode_ids:
                continue
            episode_alarm_index = int(
                episodes[episode_id]["episode_alarm_index"]
            )
            if late_start <= episode_alarm_index <= late_end:
                selected_episode_id = episode_id
                selected_alarm_index = episode_alarm_index
                break

        if selected_episode_id is not None and selected_alarm_index is not None:
            used_late_episode_ids.add(selected_episode_id)
            late_delay = selected_alarm_index - drift_point
            late_delays.append(late_delay)
            row = drift_rows[drift_position]
            row["status"] = "FN_WITH_LATE_RESPONSE"
            row["late_alarm_index"] = selected_alarm_index
            row["late_delay_samples"] = late_delay

    tp = len(matched_episode_ids)
    fp = len(episodes) - tp
    fn = len(true_drift_points) - tp
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)

    for episode_id, episode in enumerate(episodes):
        if episode_id in matched_episode_ids:
            continue
        alarm_indices = sorted(
            int(value)
            for value in episode.get(
                "episode_alarm_indices", [episode["episode_alarm_index"]]
            )
        )
        drift_rows.append(
            {
                "row_type": "false_positive_episode",
                "true_drift_point": "",
                "tolerance_start": "",
                "tolerance_end": "",
                "nearby_alarm_episode_count": "",
                "nearby_episode_starts": "",
                "nearby_episode_alarm_lists": "",
                "episode_start": int(episode["episode_start_index"]),
                "episode_end": int(episode["episode_end_index"]),
                "episode_size": int(episode.get("episode_size", 1)),
                "episode_alarm_indices": join_indices(alarm_indices),
                "matched_alarm_index": "",
                "delay_samples": "",
                "status": "FP_LATE_RESPONSE"
                if episode_id in used_late_episode_ids
                else "FP",
                "late_alarm_index": "",
                "late_delay_samples": "",
            }
        )

    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "delay_sum": sum(delays),
        "delay_count": len(delays),
        "mean_delay_samples": sum(delays) / len(delays) if delays else "",
        "late_detections": len(late_delays),
        "late_delay_sum": sum(late_delays),
        "late_delay_count": len(late_delays),
        "mean_late_delay_samples": (
            sum(late_delays) / len(late_delays) if late_delays else ""
        ),
        "missed_without_response": fn - len(late_delays),
        "a0": a0,
        "a1": a1,
        "a_gt1": a_gt1,
    }
    return result, drift_rows


def summarize_run(row: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    true_points = parse_indices(row.get("all_true_drift_points"))
    detections = parse_indices(row.get("detection_indices"))
    n_samples = parse_int(row.get("full_dataset_samples"))
    monitored_samples = parse_int(row.get("monitored_samples"))
    tolerance_ratio = parse_float(row.get("tolerance_ratio"), 0.05)
    cooldown_factor = parse_float(row.get("cooldown_factor"), 2.0)
    min_episode_size = parse_int(row.get("min_episode_size"), 2)
    increment_user_value = parse_int(row.get("increment_user_value"), 10)

    tolerance = int(round(tolerance_ratio * n_samples))
    cooldown = int(round(cooldown_factor * tolerance))
    raw_episodes = consolidate_alarm_episodes(detections, cooldown)
    retained_episodes = filter_episodes_by_size(raw_episodes, min_episode_size)
    matched, detail_rows = match_exact_protocol(
        retained_episodes, true_points, tolerance, monitored_samples
    )

    mean_delay = matched["mean_delay_samples"]
    mean_delay_batches = (
        float(mean_delay) / increment_user_value
        if mean_delay != "" and increment_user_value > 0
        else ""
    )
    practical_delay_batches = (
        max(0.0, float(mean_delay_batches) - 1.0)
        if mean_delay_batches != ""
        else ""
    )

    common = {
        "model": row.get("model", ""),
        "drift_type": row.get("drift_type", ""),
        "dataset": row.get("dataset", ""),
        "seed": parse_int(row.get("seed")),
        "baseline": row.get("baseline", ""),
        "detector": row.get("detector", ""),
        "adaptation": row.get("adaptation", ""),
        "quality_config_path": row.get("quality_config_path", ""),
        "source_script": row.get("source_script", ""),
    }
    for detail in detail_rows:
        detail.update(common)
        detail.update(
            {
                "tolerance_ratio": tolerance_ratio,
                "tolerance_samples": tolerance,
                "cooldown_factor": cooldown_factor,
                "cooldown_samples": cooldown,
                "min_episode_size": min_episode_size,
            }
        )

    summary = {
        **common,
        "runs": 1,
        "datasets": 1,
        "true_drifts": len(true_points),
        "raw_detector_detections": len(detections),
        "adaptation_activations": parse_int(row.get("adaptation_activations")),
        "candidate_episodes": len(raw_episodes),
        "retained_alarm_episodes": len(retained_episodes),
        "removed_small_episodes": len(raw_episodes) - len(retained_episodes),
        "a0": matched["a0"],
        "a1": matched["a1"],
        "a_gt1": matched["a_gt1"],
        "tp": matched["tp"],
        "fp": matched["fp"],
        "fn": matched["fn"],
        "precision": round(matched["precision"], 4),
        "recall": round(matched["recall"], 4),
        "f1": round(matched["f1"], 4),
        "delay_sum": matched["delay_sum"],
        "delay_count": matched["delay_count"],
        "mean_delay_samples": (
            round(float(mean_delay), 2) if mean_delay != "" else ""
        ),
        "mean_delay_batches": (
            round(float(mean_delay_batches), 2)
            if mean_delay_batches != ""
            else ""
        ),
        "practical_delay_batches": (
            round(float(practical_delay_batches), 2)
            if practical_delay_batches != ""
            else ""
        ),
        "late_detections": matched["late_detections"],
        "late_delay_sum": matched["late_delay_sum"],
        "late_delay_count": matched["late_delay_count"],
        "mean_late_delay_samples": (
            round(float(matched["mean_late_delay_samples"]), 2)
            if matched["mean_late_delay_samples"] != ""
            else ""
        ),
        "missed_without_response": matched["missed_without_response"],
        "tolerance_ratio": tolerance_ratio,
        "tolerance_samples": tolerance,
        "cooldown_factor": cooldown_factor,
        "cooldown_samples": cooldown,
        "min_episode_size": min_episode_size,
        "increment_user_value": increment_user_value,
        "candidate_source": row.get("candidate_source", "long_term"),
        "baseline_configuration_json": row.get("configuration_json", ""),
        "quality_configuration_json": row.get("quality_config_json", ""),
    }
    return summary, detail_rows


def aggregate_rows(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        aggregate = {field: value for field, value in zip(group_fields, key)}
        sum_fields = (
            "runs",
            "datasets",
            "true_drifts",
            "raw_detector_detections",
            "adaptation_activations",
            "candidate_episodes",
            "retained_alarm_episodes",
            "removed_small_episodes",
            "a0",
            "a1",
            "a_gt1",
            "tp",
            "fp",
            "fn",
            "delay_sum",
            "delay_count",
            "late_detections",
            "late_delay_sum",
            "late_delay_count",
            "missed_without_response",
        )
        for field in sum_fields:
            aggregate[field] = sum(parse_int(row.get(field)) for row in group)

        tp = aggregate["tp"]
        fp = aggregate["fp"]
        fn = aggregate["fn"]
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        aggregate["precision"] = round(precision, 4)
        aggregate["recall"] = round(recall, 4)
        aggregate["f1"] = round(f1_score(precision, recall), 4)
        aggregate["mean_delay_samples"] = (
            round(aggregate["delay_sum"] / aggregate["delay_count"], 2)
            if aggregate["delay_count"]
            else ""
        )
        aggregate["mean_late_delay_samples"] = (
            round(aggregate["late_delay_sum"] / aggregate["late_delay_count"], 2)
            if aggregate["late_delay_count"]
            else ""
        )

        # Match the existing SCCM aggregation: practical delay is averaged
        # over datasets and weighted by each dataset's TP count.
        practical_sum = 0.0
        practical_count = 0
        for row in group:
            dataset_tp = parse_int(row.get("tp"))
            value = row.get("practical_delay_batches", "")
            if dataset_tp > 0 and value not in ("", None):
                practical_sum += float(value) * dataset_tp
                practical_count += dataset_tp
        aggregate["practical_delay_batches"] = (
            round(practical_sum / practical_count, 2)
            if practical_count
            else ""
        )
        output.append(aggregate)

    return output


def main() -> None:
    project_root = find_project_root()
    results_root = project_root / "BenchmarkDetectionActivation" / RESULTS_DIRECTORY_NAME
    raw_root = results_root / "raw"

    raw_rows: list[dict[str, str]] = []
    for file_name in RAW_FILE_NAMES:
        path = raw_root / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing raw result file: {path}")
        raw_rows.extend(read_rows(path))

    failures = [row for row in raw_rows if row.get("status") != "ok"]
    expected_seeds = set(EVALUATION_SEEDS)
    expected_rows = 576 * len(expected_seeds)
    valid_rows = [
        row
        for row in raw_rows
        if row.get("status") == "ok"
        and parse_int(row.get("seed")) in expected_seeds
    ]
    if len(raw_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} raw benchmark rows, found {len(raw_rows)}"
        )
    if len(valid_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} successful five-seed benchmark rows, "
            f"found {len(valid_rows)}"
        )

    observed_seeds = {parse_int(row.get("seed")) for row in valid_rows}
    if observed_seeds != expected_seeds:
        raise RuntimeError(
            f"Expected seeds {sorted(expected_seeds)}, found {sorted(observed_seeds)}"
        )

    dataset_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for row in valid_rows:
        summary, details = summarize_run(row)
        dataset_rows.append(summary)
        detail_rows.extend(details)

    seed_method_rows = aggregate_rows(
        dataset_rows,
        (
            "seed",
            "model",
            "drift_type",
            "baseline",
            "detector",
            "adaptation",
        ),
    )
    method_rows = aggregate_rows(
        dataset_rows,
        ("model", "drift_type", "baseline", "detector", "adaptation"),
    )
    detector_rows = aggregate_rows(
        dataset_rows, ("model", "drift_type", "detector")
    )

    dataset_fields = list(dataset_rows[0].keys())
    detail_fields = list(detail_rows[0].keys())
    seed_method_fields = list(seed_method_rows[0].keys())
    method_fields = list(method_rows[0].keys())
    detector_fields = list(detector_rows[0].keys())

    write_rows(
        results_root / f"{RESULT_PREFIX}_by_dataset.csv",
        dataset_rows,
        dataset_fields,
    )
    write_rows(
        results_root / f"{RESULT_PREFIX}_drift_point_details.csv",
        detail_rows,
        detail_fields,
    )
    write_rows(
        results_root / f"{RESULT_PREFIX}_by_model_drift_method_seed.csv",
        seed_method_rows,
        seed_method_fields,
    )
    write_rows(
        results_root / f"{RESULT_PREFIX}_by_model_drift_method.csv",
        method_rows,
        method_fields,
    )
    write_rows(
        results_root / f"{RESULT_PREFIX}_by_model_drift_detector.csv",
        detector_rows,
        detector_fields,
    )

    failure_fields = ["model", "drift_type", "dataset", "baseline", "error"]
    write_rows(
        results_root / f"{RESULT_PREFIX}_failures.csv",
        failures,
        failure_fields,
    )

    total_true_by_type: dict[str, int] = {}
    for drift_type in ("abrupt", "incremental", "gradual"):
        representative = [
            row
            for row in method_rows
            if row["drift_type"] == drift_type
            and row["model"] == "OLR-WA"
            and row["baseline"] == "OLR-WA-ADWIN-RESET"
        ]
        total_true_by_type[drift_type] = (
            parse_int(representative[0]["true_drifts"]) if representative else 0
        )

    summary_path = results_root / f"{RESULT_PREFIX}_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Exact Benchmark Detection and Activation Summary",
                "================================================",
                "Dataset family: Datasets.Synthetic2 (same as SCCM quality runs)",
                f"Evaluation seeds: {list(EVALUATION_SEEDS)}",
                "Train percent: 90",
                f"Successful five-seed baseline runs: {len(valid_rows)}",
                f"Baseline-level errors: {len(failures)}",
                f"Abrupt true drifts per method: {total_true_by_type['abrupt']}",
                f"Incremental true drifts per method: {total_true_by_type['incremental']}",
                f"Gradual true drifts per method: {total_true_by_type['gradual']}",
                "Alignment: exact SCCM tolerance, cooldown, episode-size filtering, and chronological one-to-one matching",
                "Late detections are supplemental and do not alter TP/FP/FN.",
                "",
                "Files:",
                f"- {RESULT_PREFIX}_by_dataset.csv",
                f"- {RESULT_PREFIX}_drift_point_details.csv",
                f"- {RESULT_PREFIX}_by_model_drift_method_seed.csv",
                f"- {RESULT_PREFIX}_by_model_drift_method.csv",
                f"- {RESULT_PREFIX}_by_model_drift_detector.csv",
                f"- {RESULT_PREFIX}_failures.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Successful exact benchmark rows: {len(valid_rows)}")
    print(f"Per-dataset summaries: {len(dataset_rows)}")
    print(f"Seed/model/drift/method summaries: {len(seed_method_rows)}")
    print(f"Model/drift/method summaries: {len(method_rows)}")
    print(f"Model/drift/detector summaries: {len(detector_rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Results directory: {results_root}")


if __name__ == "__main__":
    main()
