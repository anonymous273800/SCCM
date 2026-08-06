from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


# Place this file directly inside:
# DriftDetectionQuality2/
#
# Then run:
# python aggregate_drift_point_details.py


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "drift_point_aggregates"

ALL_DETAILS_FILE = OUTPUT_DIR / "All_Drift_Point_Details.csv"
BY_DATASET_FILE = OUTPUT_DIR / "Drift_Point_Summary_By_Dataset.csv"
BY_MODEL_DRIFT_FILE = OUTPUT_DIR / "Drift_Point_Summary_By_Model_And_Drift.csv"

DETAIL_FILE_PATTERN = "*_drift_point_details.csv"


def safe_int(value, default=0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(file_path):
    with file_path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv_rows(rows, file_path, preferred_columns=None):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        file_path.write_text("", encoding="utf-8")
        return

    all_columns = []
    for row in rows:
        for key in row:
            if key not in all_columns:
                all_columns.append(key)

    preferred_columns = preferred_columns or []

    fieldnames = [
        column
        for column in preferred_columns
        if column in all_columns
    ]
    fieldnames.extend(
        column
        for column in all_columns
        if column not in fieldnames
    )

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_detail_files():
    files = []

    for file_path in BASE_DIR.rglob(DETAIL_FILE_PATTERN):
        if OUTPUT_DIR in file_path.parents:
            continue
        files.append(file_path)

    return sorted(files)


def load_all_detail_rows(detail_files):
    rows = []

    for file_path in detail_files:
        for row in read_csv_rows(file_path):
            row = dict(row)
            row["source_file"] = str(file_path.relative_to(BASE_DIR))
            rows.append(row)

    return rows


def unique_sorted_text(values):
    cleaned = {
        str(value).strip()
        for value in values
        if str(value).strip() != ""
    }

    def sort_key(value):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return ";".join(sorted(cleaned, key=sort_key))


def summarize_group(rows):
    true_drift_rows = [
        row
        for row in rows
        if row.get("row_type", "").strip() == "true_drift"
    ]

    tp_rows = [
        row
        for row in true_drift_rows
        if row.get("status", "").strip() == "TP"
    ]

    fn_rows = [
        row
        for row in true_drift_rows
        if row.get("status", "").strip() == "FN"
    ]

    fp_rows = [
        row
        for row in rows
        if (
            row.get("status", "").strip() == "FP"
            or row.get("row_type", "").strip() == "false_positive_episode"
        )
    ]

    nearby_counts = [
        safe_int(
            row.get("nearby_alarm_episode_count")
            if str(row.get("nearby_alarm_episode_count", "")).strip() != ""
            else row.get("nearby_raw_alarm_count")
        )
        for row in true_drift_rows
    ]

    delays = [
        delay
        for delay in (
            safe_float(row.get("delay_samples"))
            for row in tp_rows
        )
        if delay is not None
    ]

    true_drifts = len(true_drift_rows)
    matched_drifts = len(tp_rows)
    missed_drifts = len(fn_rows)
    false_positive_episodes = len(fp_rows)

    precision = (
        matched_drifts / (matched_drifts + false_positive_episodes)
        if (matched_drifts + false_positive_episodes) > 0
        else 0.0
    )

    recall = (
        matched_drifts / (matched_drifts + missed_drifts)
        if (matched_drifts + missed_drifts) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    total_nearby_episodes = sum(nearby_counts)

    drifts_with_no_nearby_episode = sum(
        1 for count in nearby_counts if count == 0
    )

    drifts_with_one_nearby_episode = sum(
        1 for count in nearby_counts if count == 1
    )

    drifts_with_multiple_nearby_episodes = sum(
        1 for count in nearby_counts if count > 1
    )

    mean_nearby_episodes = (
        total_nearby_episodes / true_drifts
        if true_drifts > 0
        else 0.0
    )

    mean_delay = statistics.mean(delays) if delays else ""
    median_delay = statistics.median(delays) if delays else ""
    min_delay = min(delays) if delays else ""
    max_delay = max(delays) if delays else ""

    missed_drift_points = unique_sorted_text(
        row.get("true_drift_point", "")
        for row in fn_rows
    )

    matched_drift_points = unique_sorted_text(
        row.get("true_drift_point", "")
        for row in tp_rows
    )

    false_positive_episode_starts = unique_sorted_text(
        row.get("episode_start", "")
        if str(row.get("episode_start", "")).strip() != ""
        else row.get("matched_alarm_index", "")
        for row in fp_rows
    )

    return {
        "row_count": len(rows),
        "seed_count": len({
            row.get("seed", "")
            for row in rows
            if str(row.get("seed", "")).strip() != ""
        }),
        "true_drifts": true_drifts,
        "matched_drifts_tp": matched_drifts,
        "missed_drifts_fn": missed_drifts,
        "false_positive_episodes_fp": false_positive_episodes,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_nearby_alarm_episodes": total_nearby_episodes,
        "drifts_with_no_nearby_episode": drifts_with_no_nearby_episode,
        "drifts_with_one_nearby_episode": drifts_with_one_nearby_episode,
        "drifts_with_multiple_nearby_episodes":
            drifts_with_multiple_nearby_episodes,
        "mean_nearby_alarm_episodes_per_true_drift":
            round(mean_nearby_episodes, 4),
        "maximum_nearby_alarm_episodes":
            max(nearby_counts, default=0),
        "mean_delay_samples": (
            round(mean_delay, 2)
            if mean_delay != ""
            else ""
        ),
        "median_delay_samples": (
            round(median_delay, 2)
            if median_delay != ""
            else ""
        ),
        "minimum_delay_samples": (
            round(min_delay, 2)
            if min_delay != ""
            else ""
        ),
        "maximum_delay_samples": (
            round(max_delay, 2)
            if max_delay != ""
            else ""
        ),
        "matched_drift_points": matched_drift_points,
        "missed_drift_points": missed_drift_points,
        "false_positive_episode_starts":
            false_positive_episode_starts,
    }


def build_grouped_summary(all_rows, group_columns):
    grouped = defaultdict(list)

    for row in all_rows:
        key = tuple(
            row.get(column, "")
            for column in group_columns
        )
        grouped[key].append(row)

    output_rows = []

    for key, rows in sorted(
        grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        output_row = {
            column: value
            for column, value in zip(group_columns, key)
        }
        output_row.update(summarize_group(rows))
        output_rows.append(output_row)

    return output_rows


def main():
    detail_files = find_detail_files()

    if not detail_files:
        print("No drift-point detail files were found.")
        print("Expected files named:")
        print(f"  {DETAIL_FILE_PATTERN}")
        print("Run the individual experiments first.")
        return

    all_rows = load_all_detail_rows(detail_files)

    by_dataset_rows = build_grouped_summary(
        all_rows,
        ["evaluation_level", "model", "drift_type", "dataset"],
    )

    by_model_drift_rows = build_grouped_summary(
        all_rows,
        ["evaluation_level", "model", "drift_type"],
    )

    detail_columns = [
        "evaluation_level",
        "matching_rule",
        "model",
        "drift_type",
        "dataset",
        "seed",
        "row_type",
        "status",
        "true_drift_point",
        "tolerance_start",
        "tolerance_end",
        "nearby_alarm_episode_count",
        "nearby_raw_alarm_count",
        "nearby_raw_alarm_indices",
        "nearby_episode_starts",
        "nearby_episode_alarm_lists",
        "matched_alarm_index",
        "delay_samples",
        "episode_start",
        "episode_end",
        "episode_size",
        "episode_alarm_indices",
        "source_file",
    ]

    summary_columns = [
        "evaluation_level",
        "model",
        "drift_type",
        "dataset",
        "seed_count",
        "true_drifts",
        "matched_drifts_tp",
        "missed_drifts_fn",
        "false_positive_episodes_fp",
        "precision",
        "recall",
        "f1",
        "total_nearby_alarm_episodes",
        "drifts_with_no_nearby_episode",
        "drifts_with_one_nearby_episode",
        "drifts_with_multiple_nearby_episodes",
        "mean_nearby_alarm_episodes_per_true_drift",
        "maximum_nearby_alarm_episodes",
        "mean_delay_samples",
        "median_delay_samples",
        "minimum_delay_samples",
        "maximum_delay_samples",
        "matched_drift_points",
        "missed_drift_points",
        "false_positive_episode_starts",
        "row_count",
    ]

    write_csv_rows(
        all_rows,
        ALL_DETAILS_FILE,
        detail_columns,
    )

    write_csv_rows(
        by_dataset_rows,
        BY_DATASET_FILE,
        summary_columns,
    )

    write_csv_rows(
        by_model_drift_rows,
        BY_MODEL_DRIFT_FILE,
        summary_columns,
    )

    total_true_drifts = sum(
        1
        for row in all_rows
        if row.get("row_type", "").strip() == "true_drift"
    )

    total_tp = sum(
        1
        for row in all_rows
        if row.get("status", "").strip() == "TP"
    )

    total_fn = sum(
        1
        for row in all_rows
        if row.get("status", "").strip() == "FN"
    )

    total_fp = sum(
        1
        for row in all_rows
        if row.get("status", "").strip() == "FP"
    )

    print()
    print("=" * 70)
    print("DRIFT-POINT AGGREGATION COMPLETE")
    print("=" * 70)
    print("Detail files found:", len(detail_files))
    print("Combined rows:", len(all_rows))
    print("True drift rows:", total_true_drifts)
    print("TP rows:", total_tp)
    print("FN rows:", total_fn)
    print("FP episode rows:", total_fp)
    print()
    print("Created:")
    print(" -", ALL_DETAILS_FILE)
    print(" -", BY_DATASET_FILE)
    print(" -", BY_MODEL_DRIFT_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()
