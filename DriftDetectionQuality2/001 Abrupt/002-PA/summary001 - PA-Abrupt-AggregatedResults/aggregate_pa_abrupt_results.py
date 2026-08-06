from __future__ import annotations

import csv
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SUMMARY_FOLDER = THIS_FILE.parent
MODEL_FOLDER = SUMMARY_FOLDER.parent

DATASETS = ['ADS01', 'ADS02', 'ADS03', 'ADS04', 'ADS05', 'ADS06']
DATASET_PREFIX = 'ADS'
EXPERIMENT_MODEL_NAME = 'PA'
SUMMARY_FILE_MODEL_SLUG = 'PA'
MODEL_LABEL = 'PA-SCCM'
DRIFT_TYPE = 'abrupt'
DRIFT_DISPLAY = 'Abrupt'

OUTPUT_BY_DATASET = SUMMARY_FOLDER / 'PA_Abrupt_by_dataset.csv'
OUTPUT_OVERALL = SUMMARY_FOLDER / 'PA_Abrupt_overall.csv'


def read_csv(file_path):
    with file_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(rows, file_path):
    if not rows:
        print("No rows to save:", file_path)
        return

    fieldnames = list(rows[0].keys())

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def find_dataset_summary(dataset):
    dataset_number = dataset.replace(DATASET_PREFIX, "").zfill(3)

    experiment_folder = (
        MODEL_FOLDER
        / f"{dataset_number}-{EXPERIMENT_MODEL_NAME}-{dataset}"
    )

    if not experiment_folder.is_dir():
        raise FileNotFoundError(
            f"Could not find experiment folder: {experiment_folder}"
        )

    summary_file = (
        experiment_folder
        / "quality_outputs"
        / f"{SUMMARY_FILE_MODEL_SLUG}_{dataset}_summary.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Could not find summary file: {summary_file}"
        )

    return summary_file


def get_aggregate_row(summary_file):
    rows = read_csv(summary_file)

    if not rows:
        raise ValueError(f"Summary file is empty: {summary_file}")

    for row in rows:
        if row.get("row_type") == "aggregate_for_dataset":
            return row

    # Compatibility with older single-seed summary files.
    return rows[-1]


def calculate_overall_result(dataset_rows):
    tp = sum(to_int(row.get("tp")) for row in dataset_rows)
    fp = sum(to_int(row.get("fp")) for row in dataset_rows)
    fn = sum(to_int(row.get("fn")) for row in dataset_rows)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    delay_sum = 0.0
    delay_count = 0

    practical_delay_sum = 0.0
    practical_delay_count = 0

    for row in dataset_rows:
        dataset_tp = to_int(row.get("tp"))

        mean_delay = to_float(row.get("mean_delay_samples"))
        if mean_delay is not None and dataset_tp > 0:
            delay_sum += mean_delay * dataset_tp
            delay_count += dataset_tp

        practical_delay = to_float(row.get("practical_delay_batches"))
        if practical_delay is not None and dataset_tp > 0:
            practical_delay_sum += practical_delay * dataset_tp
            practical_delay_count += dataset_tp

    mean_delay = delay_sum / delay_count if delay_count > 0 else ""

    practical_delay_batches = (
        practical_delay_sum / practical_delay_count
        if practical_delay_count > 0
        else ""
    )

    return {
        "model": MODEL_LABEL,
        "drift_type": DRIFT_TYPE,
        "dataset_count": len(dataset_rows),
        "true_drifts": sum(
            to_int(row.get("true_drifts"))
            for row in dataset_rows
        ),
        "candidate_triggers": sum(
            to_int(row.get("candidate_triggers"))
            for row in dataset_rows
        ),
        "alarm_episodes": sum(
            to_int(row.get("alarm_episodes"))
            for row in dataset_rows
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_delay_samples": (
            round(mean_delay, 2)
            if mean_delay != ""
            else ""
        ),
        "practical_delay_batches": (
            round(practical_delay_batches, 2)
            if practical_delay_batches != ""
            else ""
        ),
        "adaptations": sum(
            to_int(row.get("adaptations"))
            for row in dataset_rows
        ),
        "recalibrations": sum(
            to_int(row.get("recalibrations"))
            for row in dataset_rows
        ),
    }


def main():
    dataset_rows = []

    for dataset in DATASETS:
        summary_file = find_dataset_summary(dataset)
        row = get_aggregate_row(summary_file)

        dataset_rows.append({
            "dataset": dataset,
            "model": row.get("model", MODEL_LABEL),
            "true_drifts": row.get("true_drifts", ""),
            "candidate_triggers": row.get("candidate_triggers", ""),
            "alarm_episodes": row.get("alarm_episodes", ""),
            "tp": row.get("tp", ""),
            "fp": row.get("fp", ""),
            "fn": row.get("fn", ""),
            "precision": row.get("precision", ""),
            "recall": row.get("recall", ""),
            "f1": row.get("f1", ""),
            "mean_delay_samples": row.get("mean_delay", ""),
            "mean_delay_batches": row.get(
                "mean_delay_batches",
                ""
            ),
            "practical_delay_batches": row.get(
                "practical_delay_batches",
                ""
            ),
            "adaptations": row.get("adaptations", ""),
            "recalibrations": row.get("recalibrations", ""),
            "multiplier": row.get("multiplier", ""),
            "sccm_window_size": row.get(
                "sccm_window_size",
                ""
            ),
            "used_kpi_window_size": row.get(
                "used_kpi_window_size",
                ""
            ),
            "increment_user_value": row.get(
                "increment_user_value",
                ""
            ),
            "source_file": str(summary_file),
        })

        print(
            dataset,
            "TP:", row.get("tp"),
            "FP:", row.get("fp"),
            "FN:", row.get("fn"),
            "F1:", row.get("f1"),
        )

    overall_row = calculate_overall_result(dataset_rows)

    write_csv(dataset_rows, OUTPUT_BY_DATASET)
    write_csv([overall_row], OUTPUT_OVERALL)

    print("\n" + "=" * 40)
    print(f"{EXPERIMENT_MODEL_NAME} {DRIFT_DISPLAY.upper()} OVERALL RESULT")
    print("=" * 40)
    print("Datasets:", overall_row["dataset_count"])
    print("TP:", overall_row["tp"])
    print("FP:", overall_row["fp"])
    print("FN:", overall_row["fn"])
    print("Precision:", overall_row["precision"])
    print("Recall:", overall_row["recall"])
    print("F1:", overall_row["f1"])
    print(
        "Mean delay samples:",
        overall_row["mean_delay_samples"]
    )
    print(
        "Practical delay batches:",
        overall_row["practical_delay_batches"]
    )
    print("=" * 40)

    print("\nSaved:")
    print(" -", OUTPUT_BY_DATASET)
    print(" -", OUTPUT_OVERALL)


if __name__ == "__main__":
    main()
