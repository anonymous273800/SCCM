from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
VISUALIZATION_FOLDER = THIS_FILE.parent
OLR_WA_FOLDER = VISUALIZATION_FOLDER.parent

AGGREGATED_FOLDER = (
    OLR_WA_FOLDER
    / "summary001 - OLR-WA-Abrupt-AggregatedResults"
)

INPUT_CSV = (
    AGGREGATED_FOLDER
    / "OLR_WA_Abrupt_by_dataset.csv"
)


def read_csv(file_path):
    with file_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def add_value_labels(axis, bars):
    for bar in bars:
        height = bar.get_height()

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_precision_recall_f1(rows):
    datasets = [row["dataset"] for row in rows]
    precision = [
        to_float(row.get("precision"))
        for row in rows
    ]
    recall = [
        to_float(row.get("recall"))
        for row in rows
    ]
    f1 = [
        to_float(row.get("f1"))
        for row in rows
    ]

    x = list(range(len(datasets)))
    width = 0.25

    figure, axis = plt.subplots(figsize=(11, 6))

    precision_bars = axis.bar(
        [value - width for value in x],
        precision,
        width,
        label="Precision",
    )

    recall_bars = axis.bar(
        x,
        recall,
        width,
        label="Recall",
    )

    f1_bars = axis.bar(
        [value + width for value in x],
        f1,
        width,
        label="F1",
    )

    axis.set_title(
        "OLR-WA-SCCM Abrupt Drift Detection Quality"
    )
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels(datasets)
    axis.set_ylim(0, 1.12)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    add_value_labels(axis, precision_bars)
    add_value_labels(axis, recall_bars)
    add_value_labels(axis, f1_bars)

    figure.tight_layout()

    output_file = (
        VISUALIZATION_FOLDER
        / "OLR_WA_Abrupt_Precision_Recall_F1.png"
    )

    figure.savefig(output_file, dpi=300)
    plt.close(figure)

    print("Saved:", output_file)


def plot_tp_fp_fn(rows):
    datasets = [row["dataset"] for row in rows]
    tp = [to_float(row.get("tp")) for row in rows]
    fp = [to_float(row.get("fp")) for row in rows]
    fn = [to_float(row.get("fn")) for row in rows]

    x = list(range(len(datasets)))
    width = 0.25

    figure, axis = plt.subplots(figsize=(11, 6))

    tp_bars = axis.bar(
        [value - width for value in x],
        tp,
        width,
        label="TP",
    )

    fp_bars = axis.bar(
        x,
        fp,
        width,
        label="FP",
    )

    fn_bars = axis.bar(
        [value + width for value in x],
        fn,
        width,
        label="FN",
    )

    axis.set_title(
        "OLR-WA-SCCM Abrupt Drift Alarm Counts"
    )
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Count")
    axis.set_xticks(x)
    axis.set_xticklabels(datasets)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    add_value_labels(axis, tp_bars)
    add_value_labels(axis, fp_bars)
    add_value_labels(axis, fn_bars)

    figure.tight_layout()

    output_file = (
        VISUALIZATION_FOLDER
        / "OLR_WA_Abrupt_TP_FP_FN.png"
    )

    figure.savefig(output_file, dpi=300)
    plt.close(figure)

    print("Saved:", output_file)


def plot_practical_delay(rows):
    datasets = [row["dataset"] for row in rows]
    delays = [
        to_float(row.get("practical_delay_batches"))
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    bars = axis.bar(datasets, delays)

    axis.set_title(
        "OLR-WA-SCCM Practical Drift Detection Delay"
    )
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Practical delay in batches")
    axis.grid(axis="y", alpha=0.3)

    add_value_labels(axis, bars)

    figure.tight_layout()

    output_file = (
        VISUALIZATION_FOLDER
        / "OLR_WA_Abrupt_Practical_Delay.png"
    )

    figure.savefig(output_file, dpi=300)
    plt.close(figure)

    print("Saved:", output_file)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "Run aggregate_olr_wa_abrupt_results.py first. "
            f"Missing file: {INPUT_CSV}"
        )

    rows = read_csv(INPUT_CSV)

    if not rows:
        raise ValueError(
            f"The aggregated CSV is empty: {INPUT_CSV}"
        )

    plot_precision_recall_f1(rows)
    plot_tp_fp_fn(rows)
    plot_practical_delay(rows)


if __name__ == "__main__":
    main()