from __future__ import annotations

from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality"
for p in (str(PROJECT_ROOT), str(DDQ_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ddq_common import (  # noqa: E402
    collect_dataset_aggregate_rows,
    aggregate_rows_for_paper,
    write_csv,
    to_float,
)


def make_bar_chart(rows, drift_type, metric_names, output_file):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skipping chart", output_file)
        return

    selected = [r for r in rows if r.get("drift_type") == drift_type]
    if not selected:
        return

    models = [r.get("model") for r in selected]
    x = list(range(len(models)))
    width = 0.8 / max(1, len(metric_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metric_names):
        values = [to_float(r.get(metric), 0.0) for r in selected]
        offsets = [v + (i - (len(metric_names) - 1) / 2) * width for v in x]
        ax.bar(offsets, values, width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title(f"{drift_type.title()} alarm quality")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def main():
    out_dir = DDQ_ROOT / "AggregatedQualityResults"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = collect_dataset_aggregate_rows(DDQ_ROOT)
    paper_rows = aggregate_rows_for_paper(dataset_rows)

    by_dataset_csv = out_dir / "alarm_quality_by_dataset.csv"
    paper_csv = out_dir / "alarm_quality_for_paper.csv"
    write_csv(dataset_rows, by_dataset_csv)
    write_csv(paper_rows, paper_csv)

    for drift_type in ["abrupt", "incremental", "gradual"]:
        make_bar_chart(
            paper_rows,
            drift_type,
            ["precision", "recall", "f1"],
            out_dir / f"{drift_type}_precision_recall_f1.png",
        )
        make_bar_chart(
            paper_rows,
            drift_type,
            ["fp", "fn"],
            out_dir / f"{drift_type}_fp_fn.png",
        )
        make_bar_chart(
            paper_rows,
            drift_type,
            ["mean_delay"],
            out_dir / f"{drift_type}_mean_delay.png",
        )

    print("Saved:")
    print(" -", by_dataset_csv)
    print(" -", paper_csv)
    print(" -", out_dir)


if __name__ == "__main__":
    main()
