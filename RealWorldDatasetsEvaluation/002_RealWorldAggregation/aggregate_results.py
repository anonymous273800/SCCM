from __future__ import annotations
import itertools
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealWorldDatasetsEvaluation.config import DATASETS, MODELS, METHODS, SEEDS, EXPECTED_RUNS
from RealWorldDatasetsEvaluation.common.project import results_dir

METRICS = [
    "final_r2","avg_r2","avg_mse","runtime_seconds","runtime_per_1000_samples",
    "peak_rss_delta_mb","detector_detections","adaptation_activations",
    "sccm_adaptations","sccm_recalibrations","total_interventions","interventions_per_1000_samples"
]


def flatten(columns):
    return ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col) for col in columns]


def main() -> int:
    raw_path = results_dir("raw") / "realworld_seed_level.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run the full matrix first: {raw_path}")
    raw = pd.read_csv(raw_path)
    # Keep the last record when a failed run was later rerun successfully.
    raw = raw.drop_duplicates(["dataset","model","method","seed"], keep="last")
    good = raw[raw.status.astype(str).str.lower().eq("complete")].copy()
    out = results_dir("aggregated")
    good.to_csv(out / "realworld_seed_level_complete.csv", index=False)
    raw[~raw.status.astype(str).str.lower().eq("complete")].to_csv(out / "failed_runs.csv", index=False)

    expected = pd.DataFrame(itertools.product(DATASETS, MODELS, METHODS, SEEDS), columns=["dataset","model","method","seed"])
    completeness = expected.merge(raw[["dataset","model","method","seed","status"]], how="left")
    completeness["status"] = completeness["status"].fillna("missing")
    completeness.to_csv(out / "realworld_run_completeness.csv", index=False)

    usable = [m for m in METRICS if m in good.columns]
    by_dataset = good.groupby(["model","dataset","method"])[usable].agg(["count","mean","std","median"]).reset_index()
    by_dataset.columns = flatten(by_dataset.columns)
    by_dataset.to_csv(out / "realworld_by_dataset_mean_std.csv", index=False)

    overall = good.groupby(["model","method"])[usable].agg(["count","mean","std","median"]).reset_index()
    overall.columns = flatten(overall.columns)
    overall.to_csv(out / "realworld_all_datasets_mean_std.csv", index=False)

    activity = good.groupby(["model","dataset","method"])[[
        "detector_detections","adaptation_activations","sccm_adaptations",
        "sccm_recalibrations","total_interventions","interventions_per_1000_samples"
    ]].agg(["mean","std"]).reset_index()
    activity.columns = flatten(activity.columns)
    activity.to_csv(out / "realworld_activity_mean_std.csv", index=False)

    summary = [
        f"Expected runs: {EXPECTED_RUNS}",
        f"Complete runs: {len(good)}",
        f"Failed or incomplete records: {len(raw)-len(good)}",
        f"Missing expected combinations: {(completeness.status == 'missing').sum()}",
    ]
    (out / "AGGREGATION_SUMMARY.txt").write_text("\n".join(summary)+"\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
