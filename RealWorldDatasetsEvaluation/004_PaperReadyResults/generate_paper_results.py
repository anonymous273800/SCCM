from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from RealWorldDatasetsEvaluation.common.project import results_dir


def main() -> int:
    agg_path = results_dir("aggregated") / "realworld_by_dataset_mean_std.csv"
    stats_path = results_dir("statistics") / "paired_sccm_vs_methods.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Run aggregation first: {agg_path}")
    agg = pd.read_csv(agg_path)
    out = results_dir("paper")

    columns = [c for c in [
        "model","dataset","method","final_r2_mean","final_r2_std","avg_mse_mean","avg_mse_std",
        "runtime_per_1000_samples_mean","runtime_per_1000_samples_std","peak_rss_delta_mb_mean",
        "interventions_per_1000_samples_mean","interventions_per_1000_samples_std"
    ] if c in agg.columns]
    agg[columns].to_csv(out / "paper_full_realworld_matrix.csv", index=False)

    selected = []
    for (model, dataset), group in agg.groupby(["model","dataset"]):
        wanted = group[group.method.isin(["BASE","SCCM"])].copy()
        for prefix in ["ADWIN-", "KSWIN-"]:
            family = group[group.method.str.startswith(prefix, na=False)]
            if not family.empty:
                metric = "avg_mse_mean" if "avg_mse_mean" in family else "final_r2_mean"
                best = family.sort_values(metric, ascending=(metric=="avg_mse_mean")).head(1)
                wanted = pd.concat([wanted, best], ignore_index=True)
        selected.append(wanted)
    compact = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    compact[[c for c in columns if c in compact.columns]].to_csv(out / "paper_sccm_base_best_adwin_best_kswin.csv", index=False)

    efficiency_cols = [c for c in [
        "model","dataset","method","runtime_per_1000_samples_mean","runtime_per_1000_samples_std",
        "peak_rss_delta_mb_mean","peak_rss_delta_mb_std","interventions_per_1000_samples_mean",
        "interventions_per_1000_samples_std"
    ] if c in agg.columns]
    agg[efficiency_cols].to_csv(out / "paper_efficiency_and_interventions.csv", index=False)

    if stats_path.exists():
        stats = pd.read_csv(stats_path)
        stats[stats.scope.astype(str).str.startswith("ALL_DATASETS")].to_csv(out / "paper_paired_statistics_all_datasets.csv", index=False)
        stats[stats.significant_holm_0_05.astype(bool)].to_csv(out / "paper_significant_paired_results.csv", index=False)
        descriptive_path = results_dir("statistics") / "realworld_descriptive_sccm_vs_strongest_baseline.csv"
        if descriptive_path.exists():
            pd.read_csv(descriptive_path).to_csv(out / "paper_descriptive_strongest_baseline.csv", index=False)

    note = (
        "Real-world datasets have no ground-truth drift locations. These files report predictive performance, "
        "runtime, memory, and intervention activity only. TP, FP, FN, alarm precision/recall/F1, and detection "
        "delay must not be inferred from these results.\n"
    )
    (out / "README_PAPER_RESULTS.txt").write_text(note, encoding="utf-8")
    print(f"Paper-ready results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
