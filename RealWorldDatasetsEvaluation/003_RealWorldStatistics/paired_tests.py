"""Manuscript-aligned real-world paired predictive comparisons."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealWorldDatasetsEvaluation.config import DATASETS, MODELS, METHODS
from RealWorldDatasetsEvaluation.common.project import results_dir

BASELINES = [method for method in METHODS if method not in {"BASE", "SCCM"}]


def holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    n = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    values = differences[np.abs(differences) > 1e-15]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values))
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / (positive + negative)


def signed_rank(differences: np.ndarray) -> tuple[float, float, int]:
    nonzero = differences[np.abs(differences) > 1e-15]
    if len(nonzero) == 0:
        return 0.0, 1.0, len(differences)
    result = wilcoxon(
        nonzero,
        alternative="two-sided",
        zero_method="wilcox",
        correction=True,
        method="approx",
    )
    return float(result.statistic), float(result.pvalue), int(len(differences) - len(nonzero))


def model_metric(model: str, frame: pd.DataFrame) -> tuple[str, str]:
    if model == "OLR-WA":
        metric = "avg_r2" if "avg_r2" in frame.columns else "final_r2"
        return metric, "absolute_r2_increase"
    return "avg_mse", "relative_mse_reduction"


def oriented_improvement(model: str, sccm: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    if model == "OLR-WA":
        return sccm - comparison
    denominator = np.where(np.abs(comparison) > 1e-15, np.abs(comparison), np.nan)
    return (comparison - sccm) / denominator


def dataset_level_differences(
    frame: pd.DataFrame, model: str, comparison: str, metric: str
) -> pd.DataFrame:
    sccm = frame[frame.method.eq("SCCM")][["dataset", "seed", metric]].rename(columns={metric: "sccm"})
    other = frame[frame.method.eq(comparison)][["dataset", "seed", metric]].rename(columns={metric: "comparison"})
    paired = sccm.merge(other, on=["dataset", "seed"], how="inner").dropna()
    paired["improvement"] = oriented_improvement(
        model,
        paired.sccm.to_numpy(float),
        paired.comparison.to_numpy(float),
    )
    return paired.groupby("dataset", as_index=False).agg(
        improvement=("improvement", "mean"),
        sccm=("sccm", "mean"),
        comparison=("comparison", "mean"),
        seed_pairs=("seed", "count"),
    )


def strongest_baseline(frame: pd.DataFrame, model: str, metric: str) -> str:
    means = (
        frame[frame.method.isin(BASELINES)]
        .groupby(["dataset", "method"], as_index=False)[metric]
        .mean()
    )
    ascending = model != "OLR-WA"
    means["rank"] = means.groupby("dataset")[metric].rank(method="average", ascending=ascending)
    ranks = means.groupby("method")["rank"].mean().sort_values()
    if ranks.empty:
        raise ValueError(f"No detector-adaptation baselines found for {model}")
    return str(ranks.index[0])


def main() -> int:
    input_path = results_dir("aggregated") / "realworld_seed_level_complete.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Run aggregation first: {input_path}")
    frame = pd.read_csv(input_path)
    output_dir = results_dir("statistics")

    confirmatory: list[dict[str, object]] = []
    descriptive: list[dict[str, object]] = []
    for model in MODELS:
        model_frame = frame[frame.model.eq(model)].copy()
        metric, improvement_definition = model_metric(model, model_frame)
        if metric not in model_frame.columns:
            raise ValueError(f"Required metric {metric!r} is missing for {model}")

        base_pairs = dataset_level_differences(model_frame, model, "BASE", metric)
        if set(base_pairs.dataset) != set(DATASETS) or not (base_pairs.seed_pairs == 5).all():
            raise ValueError(
                f"{model}: standalone comparison requires 8 datasets with 5 seed pairs each; "
                f"found {len(base_pairs)} datasets."
            )
        diff = base_pairs.improvement.to_numpy(float)
        statistic, pvalue, zero_count = signed_rank(diff)
        confirmatory.append({
            "model": model,
            "metric": metric,
            "improvement_definition": improvement_definition,
            "comparison_method": "BASE",
            "n_dataset_pairs": len(diff),
            "seeds_averaged_per_dataset": 5,
            "median_improvement": float(np.nanmedian(diff)),
            "mean_improvement": float(np.nanmean(diff)),
            "wilcoxon_statistic": statistic,
            "p_value": pvalue,
            "zero_differences": zero_count,
            "rank_biserial": rank_biserial(diff),
            "positive_datasets": int(np.sum(diff > 0)),
            "negative_datasets": int(np.sum(diff < 0)),
            "ties": int(np.sum(np.abs(diff) <= 1e-15)),
        })

        selected = strongest_baseline(model_frame, model, metric)
        selected_pairs = dataset_level_differences(model_frame, model, selected, metric)
        if set(selected_pairs.dataset) != set(DATASETS) or not (selected_pairs.seed_pairs == 5).all():
            raise ValueError(f"{model}: incomplete descriptive comparison with {selected}")
        selected_diff = selected_pairs.improvement.to_numpy(float)
        statistic, pvalue, zero_count = signed_rank(selected_diff)
        descriptive.append({
            "model": model,
            "metric": metric,
            "improvement_definition": improvement_definition,
            "strongest_observed_baseline": selected,
            "selection_rule": "best mean predictive rank across eight datasets using five-seed means",
            "inference_status": "descriptive_post_hoc",
            "n_dataset_pairs": len(selected_diff),
            "seeds_averaged_per_dataset": 5,
            "median_improvement": float(np.nanmedian(selected_diff)),
            "mean_improvement": float(np.nanmean(selected_diff)),
            "wilcoxon_statistic_descriptive": statistic,
            "p_value_descriptive_unadjusted": pvalue,
            "zero_differences": zero_count,
            "rank_biserial_descriptive": rank_biserial(selected_diff),
            "positive_datasets": int(np.sum(selected_diff > 0)),
            "negative_datasets": int(np.sum(selected_diff < 0)),
            "ties": int(np.sum(np.abs(selected_diff) <= 1e-15)),
        })

    confirmatory_df = pd.DataFrame(confirmatory)
    confirmatory_df["p_holm_across_four_models"] = holm(confirmatory_df.p_value.to_numpy(float))
    confirmatory_df["significant_holm_0_05"] = confirmatory_df.p_holm_across_four_models < 0.05
    descriptive_df = pd.DataFrame(descriptive)

    confirmatory_df.to_csv(output_dir / "realworld_confirmatory_sccm_vs_standalone.csv", index=False)
    descriptive_df.to_csv(output_dir / "realworld_descriptive_sccm_vs_strongest_baseline.csv", index=False)

    # Compatibility output used by the existing paper-results generator.
    compatibility = confirmatory_df.rename(columns={
        "comparison_method": "comparison_method",
        "n_dataset_pairs": "n_pairs",
        "p_holm_across_four_models": "p_holm",
    }).copy()
    compatibility["scope"] = "ALL_DATASETS_DATASET_LEVEL"
    compatibility["sccm_method"] = "SCCM"
    compatibility["significant_holm_0_05"] = compatibility.p_holm < 0.05
    compatibility.to_csv(output_dir / "paired_sccm_vs_methods.csv", index=False)

    print(f"Wrote {len(confirmatory_df)} confirmatory and {len(descriptive_df)} descriptive comparisons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
