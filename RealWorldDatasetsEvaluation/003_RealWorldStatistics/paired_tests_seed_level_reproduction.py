"""Reproduce the legacy real-world table using 40 dataset-seed pairs.

This is intentionally separate from ``paired_tests.py``. The manuscript's
written primary protocol averages the five seed differences within each of the
eight datasets before Wilcoxon testing (n=8). Some reported p-values are too
small to arise from n=8 and are consistent with treating all 8 x 5=40
model-dataset-seed rows as paired observations. This script preserves that
legacy calculation for table reproduction, while clearly labeling it as a
repeated-run descriptive analysis rather than independent dataset-level
inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealWorldDatasetsEvaluation.config import DATASETS, MODELS, METHODS, SEEDS
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


def metric_for(model: str, frame: pd.DataFrame) -> str:
    if model == "OLR-WA":
        return "avg_r2" if "avg_r2" in frame.columns else "final_r2"
    return "avg_mse"


def oriented_improvement(model: str, sccm: np.ndarray, other: np.ndarray) -> np.ndarray:
    if model == "OLR-WA":
        return sccm - other
    denominator = np.where(np.abs(other) > 1e-15, np.abs(other), np.nan)
    return (other - sccm) / denominator


def paired_seed_differences(frame: pd.DataFrame, model: str, method: str, metric: str) -> np.ndarray:
    sccm = frame[frame.method.eq("SCCM")][["dataset", "seed", metric]].rename(columns={metric: "sccm"})
    other = frame[frame.method.eq(method)][["dataset", "seed", metric]].rename(columns={metric: "other"})
    paired = sccm.merge(other, on=["dataset", "seed"], how="inner").dropna()
    if set(paired.dataset) != set(DATASETS) or set(paired.seed) != set(SEEDS) or len(paired) != 40:
        raise ValueError(
            f"{model} vs {method}: expected 40 paired dataset-seed observations; found {len(paired)}."
        )
    return oriented_improvement(
        model, paired.sccm.to_numpy(float), paired.other.to_numpy(float)
    )


def strongest_baseline(frame: pd.DataFrame, model: str, metric: str) -> str:
    five_seed_means = (
        frame[frame.method.isin(BASELINES)]
        .groupby(["dataset", "method"], as_index=False)[metric]
        .mean()
    )
    ascending = model != "OLR-WA"
    five_seed_means["rank"] = five_seed_means.groupby("dataset")[metric].rank(
        method="average", ascending=ascending
    )
    ranks = five_seed_means.groupby("method")["rank"].mean().sort_values()
    if ranks.empty:
        raise ValueError(f"No detector-adaptation baseline rows found for {model}")
    return str(ranks.index[0])


def summarize(model: str, comparison: str, differences: np.ndarray, family: str) -> dict[str, object]:
    statistic, pvalue, zeros = signed_rank(differences)
    return {
        "model": model,
        "comparison_method": comparison,
        "analysis_unit": "model_dataset_seed_repeated_run",
        "n_pairs": len(differences),
        "dataset_count": len(DATASETS),
        "seed_count": len(SEEDS),
        "inference_status": "legacy_table_reproduction_not_independent_dataset_level",
        "correction_family": family,
        "median_improvement": float(np.nanmedian(differences)),
        "mean_improvement": float(np.nanmean(differences)),
        "wilcoxon_statistic": statistic,
        "p_value": pvalue,
        "zero_differences": zeros,
        "rank_biserial": rank_biserial(differences),
        "positive_pairs": int(np.sum(differences > 0)),
        "negative_pairs": int(np.sum(differences < 0)),
        "ties": int(np.sum(np.abs(differences) <= 1e-15)),
    }


def main() -> int:
    input_path = results_dir("aggregated") / "realworld_seed_level_complete.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Run aggregation first: {input_path}")
    frame = pd.read_csv(input_path)

    standalone_rows: list[dict[str, object]] = []
    strongest_rows: list[dict[str, object]] = []
    for model in MODELS:
        model_frame = frame[frame.model.eq(model)].copy()
        metric = metric_for(model, model_frame)
        standalone_rows.append(
            summarize(
                model,
                "BASE",
                paired_seed_differences(model_frame, model, "BASE", metric),
                "four_sccm_vs_standalone_model_comparisons",
            )
        )
        strongest = strongest_baseline(model_frame, model, metric)
        strongest_rows.append(
            summarize(
                model,
                strongest,
                paired_seed_differences(model_frame, model, strongest, metric),
                "four_post_hoc_strongest_baseline_model_comparisons",
            )
        )

    standalone = pd.DataFrame(standalone_rows)
    strongest = pd.DataFrame(strongest_rows)
    standalone["p_holm"] = holm(standalone.p_value.to_numpy(float))
    strongest["p_holm"] = holm(strongest.p_value.to_numpy(float))
    output = pd.concat([standalone, strongest], ignore_index=True)
    output["significant_holm_0_05"] = output.p_holm < 0.05

    output_path = results_dir("statistics") / "realworld_seed_level_legacy_table_reproduction.csv"
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output)} explicitly labeled legacy table-reproduction comparisons to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
