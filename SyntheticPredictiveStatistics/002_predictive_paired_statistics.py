"""Manuscript-aligned paired statistics for the synthetic predictive results."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "results" / "predictive_seed_runs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "predictive_statistics"
SEEDS = {0, 1, 42, 123, 7}
DRIFT_DATASETS = {
    "abrupt": {f"ADS{i:02d}" for i in range(1, 7)},
    "incremental": {f"IDS{i:02d}" for i in range(1, 7)},
    "gradual": {f"GDS{i:02d}" for i in range(1, 7)},
}
BASELINES = (
    "ADWIN-RESET", "ADWIN-WINDOW", "ADWIN-SSPT", "ADWIN-OHL",
    "KSWIN-RESET", "KSWIN-WINDOW", "KSWIN-SSPT", "KSWIN-OHL",
)


def canonical_model(method_names: list[str]) -> str:
    joined = " ".join(method_names).upper()
    if "OLR-WA" in joined:
        return "OLR-WA"
    if re.search(r"(^|\W)PA($|\W)", joined):
        return "PA"
    if "RLS" in joined:
        return "RLS"
    if "WIDROWHOFF" in joined or "WIDROW-HOFF" in joined or "LMS" in joined:
        return "LMS"
    raise ValueError(f"Cannot infer model from methods: {method_names}")


def normalize_drift(value: str, dataset: str) -> str:
    text = value.lower()
    if "abrupt" in text or dataset.startswith("ADS"):
        return "abrupt"
    if "incremental" in text or dataset.startswith("IDS"):
        return "incremental"
    return "gradual"


def canonical_role(method: str, model: str) -> str:
    text = method.upper().replace("_", "-")
    model_tokens = {
        "OLR-WA": ("OLR-WA",), "PA": ("PA",), "RLS": ("RLS",),
        "LMS": ("WIDROWHOFF", "WIDROW-HOFF", "LMS"),
    }[model]
    if "SCCM" in text:
        return "SCCM"
    for baseline in BASELINES:
        if baseline in text:
            return baseline
    if any(token in text for token in model_tokens):
        return "BASE"
    raise ValueError(f"Unsupported method key: {method}")


def finite_mean(values: Any) -> float:
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else math.nan


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    n = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def signed_rank_effect(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-15]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.asarray(__import__("scipy.stats", fromlist=["rankdata"]).rankdata(np.abs(nonzero)), dtype=float)
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def load_records() -> dict[tuple[str, str, str, int, str], float]:
    records: dict[tuple[str, str, str, int, str], float] = {}
    for path in sorted(INPUT_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(payload["dataset"]).upper()
        drift_type = normalize_drift(str(payload.get("drift_type", "")), dataset)
        for run in payload.get("runs", []):
            seed = int(run["seed"])
            if seed not in SEEDS:
                continue
            methods = [key for key in run if key != "seed"]
            model = canonical_model(methods)
            metric = "R2" if model == "OLR-WA" else "MSE"
            for method in methods:
                role = canonical_role(method, model)
                values = run[method].get(metric, [])
                performance = finite_mean(values)
                key = (model, drift_type, dataset, seed, role)
                if key in records and not math.isclose(records[key], performance, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Conflicting duplicate record for {key}")
                records[key] = performance
    return records


def paired_test(differences: np.ndarray) -> tuple[float, float, int]:
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


def main() -> int:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Run the synthetic experiments first: {INPUT_DIR}")
    records = load_records()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_roles = {"BASE", "SCCM", *BASELINES}
    missing: list[tuple[str, str, str, int, str]] = []
    for model in ("OLR-WA", "PA", "RLS", "LMS"):
        for drift_type, datasets in DRIFT_DATASETS.items():
            for dataset in datasets:
                for seed in SEEDS:
                    for role in expected_roles:
                        key = (model, drift_type, dataset, seed, role)
                        if key not in records or not math.isfinite(records.get(key, math.nan)):
                            missing.append(key)
    if missing:
        report = OUTPUT_DIR / "missing_predictive_records.csv"
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "drift_type", "dataset", "seed", "method"])
            writer.writerows(missing)
        raise RuntimeError(f"Missing {len(missing)} required records. See {report}")

    rows: list[dict[str, Any]] = []
    for model in ("OLR-WA", "PA", "RLS", "LMS"):
        metric = "R2" if model == "OLR-WA" else "MSE"
        for drift_type, datasets in DRIFT_DATASETS.items():
            keys = [(dataset, seed) for dataset in sorted(datasets) for seed in sorted(SEEDS)]
            comparisons = ["BASE", *BASELINES]
            family_rows: list[dict[str, Any]] = []
            for comparison in comparisons:
                sccm = np.asarray([records[(model, drift_type, d, s, "SCCM")] for d, s in keys])
                other = np.asarray([records[(model, drift_type, d, s, comparison)] for d, s in keys])
                differences = sccm - other if metric == "R2" else other - sccm
                statistic, pvalue, zeros = paired_test(differences)
                row = {
                    "model": model,
                    "drift_type": drift_type,
                    "metric": metric,
                    "comparison": comparison,
                    "n_pairs": len(differences),
                    "zero_differences": zeros,
                    "sccm_mean": float(np.mean(sccm)),
                    "comparison_mean": float(np.mean(other)),
                    "mean_oriented_improvement": float(np.mean(differences)),
                    "median_oriented_improvement": float(np.median(differences)),
                    "wilcoxon_statistic": statistic,
                    "p_value": pvalue,
                    "rank_biserial": signed_rank_effect(differences),
                    "positive_pairs": int(np.sum(differences > 0)),
                    "negative_pairs": int(np.sum(differences < 0)),
                    "ties": int(np.sum(np.abs(differences) <= 1e-15)),
                    "p_holm": pvalue if comparison == "BASE" else math.nan,
                    "correction_family": "standalone_reported_separately" if comparison == "BASE" else "eight_baselines",
                }
                rows.append(row)
                if comparison != "BASE":
                    family_rows.append(row)
            adjusted = holm_adjust([row["p_value"] for row in family_rows])
            for row, p_holm in zip(family_rows, adjusted):
                row["p_holm"] = p_holm

    output = OUTPUT_DIR / "synthetic_predictive_paired_statistics.csv"
    fieldnames = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} manuscript-aligned comparisons to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
