from __future__ import annotations

import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality2"
for path in (str(PROJECT_ROOT), str(DDQ_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ddq_common import write_csv  # noqa: E402
from ddq2_statistics import (  # noqa: E402
    as_float,
    descriptive,
    holm_adjust,
    paired_wilcoxon,
)

TOLERANCE_RATIO = 0.05
COOLDOWN_FACTOR = 2.0
MIN_EPISODE_SIZE = 2

BASE_MODELS = ["OLR-WA", "PA", "RLS", "WidrowHoff"]
BASELINES = [
    "ADWIN-RESET",
    "ADWIN-WINDOW",
    "ADWIN-SSPT",
    "ADWIN-OHL",
    "KSWIN-RESET",
    "KSWIN-WINDOW",
    "KSWIN-SSPT",
    "KSWIN-OHL",
]
DRIFT_TYPES = ["abrupt", "incremental", "gradual"]
METRICS = {
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
    "fp": "lower",
    "fn": "lower",
    "mean_delay_samples": "lower",
    "mean_delay_increments": "lower",
}
BOUNDED_METRICS = {"precision", "recall", "f1"}


def load_sensitivity_module():
    path = DDQ_ROOT / "006parameter_sensitivity.py"
    spec = importlib.util.spec_from_file_location("ddq2_parameter_sensitivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_basis(rows: list[dict[str, Any]], basis: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item["evaluation_basis"] = basis
        item["fixed_protocol"] = True
        item["protocol_tolerance_ratio"] = TOLERANCE_RATIO
        item["protocol_cooldown_factor"] = (
            COOLDOWN_FACTOR if "episode" in basis else ""
        )
        item["protocol_min_episode_size"] = (
            MIN_EPISODE_SIZE if "episode" in basis else ""
        )
        item["status"] = "ok"
        output.append(item)
    return output


def clipped_descriptive(values: list[Any], metric: str) -> dict[str, Any]:
    stats = descriptive([as_float(value) for value in values])
    if metric in BOUNDED_METRICS:
        if stats.get("ci95_low") != "":
            stats["ci95_low"] = max(0.0, float(stats["ci95_low"]))
        if stats.get("ci95_high") != "":
            stats["ci95_high"] = min(1.0, float(stats["ci95_high"]))
    return stats


def summarize_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["evaluation_basis"],
            row["method_family"],
            row["base_model"],
            row["method"],
            row["dataset"],
            row["drift_type"],
        )
        grouped[key].append(row)

    output = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        item = {
            "evaluation_basis": key[0],
            "method_family": key[1],
            "base_model": key[2],
            "method": key[3],
            "dataset": key[4],
            "drift_type": key[5],
            "seed_count": len({int(float(row["seed"])) for row in members}),
            "variability_unit": "seed",
            "tolerance_ratio": TOLERANCE_RATIO,
            "cooldown_factor": COOLDOWN_FACTOR if "episode" in key[0] else "",
            "min_episode_size": MIN_EPISODE_SIZE if "episode" in key[0] else "",
        }
        for metric in METRICS:
            stats = clipped_descriptive([row.get(metric) for row in members], metric)
            for name, value in stats.items():
                item[f"{metric}_{name}"] = value
        output.append(item)
    return output


def micro_aggregate(members: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(float(row.get("tp", 0) or 0)) for row in members)
    fp = sum(int(float(row.get("fp", 0) or 0)) for row in members)
    fn = sum(int(float(row.get("fn", 0) or 0)) for row in members)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delay_sum = sum(float(row.get("delay_sum", 0) or 0) for row in members)
    delay_count = sum(int(float(row.get("delay_count", 0) or 0)) for row in members)
    increment_sum = sum(
        float(row.get("mean_delay_increments", 0) or 0)
        * int(float(row.get("delay_count", 0) or 0))
        for row in members
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "delay_sum": delay_sum,
        "delay_count": delay_count,
        "mean_delay_samples": delay_sum / delay_count if delay_count else "",
        "mean_delay_increments": increment_sum / delay_count if delay_count else "",
    }


def pooled_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["evaluation_basis"],
            row["method_family"],
            row["base_model"],
            row["method"],
            row["drift_type"],
            int(float(row["seed"])),
        )
        grouped[key].append(row)

    output = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        item = {
            "evaluation_basis": key[0],
            "method_family": key[1],
            "base_model": key[2],
            "method": key[3],
            "drift_type": key[4],
            "seed": key[5],
            "dataset_count": len(members),
            "tolerance_ratio": TOLERANCE_RATIO,
            "cooldown_factor": COOLDOWN_FACTOR if "episode" in key[0] else "",
            "min_episode_size": MIN_EPISODE_SIZE if "episode" in key[0] else "",
        }
        item.update(micro_aggregate(members))
        output.append(item)
    return output


def summarize_for_paper(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = (
            row["evaluation_basis"],
            row["method_family"],
            row["base_model"],
            row["method"],
            row["drift_type"],
        )
        grouped[key].append(row)

    output = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        item = {
            "evaluation_basis": key[0],
            "method_family": key[1],
            "base_model": key[2],
            "method": key[3],
            "drift_type": key[4],
            "seed_count": len(members),
            "dataset_count_per_seed": members[0].get("dataset_count", "") if members else "",
            "variability_unit": "seed_after_pooling_six_datasets",
            "tolerance_ratio": TOLERANCE_RATIO,
            "cooldown_factor": COOLDOWN_FACTOR if "episode" in key[0] else "",
            "min_episode_size": MIN_EPISODE_SIZE if "episode" in key[0] else "",
        }
        for metric in METRICS:
            stats = clipped_descriptive([row.get(metric) for row in members], metric)
            for name, value in stats.items():
                item[f"{metric}_{name}"] = value
        output.append(item)
    return output


def comparison_result(
    *,
    a_values: list[float],
    b_values: list[float],
    metric: str,
    direction: str,
    basis: str,
    scope: str,
    base_model: str,
    baseline: str,
    pairing_unit: str,
    paired_keys: list[str],
) -> dict[str, Any]:
    result = paired_wilcoxon(a_values, b_values)
    a_stats = clipped_descriptive(a_values, metric)
    b_stats = clipped_descriptive(b_values, metric)
    result.update({
        "evaluation_basis": basis,
        "scope": scope,
        "base_model": base_model,
        "sccm_method": f"{base_model}-SCCM" if base_model != "ALL_MODELS" else "ALL-SCCM",
        "baseline_method": f"{base_model}-{baseline}" if base_model != "ALL_MODELS" else baseline,
        "baseline": baseline,
        "metric": metric,
        "preferred_direction": direction,
        "pairing_unit": pairing_unit,
        "paired_keys": ";".join(paired_keys),
        "sccm_mean": a_stats["mean"],
        "sccm_std": a_stats["std"],
        "baseline_mean": b_stats["mean"],
        "baseline_std": b_stats["std"],
        "difference_definition": "SCCM minus baseline",
        "effect_interpretation": (
            "positive rank-biserial favors SCCM"
            if direction == "higher"
            else "negative rank-biserial favors SCCM"
        ),
    })
    if direction == "higher":
        result["sccm_better_pairs"] = result["wins_a"]
        result["baseline_better_pairs"] = result["wins_b"]
    else:
        result["sccm_better_pairs"] = result["wins_b"]
        result["baseline_better_pairs"] = result["wins_a"]
    result["tie_pairs"] = result["ties"]
    return result


def paired_dataset_seed_tests(
    sccm_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    basis: str,
) -> list[dict[str, Any]]:
    sccm_lookup = {
        (row["base_model"], row["dataset"], int(float(row["seed"]))): row
        for row in sccm_rows
    }
    baseline_lookup = {
        (row["base_model"], row["method"], row["dataset"], int(float(row["seed"]))): row
        for row in baseline_rows
    }
    output = []
    scopes = DRIFT_TYPES + ["all"]

    for base_model in BASE_MODELS:
        for baseline in BASELINES:
            for scope in scopes:
                keys = sorted({
                    (row["dataset"], int(float(row["seed"])))
                    for row in sccm_rows
                    if row["base_model"] == base_model
                    and (scope == "all" or row["drift_type"] == scope)
                })
                for metric, direction in METRICS.items():
                    a_values, b_values, used = [], [], []
                    for dataset, seed in keys:
                        a = sccm_lookup.get((base_model, dataset, seed))
                        b = baseline_lookup.get((base_model, baseline, dataset, seed))
                        if a is None or b is None:
                            continue
                        av = as_float(a.get(metric))
                        bv = as_float(b.get(metric))
                        if av is None or bv is None:
                            continue
                        a_values.append(av)
                        b_values.append(bv)
                        used.append(f"{dataset}:{seed}")
                    output.append(comparison_result(
                        a_values=a_values,
                        b_values=b_values,
                        metric=metric,
                        direction=direction,
                        basis=basis,
                        scope=scope,
                        base_model=base_model,
                        baseline=baseline,
                        pairing_unit="dataset_seed",
                        paired_keys=used,
                    ))

    for baseline in BASELINES:
        for scope in scopes:
            keys = sorted({
                (row["base_model"], row["dataset"], int(float(row["seed"])))
                for row in sccm_rows
                if scope == "all" or row["drift_type"] == scope
            })
            for metric, direction in METRICS.items():
                a_values, b_values, used = [], [], []
                for base_model, dataset, seed in keys:
                    a = sccm_lookup.get((base_model, dataset, seed))
                    b = baseline_lookup.get((base_model, baseline, dataset, seed))
                    if a is None or b is None:
                        continue
                    av = as_float(a.get(metric))
                    bv = as_float(b.get(metric))
                    if av is None or bv is None:
                        continue
                    a_values.append(av)
                    b_values.append(bv)
                    used.append(f"{base_model}:{dataset}:{seed}")
                output.append(comparison_result(
                    a_values=a_values,
                    b_values=b_values,
                    metric=metric,
                    direction=direction,
                    basis=basis,
                    scope=scope,
                    base_model="ALL_MODELS",
                    baseline=baseline,
                    pairing_unit="model_dataset_seed",
                    paired_keys=used,
                ))
    return output


def paired_seed_pooled_tests(
    sccm_seed_rows: list[dict[str, Any]],
    baseline_seed_rows: list[dict[str, Any]],
    basis: str,
) -> list[dict[str, Any]]:
    sccm_lookup = {
        (row["base_model"], row["drift_type"], int(float(row["seed"]))): row
        for row in sccm_seed_rows
    }
    baseline_lookup = {
        (row["base_model"], row["method"], row["drift_type"], int(float(row["seed"]))): row
        for row in baseline_seed_rows
    }
    output = []
    for base_model in BASE_MODELS:
        for baseline in BASELINES:
            for scope in DRIFT_TYPES:
                seeds = sorted({
                    int(float(row["seed"]))
                    for row in sccm_seed_rows
                    if row["base_model"] == base_model and row["drift_type"] == scope
                })
                for metric, direction in METRICS.items():
                    a_values, b_values, used = [], [], []
                    for seed in seeds:
                        a = sccm_lookup.get((base_model, scope, seed))
                        b = baseline_lookup.get((base_model, baseline, scope, seed))
                        if a is None or b is None:
                            continue
                        av = as_float(a.get(metric))
                        bv = as_float(b.get(metric))
                        if av is None or bv is None:
                            continue
                        a_values.append(av)
                        b_values.append(bv)
                        used.append(str(seed))
                    output.append(comparison_result(
                        a_values=a_values,
                        b_values=b_values,
                        metric=metric,
                        direction=direction,
                        basis=basis,
                        scope=scope,
                        base_model=base_model,
                        baseline=baseline,
                        pairing_unit="seed_after_pooling_six_datasets",
                        paired_keys=used,
                    ))
    return output


def protocol_manifest() -> list[dict[str, Any]]:
    return [{
        "protocol_name": "fixed_alarm_matching_protocol",
        "tolerance_ratio": TOLERANCE_RATIO,
        "cooldown_factor": COOLDOWN_FACTOR,
        "min_episode_size": MIN_EPISODE_SIZE,
        "episode_alarm_time": "first episode trigger inside the post-drift matching window",
        "matching_window": "[true drift, true drift + tolerance]",
        "matching_type": "chronological one-to-one; an episode is eligible when any member falls inside the post-drift window",
        "raw_comparison": "SCCM raw candidate triggers versus baseline raw detector alarms",
        "episode_comparison": "SCCM episodes versus baseline detector episodes",
        "models_rerun": False,
    }]


def require_count(name: str, rows: list[dict[str, Any]], expected: int) -> None:
    actual = len(rows)
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected} rows, found {actual}")


def main() -> None:
    sensitivity = load_sensitivity_module()

    sccm_raw, sccm_episode = sensitivity.load_sccm_sensitivity(
        [TOLERANCE_RATIO], [COOLDOWN_FACTOR], [MIN_EPISODE_SIZE]
    )
    baseline_raw, baseline_episode = sensitivity.load_baseline_sensitivity(
        [TOLERANCE_RATIO], [COOLDOWN_FACTOR], [MIN_EPISODE_SIZE]
    )

    require_count("SCCM raw", sccm_raw, 360)
    require_count("SCCM episode", sccm_episode, 360)
    require_count("Baseline raw", baseline_raw, 2880)
    require_count("Baseline episode", baseline_episode, 2880)

    sccm_raw = add_basis(sccm_raw, "raw_to_raw_fixed_tolerance")
    baseline_raw = add_basis(baseline_raw, "raw_to_raw_fixed_tolerance")
    sccm_episode = add_basis(sccm_episode, "episode_to_episode_fixed_protocol")
    baseline_episode = add_basis(baseline_episode, "episode_to_episode_fixed_protocol")

    all_seed_rows = sccm_raw + baseline_raw + sccm_episode + baseline_episode
    pooled_rows = pooled_seed_rows(all_seed_rows)

    tests = []
    for basis, sccm_rows, baseline_rows in (
        ("raw_to_raw_fixed_tolerance", sccm_raw, baseline_raw),
        ("episode_to_episode_fixed_protocol", sccm_episode, baseline_episode),
    ):
        tests.extend(paired_dataset_seed_tests(sccm_rows, baseline_rows, basis))
        sccm_pooled = [row for row in pooled_rows if row["evaluation_basis"] == basis and row["method_family"] == "SCCM"]
        baseline_pooled = [row for row in pooled_rows if row["evaluation_basis"] == basis and row["method_family"] != "SCCM"]
        tests.extend(paired_seed_pooled_tests(sccm_pooled, baseline_pooled, basis))

    tests = holm_adjust(
        tests,
        group_fields=("evaluation_basis", "pairing_unit", "base_model", "scope", "metric"),
    )

    out_dir = DDQ_ROOT / "FixedProtocolResults"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(protocol_manifest(), out_dir / "fixed_protocol_manifest.csv")
    write_csv(sccm_raw, out_dir / "sccm_raw_fixed_seed_level.csv")
    write_csv(baseline_raw, out_dir / "baseline_raw_fixed_seed_level.csv")
    write_csv(sccm_episode, out_dir / "sccm_episode_fixed_seed_level.csv")
    write_csv(baseline_episode, out_dir / "baseline_episode_fixed_seed_level.csv")
    write_csv(all_seed_rows, out_dir / "fixed_protocol_all_seed_level.csv")
    write_csv(summarize_by_dataset(all_seed_rows), out_dir / "fixed_protocol_by_dataset_mean_std.csv")
    write_csv(pooled_rows, out_dir / "fixed_protocol_pooled_by_seed.csv")
    write_csv(summarize_for_paper(pooled_rows), out_dir / "fixed_protocol_for_paper_mean_std.csv")
    write_csv(tests, out_dir / "paired_wilcoxon_fixed_protocol.csv")

    print("Fixed protocol:")
    print(" tolerance_ratio =", TOLERANCE_RATIO)
    print(" cooldown_factor =", COOLDOWN_FACTOR)
    print(" min_episode_size =", MIN_EPISODE_SIZE)
    print("SCCM raw rows:", len(sccm_raw), "expected 360")
    print("SCCM episode rows:", len(sccm_episode), "expected 360")
    print("Baseline raw rows:", len(baseline_raw), "expected 2880")
    print("Baseline episode rows:", len(baseline_episode), "expected 2880")
    print("Paired comparison rows:", len(tests))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
