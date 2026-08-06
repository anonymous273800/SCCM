from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality2"
for path in (str(PROJECT_ROOT), str(DDQ_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ddq_common import read_csv, write_csv  # noqa: E402
from ddq2_statistics import as_float, descriptive, holm_adjust, paired_wilcoxon  # noqa: E402

EXPECTED_SEEDS = [0, 1, 42, 123, 7]
MODELS = ["OLR-WA-SCCM", "PA-SCCM", "RLS-SCCM", "WidrowHoff-SCCM"]
DRIFT_TYPES = ["abrupt", "incremental", "gradual"]
DATASETS_BY_DRIFT = {
    "abrupt": [f"ADS{i:02d}" for i in range(1, 7)],
    "incremental": [f"IDS{i:02d}" for i in range(1, 7)],
    "gradual": [f"GDS{i:02d}" for i in range(1, 7)],
}
COUNT_METRICS = [
    "true_drifts",
    "candidate_triggers",
    "raw_alarm_episodes",
    "alarm_episodes",
    "removed_small_episodes",
    "tp", "fp", "fn",
    "raw_tp", "raw_fp", "raw_fn",
    "adaptations", "recalibrations", "recalibration_batches",
    "interventions", "processed_samples",
]
SUMMARY_METRICS = COUNT_METRICS + [
    "precision", "recall", "f1",
    "raw_precision", "raw_recall", "raw_f1",
    "mean_delay_samples", "mean_delay_increments",
    "raw_mean_delay_samples", "raw_mean_delay_increments",
    "runtime_seconds", "runtime_per_1000_samples",
    "peak_rss_mb", "peak_rss_delta_mb",
    "adaptations_per_1000", "recalibrations_per_1000",
    "interventions_per_1000",
]

PAIRWISE_METRICS = {
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
    "fp": "lower",
    "fn": "lower",
    "mean_delay": "lower",
}


def load_seed_rows() -> list[dict]:
    rows: list[dict] = []
    for summary_file in DDQ_ROOT.rglob("quality_outputs/*_summary.csv"):
        for row in read_csv(summary_file):
            if row.get("row_type") == "aggregate_for_dataset":
                continue
            if row.get("seed") in (None, ""):
                continue
            item = dict(row)
            item["seed"] = int(float(item["seed"]))
            item["source_file"] = str(summary_file.relative_to(DDQ_ROOT))
            rows.append(item)
    return rows


def add_descriptives(output: dict, rows: list[dict], metrics=SUMMARY_METRICS) -> None:
    for metric in metrics:
        stats = descriptive([as_float(row.get(metric)) for row in rows])
        for stat_name, value in stats.items():
            output[f"{metric}_{stat_name}"] = value


def micro_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def aggregate_member_rows(
    members: list[dict],
    model: str,
    drift_type: str,
    seed: int,
    scope: str,
) -> dict:
    item: dict = {
        "model": model,
        "base_model": members[0].get("base_model", "") if members else "",
        "method": members[0].get("method", model) if members else model,
        "drift_type": drift_type,
        "seed": seed,
        "scope": scope,
        "dataset_count": len({row["dataset"] for row in members}),
        "datasets": ";".join(sorted({row["dataset"] for row in members})),
    }

    for metric in COUNT_METRICS:
        item[metric] = sum(int(as_float(row.get(metric)) or 0) for row in members)

    precision, recall, f1 = micro_metrics(item["tp"], item["fp"], item["fn"])
    item["precision"] = precision
    item["recall"] = recall
    item["f1"] = f1
    raw_precision, raw_recall, raw_f1 = micro_metrics(
        item["raw_tp"], item["raw_fp"], item["raw_fn"]
    )
    item["raw_precision"] = raw_precision
    item["raw_recall"] = raw_recall
    item["raw_f1"] = raw_f1

    delay_sum = sum(float(as_float(row.get("delay_sum")) or 0.0) for row in members)
    delay_count = sum(int(as_float(row.get("delay_count")) or 0) for row in members)
    item["delay_sum"] = delay_sum
    item["delay_count"] = delay_count
    item["mean_delay_samples"] = delay_sum / delay_count if delay_count else ""
    item["mean_delay"] = item["mean_delay_samples"]

    delay_increment_sum = sum(
        float(as_float(row.get("mean_delay_increments")) or 0.0)
        * int(as_float(row.get("delay_count")) or 0)
        for row in members
    )
    item["mean_delay_increments"] = (
        delay_increment_sum / delay_count if delay_count else ""
    )
    item["mean_delay_batches"] = item["mean_delay_increments"]

    raw_delay_sum = sum(float(as_float(row.get("raw_delay_sum")) or 0.0) for row in members)
    raw_delay_count = sum(int(as_float(row.get("raw_delay_count")) or 0) for row in members)
    item["raw_delay_sum"] = raw_delay_sum
    item["raw_delay_count"] = raw_delay_count
    item["raw_mean_delay_samples"] = (
        raw_delay_sum / raw_delay_count if raw_delay_count else ""
    )
    raw_increment_sum = sum(
        float(as_float(row.get("raw_mean_delay_increments")) or 0.0)
        * int(as_float(row.get("raw_delay_count")) or 0)
        for row in members
    )
    item["raw_mean_delay_increments"] = (
        raw_increment_sum / raw_delay_count if raw_delay_count else ""
    )

    item["runtime_seconds"] = sum(
        float(as_float(row.get("runtime_seconds")) or 0.0) for row in members
    )
    processed = item.get("processed_samples", 0)
    item["runtime_per_1000_samples"] = (
        item["runtime_seconds"] * 1000.0 / processed if processed else 0.0
    )
    item["adaptations_per_1000"] = (
        item["adaptations"] * 1000.0 / processed if processed else 0.0
    )
    item["recalibrations_per_1000"] = (
        item["recalibrations"] * 1000.0 / processed if processed else 0.0
    )
    item["interventions_per_1000"] = (
        item["interventions"] * 1000.0 / processed if processed else 0.0
    )
    item["peak_rss_mb"] = max(
        (float(as_float(row.get("peak_rss_mb")) or 0.0) for row in members),
        default=0.0,
    )
    item["peak_rss_delta_mb"] = max(
        (float(as_float(row.get("peak_rss_delta_mb")) or 0.0) for row in members),
        default=0.0,
    )
    return item

def aggregate_by_dataset(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["dataset"], row["drift_type"])].append(row)

    output: list[dict] = []
    for (model, dataset, drift_type), members in sorted(grouped.items()):
        item = {
            "model": model,
            "dataset": dataset,
            "drift_type": drift_type,
            "seed_count": len(members),
            "seeds": ";".join(
                str(seed) for seed in sorted({row["seed"] for row in members})
            ),
        }
        add_descriptives(item, members)
        output.append(item)
    return output


def aggregate_model_drift_seed(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["drift_type"], row["seed"])].append(row)

    output: list[dict] = []
    for (model, drift_type, seed), members in sorted(grouped.items()):
        output.append(
            aggregate_member_rows(
                members,
                model=model,
                drift_type=drift_type,
                seed=seed,
                scope=drift_type,
            )
        )
    return output


def aggregate_model_all_seed(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["seed"])].append(row)

    output: list[dict] = []
    for (model, seed), members in sorted(grouped.items()):
        output.append(
            aggregate_member_rows(
                members,
                model=model,
                drift_type="all",
                seed=seed,
                scope="all",
            )
        )
    return output


def aggregate_for_paper(seed_aggregate_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_aggregate_rows:
        grouped[(row["model"], row["drift_type"])].append(row)

    output: list[dict] = []
    for (model, drift_type), members in sorted(grouped.items()):
        pooled_tp = sum(int(row["tp"]) for row in members)
        pooled_fp = sum(int(row["fp"]) for row in members)
        pooled_fn = sum(int(row["fn"]) for row in members)
        pooled_precision, pooled_recall, pooled_f1 = micro_metrics(
            pooled_tp, pooled_fp, pooled_fn
        )
        item = {
            "model": model,
            "drift_type": drift_type,
            "seed_count": len(members),
            "seeds": ";".join(str(seed) for seed in sorted(row["seed"] for row in members)),
            "dataset_count_per_seed_min": min(row["dataset_count"] for row in members),
            "dataset_count_per_seed_max": max(row["dataset_count"] for row in members),
            "pooled_tp": pooled_tp,
            "pooled_fp": pooled_fp,
            "pooled_fn": pooled_fn,
            "pooled_precision": pooled_precision,
            "pooled_recall": pooled_recall,
            "pooled_f1": pooled_f1,
            "variability_unit": "seed",
        }
        add_descriptives(item, members)
        output.append(item)
    return output


def build_completeness(rows: list[dict]) -> list[dict]:
    observed = {(row["model"], row["dataset"], row["seed"]) for row in rows}
    output: list[dict] = []
    for drift_type, datasets in DATASETS_BY_DRIFT.items():
        for model in MODELS:
            for dataset in datasets:
                for seed in EXPECTED_SEEDS:
                    output.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "drift_type": drift_type,
                            "seed": seed,
                            "status": (
                                "complete"
                                if (model, dataset, seed) in observed
                                else "missing"
                            ),
                        }
                    )
    return output


def compare_lookup(
    lookup: dict[tuple, dict],
    scopes: list[tuple[str, list[str]]],
    key_builder,
    pairing_unit: str,
) -> list[dict]:
    output: list[dict] = []
    for scope, pair_keys in scopes:
        for model_a, model_b in combinations(MODELS, 2):
            for metric, direction in PAIRWISE_METRICS.items():
                a_values: list[float] = []
                b_values: list[float] = []
                used_keys: list[str] = []
                for pair_key in pair_keys:
                    row_a = lookup.get(key_builder(model_a, scope, pair_key))
                    row_b = lookup.get(key_builder(model_b, scope, pair_key))
                    if row_a is None or row_b is None:
                        continue
                    value_a = as_float(row_a.get(metric))
                    value_b = as_float(row_b.get(metric))
                    if value_a is None or value_b is None:
                        continue
                    a_values.append(value_a)
                    b_values.append(value_b)
                    used_keys.append(str(pair_key))

                result = paired_wilcoxon(a_values, b_values)
                result.update(
                    {
                        "scope": scope,
                        "metric": metric,
                        "preferred_direction": direction,
                        "model_a": model_a,
                        "model_b": model_b,
                        "pairing_unit": pairing_unit,
                        "paired_keys": ";".join(used_keys),
                        "interpretation": (
                            "positive difference favors model_a"
                            if direction == "higher"
                            else "negative difference favors model_a"
                        ),
                    }
                )
                if direction == "higher":
                    result["better_a"] = result["wins_a"]
                    result["better_b"] = result["wins_b"]
                else:
                    result["better_a"] = result["wins_b"]
                    result["better_b"] = result["wins_a"]
                result["better_ties"] = result["ties"]
                output.append(result)
    return holm_adjust(output, group_fields=("scope", "metric"))


def paired_dataset_seed_comparisons(rows: list[dict]) -> list[dict]:
    lookup = {(row["model"], row["dataset"], row["seed"]): row for row in rows}
    scopes: list[tuple[str, list[tuple[str, int]]]] = []
    for drift_type in DRIFT_TYPES:
        keys = [
            (dataset, seed)
            for dataset in DATASETS_BY_DRIFT[drift_type]
            for seed in EXPECTED_SEEDS
        ]
        scopes.append((drift_type, keys))
    all_keys = [
        (dataset, seed)
        for drift_type in DRIFT_TYPES
        for dataset in DATASETS_BY_DRIFT[drift_type]
        for seed in EXPECTED_SEEDS
    ]
    scopes.append(("all", all_keys))

    return compare_lookup(
        lookup=lookup,
        scopes=scopes,
        key_builder=lambda model, _scope, key: (model, key[0], key[1]),
        pairing_unit="dataset_seed",
    )


def paired_seed_aggregate_comparisons(
    drift_seed_rows: list[dict], all_seed_rows: list[dict]
) -> list[dict]:
    combined = drift_seed_rows + all_seed_rows
    lookup = {(row["model"], row["drift_type"], row["seed"]): row for row in combined}
    scopes = [(scope, EXPECTED_SEEDS) for scope in DRIFT_TYPES + ["all"]]
    return compare_lookup(
        lookup=lookup,
        scopes=scopes,
        key_builder=lambda model, scope, seed: (model, scope, seed),
        pairing_unit="seed_after_pooling_datasets",
    )


def fmt(value, digits=3) -> str:
    number = as_float(value)
    return "--" if number is None else f"{number:.{digits}f}"


def write_latex(paper_rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Drift & Precision & Recall & $F_1$ & FP & Delay \\",
        r"\midrule",
    ]
    for row in paper_rows:
        line = (
            f"{row['model']} & {row['drift_type'].title()} & "
            f"{fmt(row.get('precision_mean'))} $\\pm$ {fmt(row.get('precision_std'))} & "
            f"{fmt(row.get('recall_mean'))} $\\pm$ {fmt(row.get('recall_std'))} & "
            f"{fmt(row.get('f1_mean'))} $\\pm$ {fmt(row.get('f1_std'))} & "
            f"{fmt(row.get('fp_mean'), 2)} $\\pm$ {fmt(row.get('fp_std'), 2)} & "
            f"{fmt(row.get('mean_delay_samples_mean'), 2)} $\\pm$ {fmt(row.get('mean_delay_samples_std'), 2)}"
        )
        lines.append(line + " " + "\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DDQ_ROOT / "AggregatedQualityResults"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_seed_rows()
    if not rows:
        print("No seed-level result rows were found. Run the quality experiments first.")

    dataset_rows = aggregate_by_dataset(rows)
    drift_seed_rows = aggregate_model_drift_seed(rows)
    all_seed_rows = aggregate_model_all_seed(rows)
    paper_rows = aggregate_for_paper(drift_seed_rows)
    completeness_rows = build_completeness(rows)

    write_csv(rows, out_dir / "alarm_quality_seed_level.csv")
    write_csv(dataset_rows, out_dir / "alarm_quality_by_dataset_mean_std.csv")
    write_csv(drift_seed_rows, out_dir / "alarm_quality_by_model_drift_seed.csv")
    write_csv(paper_rows, out_dir / "alarm_quality_for_paper_mean_std.csv")
    write_csv(completeness_rows, out_dir / "run_completeness.csv")
    write_latex(paper_rows, out_dir / "alarm_quality_for_paper_mean_std.tex")

    completed = sum(row["status"] == "complete" for row in completeness_rows)
    print("Seed-level rows:", len(rows))
    print("Expected rows:", len(completeness_rows))
    print("Complete rows:", completed)
    print("Missing rows:", len(completeness_rows) - completed)
    print("Paper variability unit: seed")
    print("SCCM-versus-baseline tests are generated by 007paired_sccm_vs_baselines.py")
    print("Saved results in", out_dir)


if __name__ == "__main__":
    main()
