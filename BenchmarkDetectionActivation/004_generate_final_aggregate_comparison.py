from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark_detection_common import EVALUATION_SEEDS, find_project_root


THIS_FILE = Path(__file__).resolve()
BENCHMARK_ROOT = THIS_FILE.parent
RESULTS_DIR = BENCHMARK_ROOT / "results"
DEFAULT_BASELINE_CSV = RESULTS_DIR / "benchmark_exact_by_dataset.csv"

OUTPUT_CSV = RESULTS_DIR / "final_aggregate_sccm_vs_baselines.csv"
OUTPUT_TEX = RESULTS_DIR / "final_aggregate_sccm_vs_baselines.tex"
OUTPUT_SUMMARY = RESULTS_DIR / "final_aggregate_sccm_vs_baselines_summary.txt"

DRIFT_ORDER = ("abrupt", "incremental", "gradual")
DISPLAY_DRIFT = {
    "abrupt": "Abrupt",
    "incremental": "Incremental",
    "gradual": "Alternating gradual",
}
METHOD_ORDER = (
    "SCCM",
    "ADWIN + RESET",
    "ADWIN + WINDOW",
    "ADWIN + SSPT",
    "ADWIN + OHL",
    "KSWIN + RESET",
    "KSWIN + WINDOW",
    "KSWIN + SSPT",
    "KSWIN + OHL",
)
EXPECTED_MODELS = {"OLR-WA", "PA", "RLS", "LMS"}
EXPECTED_DATASETS_BY_DRIFT = {
    "abrupt": {f"ADS{i:02d}" for i in range(1, 7)},
    "incremental": {f"IDS{i:02d}" for i in range(1, 7)},
    "gradual": {f"GDS{i:02d}" for i in range(1, 7)},
}

MODEL_ALIASES = {
    "OLRWA": "OLR-WA",
    "OLR-WA": "OLR-WA",
    "PA": "PA",
    "RLS": "RLS",
    "LMS": "LMS",
    "WIDROWHOFF": "LMS",
    "WIDROW-HOFF": "LMS",
    "WIDROWHOFFLMS": "LMS",
}

COLUMN_ALIASES = {
    "model": ("model", "base_model", "learner", "regression_model"),
    "dataset": ("dataset", "dataset_name", "data_set"),
    "drift_type": ("drift_type", "drift", "category", "drift_category"),
    "seed": ("seed", "random_seed", "evaluation_seed"),
    "method": ("method", "baseline", "approach", "configuration"),
    "detector": ("detector", "drift_detector"),
    "adaptation": ("adaptation", "adaptation_method", "strategy"),
    "true_drifts": ("true_drifts", "total_true_drifts", "ground_truth_drifts"),
    "raw_alarms": (
        "raw_detector_detections",
        "raw_alarms",
        "raw_detections",
        "detector_detections",
    ),
    "candidate_episodes": ("candidate_episodes", "alarm_episodes"),
    "retained_episodes": (
        "retained_alarm_episodes",
        "retained_episodes",
        "retained_alarms",
    ),
    "tp": ("tp", "true_positives", "true_positive"),
    "fp": ("fp", "false_positives", "false_positive"),
    "fn": ("fn", "false_negatives", "false_negative", "missed_drifts"),
    "delay_sum": ("delay_sum", "detection_delay_sum", "matched_delay_sum"),
    "delay_count": ("delay_count", "matched_delay_count", "detected_drifts"),
    "mean_delay": (
        "mean_delay_batches",
        "mean_delay_increments",
        "mean_detection_delay",
        "detection_delay",
        "delay",
    ),
    "adaptations": (
        "adaptation_activations",
        "adaptation_events",
        "adaptations",
        "num_adaptations",
    ),
    "recalibrations": (
        "recalibration_events",
        "recalibrations",
        "num_recalibrations",
    ),
}


@dataclass(frozen=True)
class RunKey:
    model: str
    dataset: str
    drift_type: str
    seed: int
    method: str


@dataclass
class RunRecord:
    key: RunKey
    true_drifts: float
    raw_alarms: float | None
    candidate_episodes: float | None
    retained_episodes: float | None
    tp: float
    fp: float
    fn: float
    delay_sum: float | None
    delay_count: float | None
    mean_delay: float | None
    adaptations: float | None
    recalibrations: float | None


@dataclass
class SeedAggregate:
    drift_type: str
    method: str
    seed: int
    true_drifts: float
    raw_alarms: float | None
    candidate_episodes: float | None
    retained_episodes: float | None
    tp: float
    fp: float
    fn: float
    precision: float
    recall: float
    f1: float
    mean_delay: float | None
    adaptations: float | None
    recalibrations: float | None


def canonical_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_column(fieldnames: Sequence[str], logical_name: str) -> str | None:
    normalized = {canonical_header(name): name for name in fieldnames}
    for alias in COLUMN_ALIASES[logical_name]:
        match = normalized.get(canonical_header(alias))
        if match is not None:
            return match
    return None


def parse_int(value: Any) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError("Missing integer value")
    return int(float(str(value).strip()))


def parse_float_optional(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(str(value).strip())
    if not math.isfinite(result):
        return None
    return result


def normalize_model(value: Any) -> str:
    compact = re.sub(r"[^A-Z0-9-]+", "", str(value).strip().upper())
    no_hyphen = compact.replace("-", "")
    if compact in MODEL_ALIASES:
        return MODEL_ALIASES[compact]
    if no_hyphen in MODEL_ALIASES:
        return MODEL_ALIASES[no_hyphen]
    raise ValueError(f"Unsupported model name: {value!r}")


def normalize_dataset(value: Any) -> str:
    dataset = re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())
    match = re.fullmatch(r"([AIG]DS)0?([1-6])", dataset)
    if not match:
        raise ValueError(f"Unsupported synthetic dataset name: {value!r}")
    return f"{match.group(1)}{int(match.group(2)):02d}"


def infer_drift_type(dataset: str, explicit: Any = "") -> str:
    explicit_value = str(explicit).strip().lower().replace("-", "_")
    if "abrupt" in explicit_value:
        return "abrupt"
    if "incremental" in explicit_value:
        return "incremental"
    if "gradual" in explicit_value or "alternating" in explicit_value:
        return "gradual"
    if dataset.startswith("ADS"):
        return "abrupt"
    if dataset.startswith("IDS"):
        return "incremental"
    if dataset.startswith("GDS"):
        return "gradual"
    raise ValueError(f"Cannot infer drift type for {dataset}")


def normalize_method(row: dict[str, str], columns: dict[str, str | None], *, sccm: bool) -> str:
    if sccm:
        return "SCCM"
    detector_col = columns.get("detector")
    adaptation_col = columns.get("adaptation")
    if detector_col and adaptation_col:
        detector = str(row.get(detector_col, "")).strip().upper()
        adaptation = str(row.get(adaptation_col, "")).strip().upper()
        if detector and adaptation:
            return f"{detector} + {adaptation}"
    method_col = columns.get("method")
    raw = str(row.get(method_col or "", "")).strip().upper()
    detector = "ADWIN" if "ADWIN" in raw else "KSWIN" if "KSWIN" in raw else ""
    adaptation = next((name for name in ("RESET", "WINDOW", "SSPT", "OHL") if name in raw), "")
    if detector and adaptation:
        return f"{detector} + {adaptation}"
    raise ValueError(f"Cannot determine baseline method from row: {row}")


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def sum_optional(values: Iterable[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return sum(available) if available else None


def read_csv_records(paths: Sequence[Path], *, sccm: bool) -> dict[RunKey, RunRecord]:
    records: dict[RunKey, RunRecord] = {}
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            columns = {name: resolve_column(reader.fieldnames, name) for name in COLUMN_ALIASES}
            required = ("model", "dataset", "seed", "tp", "fp", "fn")
            missing = [name for name in required if columns[name] is None]
            if missing:
                raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

            for row_number, row in enumerate(reader, start=2):
                try:
                    model = normalize_model(row[columns["model"] or ""])
                    dataset = normalize_dataset(row[columns["dataset"] or ""])
                    seed = parse_int(row[columns["seed"] or ""])
                    if seed not in EVALUATION_SEEDS:
                        continue
                    drift_value = row.get(columns["drift_type"] or "", "")
                    drift_type = infer_drift_type(dataset, drift_value)
                    method = normalize_method(row, columns, sccm=sccm)
                    tp = float(parse_int(row[columns["tp"] or ""]))
                    fp = float(parse_int(row[columns["fp"] or ""]))
                    fn = float(parse_int(row[columns["fn"] or ""]))
                    true_drifts = parse_float_optional(row.get(columns["true_drifts"] or ""))
                    if true_drifts is None:
                        true_drifts = tp + fn
                    key = RunKey(model, dataset, drift_type, seed, method)
                    record = RunRecord(
                        key=key,
                        true_drifts=true_drifts,
                        raw_alarms=parse_float_optional(row.get(columns["raw_alarms"] or "")),
                        candidate_episodes=parse_float_optional(row.get(columns["candidate_episodes"] or "")),
                        retained_episodes=parse_float_optional(row.get(columns["retained_episodes"] or "")),
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        delay_sum=parse_float_optional(row.get(columns["delay_sum"] or "")),
                        delay_count=parse_float_optional(row.get(columns["delay_count"] or "")),
                        mean_delay=parse_float_optional(row.get(columns["mean_delay"] or "")),
                        adaptations=parse_float_optional(row.get(columns["adaptations"] or "")),
                        recalibrations=parse_float_optional(row.get(columns["recalibrations"] or "")),
                    )
                    if key in records:
                        raise ValueError(f"Duplicate key {key} in {path}")
                    records[key] = record
                except Exception as exc:
                    raise ValueError(f"Invalid row {row_number} in {path}: {exc}") from exc
    return records


def validate_coverage(records: dict[RunKey, RunRecord], *, sccm: bool) -> None:
    expected_methods = ("SCCM",) if sccm else METHOD_ORDER[1:]
    expected = {
        RunKey(model, dataset, drift_type, seed, method)
        for drift_type in DRIFT_ORDER
        for dataset in EXPECTED_DATASETS_BY_DRIFT[drift_type]
        for model in EXPECTED_MODELS
        for seed in EVALUATION_SEEDS
        for method in expected_methods
    }
    actual = set(records)
    missing = sorted(expected - actual, key=lambda key: (key.method, key.drift_type, key.model, key.dataset, key.seed))
    extra = sorted(actual - expected, key=lambda key: (key.method, key.drift_type, key.model, key.dataset, key.seed))
    if missing or extra:
        message = [f"Coverage mismatch for {'SCCM' if sccm else 'baseline'} records."]
        if missing:
            message.append(f"Missing {len(missing)} rows; first examples: {missing[:8]}")
        if extra:
            message.append(f"Unexpected {len(extra)} rows; first examples: {extra[:8]}")
        raise ValueError("\n".join(message))


def aggregate_by_seed(records: Iterable[RunRecord]) -> list[SeedAggregate]:
    groups: dict[tuple[str, str, int], list[RunRecord]] = defaultdict(list)
    for record in records:
        groups[(record.key.drift_type, record.key.method, record.key.seed)].append(record)

    output: list[SeedAggregate] = []
    for (drift_type, method, seed), group in groups.items():
        tp = sum(record.tp for record in group)
        fp = sum(record.fp for record in group)
        fn = sum(record.fn for record in group)
        precision, recall, f1 = compute_metrics(tp, fp, fn)

        delay_sum_values = [record.delay_sum for record in group if record.delay_sum is not None]
        delay_count_values = [record.delay_count for record in group if record.delay_count is not None]
        if delay_sum_values and delay_count_values and sum(delay_count_values) > 0:
            mean_delay = sum(delay_sum_values) / sum(delay_count_values)
        else:
            weighted_numerator = 0.0
            weighted_denominator = 0.0
            for record in group:
                if record.mean_delay is None or record.tp <= 0:
                    continue
                weighted_numerator += record.mean_delay * record.tp
                weighted_denominator += record.tp
            mean_delay = weighted_numerator / weighted_denominator if weighted_denominator else None

        output.append(
            SeedAggregate(
                drift_type=drift_type,
                method=method,
                seed=seed,
                true_drifts=sum(record.true_drifts for record in group),
                raw_alarms=sum_optional(record.raw_alarms for record in group),
                candidate_episodes=sum_optional(record.candidate_episodes for record in group),
                retained_episodes=sum_optional(record.retained_episodes for record in group),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                mean_delay=mean_delay,
                adaptations=sum_optional(record.adaptations for record in group),
                recalibrations=sum_optional(record.recalibrations for record in group),
            )
        )
    return output


def mean_sd(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    mean_value = statistics.fmean(clean)
    sd_value = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean_value, sd_value


def summarize_across_seeds(seed_rows: Sequence[SeedAggregate]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[SeedAggregate]] = defaultdict(list)
    for row in seed_rows:
        groups[(row.drift_type, row.method)].append(row)

    metric_names = (
        "true_drifts",
        "raw_alarms",
        "candidate_episodes",
        "retained_episodes",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_delay",
        "adaptations",
        "recalibrations",
    )
    output: list[dict[str, Any]] = []
    for drift_type in DRIFT_ORDER:
        for method in METHOD_ORDER:
            rows = groups.get((drift_type, method), [])
            observed_seeds = {row.seed for row in rows}
            if observed_seeds != set(EVALUATION_SEEDS):
                raise ValueError(
                    f"Expected seeds {list(EVALUATION_SEEDS)} for {drift_type}/{method}, "
                    f"found {sorted(observed_seeds)}"
                )
            summary: dict[str, Any] = {
                "drift_type": drift_type,
                "method": method,
                "n_seeds": len(rows),
                "seeds": ";".join(str(seed) for seed in EVALUATION_SEEDS),
            }
            for metric in metric_names:
                mean_value, sd_value = mean_sd(getattr(row, metric) for row in rows)
                summary[f"{metric}_mean"] = mean_value
                summary[f"{metric}_sd"] = sd_value
            output.append(summary)
    return output


def format_mean_sd(row: dict[str, Any], metric: str, decimals: int) -> str:
    mean_value = row.get(f"{metric}_mean")
    sd_value = row.get(f"{metric}_sd")
    if mean_value is None or sd_value is None:
        return "--"
    return f"{float(mean_value):.{decimals}f}\\(\\pm{float(sd_value):.{decimals}f}\\)"


def latex_method(method: str) -> str:
    return r"\textbf{\modelname{}}" if method == "SCCM" else method


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        r"\begin{sidewaystable}[p]",
        r"\centering",
        r"\captionsetup{justification=centering}",
        r"\caption{Five-Seed Episode-Level Alarm Quality and Intervention Comparison}",
        r"\label{tab:final_aggregate_detection_benchmark}",
        r"\caption*{\footnotesize Results are reported as mean $\pm$ standard deviation over the five evaluation seeds $\{0,1,42,123,7\}$. Within each seed and drift category, counts are pooled across six datasets and four online regression models. Delay is the conventional matched-alarm delay in processing increments; no increment is subtracted. Raw ADWIN/KSWIN alarms correspond one-to-one with baseline adaptations. SCCM raw-alarm entries are shown as dashes when the SCCM input reports only first-trigger alarm episodes.}",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\resizebox{\textheight}{!}{%",
        r"\begin{tabular}{llccccccccccccc}",
        r"\toprule",
        r"\textbf{Drift Type} & \textbf{Method} & \textbf{True Drifts} & \textbf{Raw Alarms} & \textbf{Retained Episodes} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{Precision} & \textbf{Recall} & \textbf{$F_1$} & \textbf{Delay} & \textbf{Adaptations} & \textbf{Recalibrations} \\",
        r"\midrule",
    ]
    previous_drift = None
    for row in rows:
        drift_type = str(row["drift_type"])
        if previous_drift is not None and drift_type != previous_drift:
            lines.append(r"\midrule")
        display = DISPLAY_DRIFT[drift_type] if drift_type != previous_drift else ""
        cells = [
            display,
            latex_method(str(row["method"])),
            format_mean_sd(row, "true_drifts", 1),
            format_mean_sd(row, "raw_alarms", 1),
            format_mean_sd(row, "retained_episodes", 1),
            format_mean_sd(row, "tp", 1),
            format_mean_sd(row, "fp", 1),
            format_mean_sd(row, "fn", 1),
            format_mean_sd(row, "precision", 3),
            format_mean_sd(row, "recall", 3),
            format_mean_sd(row, "f1", 3),
            format_mean_sd(row, "mean_delay", 2),
            format_mean_sd(row, "adaptations", 1),
            format_mean_sd(row, "recalibrations", 1),
        ]
        lines.append(" & ".join(cells) + r" \\")
        previous_drift = drift_type
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{sidewaystable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    baseline_path: Path,
    sccm_paths: Sequence[Path],
    rows: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "Five-Seed SCCM vs Baseline Aggregate Comparison",
        "================================================",
        f"Evaluation seeds: {list(EVALUATION_SEEDS)}",
        f"Baseline input: {baseline_path}",
        "SCCM input(s):",
        *[f"- {item}" for item in sccm_paths],
        f"Output rows: {len(rows)}",
        "Delay: conventional processing-increment delay; no subtraction applied.",
        "",
    ]
    for row in rows:
        lines.append(
            f"{DISPLAY_DRIFT[str(row['drift_type'])]} | {row['method']} | "
            f"F1={row['f1_mean']:.4f}±{row['f1_sd']:.4f} | "
            f"TP={row['tp_mean']:.1f}±{row['tp_sd']:.1f} | "
            f"FP={row['fp_mean']:.1f}±{row['fp_sd']:.1f} | "
            f"FN={row['fn_mean']:.1f}±{row['fn_sd']:.1f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def discover_sccm_csvs(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    excluded_names = {
        DEFAULT_BASELINE_CSV.name,
        OUTPUT_CSV.name,
        "alarm_quality_paired_wilcoxon.csv",
    }
    for path in project_root.rglob("*.csv"):
        if path.name in excluded_names or RESULTS_DIR in path.parents:
            continue
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
        except (OSError, UnicodeError):
            continue
        required = all(resolve_column(header, name) is not None for name in ("model", "dataset", "seed", "tp", "fp", "fn"))
        if required:
            candidates.append(path.resolve())
    return sorted(set(candidates))


def resolve_sccm_paths(project_root: Path, explicit: Sequence[str] | None) -> list[Path]:
    if explicit:
        paths = [Path(value).expanduser().resolve() for value in explicit]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing SCCM CSV file(s): {missing}")
        return paths
    candidates = discover_sccm_csvs(project_root)
    if len(candidates) == 1:
        print(f"Auto-detected SCCM CSV: {candidates[0]}")
        return candidates
    if not candidates:
        raise FileNotFoundError(
            "No compatible five-seed SCCM CSV was found. Use --sccm-csv PATH. "
            "The file must contain model, dataset, seed, TP, FP, and FN."
        )
    formatted = "\n".join(f"- {path}" for path in candidates)
    raise RuntimeError(
        "Multiple compatible SCCM CSV files were found. Select the correct file(s) "
        f"with --sccm-csv PATH:\n{formatted}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the five-seed aggregate SCCM-versus-baseline alarm-quality "
            "and intervention table."
        )
    )
    parser.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE_CSV))
    parser.add_argument(
        "--sccm-csv",
        action="append",
        help="Five-seed SCCM per-model/per-dataset/per-seed CSV; repeat for multiple files.",
    )
    parser.add_argument("--output-csv", default=str(OUTPUT_CSV))
    parser.add_argument("--output-tex", default=str(OUTPUT_TEX))
    parser.add_argument("--output-summary", default=str(OUTPUT_SUMMARY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_csv).expanduser().resolve()
    if args.sccm_csv:
        # An explicit SCCM path does not require project-root discovery.
        project_root = Path.cwd().resolve()
    else:
        project_root = find_project_root()
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found: {baseline_path}\nRun 002_aggregate_and_align.py first."
        )
    sccm_paths = resolve_sccm_paths(project_root, args.sccm_csv)

    baseline_records = read_csv_records([baseline_path], sccm=False)
    sccm_records = read_csv_records(sccm_paths, sccm=True)
    validate_coverage(baseline_records, sccm=False)
    validate_coverage(sccm_records, sccm=True)

    seed_rows = aggregate_by_seed([*sccm_records.values(), *baseline_records.values()])
    rows = summarize_across_seeds(seed_rows)

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_tex = Path(args.output_tex).expanduser().resolve()
    output_summary = Path(args.output_summary).expanduser().resolve()
    write_csv(output_csv, rows)
    write_latex(output_tex, rows)
    write_summary(output_summary, baseline_path, sccm_paths, rows)

    print(f"Evaluation seeds: {list(EVALUATION_SEEDS)}")
    print(f"Baseline detailed rows: {len(baseline_records)}")
    print(f"SCCM detailed rows: {len(sccm_records)}")
    print(f"Aggregate rows: {len(rows)}")
    print(f"CSV: {output_csv}")
    print(f"LaTeX: {output_tex}")
    print(f"Summary: {output_summary}")


if __name__ == "__main__":
    main()
