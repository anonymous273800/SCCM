from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from scipy.stats import rankdata, wilcoxon
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "SciPy is required. Install it with: python -m pip install scipy"
    ) from exc

from benchmark_detection_common import EVALUATION_SEEDS, find_project_root


THIS_FILE = Path(__file__).resolve()
BENCHMARK_ROOT = THIS_FILE.parent
RESULTS_DIR = BENCHMARK_ROOT / "results"
DEFAULT_BASELINE_CSV = RESULTS_DIR / "benchmark_exact_by_dataset.csv"

OUTPUT_CSV = RESULTS_DIR / "alarm_quality_paired_wilcoxon.csv"
OUTPUT_TEX = RESULTS_DIR / "alarm_quality_paired_wilcoxon.tex"
OUTPUT_SUMMARY = RESULTS_DIR / "alarm_quality_paired_wilcoxon_summary.txt"

DRIFT_ORDER = ("abrupt", "incremental", "gradual")
DISPLAY_DRIFT = {
    "abrupt": "Abrupt",
    "incremental": "Incremental",
    "gradual": "Alternating gradual",
}

COMPARATOR_ORDER = ("ADWIN", "KSWIN")

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
    "f1": ("f1", "f1_score", "f1-score", "F1", "F1-score"),
    "tp": ("tp", "true_positives", "true_positive"),
    "fp": ("fp", "false_positives", "false_positive"),
    "fn": ("fn", "false_negatives", "false_negative", "missed_drifts"),
}


@dataclass(frozen=True)
class AlarmKey:
    model: str
    dataset: str
    drift_type: str
    seed: int


@dataclass(frozen=True)
class BaselineKey:
    model: str
    dataset: str
    drift_type: str
    seed: int
    comparator: str


@dataclass
class TestResult:
    drift_type: str
    comparator: str
    n_pairs: int
    zero_differences: int
    mean_sccm_f1: float
    mean_baseline_f1: float
    mean_difference: float
    median_difference: float
    wilcoxon_statistic: float
    raw_p_value: float
    holm_p_value: float = math.nan
    rank_biserial: float = 0.0

    @property
    def significant(self) -> bool:
        return bool(self.holm_p_value < 0.05)

    def as_row(self) -> dict[str, Any]:
        return {
            "drift_type": self.drift_type,
            "comparator": self.comparator,
            "n_pairs": self.n_pairs,
            "zero_differences": self.zero_differences,
            "mean_sccm_f1": round(self.mean_sccm_f1, 6),
            "mean_baseline_f1": round(self.mean_baseline_f1, 6),
            "mean_difference": round(self.mean_difference, 6),
            "median_difference": round(self.median_difference, 6),
            "wilcoxon_statistic": round(self.wilcoxon_statistic, 6),
            "raw_p_value": self.raw_p_value,
            "holm_p_value": self.holm_p_value,
            "rank_biserial": round(self.rank_biserial, 6),
            "significant_0_05": self.significant,
        }


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


def parse_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError("Missing numeric value")
    result = float(str(value).strip())
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value}")
    return result


def safe_f1(tp: float, fp: float, fn: float) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def normalize_model(value: Any) -> str:
    compact = re.sub(r"[^A-Z0-9-]+", "", str(value).strip().upper())
    compact_without_hyphen = compact.replace("-", "")
    if compact in MODEL_ALIASES:
        return MODEL_ALIASES[compact]
    if compact_without_hyphen in MODEL_ALIASES:
        return MODEL_ALIASES[compact_without_hyphen]
    raise ValueError(f"Unsupported model name: {value!r}")


def normalize_dataset(value: Any) -> str:
    dataset = str(value).strip().upper()
    match = re.search(r"(ADS|IDS|GDS)\s*0*([1-6])", dataset)
    if not match:
        raise ValueError(f"Unsupported synthetic dataset name: {value!r}")
    return f"{match.group(1)}{int(match.group(2)):02d}"


def infer_drift_type(dataset: str) -> str:
    if dataset.startswith("ADS"):
        return "abrupt"
    if dataset.startswith("IDS"):
        return "incremental"
    if dataset.startswith("GDS"):
        return "gradual"
    raise ValueError(f"Cannot infer drift type from dataset: {dataset}")


def normalize_drift_type(value: Any, dataset: str) -> str:
    if value is None or str(value).strip() == "":
        return infer_drift_type(dataset)
    text = re.sub(r"[^a-z]+", " ", str(value).strip().lower()).strip()
    if "abrupt" in text:
        return "abrupt"
    if "incremental" in text:
        return "incremental"
    if "gradual" in text or "alternating" in text or "recurring" in text:
        return "gradual"
    return infer_drift_type(dataset)


def normalize_comparator(
    method: Any = "", detector: Any = "", adaptation: Any = ""
) -> str:
    detector_text = str(detector).strip().upper()
    adaptation_text = str(adaptation).strip().upper()
    method_text = str(method).strip().upper().replace("_", "-")

    if detector_text not in {"ADWIN", "KSWIN"}:
        detector_match = re.search(r"(ADWIN|KSWIN)", method_text)
        if detector_match:
            detector_text = detector_match.group(1)

    if adaptation_text not in {"RESET", "WINDOW", "SSPT", "OHL"}:
        adaptation_match = re.search(r"(RESET|WINDOW|SSPT|OHL)", method_text)
        if adaptation_match:
            adaptation_text = adaptation_match.group(1)

    if detector_text not in COMPARATOR_ORDER:
        raise ValueError(
            f"Could not identify detector family from method={method!r}, "
            f"detector={detector!r}, adaptation={adaptation!r}"
        )
    # RESET, WINDOW, SSPT, and OHL share the same detector alarm sequence.
    # Alarm-quality inference therefore compares SCCM once with ADWIN and once
    # with KSWIN, exactly as specified in the manuscript.
    return detector_text


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def extract_f1(
    row: dict[str, str],
    f1_column: str | None,
    tp_column: str | None,
    fp_column: str | None,
    fn_column: str | None,
) -> float:
    if f1_column and str(row.get(f1_column, "")).strip() != "":
        return parse_float(row[f1_column])
    if tp_column and fp_column and fn_column:
        return safe_f1(
            parse_float(row[tp_column]),
            parse_float(row[fp_column]),
            parse_float(row[fn_column]),
        )
    raise ValueError("No F1 column and no TP/FP/FN columns were found")


def load_baseline_results(path: Path) -> dict[BaselineKey, float]:
    fieldnames, rows = read_csv_rows(path)
    required = {
        name: resolve_column(fieldnames, name)
        for name in ("model", "dataset", "seed")
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"Baseline CSV is missing columns {missing}: {path}")

    drift_column = resolve_column(fieldnames, "drift_type")
    method_column = resolve_column(fieldnames, "method")
    detector_column = resolve_column(fieldnames, "detector")
    adaptation_column = resolve_column(fieldnames, "adaptation")
    f1_column = resolve_column(fieldnames, "f1")
    tp_column = resolve_column(fieldnames, "tp")
    fp_column = resolve_column(fieldnames, "fp")
    fn_column = resolve_column(fieldnames, "fn")

    if method_column is None and (detector_column is None or adaptation_column is None):
        raise ValueError(
            "Baseline CSV must contain either a method/baseline column or both "
            f"detector and adaptation columns: {path}"
        )

    output: dict[BaselineKey, float] = {}
    for line_number, row in enumerate(rows, start=2):
        try:
            model = normalize_model(row[required["model"]])
            dataset = normalize_dataset(row[required["dataset"]])
            seed = parse_int(row[required["seed"]])
            if seed not in EVALUATION_SEEDS:
                continue
            drift_type = normalize_drift_type(
                row.get(drift_column, "") if drift_column else "", dataset
            )
            comparator = normalize_comparator(
                row.get(method_column, "") if method_column else "",
                row.get(detector_column, "") if detector_column else "",
                row.get(adaptation_column, "") if adaptation_column else "",
            )
            f1 = extract_f1(row, f1_column, tp_column, fp_column, fn_column)
            key = BaselineKey(model, dataset, drift_type, seed, comparator)
            if key in output and not math.isclose(output[key], f1, abs_tol=1e-12):
                raise ValueError(f"Conflicting duplicate baseline row for {key}")
            output[key] = f1
        except Exception as exc:
            raise ValueError(f"{path}, line {line_number}: {exc}") from exc
    return output


def looks_like_sccm_method(value: Any) -> bool:
    text = str(value).strip().upper()
    return text == "" or "SCCM" in text or text.endswith("*")


def load_sccm_results(paths: Sequence[Path]) -> dict[AlarmKey, float]:
    output: dict[AlarmKey, float] = {}
    for path in paths:
        fieldnames, rows = read_csv_rows(path)
        columns = {
            name: resolve_column(fieldnames, name)
            for name in ("model", "dataset", "seed")
        }
        missing = [name for name, column in columns.items() if column is None]
        if missing:
            raise ValueError(f"SCCM CSV is missing columns {missing}: {path}")

        drift_column = resolve_column(fieldnames, "drift_type")
        method_column = resolve_column(fieldnames, "method")
        f1_column = resolve_column(fieldnames, "f1")
        tp_column = resolve_column(fieldnames, "tp")
        fp_column = resolve_column(fieldnames, "fp")
        fn_column = resolve_column(fieldnames, "fn")

        for line_number, row in enumerate(rows, start=2):
            try:
                if method_column and not looks_like_sccm_method(row.get(method_column, "")):
                    continue
                model = normalize_model(row[columns["model"]])
                dataset = normalize_dataset(row[columns["dataset"]])
                seed = parse_int(row[columns["seed"]])
                if seed not in EVALUATION_SEEDS:
                    continue
                drift_type = normalize_drift_type(
                    row.get(drift_column, "") if drift_column else "", dataset
                )
                f1 = extract_f1(row, f1_column, tp_column, fp_column, fn_column)
                key = AlarmKey(model, dataset, drift_type, seed)
                if key in output and not math.isclose(output[key], f1, abs_tol=1e-12):
                    raise ValueError(f"Conflicting duplicate SCCM row for {key}")
                output[key] = f1
            except Exception as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
    return output


def compatible_sccm_csv(path: Path) -> bool:
    try:
        fieldnames, _ = read_csv_rows(path)
    except Exception:
        return False
    has_keys = all(
        resolve_column(fieldnames, name) is not None
        for name in ("model", "dataset", "seed")
    )
    has_metric = resolve_column(fieldnames, "f1") is not None or all(
        resolve_column(fieldnames, name) is not None for name in ("tp", "fp", "fn")
    )
    return has_keys and has_metric


def discover_sccm_csvs(project_root: Path) -> list[Path]:
    search_roots = [
        project_root / "DriftDetectionQuality" / "results",
        project_root / "results",
        RESULTS_DIR,
    ]
    candidates: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.csv"):
            lower_name = path.name.lower()
            if "benchmark_exact" in lower_name or "wilcoxon" in lower_name:
                continue
            if not any(token in lower_name for token in ("sccm", "alarm", "quality", "drift")):
                continue
            if compatible_sccm_csv(path):
                candidates.append(path.resolve())
    return sorted(set(candidates))


def validate_coverage(
    sccm: dict[AlarmKey, float], baseline: dict[BaselineKey, float]
) -> None:
    expected_seeds = set(EVALUATION_SEEDS)
    problems: list[str] = []

    for drift_type, datasets in EXPECTED_DATASETS_BY_DRIFT.items():
        expected_alarm_keys = {
            AlarmKey(model, dataset, drift_type, seed)
            for model in EXPECTED_MODELS
            for dataset in datasets
            for seed in expected_seeds
        }
        missing_sccm = expected_alarm_keys.difference(sccm)
        if missing_sccm:
            examples = sorted(missing_sccm, key=str)[:5]
            problems.append(
                f"SCCM {drift_type}: missing {len(missing_sccm)} of "
                f"{len(expected_alarm_keys)} rows; examples={examples}"
            )

        for comparator in COMPARATOR_ORDER:
            expected_baseline_keys = {
                BaselineKey(
                    key.model,
                    key.dataset,
                    key.drift_type,
                    key.seed,
                    comparator,
                )
                for key in expected_alarm_keys
            }
            missing_baseline = expected_baseline_keys.difference(baseline)
            if missing_baseline:
                examples = sorted(missing_baseline, key=str)[:3]
                problems.append(
                    f"{comparator}, {drift_type}: missing "
                    f"{len(missing_baseline)} rows; examples={examples}"
                )

    if problems:
        raise RuntimeError("Incomplete paired-test inputs:\n- " + "\n- ".join(problems))


def rank_biserial_signed(differences: Sequence[float]) -> float:
    nonzero = [float(value) for value in differences if not math.isclose(value, 0.0, abs_tol=1e-15)]
    if not nonzero:
        return 0.0
    ranks = rankdata([abs(value) for value in nonzero], method="average")
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    denominator = positive + negative
    return float((positive - negative) / denominator) if denominator else 0.0


def median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def run_wilcoxon(differences: Sequence[float]) -> tuple[float, float]:
    nonzero = [float(value) for value in differences if not math.isclose(value, 0.0, abs_tol=1e-15)]
    if not nonzero:
        return 0.0, 1.0
    result = wilcoxon(
        nonzero,
        zero_method="wilcox",
        correction=True,
        alternative="two-sided",
        method="approx",
    )
    return float(result.statistic), float(result.pvalue)


def holm_adjust(results: list[TestResult]) -> None:
    ordered = sorted(enumerate(results), key=lambda item: item[1].raw_p_value)
    m = len(ordered)
    running_max = 0.0
    adjusted: dict[int, float] = {}
    for rank, (original_index, result) in enumerate(ordered):
        value = min(1.0, (m - rank) * result.raw_p_value)
        running_max = max(running_max, value)
        adjusted[original_index] = running_max
    for index, result in enumerate(results):
        result.holm_p_value = adjusted[index]


def calculate_tests(
    sccm: dict[AlarmKey, float], baseline: dict[BaselineKey, float]
) -> list[TestResult]:
    all_results: list[TestResult] = []
    for drift_type in DRIFT_ORDER:
        category_results: list[TestResult] = []
        alarm_keys = sorted(
            (
                key
                for key in sccm
                if key.drift_type == drift_type
                and key.model in EXPECTED_MODELS
                and key.dataset in EXPECTED_DATASETS_BY_DRIFT[drift_type]
                and key.seed in EVALUATION_SEEDS
            ),
            key=lambda key: (key.model, key.dataset, key.seed),
        )
        for comparator in COMPARATOR_ORDER:
            sccm_values: list[float] = []
            baseline_values: list[float] = []
            for key in alarm_keys:
                baseline_key = BaselineKey(
                    key.model,
                    key.dataset,
                    key.drift_type,
                    key.seed,
                    comparator,
                )
                sccm_values.append(sccm[key])
                baseline_values.append(baseline[baseline_key])

            differences = [a - b for a, b in zip(sccm_values, baseline_values)]
            statistic, raw_p = run_wilcoxon(differences)
            category_results.append(
                TestResult(
                    drift_type=drift_type,
                    comparator=comparator,
                    n_pairs=len(differences),
                    zero_differences=sum(
                        math.isclose(value, 0.0, abs_tol=1e-15)
                        for value in differences
                    ),
                    mean_sccm_f1=sum(sccm_values) / len(sccm_values),
                    mean_baseline_f1=sum(baseline_values) / len(baseline_values),
                    mean_difference=sum(differences) / len(differences),
                    median_difference=median(differences),
                    wilcoxon_statistic=statistic,
                    raw_p_value=raw_p,
                    rank_biserial=rank_biserial_signed(differences),
                )
            )
        holm_adjust(category_results)
        all_results.extend(category_results)
    return all_results


def format_p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    output = value
    for old, new in replacements.items():
        output = output.replace(old, new)
    return output


def write_csv(path: Path, results: Sequence[TestResult]) -> None:
    rows = [result.as_row() for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, results: Sequence[TestResult]) -> None:
    lines = [
        r"\begin{sidewaystable}[p]",
        r"\centering",
        r"\captionsetup{justification=centering}",
        r"\caption{Paired Statistical Comparison of Episode-Level Alarm $F_1$ Scores}",
        r"\label{tab:alarm_quality_paired_wilcoxon}",
        r"\caption*{\footnotesize Two-sided Wilcoxon signed-rank tests use 120 paired model--dataset--seed observations per comparison (four models, six datasets, and five seeds). Holm correction is applied across the eight detector--adaptation comparisons within each drift category. Positive $\Delta F_1$ and rank-biserial correlation favor \modelname{}.}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Drift Type} & \textbf{Comparator} & \textbf{$n$} & \textbf{\modelname{} $F_1$} & \textbf{Baseline $F_1$} & \textbf{$\Delta F_1$} & \textbf{$W$} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{$r_{rb}$} \\",
        r"\midrule",
    ]

    previous_drift = None
    for result in results:
        drift_label = DISPLAY_DRIFT[result.drift_type]
        if previous_drift is not None and result.drift_type != previous_drift:
            lines.append(r"\midrule")
        display_drift = drift_label if result.drift_type != previous_drift else ""
        holm_text = format_p(result.holm_p_value)
        if result.significant:
            holm_text = rf"\textbf{{{holm_text}}}"
        lines.append(
            " & ".join(
                [
                    latex_escape(display_drift),
                    latex_escape(result.comparator),
                    str(result.n_pairs),
                    f"{result.mean_sccm_f1:.4f}",
                    f"{result.mean_baseline_f1:.4f}",
                    f"{result.mean_difference:+.4f}",
                    f"{result.wilcoxon_statistic:.1f}",
                    format_p(result.raw_p_value),
                    holm_text,
                    f"{result.rank_biserial:+.3f}",
                ]
            )
            + r" \\"
        )
        previous_drift = result.drift_type

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    baseline_path: Path,
    sccm_paths: Sequence[Path],
    results: Sequence[TestResult],
) -> None:
    significant = sum(result.significant for result in results)
    lines = [
        "Alarm-Quality Paired Wilcoxon Summary",
        "=======================================",
        f"Evaluation seeds: {list(EVALUATION_SEEDS)}",
        f"Baseline input: {baseline_path}",
        "SCCM input(s):",
        *[f"- {path}" for path in sccm_paths],
        f"Comparisons: {len(results)}",
        f"Holm-significant comparisons at alpha=0.05: {significant}",
        "",
    ]
    for drift_type in DRIFT_ORDER:
        lines.append(DISPLAY_DRIFT[drift_type])
        for result in results:
            if result.drift_type != drift_type:
                continue
            lines.append(
                f"- {result.comparator}: n={result.n_pairs}, "
                f"delta_F1={result.mean_difference:+.4f}, "
                f"Holm p={format_p(result.holm_p_value)}, "
                f"r_rb={result.rank_biserial:+.3f}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_sccm_paths(
    project_root: Path, explicit_paths: Sequence[str] | None
) -> list[Path]:
    if explicit_paths:
        paths = [Path(value).expanduser().resolve() for value in explicit_paths]
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
            "No compatible SCCM per-dataset/per-seed CSV was found. Run this "
            "script with --sccm-csv PATH. The CSV must contain model, dataset, "
            "seed, and either F1 or TP/FP/FN."
        )
    formatted = "\n".join(f"- {path}" for path in candidates)
    raise RuntimeError(
        "Multiple compatible SCCM CSV files were found. Select the correct one "
        f"with --sccm-csv PATH:\n{formatted}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the six detector-family paired Wilcoxon comparisons for "
            "episode-level synthetic alarm F1."
        )
    )
    parser.add_argument(
        "--baseline-csv",
        default=str(DEFAULT_BASELINE_CSV),
        help="Five-seed baseline by-dataset CSV produced by 002_aggregate_and_align.py.",
    )
    parser.add_argument(
        "--sccm-csv",
        action="append",
        help=(
            "SCCM per-model/per-dataset/per-seed CSV. Repeat this option when "
            "the four models are stored in separate CSV files."
        ),
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
    sccm_paths = resolve_sccm_paths(project_root, args.sccm_csv)

    baseline = load_baseline_results(baseline_path)
    sccm = load_sccm_results(sccm_paths)
    validate_coverage(sccm, baseline)
    results = calculate_tests(sccm, baseline)

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_tex = Path(args.output_tex).expanduser().resolve()
    output_summary = Path(args.output_summary).expanduser().resolve()
    write_csv(output_csv, results)
    write_latex(output_tex, results)
    write_summary(output_summary, baseline_path, sccm_paths, results)

    print(f"SCCM paired rows: {len(sccm)}")
    print(f"Baseline paired rows: {len(baseline)}")
    print(f"Statistical comparisons: {len(results)}")
    print(f"CSV: {output_csv}")
    print(f"LaTeX: {output_tex}")
    print(f"Summary: {output_summary}")


if __name__ == "__main__":
    main()
