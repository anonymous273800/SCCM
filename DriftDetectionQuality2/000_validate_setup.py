from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

EXPECTED_SEEDS = [0, 1, 42, 123, 7]
EXPECTED_DATASETS = {
    *(f"ADS{i:02d}" for i in range(1, 7)),
    *(f"IDS{i:02d}" for i in range(1, 7)),
    *(f"GDS{i:02d}" for i in range(1, 7)),
}
EXPECTED_MODELS = {"OLR-WA", "PA", "RLS", "WidrowHoff"}
EXPECTED_EXPERIMENTS = len(EXPECTED_DATASETS) * len(EXPECTED_MODELS)
EXPECTED_SCCM_SEED_RUNS = EXPECTED_EXPERIMENTS * len(EXPECTED_SEEDS)
EXPECTED_BASELINE_SEED_RUNS = EXPECTED_EXPERIMENTS * 8 * len(EXPECTED_SEEDS)
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for path in (str(PROJECT_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def read_config(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    configs = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    configs.append(ast.literal_eval(node.value))
    if not configs:
        raise ValueError("No active CONFIG dictionary found")
    return configs[-1]


def validate_matching(errors: list[str]) -> None:
    try:
        from ddq_common import (
            match_alarm_indices_to_true_drifts,
            match_episodes_to_true_drifts,
        )
        episode = {
            "episode_alarm_index": 450,
            "episode_start_index": 450,
            "episode_end_index": 510,
            "episode_size": 3,
            "episode_alarm_indices": [450, 490, 510],
        }
        episode_result = match_episodes_to_true_drifts([episode], [500], 50)
        raw_result = match_alarm_indices_to_true_drifts([450, 490, 510], [500], 50)
        if (episode_result["tp"], episode_result["fp"], episode_result["fn"]) != (1, 0, 0):
            errors.append(
                "Episode-window matching test failed: expected TP=1, FP=0, FN=0."
            )
        if episode_result["matched_alarm_indices"] != "510":
            errors.append(
                "Episode-window matching test failed: expected matched alarm 510."
            )
        if episode_result["delay_sum"] != 10:
            errors.append(
                "Episode-window matching test failed: expected delay 10 samples."
            )
        if (raw_result["tp"], raw_result["fp"], raw_result["fn"]) != (1, 2, 0):
            errors.append(
                "Raw-trigger supplemental test failed: expected TP=1, FP=2, FN=0."
            )
    except Exception as exc:
        errors.append(f"Matching self-test failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DriftDetectionQuality2 before execution.")
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Validate code and configuration without requiring runtime packages.",
    )
    args = parser.parse_args()

    scripts = sorted(ROOT.rglob("quality_run.py"))
    errors: list[str] = []
    warnings: list[str] = []
    observed_pairs: set[tuple[str, str]] = set()

    if len(scripts) != EXPECTED_EXPERIMENTS:
        errors.append(
            f"Expected {EXPECTED_EXPERIMENTS} quality_run.py files, found {len(scripts)}."
        )

    for script in scripts:
        try:
            config = read_config(script)
            model = config.get("model")
            dataset = config.get("dataset")
            seeds = config.get("seeds")
            observed_pairs.add((model, dataset))
            if model not in EXPECTED_MODELS:
                errors.append(f"{script}: unknown model {model!r}")
            if dataset not in EXPECTED_DATASETS:
                errors.append(f"{script}: unknown dataset {dataset!r}")
            if seeds != EXPECTED_SEEDS:
                errors.append(
                    f"{script}: seeds must be exactly {EXPECTED_SEEDS}, found {seeds!r}"
                )
        except Exception as exc:
            errors.append(f"{script}: {exc}")

    expected_pairs = {
        (model, dataset)
        for model in EXPECTED_MODELS
        for dataset in EXPECTED_DATASETS
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    duplicate_count = len(scripts) - len(observed_pairs)
    if missing_pairs:
        errors.append(f"Missing model-dataset pairs: {missing_pairs}")
    if duplicate_count:
        errors.append(f"Found {duplicate_count} duplicate model-dataset configurations.")

    required_files = [
        "ddq_common.py",
        "ddq2_baseline_runner.py",
        "resource_metrics.py",
        "005aggregate_baseline_results.py",
        "006parameter_sensitivity.py",
        "007paired_sccm_vs_baselines.py",
    ]
    for name in required_files:
        if not (ROOT / name).is_file():
            errors.append(f"Missing required file: {name}")

    validate_matching(errors)

    dependencies = {
        "river": "required by the existing ADWIN/KSWIN implementations",
        "psutil": "required for sampled process-RSS memory measurements",
        "scipy": "recommended for Wilcoxon tests and confidence intervals",
    }
    for package, purpose in dependencies.items():
        if importlib.util.find_spec(package) is None:
            message = f"Missing Python package '{package}', {purpose}."
            if args.skip_dependency_check:
                warnings.append(message)
            else:
                errors.append(message)

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(" -", error)
        if warnings:
            print("WARNINGS")
            for warning in warnings:
                print(" -", warning)
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print("Experiment files:", len(scripts))
    print("Runs per SCCM experiment:", len(EXPECTED_SEEDS))
    print("Expected SCCM seed runs:", EXPECTED_SCCM_SEED_RUNS)
    print("Expected baseline method-seed runs:", EXPECTED_BASELINE_SEED_RUNS)
    print("Expected total evaluated method-seed runs:", EXPECTED_SCCM_SEED_RUNS + EXPECTED_BASELINE_SEED_RUNS)
    print("Seeds:", EXPECTED_SEEDS)
    print("Episode matching self-test: passed")
    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print(" -", warning)


if __name__ == "__main__":
    main()
