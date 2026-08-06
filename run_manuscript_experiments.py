"""Top-level reproducibility entry point for every experiment claimed in the manuscript."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(relative: str, extra: list[str] | None = None) -> int:
    command = [sys.executable, str(ROOT / relative), *(extra or [])]
    print("\n" + "=" * 96)
    print("Running:", " ".join(command))
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-synthetic-predictive", action="store_true")
    parser.add_argument("--skip-alarm-quality", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-real-world", action="store_true")
    parser.add_argument("--skip-computational-cost", action="store_true")
    parser.add_argument("--allow-missing-real-datasets", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    stages: list[tuple[str, list[str]]] = []
    if not args.skip_synthetic_predictive:
        stages.append(("SyntheticPredictiveStatistics/001_run_all_synthetic_predictive.py", []))
    if not args.skip_alarm_quality:
        stages.append(("BenchmarkDetectionActivation/001_run_all_benchmarks.py", []))
        stages.extend([
            ("BenchmarkDetectionActivation/002_aggregate_and_align.py", []),
            ("BenchmarkDetectionActivation/003_alarm_quality_paired_significance.py", []),
            ("BenchmarkDetectionActivation/004_generate_final_aggregate_comparison.py", []),
        ])
    if not args.skip_ablation:
        stages.append(("AblationSensitivity/001_run_olrwa_ablation.py", []))
    if not args.skip_computational_cost:
        stages.extend([
            ("DriftDetectionQuality2/001run_all_quality_experiments.py", ["--skip-sensitivity", "--skip-paired-statistics"]),
            ("ComputationalCost/001_run_standalone_resources.py", []),
            ("ComputationalCost/002_generate_computational_table.py", []),
        ])
    if not args.skip_real_world:
        validate_args = ["--allow-missing-datasets"] if args.allow_missing_real_datasets else []
        stages.append(("RealWorldDatasetsEvaluation/000_validate_setup.py", validate_args))
        if not args.allow_missing_real_datasets:
            stages.append(("RealWorldDatasetsEvaluation/001run_all_real_world_experiments.py", []))

    failures: list[str] = []
    for stage, extra in stages:
        if run(stage, extra) != 0:
            failures.append(stage)
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailed stages:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nAll requested manuscript experiment stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
