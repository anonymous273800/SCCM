from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality2"


def run_script(script: Path, extra_args: list[str] | None = None) -> int:
    print("\n" + "=" * 80)
    print("Running", script.relative_to(PROJECT_ROOT))
    command = [sys.executable, str(script), *(extra_args or [])]
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete five-seed alarm-quality study: SCCM, all eight "
            "ADWIN/KSWIN baselines, aggregation, sensitivity analysis, resource "
            "reporting, and paired SCCM-versus-baseline statistics."
        )
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-sccm", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument(
        "--skip-paired-statistics",
        action="store_true",
        help="Skip the legacy eight-baseline paired analysis; use BenchmarkDetectionActivation/003 for the manuscript-aligned ADWIN/KSWIN family test.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    failed: list[str] = []

    if not args.skip_validation:
        validator = DDQ_ROOT / "000_validate_setup.py"
        if run_script(validator) != 0:
            raise SystemExit("Validation failed. No experiments were started.")

    if not args.skip_sccm:
        scripts = sorted(DDQ_ROOT.rglob("quality_run.py"))
        print("Found", len(scripts), "SCCM quality_run.py scripts")
        for script in scripts:
            code = run_script(script)
            if code != 0:
                failed.append(str(script.relative_to(PROJECT_ROOT)))
                if not args.continue_on_error:
                    break

    if not args.skip_baselines and (not failed or args.continue_on_error):
        script = DDQ_ROOT / "004run_all_baselines.py"
        if run_script(script) != 0:
            failed.append(str(script.relative_to(PROJECT_ROOT)))

    analysis_scripts = [
        DDQ_ROOT / "aggregate_quality_results.py",
        DDQ_ROOT / "002aggregate_drift_point_details.py",
        DDQ_ROOT / "005aggregate_baseline_results.py",
    ]
    if not args.skip_sensitivity:
        analysis_scripts.append(DDQ_ROOT / "006parameter_sensitivity.py")
    if not args.skip_paired_statistics:
        analysis_scripts.append(DDQ_ROOT / "008fixed_protocol_analysis.py")

    for script in analysis_scripts:
        code = run_script(script)
        if code != 0:
            failed.append(str(script.relative_to(PROJECT_ROOT)))
            if not args.continue_on_error:
                break

    print("\nFinished DriftDetectionQuality2.")
    if failed:
        print("Failed stages:")
        for item in failed:
            print(" -", item)
        raise SystemExit(1)

    print("All requested experiment and analysis stages completed successfully.")


if __name__ == "__main__":
    main()
