from __future__ import annotations
import argparse
import os
import subprocess
import sys
sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def call(relative, extra=None):
    command = [sys.executable, str(ROOT / relative)] + list(extra or [])
    print("\n" + "="*80)
    print("Running", " ".join(command))
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, cwd=ROOT.parent, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Validate, run, aggregate, test, and prepare the complete real-world evaluation.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--datasets")
    parser.add_argument("--models")
    parser.add_argument("--methods")
    parser.add_argument("--seeds")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--echo-model-output", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    call("000_validate_setup.py")
    run_args = []
    for name in ["datasets","models","methods","seeds"]:
        value = getattr(args, name)
        if value:
            run_args += [f"--{name}", value]
    if args.continue_on_error: run_args.append("--continue-on-error")
    if args.no_resume: run_args.append("--no-resume")
    if args.echo_model_output: run_args.append("--echo-model-output")
    if args.replace_existing: run_args.append("--replace-existing")
    call("001_RealWorldFullMatrix/run_full_matrix.py", run_args)
    call("002_RealWorldAggregation/aggregate_results.py")
    call("003_RealWorldStatistics/paired_tests.py")
    call("003_RealWorldStatistics/paired_tests_seed_level_reproduction.py")
    call("004_PaperReadyResults/generate_paper_results.py")
    call("005_ReviewerResponseMapping/generate_mapping.py")
    print("\nAll RealWorldDatasetsEvaluation stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
