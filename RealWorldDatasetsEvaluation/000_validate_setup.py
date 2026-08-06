from __future__ import annotations
import argparse
import importlib
import sys
sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealWorldDatasetsEvaluation.config import DATASETS, EXPECTED_RUNS
from RealWorldDatasetsEvaluation.common.project import ensure_project_importable
from RealWorldDatasetsEvaluation.common.data_registry import required_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-datasets", action="store_true")
    args = parser.parse_args()

    errors = []
    ensure_project_importable()
    for package in ["numpy", "pandas", "scipy", "sklearn", "psutil"]:
        try:
            importlib.import_module(package)
        except Exception as exc:
            errors.append(f"Missing Python package '{package}': {exc}")

    try:
        from Utils import DriftDetectors
        if DriftDetectors.BACKEND == "unavailable":
            errors.append("Neither scikit-multiflow nor river is available for drift detection.")
    except Exception as exc:
        errors.append(f"Could not initialize drift-detector backend: {exc}")

    for dataset, paths in required_paths().items():
        missing = [str(p) for p in paths if not p.exists()]
        if missing and not args.allow_missing_datasets:
            errors.append(f"{dataset}: missing " + "; ".join(missing))

    try:
        from Models.OLR_WA import OLR_WA
        from Models.PA import PA
        from Models.RLS import RLS
        from Models.WidrowHoff import WidrowHoff
        assert OLR_WA and PA and RLS and WidrowHoff
    except Exception as exc:
        errors.append(f"Could not import existing model code: {exc}")

    report = ROOT / "VALIDATION_REPORT.txt"
    lines = [
        "RealWorldDatasetsEvaluation validation",
        f"Expected complete runs: {EXPECTED_RUNS}",
        f"Datasets: {', '.join(DATASETS)}",
        f"Drift detector backend: {getattr(DriftDetectors, 'BACKEND', 'unavailable')}",
    ]
    if errors:
        lines.append("VALIDATION FAILED")
        lines.extend(f" - {error}" for error in errors)
    else:
        lines.append("VALIDATION PASSED")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
