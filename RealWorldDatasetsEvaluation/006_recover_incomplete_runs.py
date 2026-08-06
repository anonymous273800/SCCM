from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealWorldDatasetsEvaluation.config import DATASETS, MODELS, METHODS, SEEDS

RUNNER = ROOT / "001_RealWorldFullMatrix" / "run_full_matrix.py"

def call(args):
    cmd = [sys.executable, str(args[0]), *map(str, args[1:])]
    print("\n" + "=" * 88)
    print("Running:", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(cmd, cwd=ROOT.parent, env=env, check=True)


def validate_complete_matrix():
    raw = ROOT / "Results" / "raw" / "realworld_seed_level.csv"
    if not raw.exists():
        raise FileNotFoundError(f"Recovery output not found: {raw}")

    df = pd.read_csv(raw)
    key_cols = ["dataset", "model", "method", "seed"]
    duplicate_count = int(df.duplicated(key_cols, keep=False).sum())
    if duplicate_count:
        raise RuntimeError(
            f"Recovery produced {duplicate_count} duplicate rows across the matrix keys."
        )

    expected = {
        (dataset, model, method, int(seed))
        for dataset in DATASETS for model in MODELS
        for method in METHODS for seed in SEEDS
    }
    rows = {
        (str(row.dataset), str(row.model), str(row.method), int(row.seed)): str(row.status).lower()
        for row in df.itertuples(index=False)
    }
    missing = sorted(expected - set(rows))
    failed = sorted(key for key in expected if rows.get(key) != "complete")

    if missing or failed or len(rows) != len(expected):
        raise RuntimeError(
            "Recovery is still incomplete: "
            f"unique_rows={len(rows)}/{len(expected)}, "
            f"missing={len(missing)}, noncomplete={len(failed)}. "
            "Inspect Results/raw/realworld_seed_level.csv before aggregation."
        )

    print(f"Matrix validation passed: {len(expected)} complete unique runs.")

def main():
    # 200 combinations that never produced rows.
    call([RUNNER, "--datasets", "UCIAQD", "--replace-existing", "--continue-on-error"])

    # Rerun every method for these model-dataset pairs because their shared
    # numerical configuration changed. This keeps comparisons fair.
    call([RUNNER, "--datasets", "GASD", "--models", "RLS",
          "--replace-existing", "--continue-on-error"])
    call([RUNNER, "--datasets", "GASD", "--models", "WidrowHoff",
          "--replace-existing", "--continue-on-error"])
    call([RUNNER, "--datasets", "WSSF", "--models", "RLS",
          "--replace-existing", "--continue-on-error"])

    validate_complete_matrix()

    call([ROOT / "002_RealWorldAggregation" / "aggregate_results.py"])
    call([ROOT / "003_RealWorldStatistics" / "paired_tests.py"])
    call([ROOT / "004_PaperReadyResults" / "generate_paper_results.py"])
    call([ROOT / "005_ReviewerResponseMapping" / "generate_mapping.py"])
    print("\nRecovery completed. Check AGGREGATION_SUMMARY.txt for 1600 complete runs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
