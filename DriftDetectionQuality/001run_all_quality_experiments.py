from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality"


def main():
    scripts = sorted(DDQ_ROOT.rglob("quality_run.py"))
    print("Found", len(scripts), "quality_run.py scripts")
    failed = []
    for script in scripts:
        print("\n" + "=" * 80)
        print("Running", script.relative_to(PROJECT_ROOT))
        result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            failed.append(str(script.relative_to(PROJECT_ROOT)))
            print("FAILED", script)

    print("\nFinished all scripts.")
    if failed:
        print("Failed scripts:")
        for item in failed:
            print(" -", item)
        sys.exit(1)

    aggregate = DDQ_ROOT / "aggregate_quality_results.py"
    subprocess.run([sys.executable, str(aggregate)], cwd=str(PROJECT_ROOT), check=False)


if __name__ == "__main__":
    main()
