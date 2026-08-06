from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "Experiments" / "001 Synthetic"


def discover_scripts() -> list[Path]:
    scripts: list[Path] = []
    for path in EXPERIMENT_ROOT.rglob("*.py"):
        lower = path.name.lower()
        if any(token in lower for token in ("aggregate", "visualized", "visualize", "delme")):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "def run_multi_seed_experiment" in text and "run_single_seed_experiment" in text:
            scripts.append(path)
    return sorted(scripts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all 72 synthetic predictive experiments with the manuscript's five seeds."
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--filter", default="", help="Optional case-insensitive path substring.")
    parser.add_argument("--skip-statistics", action="store_true")
    args = parser.parse_args()

    scripts = discover_scripts()
    if args.filter:
        scripts = [p for p in scripts if args.filter.lower() in str(p).lower()]
    print(f"Discovered {len(scripts)} synthetic predictive scripts.")
    if not scripts:
        raise SystemExit("No matching experiment scripts were found.")

    failures: list[str] = []
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for index, script in enumerate(scripts, start=1):
        rel = script.relative_to(PROJECT_ROOT)
        print("\n" + "=" * 88)
        print(f"[{index}/{len(scripts)}] {rel}")
        result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            failures.append(str(rel))
            if not args.continue_on_error:
                break

    if not args.skip_statistics and (not failures or args.continue_on_error):
        stats = Path(__file__).with_name("002_predictive_paired_statistics.py")
        result = subprocess.run([sys.executable, str(stats)], cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            failures.append(str(stats.relative_to(PROJECT_ROOT)))

    if failures:
        print("\nFailed stages:")
        for item in failures:
            print(" -", item)
        return 1
    print("\nSynthetic predictive pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
