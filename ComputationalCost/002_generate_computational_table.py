"""Generate the complete five-seed computational-cost table."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality2"
OUTPUT_DIR = PROJECT_ROOT / "ComputationalCost" / "results"


def pool_seed(frame: pd.DataFrame, method_column: str, method_name: str | None = None) -> pd.DataFrame:
    data = frame.copy()
    if method_name is not None:
        data["paper_method"] = method_name
    else:
        data["paper_method"] = data[method_column]
    grouped = []
    for (method, seed), members in data.groupby(["paper_method", "seed"]):
        processed = members.processed_samples.astype(float).sum()
        runtime = members.runtime_seconds.astype(float).sum()
        grouped.append({
            "method": method, "seed": int(seed),
            "runtime_per_1000_samples": runtime * 1000.0 / processed if processed else 0.0,
            "peak_memory_mb": members.peak_rss_delta_mb.astype(float).max(),
            "adaptations_per_1000": (
                (members.adaptations_per_1000.astype(float) * members.processed_samples.astype(float)).sum() / processed
                if processed else 0.0
            ),
            "recalibrations_per_1000": (
                (members.recalibrations_per_1000.astype(float) * members.processed_samples.astype(float)).sum() / processed
                if processed else 0.0
            ),
            "processed_samples": processed,
        })
    return pd.DataFrame(grouped)


def main() -> int:
    standalone_path = OUTPUT_DIR / "standalone_resource_seed_level.csv"
    sccm_path = DDQ_ROOT / "AggregatedQualityResults" / "alarm_quality_seed_level.csv"
    baseline_path = DDQ_ROOT / "BaselineResults" / "aggregated" / "baseline_alarm_quality_seed_level.csv"
    missing = [str(p) for p in (standalone_path, sccm_path, baseline_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing resource inputs:\n- " + "\n- ".join(missing))

    standalone = pool_seed(pd.read_csv(standalone_path), "method", "Standalone learner")
    sccm_raw = pd.read_csv(sccm_path)
    sccm = pool_seed(sccm_raw, "method", "SCCM")
    baseline_raw = pd.read_csv(baseline_path)
    baseline_raw = baseline_raw.rename(columns={"method": "baseline_method"})
    baselines = pool_seed(baseline_raw, "baseline_method")
    pooled = pd.concat([standalone, sccm, baselines], ignore_index=True)
    pooled.to_csv(OUTPUT_DIR / "computational_cost_seed_level_pooled.csv", index=False)

    summary = pooled.groupby("method", as_index=False).agg(
        runtime_ms_per_1000_mean=("runtime_per_1000_samples", lambda s: 1000.0 * s.mean()),
        runtime_ms_per_1000_std=("runtime_per_1000_samples", lambda s: 1000.0 * s.std(ddof=1)),
        peak_memory_mb_mean=("peak_memory_mb", "mean"),
        peak_memory_mb_std=("peak_memory_mb", "std"),
        adaptations_per_1000_mean=("adaptations_per_1000", "mean"),
        adaptations_per_1000_std=("adaptations_per_1000", "std"),
        recalibrations_per_1000_mean=("recalibrations_per_1000", "mean"),
        recalibrations_per_1000_std=("recalibrations_per_1000", "std"),
        seed_count=("seed", "count"),
    )
    base_runtime = float(summary.loc[summary.method.eq("Standalone learner"), "runtime_ms_per_1000_mean"].iloc[0])
    summary["runtime_overhead_vs_standalone_percent"] = 100.0 * (summary.runtime_ms_per_1000_mean / base_runtime - 1.0)
    summary.to_csv(OUTPUT_DIR / "computational_cost_for_paper.csv", index=False)
    print(f"Computational table written to {OUTPUT_DIR / 'computational_cost_for_paper.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
