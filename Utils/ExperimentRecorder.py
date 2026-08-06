"""Write per-seed predictive results before any cross-seed averaging."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_seed_runs(
    *,
    experiment_name: str,
    dataset_name: str,
    drift_type: str,
    runs: Iterable[dict[str, Any]],
) -> Path:
    """Persist one JSON file containing all five seed-level method trajectories."""
    output_dir = _project_root() / "results" / "predictive_seed_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset_name}_{experiment_name}_seed_runs.json"
    payload = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "drift_type": drift_type,
        "runs": _clean(list(runs)),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved per-seed predictive records to: {path}")
    return path
