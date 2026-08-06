from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def evaluation_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    root = evaluation_root().parent
    required = [root / "Models", root / "Datasets", root / "Utils"]
    if not all(p.exists() for p in required):
        raise RuntimeError(
            "RealWorldDatasetsEvaluation must be placed directly inside the "
            "SCCM-StreamCruiseControlMethod repository root."
        )
    return root


def ensure_project_importable() -> Path:
    root = project_root()
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def results_dir(name: str) -> Path:
    path = evaluation_root() / "Results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
