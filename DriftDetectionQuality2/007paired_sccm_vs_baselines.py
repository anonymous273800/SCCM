from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    corrected_script = root / "008fixed_protocol_analysis.py"
    spec = importlib.util.spec_from_file_location(
        "ddq2_fixed_protocol_analysis", corrected_script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {corrected_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(
        "007paired_sccm_vs_baselines.py now delegates to "
        "008fixed_protocol_analysis.py so the corrected episode-to-episode "
        "protocol is always used."
    )
    module.main()


if __name__ == "__main__":
    main()
