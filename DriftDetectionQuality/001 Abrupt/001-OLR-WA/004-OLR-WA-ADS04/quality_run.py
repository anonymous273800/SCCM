from pathlib import Path
import sys
from Hyperparameters import Hyperparameter

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE
while not (PROJECT_ROOT / "DriftDetectionQuality").is_dir():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not find project root.")
    PROJECT_ROOT = PROJECT_ROOT.parent

DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality"
for p in (str(PROJECT_ROOT), str(DDQ_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from DriftDetectionQuality.ddq_common import run_quality_experiment


# Edit only this block when tuning this one experiment.
CONFIG = {
    "model": 'OLR-WA',
    "dataset": 'ADS04',
    "seeds": [42], #[0,1,42,123,7],
    "train_percent": 90,
    "candidate_source": "long_term",
    "tolerance_ratio": 0.05,
    "cooldown_factor": 2.0,
    "min_episode_size": 2,
    "kpi": 'R2',
    "multiplier": 2,
    "sccm_window_size": 10,
    "used_kpi_window_size": 4,
    "report_interval": 1,
    "increment_user_value":50,# Hyperparameter.olr_wa_increment_size(n_features=10, user_defined_val=10)
}


if __name__ == "__main__":
    run_quality_experiment(CONFIG, script_file=__file__)
