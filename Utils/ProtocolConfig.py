"""Canonical protocol values used by the manuscript experiments.

Keep detector parameters and evaluation seeds in one place so synthetic,
real-world, alarm-quality, and statistical pipelines cannot silently diverge.
"""
from __future__ import annotations

EVALUATION_SEEDS = (0, 1, 42, 123, 7)

# Documented scikit-multiflow defaults used in the manuscript.
ADWIN_DELTA = 0.002
KSWIN_ALPHA = 0.005
KSWIN_WINDOW_SIZE = 100
KSWIN_STAT_SIZE = 30

# Alarm alignment protocol.
ALARM_TOLERANCE_RATIO = 0.05
ALARM_COOLDOWN_FACTOR = 2.0
ALARM_MIN_EPISODE_SIZE = 2

# SCCM deployment settings reported in the manuscript.
SCCM_Z = 1.5
SCCM_RHO = 0.06680720126885809  # 1 - Phi(1.5)
SCCM_SAFE_BAND = 0.005
