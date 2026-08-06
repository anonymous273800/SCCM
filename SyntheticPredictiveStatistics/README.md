# Synthetic predictive protocol

1. Run `python SyntheticPredictiveStatistics/001_run_all_synthetic_predictive.py`.
2. Every dataset script writes its unaveraged five-seed trajectories to `results/predictive_seed_runs/`.
3. `002_predictive_paired_statistics.py` constructs 30 paired observations per model and drift category (6 datasets × 5 seeds).
4. SCCM is compared separately with the standalone learner and with all eight detector-adaptation baselines. The eight baseline p-values are Holm-adjusted within each model and drift category. Rank-biserial correlation is computed from signed ranks.
