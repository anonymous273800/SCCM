# SCCM revised reproducibility package

This revision closes the experiment-coverage gaps identified during the manuscript-to-code audit.

## Corrections included

- All synthetic predictive scripts now use the five evaluation seeds `{0, 1, 42, 123, 7}`.
- Every synthetic predictive run is saved before cross-seed averaging.
- ADWIN uses `delta=0.002`; KSWIN uses `alpha=0.005`, `window_size=100`, and `stat_size=30` throughout the project.
- `Utils/DriftDetectors.py` prefers the scikit-multiflow reference implementation specified in the manuscript and explicitly reports when the River fallback is active.
- Synthetic predictive statistics use 30 paired dataset-seed observations per model and drift category, Wilcoxon signed-rank tests with continuity correction, Holm correction across all eight baselines, and signed-rank rank-biserial correlation.
- Real-world confirmatory statistics average the five seed differences within each dataset, use eight dataset-level observations, and apply Holm correction across the four SCCM-versus-standalone comparisons.
- Strongest real-world baseline comparisons are output separately as descriptive post hoc analyses.
- Because the manuscript text specifies 8 dataset-level Wilcoxon pairs while its reported p-values imply 40 dataset-seed pairs, both analyses are generated and explicitly labeled. See `RealWorldDatasetsEvaluation/003_RealWorldStatistics/STATISTICAL_PROTOCOL_NOTE.md`.
- Alarm episodes are matched by their first trigger only. A later trigger cannot rescue an episode that began before the true drift.
- Alarm-quality significance compares SCCM once with ADWIN and once with KSWIN, avoiding duplicated RESET/WINDOW/SSPT/OHL detector sequences.
- An executable OLR-WA SCCM ablation/sensitivity runner covers recalibration, safe band, rho/z sensitivity, and KPI-window size.
- Dependency and real-world data manifests are included.
- The DriftDetectionQuality2 resource pipeline uses the fixed alarm protocol `r_tol=0.05`, `c_cool=2.0`, and `m_ep=2`; its legacy duplicated eight-baseline significance stage is skipped by the master runner in favor of the detector-family analysis.
- A complete synthetic computational-cost pipeline records runtime and peak RSS over all 18 datasets, four models, and five seeds.

## Main commands

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run synthetic predictive experiments and paired statistics:

```bash
python SyntheticPredictiveStatistics/001_run_all_synthetic_predictive.py
```

Run alarm-quality experiments and statistics:

```bash
python BenchmarkDetectionActivation/001_run_all_benchmarks.py
python BenchmarkDetectionActivation/002_aggregate_and_align.py
python BenchmarkDetectionActivation/003_alarm_quality_paired_significance.py
python BenchmarkDetectionActivation/004_generate_final_aggregate_comparison.py
```

Run the ablation study:

```bash
python AblationSensitivity/001_run_olrwa_ablation.py
```

Run the complete synthetic computational-cost analysis:

```bash
python DriftDetectionQuality2/001run_all_quality_experiments.py --skip-sensitivity --skip-paired-statistics
python ComputationalCost/001_run_standalone_resources.py
python ComputationalCost/002_generate_computational_table.py
```

Run the real-world matrix after supplying the files listed in `DATASETS_REQUIRED.md`:

```bash
python RealWorldDatasetsEvaluation/001run_all_real_world_experiments.py
```

Run everything:

```bash
python run_manuscript_experiments.py
```

For a synthetic-only/code-level check when real files are not present:

```bash
python run_manuscript_experiments.py --skip-real-world
```

## Validation

See `REVISION_VALIDATION_REPORT.md` for the static checks, fixture tests, and the limitations of the delivered archive.
