# RealWorldDatasetsEvaluation

This directory is a separate real-world evaluation pipeline. Place it directly inside the repository root. It reads the repository's existing dataset loaders and model implementations but does not edit them.

Expected location:

```text
C:\New\004\SCCM-StreamCruiseControlMethod\RealWorldDatasetsEvaluation
```

## Exact matrix

- 4 models: OLR-WA, PA, RLS, Widrow-Hoff
- 8 real datasets: CCPP, MCPD, KCHSD, 1KC, UCIAQD, GASD, CalCOFI, WSSF
- 10 methods: BASE, SCCM, and eight ADWIN/KSWIN adaptation baselines
- 5 seeds: 0, 1, 42, 123, 7
- Total expected method runs: 1,600

## Recommended one-command run

From the repository root:

```bat
cd /d C:\New\004\SCCM-StreamCruiseControlMethod
python RealWorldDatasetsEvaluation\001run_all_real_world_experiments.py --continue-on-error
```

This master script runs the following stages automatically:

1. `000_validate_setup.py`
2. `001_RealWorldFullMatrix\run_full_matrix.py`
3. `002_RealWorldAggregation\aggregate_results.py`
4. `003_RealWorldStatistics\paired_tests.py`
5. `004_PaperReadyResults\generate_paper_results.py`
6. `005_ReviewerResponseMapping\generate_mapping.py`

## Manual sequence

Use this sequence only when running the stages separately or resuming after a stopped stage:

```bat
python RealWorldDatasetsEvaluation\000_validate_setup.py
python RealWorldDatasetsEvaluation\001_RealWorldFullMatrix\run_full_matrix.py --continue-on-error
python RealWorldDatasetsEvaluation\002_RealWorldAggregation\aggregate_results.py
python RealWorldDatasetsEvaluation\003_RealWorldStatistics\paired_tests.py
python RealWorldDatasetsEvaluation\004_PaperReadyResults\generate_paper_results.py
python RealWorldDatasetsEvaluation\005_ReviewerResponseMapping\generate_mapping.py
```

The matrix runner resumes combinations already recorded as complete. A subset can be run with:

```bat
python RealWorldDatasetsEvaluation\001_RealWorldFullMatrix\run_full_matrix.py --datasets CCPP,MCPD --models PA,RLS --seeds 0 --continue-on-error
```

## Output locations

- `Results/raw/realworld_seed_level.csv`: one row per method, dataset, model, and seed
- `Results/aggregated/realworld_run_completeness.csv`: checks all 1,600 expected combinations
- `Results/aggregated/realworld_by_dataset_mean_std.csv`: five-seed mean and standard deviation
- `Results/statistics/paired_sccm_vs_methods.csv`: paired Wilcoxon tests, Holm correction, and directional effect sizes
- `Results/paper/`: compact manuscript-ready result files
- `Results/reviewer/`: reviewer concern-to-evidence mapping

## Important interpretation

The real datasets do not contain ground-truth drift locations. This pipeline therefore does not calculate TP, FP, FN, alarm precision, recall, F1, or detection delay. It reports predictive performance, runtime, sampled process memory, detector/adaptation activity, and interventions per 1,000 processed samples.

The `report_interval` controls only how frequently pointwise-model predictive metrics are stored. PA, RLS, and Widrow-Hoff still update on every observation.


## Recover the incomplete July run

Apply the numerical/model patches once, then rerun only the affected matrix:

```bat
python RealWorldDatasetsEvaluation\apply_model_patches.py
python RealWorldDatasetsEvaluation\006_recover_incomplete_runs.py
```

The recovery runner backs up and replaces affected raw rows rather than appending duplicates. See `REALWORLD_RECOVERY_NOTES.md`.
