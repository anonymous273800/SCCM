# Benchmark Detection and Activation Analysis

This directory evaluates the eight ADWIN/KSWIN detector-adaptation baselines under the same synthetic alarm-quality basis used by the SCCM evaluation.

## Evaluation seeds

All baseline experiments are run using the same five final evaluation seeds:

```text
0, 1, 42, 123, 7
```

The seed stored in the original SCCM quality configuration is not used to limit the benchmark run. The quality configuration supplies the model and episode-alignment settings, while `EVALUATION_SEEDS` in `benchmark_detection_common.py` controls the final repeated evaluation.

## Consistency rules

For every model-dataset pair, the runner uses:

- the same 18 `Datasets.Synthetic2` datasets;
- the same 90% chronological training stream;
- the same base-model settings from the corresponding `quality_run.py` file;
- the same per-experiment `tolerance_ratio`;
- the same per-experiment `cooldown_factor`;
- the same per-experiment `min_episode_size`;
- fixed-boundary candidate-episode construction;
- the episode's first trigger as its alarm time;
- minimum episode-size filtering for the secondary episode analysis;
- chronological one-to-one drift matching;
- all original ADWIN, KSWIN, RESET, WINDOW, SSPT, and OHL settings from the original experiment scripts.

Using the first trigger as the episode alarm time prevents a pre-drift episode from becoming a true positive because of a later post-drift trigger.

## Base-model overrides

The runner temporarily applies the corresponding SCCM quality configuration without modifying the original experiment files. This includes:

- OLR-WA `increment_user_value`;
- PA `C`, epsilon, bounds, and report interval;
- RLS lambda, delta, bounds, and report interval;
- Widrow-Hoff learning rate and report interval.

All temporary patches are restored after each experiment.

## Execution

Run from the project root:

```bash
python BenchmarkDetectionActivation/000_validate_setup.py
python BenchmarkDetectionActivation/001_run_all_benchmarks.py
python BenchmarkDetectionActivation/002_aggregate_and_align.py
```

The expected baseline workload is:

```text
72 experiment scripts x 8 detector-adaptation baselines x 5 seeds = 2,880 runs
```

The validation script writes:

```text
BenchmarkDetectionActivation/results/benchmark_exact_configuration_audit.csv
```

## Main baseline result files

- `benchmark_exact_summary.txt`
- `benchmark_exact_by_dataset.csv`
- `benchmark_exact_by_model_drift_method_seed.csv`
- `benchmark_exact_by_model_drift_method.csv`
- `benchmark_exact_by_model_drift_detector.csv`
- `benchmark_exact_drift_point_details.csv`
- `benchmark_exact_failures.csv`

The file required for the paired statistical test is:

```text
BenchmarkDetectionActivation/results/benchmark_exact_by_dataset.csv
```

It preserves one row for every:

```text
model x dataset x seed x detector-adaptation baseline
```

## Paired Wilcoxon significance test

The significance script requires SciPy:

```bash
python -m pip install scipy
```

After the five-seed SCCM alarm-quality experiment has produced a per-model, per-dataset, per-seed CSV, run:

```bash
python BenchmarkDetectionActivation/003_alarm_quality_paired_significance.py \
  --sccm-csv /path/to/latest_sccm_alarm_quality_by_dataset_seed.csv
```

The SCCM input CSV must contain:

```text
model, dataset, seed
```

and either:

```text
f1
```

or:

```text
tp, fp, fn
```

The script accepts repeated `--sccm-csv` arguments when the four regression models are stored in separate files.

It produces:

- `alarm_quality_paired_wilcoxon.csv`
- `alarm_quality_paired_wilcoxon.tex`
- `alarm_quality_paired_wilcoxon_summary.txt`

The analysis performs 24 two-sided paired Wilcoxon signed-rank comparisons:

```text
3 drift categories x 8 detector-adaptation baselines
```

Each comparison uses 120 paired observations:

```text
4 models x 6 datasets x 5 seeds
```

Holm correction is applied across the eight baseline comparisons within each drift category. Rank-biserial correlation is reported with positive values favoring SCCM.

## Five-seed final aggregate table

After generating the baseline CSV and the detailed five-seed SCCM alarm CSV, run:

```bash
python BenchmarkDetectionActivation/004_generate_final_aggregate_comparison.py \
  --sccm-csv /path/to/latest_sccm_alarm_quality_by_dataset_seed.csv
```

This script reads both five-seed inputs directly. It does not contain hardcoded SCCM results and reports conventional detection delay without subtracting a processing increment.
