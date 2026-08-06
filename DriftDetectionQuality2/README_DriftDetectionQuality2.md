# DriftDetectionQuality2

This directory is a separate replacement for the alarm-quality evaluation. The original `DriftDetectionQuality` directory is not modified.

## What this version addresses

1. SCCM and all eight ADWIN/KSWIN detector-adaptation baselines run with seeds `[0, 1, 42, 123, 7]`.
2. A retained episode is eligible when any alarm in that episode falls inside the post-drift tolerance window; delay uses the first in-window alarm.
3. SCCM raw trigger-level results are reported as a supplemental analysis.
4. Every raw ADWIN/KSWIN detection is treated as a primary baseline alarm because every detection triggers an adaptation.
5. Sensitivity results are recomputed from saved raw events without rerunning model training.
6. Conventional delay is reported in samples and model processing increments. Practical delay is retained only as supplemental output.
7. Each method run records wall-clock runtime, sampled process RSS, and intervention rates per 1,000 processed stream samples.
8. Paired Wilcoxon tests compare SCCM with each baseline using the same base model, dataset, and seed. Holm-adjusted p-values and rank-biserial effect sizes are reported.

## Required dependency

The existing ADWIN/KSWIN implementations import `river`. Install the project dependencies before running. This directory also uses `psutil` for sampled process-RSS measurements and `scipy` for Wilcoxon tests and confidence intervals.

A minimal dependency list is provided in:

```text
requirements_DriftDetectionQuality2.txt
```

## Validation

Run validation before starting the experiments:

```bash
python "DriftDetectionQuality2/000_validate_setup.py"
```

The validator confirms:

- 72 SCCM model-dataset configurations;
- five seeds in every configuration;
- 360 expected SCCM seed runs;
- 2,880 expected baseline method-seed runs;
- the corrected episode-window matching behavior;
- required Python dependencies.

## One complete run

Run the full evaluation once with:

```bash
python "DriftDetectionQuality2/001run_all_quality_experiments.py" --continue-on-error
```

The command executes:

1. validation;
2. all 360 SCCM seed runs;
3. all 2,880 ADWIN/KSWIN method-seed runs;
4. SCCM and baseline aggregation;
5. sensitivity post-processing;
6. paired SCCM-versus-baseline statistical comparisons.

The command returns a nonzero exit status when SCCM seeds, baseline runs, or analysis stages are incomplete.

## Alarm definitions

### SCCM episode-level primary analysis

For an episode `E`, the alarms eligible for drift `d` are:

```text
M(d, E) = {e in E : d <= e <= d + T}
```

The episode matches the drift when `M(d, E)` is nonempty. The alarm used for delay is:

```text
a(d, E) = min M(d, E)
```

For the example episode `[450, 490, 510]`, drift `500`, and tolerance window `[500, 550]`, trigger `510` is inside the window. The episode is therefore a true positive and its detection delay is `10` samples.

### SCCM raw-trigger supplemental analysis

Every selected SCCM candidate trigger is evaluated individually using chronological one-to-one matching. In the same example, triggers `450` and `490` are false positives, while `510` is a true positive with a 10-sample delay.

### ADWIN/KSWIN primary analysis

Every raw detector detection is evaluated directly as one alarm. Detector episode consolidation is generated only as a supplemental analysis.

## Delay measures

Primary delay outputs are:

- `mean_delay_samples`;
- `mean_delay_increments`;
- corresponding median fields where available.

The processing increment is:

- the OLR-WA mini-batch increment for OLR-WA;
- one sample for PA and RLS;
- the configured reporting or processing interval for Widrow-Hoff.

## Resource and intervention measures

Each SCCM or baseline method-seed run records:

- `runtime_seconds`;
- `runtime_per_1000_samples`;
- `rss_before_mb`;
- `peak_rss_mb`;
- `peak_rss_delta_mb`;
- `adaptations_per_1000`;
- `recalibrations_per_1000`;
- `interventions_per_1000`.

Memory is measured by sampling the process resident set size with `psutil`, which includes Python and native numerical-library allocations visible in process RSS.

## Sensitivity analysis

The default post-processing grid is:

```text
tolerance_ratio    = [0.025, 0.05, 0.10]
cooldown_factor    = [1.0, 2.0, 3.0]
min_episode_size   = [1, 2, 3]
```

It can be changed without rerunning the models:

```bash
python "DriftDetectionQuality2/006parameter_sensitivity.py" \
  --tolerances 0.025,0.05,0.10 \
  --cooldowns 1,2,3 \
  --min-episode-sizes 1,2,3
```

## Main output directories

### SCCM

```text
DriftDetectionQuality2/AggregatedQualityResults/
DriftDetectionQuality2/drift_point_aggregates/
```

### Baselines

```text
DriftDetectionQuality2/BaselineResults/raw/
DriftDetectionQuality2/BaselineResults/aggregated/
```

### Sensitivity

```text
DriftDetectionQuality2/SensitivityResults/
```

### Unified statistics

```text
DriftDetectionQuality2/UnifiedResults/
```

The corrected paired-test file is:

```text
FixedProtocolResults/paired_wilcoxon_fixed_protocol.csv
```

It contains two alarm comparison bases:

- `episode_to_episode_fixed_protocol`: SCCM episodes versus baseline detector episodes, with an episode matched when any member falls inside the post-drift window;
- `raw_to_raw_fixed_tolerance`: raw SCCM candidate triggers versus raw baseline detector alarms.

It also contains two pairing strategies:

- `dataset_seed` or `model_dataset_seed`;
- `seed_after_pooling_datasets`.

## Important interpretation

Do not combine SCCM episode-level and raw-trigger rows as though they were independent observations. They are two views of the same saved trigger sequence. The episode-level result is the SCCM primary analysis; the raw-trigger result is supplemental and provides a direct raw-alarm fairness check against ADWIN/KSWIN.
