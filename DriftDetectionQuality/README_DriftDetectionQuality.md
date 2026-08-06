# DriftDetectionQuality

This folder contains standalone drift-detection quality experiments.

Important rule:

- No original SCCM model, dataset, or utility file is edited.
- The code here runs the DriftQuality SCCM model copies and parses their console output. The original SCCM model files are not edited.

## Structure

Each model + dataset folder contains a small `quality_run.py` file.

Example:

```bash
python "DriftDetectionQuality/001 Abrupt/001-OLR-WA/001-OLR-WA-ADS01/quality_run.py"
```

Each `quality_run.py` runs:

- one model
- one dataset
- all 5 seeds by default

Outputs are saved beside the script in:

```text
quality_outputs/
```

## Edit configuration per experiment

Open the specific `quality_run.py` and edit only the `CONFIG` block.

Useful fields:

```python
"multiplier": 3.0,
"cooldown_factor": 2.0,
"min_episode_size": 1,
"tolerance_ratio": 0.05,
"sccm_window_size": 4,
"used_kpi_window_size": 4,
"candidate_source": "long_term",
```

This lets each model + dataset have its own configuration.

## Aggregate results

After running individual experiments, run:

```bash
python "DriftDetectionQuality/aggregate_quality_results.py"
```

It creates:

```text
DriftDetectionQuality/AggregatedQualityResults/alarm_quality_by_dataset.csv
DriftDetectionQuality/AggregatedQualityResults/alarm_quality_for_paper.csv
DriftDetectionQuality/AggregatedQualityResults/*.png
```

## Run everything

Only after testing several individual scripts, you can run:

```bash
python "DriftDetectionQuality/run_all_quality_experiments.py"
```

This runs all generated `quality_run.py` scripts and then aggregates the results.


## Configurable SCCM short-term window

For drift-quality tuning, each `quality_run.py` can now set:

```python
"sccm_window_size": 4,       # try 4, 10, 15, 20, 30
"used_kpi_window_size": 4,   # original SCCM KPI window length
```

This is passed from `quality_run.py` -> `ddq_common.py` -> the new `*_DriftQuality.py` model files.

The original files remain untouched. New files added:

```text
ConceptDriftManager/ConceptDriftDetector/ConceptDriftDetector_DriftQuality.py
Models/OLR_WA/OLR_WA_SCCM_DriftQuality.py
Models/PA/PA_SCCM_DriftQuality.py
Models/RLS/RLS_SCCM_DriftQuality.py
Models/WidrowHoff/WidrowHoff_SCCM_DriftQuality.py
```


### Note about `sccm_window_size` and `used_kpi_window_size`

In this DriftQuality version, `CONFIG["sccm_window_size"]` controls when SCCM starts checking for short-term drift. `CONFIG["used_kpi_window_size"]` controls how many recent KPI values are used in the short-term KPI window. To reproduce the previous best ADS01 behavior, use `sccm_window_size = 30` and `used_kpi_window_size = 4`.
