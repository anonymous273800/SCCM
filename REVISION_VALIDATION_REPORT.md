# SCCM revision validation report

## Static validation

- All Python files compile successfully with `python -m compileall`.
- The synthetic predictive runner discovers exactly **72** primary scripts: 4 regression models × 18 synthetic datasets.
- All 72 primary synthetic scripts save their five seed-level runs before averaging.
- The central evaluation seeds are `{0, 1, 42, 123, 7}`.
- No direct `river.drift` imports remain outside the compatibility layer.
- No Python constructor retains the former ADWIN `delta=0.1` or KSWIN `alpha=0.01`, `window_size=50`, or `stat_size=20` settings.
- All executable DriftDetectionQuality2 `quality_run.py` files use:
  - `tolerance_ratio = 0.05`
  - `cooldown_factor = 2.0`
  - `min_episode_size = 2`

## Functional smoke and fixture tests

The following code paths were executed successfully without using manuscript result values:

1. **Synthetic predictive statistics fixture**
   - complete 4-model × 18-dataset × 5-seed × 10-method input;
   - produced 108 comparisons;
   - every comparison used 30 dataset-seed pairs.

2. **Alarm-quality significance fixture**
   - produced six comparisons: SCCM versus ADWIN and SCCM versus KSWIN for each of three drift categories;
   - every comparison used 120 model-dataset-seed pairs.

3. **Episode first-trigger protocol**
   - an episode beginning before the true drift remained a false positive even when it contained a later post-drift trigger;
   - an episode whose first trigger occurred inside the post-drift window matched correctly.

4. **Real-world statistical fixtures**
   - written primary protocol produced four SCCM-versus-standalone tests with eight dataset-level pairs;
   - explicitly labeled legacy table-reproduction protocol produced eight comparisons with 40 dataset-seed pairs.

5. **Ablation smoke test**
   - one ADS05 seed executed the base model and all eight SCCM variants successfully.

6. **Computational-cost aggregation fixture**
   - produced the standalone, SCCM, and eight baseline rows;
   - every summary row contained five seed-level measurements.

## Full-run limitations in this delivered archive

- Seven external real-world datasets were not included in the supplied archive and cannot be reconstructed from the code. Their required locations are listed in `DATASETS_REQUIRED.md`.
- The current packaging environment did not have `scikit-multiflow` or `river` installed, so the complete detector-based experiments were not rerun here. `requirements.txt` supplies both the preferred backend and fallback.
- Existing manuscript numerical tables were not overwritten. The revised scripts generate fresh outputs after the user runs the complete experiment matrix.

## Manuscript inconsistency retained transparently

The real-world manuscript text specifies eight dataset-level Wilcoxon pairs, but its reported p-values are too small to arise from `n=8`. The package therefore provides both:

- `paired_tests.py`: the written eight-dataset primary protocol;
- `paired_tests_seed_level_reproduction.py`: the explicitly labeled 40 dataset-seed legacy calculation.

See `RealWorldDatasetsEvaluation/003_RealWorldStatistics/STATISTICAL_PROTOCOL_NOTE.md` before updating the paper.
