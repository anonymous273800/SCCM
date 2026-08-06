# SCCM Manuscript-to-Code Experiment Coverage Audit

## Scope and method

This is a static coverage and reproducibility audit of:

- Manuscript: `Pasted text(920).txt`
- Code archive: `SCCM-StreamCruiseControlMethod(6).zip`

I mapped the manuscript's experimental claims to runners, configurations, statistical-analysis scripts, data files, and archived results. I also compiled all Python files and ran the included setup validators where possible. I did not rerun the complete experiment matrix because the archive is missing required dependencies and most real-world raw datasets.

## Overall verdict

**The core experiment matrix is largely represented in the code, but the archive does not currently reproduce all experiments and protocols claimed in the manuscript.** Several issues are critical because they change the reported experimental design or statistical conclusions.

## Coverage matrix

| Manuscript component | Status | Audit finding |
|---|---:|---|
| Four learners: OLR-WA, PA, RLS, LMS | Covered | Implementations and synthetic/real runners are present. |
| Base + SCCM + eight detector-adaptation baselines | Covered structurally | The expected method matrix is represented. |
| 18 synthetic datasets | Covered structurally | Six abrupt, six incremental, and six alternating-gradual scripts exist for each learner. |
| Eight real-world datasets × four learners × ten methods × five seeds | Partially covered | The matrix is defined as 1,600 runs, but seven datasets' raw files are absent and `river` is not installed/pinned. |
| Five-seed synthetic predictive evaluation | **Not matched** | Main synthetic predictive scripts use `Constants.SEEDS = [42, 0]`, not the five manuscript seeds. |
| Fixed ADWIN/KSWIN configurations | **Not matched** | Code commonly uses ADWIN `0.1`; RLS/LMS use KSWIN `0.01/50/20`; manuscript specifies `0.002` and `0.005/100/30`. Code also uses `river`, while manuscript identifies scikit-multiflow. |
| Synthetic predictive Wilcoxon/Holm/effect-size tests | **Missing** | No matching analysis script was found, and main synthetic outputs save seed-averaged traces rather than the per-seed observations needed for the tests. |
| Real-world paired statistical protocol | **Not matched** | Existing script tests 40 dataset-seed pairs and all methods, uses a sign-count statistic labeled rank-biserial, and does not implement the manuscript's eight dataset-level confirmatory comparisons. |
| Raw alarm evaluation | Covered | Raw-event matching and standard alarm metrics are represented. |
| Episode-level alarm protocol | **Conflicting implementations** | One pipeline correctly uses the episode's first trigger; another allows any later trigger in the episode to match a drift. The latter contradicts the manuscript. |
| Alarm-quality Wilcoxon tests | **Not matched** | Existing scripts compare all eight detector-adaptation variants; manuscript requires two nonduplicated detector-family comparisons, ADWIN and KSWIN. |
| Alarm-protocol sensitivity | Partially covered | Tolerance, cooldown, and minimum episode size are swept, but the active sensitivity implementation uses the conflicting any-trigger matching rule. |
| SCCM component ablation | **Missing** | No complete runner was found for no recalibration, no safe band, rho sweep, and KPI-window sweep on representative OLR-WA streams. |
| Runtime, peak memory, intervention rates | Partially covered | Instrumentation exists, but the exact manuscript table is not generated from complete archived five-seed raw measurements; no complete result set is included. |
| Dataset drift-characterization statistics | Covered structurally | Utilities exist for synthetic and real first-half/second-half distribution diagnostics; most real raw files are absent. |
| Reproducible environment | **Missing** | No `requirements.txt`, `pyproject.toml`, `environment.yml`, or equivalent dependency lock was found. |

## Critical discrepancies

### 1. Synthetic predictive experiments run only two seeds

- `Utils/Constants.py:10`: `SEEDS = [42, 0]`
- `Utils/Constants.py:12`: the intended five-seed list exists separately as `SEEDS5 = [0, 1, 42, 123, 7]`.
- Main synthetic scripts reference `Constants.SEEDS`.
- Representative scripts aggregate seed results before saving them, so the archive does not retain the paired per-seed observations required for the manuscript's predictive significance tests.

**Required correction:** use one canonical five-seed constant everywhere and save one row per model, dataset, method, seed, and metric before aggregation.

### 2. Baseline detector settings do not match the manuscript

Examples:

- `RealWorldDatasetsEvaluation/config.py:39,53,67,80`: ADWIN delta is `0.1`.
- `RealWorldDatasetsEvaluation/config.py:68-70,81-83`: RLS and LMS use KSWIN `alpha=0.01`, `window=50`, `stat=20`.
- The model classes and synthetic runners also commonly default to ADWIN `0.1`.
- Baseline implementations import `river.drift`, not scikit-multiflow.

**Required correction:** either change all code to the manuscript's fixed parameters/library or revise the manuscript to accurately describe the implementation actually used. A single central detector configuration should be imported by every runner.

### 3. Synthetic predictive statistical analysis is absent

The manuscript requires, per model and drift category:

- 30 paired observations: six datasets × five seeds;
- SCCM versus standalone, plus SCCM versus each of eight baselines;
- two-sided Wilcoxon signed-rank;
- normal approximation with continuity correction;
- zero differences excluded;
- Holm correction across the eight baseline comparisons;
- true rank-biserial correlation.

No code implementing this complete protocol was found. Existing Wilcoxon code targets alarm quality or real-world results.

### 4. Real-world statistical script implements a different protocol

`RealWorldDatasetsEvaluation/003_RealWorldStatistics/paired_tests.py`:

- pairs at `dataset + seed`, resulting in 40 pairs for all eight datasets rather than first averaging the five seeds within each dataset;
- tests SCCM against every method, whereas the manuscript defines SCCM-versus-standalone as confirmatory and the strongest observed baseline as descriptive;
- applies Holm within each model/metric over all methods, rather than across four prespecified standalone-learner hypotheses;
- computes `(positive_count - negative_count) / n` and labels it rank-biserial, but this is a directional sign statistic, not Wilcoxon rank-biserial correlation;
- does not explicitly request the manuscript's normal approximation and continuity correction;
- uses raw MSE orientation instead of the manuscript's dataset-level relative MSE reduction for PA, RLS, and LMS.

### 5. Two alarm-episode definitions coexist

Correct implementation:

- `BenchmarkDetectionActivation/002_aggregate_and_align.py:112-157`
- It uses the episode's first alarm as the episode time and explicitly prevents a later trigger from rescuing a pre-drift episode.

Conflicting implementation:

- `DriftDetectionQuality2/ddq_common.py:679-695,750-767,802`
- It allows an episode to match when any member falls in the post-drift window and calculates delay from that later member.

**Required correction:** consolidate all alarm aggregation and sensitivity code on the fixed-first-trigger implementation.

### 6. Alarm-quality statistical comparison is broader than the manuscript

- `BenchmarkDetectionActivation/003_alarm_quality_paired_significance.py:37-46` defines all eight ADWIN/KSWIN × RESET/WINDOW/SSPT/OHL comparators.
- The manuscript states that alarm sequences are duplicated within a detector family and therefore compares SCCM only with one ADWIN result and one KSWIN result, applying Holm over two comparisons per drift category.

**Required correction:** deduplicate detector families before testing and generate exactly six tests: three drift categories × two detector families.

### 7. SCCM component ablation is not implemented

The manuscript describes:

- base learner;
- full SCCM;
- SCCM without recalibration;
- SCCM without the safe band;
- several sensitivity values for rho;
- several KPI-window sizes;
- representative OLR-WA abrupt and alternating-gradual streams.

The archive contains alarm-evaluation-parameter sensitivity, but no complete internal SCCM component-ablation runner and result generator matching this design.

### 8. Real-world archive is not self-contained

The included validator identifies the intended 1,600-run matrix, but execution is blocked by:

- missing `river` dependency;
- missing raw files for CCPP, MCPD, KCHSD, 1KC, UCIAQD, CalCOFI, and WSSF;
- absence of a pinned environment/dependency file.

Only GASD raw files appear to be included. When datasets cannot legally be redistributed, add a download/preparation script, expected filenames, checksums, and source instructions.

## What is already good

- All Python files pass a syntax compilation check.
- The core model/method/dataset directory structure is comprehensive.
- The benchmark validator recognizes 72 synthetic scripts, eight baselines per script, 72 SCCM alarm configurations, and five alarm seeds.
- The real-world configuration explicitly defines the complete 1,600-run matrix.
- Runtime and RSS instrumentation exists.
- A correct fixed-first-trigger episode matcher exists and can become the canonical implementation.

## Recommended repair order

1. **Freeze one authoritative experiment specification:** seeds, detector library, detector settings, adaptation settings, and metric orientation.
2. **Change synthetic predictive runners to five seeds and persist seed-level tidy CSVs.**
3. **Centralize ADWIN/KSWIN settings and align them with the manuscript.**
4. **Replace the real-world statistics script with the exact eight-dataset protocol.**
5. **Add the missing synthetic predictive Wilcoxon/Holm/rank-biserial analysis.**
6. **Delete or deprecate the any-trigger episode matcher; reuse the fixed-first-trigger matcher everywhere.**
7. **Change alarm significance to two detector-family comparisons per drift category.**
8. **Implement the SCCM component-ablation runner and paper-table generator.**
9. **Add a computational-cost merger/table generator using complete five-seed raw measurements, including standalone runs.**
10. **Add a pinned environment and dataset acquisition/preparation documentation.**
11. **Add a final audit script that asserts expected counts and parameters before paper results are generated.**

## Suggested final expected-count checks

- Synthetic predictive raw runs: `4 models × 18 datasets × 10 methods × 5 seeds = 3,600`.
- Real-world raw runs: `4 × 8 × 10 × 5 = 1,600`.
- Synthetic predictive statistical pairs: `30` per model × drift × comparator.
- Alarm statistical pairs: `120` per drift × detector-family comparison.
- Alarm significance rows: `3 drift types × 2 detector families = 6`.
- Real-world confirmatory tests: `4` SCCM-versus-standalone tests, each based on `8` dataset-level paired differences.

