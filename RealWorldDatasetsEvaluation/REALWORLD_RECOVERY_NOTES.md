# Real-world matrix recovery fix

## Failures found

- UCIAQD: 200 missing combinations because dataset loading failed before method rows were written.
- GASD + RLS: SCCM produced non-finite predictions.
- GASD + Widrow-Hoff: BASE and eight detector baselines diverged numerically.
- WSSF + RLS: BASE, SCCM, SSPT, and OHL variants produced non-finite predictions.

## Fixes

1. UCIAQD now uses a robust local chronological loader that supports both date orders, semicolon/decimal-comma files, `-200` missing values, causal forward filling, and training-segment-only scaling.
2. GASD/WSSF RLS use near-unity forgetting factors and conservative ranges shared across all RLS methods for the affected dataset.
3. GASD Widrow-Hoff uses fixed feature clipping and a conservative learning-rate range shared across all methods.
4. RLS and Widrow-Hoff common updates reject only non-finite/divergent numerical updates.
5. SCCM Widrow-Hoff clips tuned learning rates to the same predefined bounds as the competing variants.
6. Recovery reruns remove old rows first, preventing duplicates.

## Run

From the repository root:

```bat
python RealWorldDatasetsEvaluation\apply_model_patches.py
python RealWorldDatasetsEvaluation\006_recover_incomplete_runs.py
```

The recovery script reruns 350 combinations: UCIAQD full matrix plus all methods for GASD/RLS, GASD/Widrow-Hoff, and WSSF/RLS. It then regenerates all aggregate, statistical, paper, and reviewer files.

Final check:

```text
Expected runs: 1600
Complete runs: 1600
Failed or incomplete records: 0
Missing expected combinations: 0
```
