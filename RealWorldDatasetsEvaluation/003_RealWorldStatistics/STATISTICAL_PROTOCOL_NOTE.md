# Real-world Wilcoxon protocol note

The manuscript text and its reported real-world significance table do not describe the same analysis unit.

## Written primary protocol

The text says to:

1. calculate the SCCM-minus-comparator difference for each of five seeds;
2. average those differences within each real-world dataset; and
3. apply Wilcoxon to the resulting eight dataset-level differences.

`paired_tests.py` implements this protocol. It produces `n=8` paired observations per regression model and applies Holm correction across the four SCCM-versus-standalone comparisons.

## Existing table values

The table reports adjusted p-values near `10^-7`. Such p-values cannot arise from a two-sided Wilcoxon signed-rank test with only eight nonzero pairs. Even the most extreme possible result has:

- exact two-sided p-value: `0.0078125`;
- normal approximation with continuity correction: approximately `0.01427`;
- Holm adjustment: equal or larger.

The small reported p-values are consistent with using all `8 datasets × 5 seeds = 40` dataset-seed differences as paired observations. `paired_tests_seed_level_reproduction.py` implements that legacy calculation and labels it as repeated-run/table-reproduction analysis, not independent dataset-level inference.

## Required manuscript decision

Choose one consistent presentation before resubmission:

- **Dataset-level inference:** retain the written `n=8` protocol and replace the table p-values with the output of `paired_tests.py`.
- **Repeated-run analysis:** retain the legacy table-style p-values, change the manuscript to state `n=40` model-dataset-seed paired observations, and explicitly note that observations sharing a dataset are not independent experimental units.

The revised code keeps both outputs so no calculation is hidden or silently redefined.
