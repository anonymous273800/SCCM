# Revision changelog

## Protocol fixes

1. Changed `Utils.Constants.SEEDS` from two seeds to the five manuscript seeds.
2. Replaced inconsistent detector settings across synthetic, alarm, and real-world code with the manuscript values.
3. Added a detector compatibility layer that prefers scikit-multiflow.
4. Added per-seed predictive record persistence to all 72 synthetic model-dataset scripts.
5. Added the complete synthetic paired statistical protocol.
6. Reimplemented the real-world statistical protocol at the dataset level.
7. Corrected episode matching to use the episode's first trigger.
8. Collapsed duplicated alarm-quality baselines into ADWIN and KSWIN detector-family comparisons.
9. Added the missing SCCM ablation and sensitivity experiment.
10. Added the complete five-seed computational-cost pipeline.
11. Added both the written n=8 real-world protocol and an explicitly labeled n=40 legacy table-reproduction analysis because the manuscript text and reported p-values are internally inconsistent.
12. Added requirements, dataset placement instructions, validation improvements, and a top-level runner.

## Data limitation

Seven external real-world datasets were not present in the supplied archive and therefore could not be redistributed in this revision. The exact required paths and dataset identities are documented in `DATASETS_REQUIRED.md`.
