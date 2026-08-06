# Computational cost reproduction

1. Run the SCCM and baseline resource-instrumented study:

```bash
python DriftDetectionQuality2/001run_all_quality_experiments.py --skip-sensitivity
```

2. Measure the four standalone learners on the same 18 datasets and five seeds:

```bash
python ComputationalCost/001_run_standalone_resources.py
```

3. Generate the pooled five-seed table:

```bash
python ComputationalCost/002_generate_computational_table.py
```

Runtime is pooled by total processed samples within each seed. Peak memory is the maximum sampled RSS increase among the model-dataset runs within that seed. The final table reports mean and standard deviation across five seeds.
