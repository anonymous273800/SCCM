# SCCM ablation and sensitivity analysis

The runner evaluates OLR-WA on representative abrupt (`ADS05`) and alternating-gradual (`GDS05`) streams using the five manuscript seeds. It includes the base learner, full SCCM, SCCM without bounded recalibration, SCCM without the safe band, conservative/sensitive rho settings, and KPI-window sizes 10, 20, and 30. The full setting uses z=1.5, safe band 0.005, and a four-entry local KPI window.

Run:

```bash
python AblationSensitivity/001_run_olrwa_ablation.py
```
