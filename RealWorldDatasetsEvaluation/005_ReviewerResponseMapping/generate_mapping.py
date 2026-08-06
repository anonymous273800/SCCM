from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from RealWorldDatasetsEvaluation.common.project import results_dir


def main() -> int:
    rows = [
        {
            "reviewer_concern":"Real-world evaluation used selected model-dataset pairings.",
            "new_evidence":"All four regression models are evaluated on all eight real-world datasets with BASE, SCCM, and eight detector-adaptation baselines over five seeds.",
            "result_file":"Results/paper/paper_full_realworld_matrix.csv",
            "paper_action":"Replace selected-pairing language with the complete 4x8 matrix and discuss cross-model consistency."
        },
        {
            "reviewer_concern":"Uncertainty and paired evidence were insufficient.",
            "new_evidence":"Mean and standard deviation across five seeds plus paired SCCM-versus-method Wilcoxon tests, Holm correction, and directional effect sizes.",
            "result_file":"Results/statistics/paired_sccm_vs_methods.csv",
            "paper_action":"Report mean ± SD and the adjusted paired comparisons."
        },
        {
            "reviewer_concern":"Computational and adaptation activity costs were unclear.",
            "new_evidence":"Runtime, sampled process RSS, detector detections, adaptations, recalibrations, and interventions per 1,000 processed samples.",
            "result_file":"Results/paper/paper_efficiency_and_interventions.csv",
            "paper_action":"Add an efficiency and normalized-intervention table."
        },
        {
            "reviewer_concern":"Real datasets do not provide ground-truth drift labels.",
            "new_evidence":"The evaluation explicitly avoids TP, FP, FN, alarm F1, and delay on real streams.",
            "result_file":"Results/paper/README_PAPER_RESULTS.txt",
            "paper_action":"State that real streams assess predictive robustness and cost, not alarm correctness."
        },
    ]
    out = results_dir("reviewer")
    pd.DataFrame(rows).to_csv(out / "reviewer4_realworld_evidence_mapping.csv", index=False)
    markdown = ["# Reviewer 4 real-world evidence mapping", ""]
    for i, row in enumerate(rows, 1):
        markdown.extend([f"## {i}. {row['reviewer_concern']}", row["new_evidence"], f"Evidence: `{row['result_file']}`", f"Paper action: {row['paper_action']}", ""])
    (out / "reviewer4_realworld_evidence_mapping.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Reviewer mapping: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
