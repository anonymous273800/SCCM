import ast
import numpy as np
import pandas as pd
from pathlib import Path


def save_aggregate_results(df, gen_path, filename="aggregated_results.csv"):
    agg_dir = Path(gen_path) / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    output_file = agg_dir / filename
    df.to_csv(output_file, index=False)

    print(f"Saved to: {output_file}")
    return output_file


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def clean_values(x):
    return [v for v in x if v is not None and not np.isnan(v)]


def safe_mean(x):
    x = clean_values(x)
    return float(np.mean(x)) if x else np.nan


def safe_min(x):
    x = clean_values(x)
    return float(np.min(x)) if x else np.nan


def safe_max(x):
    x = clean_values(x)
    return float(np.max(x)) if x else np.nan


def get_report_idx(drift_location, report_interval):
    """
    Convert the drift location in data-point space
    into the corresponding index in the reported MSE array.

    Example:
        drift_location = 300
        report_interval = 10
        index = 30
    """
    return int(drift_location // report_interval)


def get_val(arr, idx):
    if idx < 0 or idx >= len(arr):
        return np.nan
    return arr[idx]


def get_incremental_drift_locations(dataset_name):
    """
    Incremental drift locations:

    IDS01, IDS02, IDS03:
        drift occurs every 100 points:
        [100, 200, 300, ..., 900]

    IDS04, IDS05, IDS06:
        drift occurs every 200 points:
        [200, 400, 600, ..., 1800]
    """

    if dataset_name in ["GDS01", "GDS02", "GDS03"]:
        return list(range(100, 1000, 100))

    elif dataset_name in ["GDS04", "GDS05", "GDS06"]:
        return list(range(200, 2000, 200))

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# =========================
# 1) READ FILES
# =========================

gen_path = r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\001 Synthetic\003 Gradual\003-RLS"

expr1 = read_file(gen_path + r"\001-RLS-GDS01\001-RLS-GDS01\GDS01_gradual_exp1_RLS_expr_data.txt")
expr2 = read_file(gen_path + r"\002-RLS-GDS02\002-RLS-GDS02\GDS02_gradual_exp2_RLS_expr_data.txt")
expr3 = read_file(gen_path + r"\003-RLS-GDS03\003-RLS-GDS03\GDS03_gradual_exp3_RLS_expr_data.txt")
expr4 = read_file(gen_path + r"\004-RLS-GDS04\004-RLS-GDS04\GDS04_gradual_exp4_RLS_expr_data.txt")
expr5 = read_file(gen_path + r"\005-RLS-GDS05\005-RLS-GDS05\GDS05_gradual_exp5_RLS_expr_data.txt")
expr6 = read_file(gen_path + r"\006-RLS-GDS06\006-RLS-GDS06\GDS06_gradual_exp6_RLS_expr_data.txt")

experiments = [expr1, expr2, expr3, expr4, expr5, expr6]


# =========================
# 2) AGGREGATION
# =========================

methods_data = {}

offsets = [-3, -2, -1, 0, 1, 2, 3]

for exp in experiments:
    dataset_name = exp["dataset_name"]

    # PA reports MSE every report_interval points.
    # If report_interval = 1, then each MSE entry corresponds to one data point.
    report_interval = exp.get("report_interval", 10)

    drift_locations = get_incremental_drift_locations(dataset_name)
    drift_indices = [get_report_idx(loc, report_interval) for loc in drift_locations]

    for method, metrics in exp["methods"].items():

        if method not in methods_data:
            methods_data[method] = {
                "MSE": [],
                "MSE_pos": [[] for _ in range(7)],
            }

        mse = metrics["MSE"]

        methods_data[method]["MSE"].extend(mse)

        for drift_idx in drift_indices:
            for i, off in enumerate(offsets):
                idx = drift_idx + off
                methods_data[method]["MSE_pos"][i].append(get_val(mse, idx))


# =========================
# 3) BUILD TABLE
# =========================

rows = []

labels = ["-3", "-2", "-1", "Drift", "+1", "+2", "+3"]

for method, data in methods_data.items():

    row = {
        "Method": method,

        "Avg_MSE": safe_mean(data["MSE"]),
        "Min_MSE": safe_min(data["MSE"]),
        "Max_MSE": safe_max(data["MSE"]),
    }

    for i, label in enumerate(labels):
        row[f"MSE_{label}"] = safe_mean(data["MSE_pos"][i])

    rows.append(row)


df = pd.DataFrame(rows)


# =========================
# 4) SORT BEST FIRST
# =========================

# For PA, lower MSE is better.
df = df.sort_values(
    ["Avg_MSE"],
    ascending=[True]
).reset_index(drop=True)


# =========================
# 5) REORDER COLUMNS
# =========================

df = df[
    [
        "Method",

        "Avg_MSE", "Min_MSE", "Max_MSE",

        "MSE_-3", "MSE_-2", "MSE_-1", "MSE_Drift",
        "MSE_+1", "MSE_+2", "MSE_+3",
    ]
]


# =========================
# 6) PRINT AND SAVE
# =========================

print(df.round(6))

save_aggregate_results(df, gen_path)