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


def get_idx(drift_location, increment_size):
    return int(drift_location // increment_size)


def get_val(arr, idx):
    if idx < 0 or idx >= len(arr):
        return np.nan
    return arr[idx]


def get_incremental_drift_locations(dataset_name):
    """
    IDS01, IDS02, IDS03: drift every 100 points
    IDS04, IDS05, IDS06: drift every 200 points
    """

    if dataset_name in ["IDS01", "IDS02", "IDS03"]:
        return list(range(100, 1000, 100))

    elif dataset_name in ["IDS04", "IDS05", "IDS06"]:
        return list(range(200, 2000, 200))

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# =========================
# 1) READ FILES
# =========================

gen_path = r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\001 Synthetic\002 Incremental\001-OLR-WA"

expr1 = read_file(gen_path + r"\001-OLR-WA-IDS01\001-OLR-WA-IDS01\IDS01_incremental_exp1_expr_data.txt")
expr2 = read_file(gen_path + r"\002-OLR-WA-IDS02\002-OLR-WA-IDS02\IDS02_incremental_exp2_expr_data.txt")
expr3 = read_file(gen_path + r"\003-OLR-WA-IDS03\003-OLR-WA-IDS03\IDS03_incremental_exp3_expr_data.txt")
expr4 = read_file(gen_path + r"\004-OLR-WA-IDS04\004-OLR-WA-IDS04\IDS04_incremental_exp4_expr_data.txt")
expr5 = read_file(gen_path + r"\005-OLR-WA-IDS05\005-OLR-WA-IDS05\IDS05_incremental_exp5_expr_data.txt")
expr6 = read_file(gen_path + r"\006-OLR-WA-IDS06\006-OLR-WA-IDS06\IDS06_incremental_exp6_expr_data.txt")

experiments = [expr1, expr2, expr3, expr4, expr5, expr6]


# =========================
# 2) AGGREGATION
# =========================

methods_data = {}

offsets = [-3, -2, -1, 0, 1, 2, 3]

for exp in experiments:
    dataset_name = exp["dataset_name"]
    inc = exp["increment_size"]

    drift_locations = get_incremental_drift_locations(dataset_name)
    drift_indices = [get_idx(loc, inc) for loc in drift_locations]

    for method, metrics in exp["methods"].items():

        if method not in methods_data:
            methods_data[method] = {
                "R2": [],
                "MSE": [],
                "R2_pos": [[] for _ in range(7)],
                "MSE_pos": [[] for _ in range(7)],
            }

        r2 = metrics["R2"]
        mse = metrics["MSE"]

        methods_data[method]["R2"].extend(r2)
        methods_data[method]["MSE"].extend(mse)

        for drift_idx in drift_indices:
            for i, off in enumerate(offsets):
                idx = drift_idx + off
                methods_data[method]["R2_pos"][i].append(get_val(r2, idx))
                methods_data[method]["MSE_pos"][i].append(get_val(mse, idx))


# =========================
# 3) BUILD TABLE
# =========================

rows = []

labels = ["-3", "-2", "-1", "Drift", "+1", "+2", "+3"]

for method, data in methods_data.items():

    row = {
        "Method": method,

        "Avg_R2": safe_mean(data["R2"]),
        "Min_R2": safe_min(data["R2"]),
        "Max_R2": safe_max(data["R2"]),

        "Avg_MSE": safe_mean(data["MSE"]),
        "Min_MSE": safe_min(data["MSE"]),
        "Max_MSE": safe_max(data["MSE"]),
    }

    for i, label in enumerate(labels):
        row[f"R2_{label}"] = safe_mean(data["R2_pos"][i])
        row[f"MSE_{label}"] = safe_mean(data["MSE_pos"][i])

    rows.append(row)


df = pd.DataFrame(rows)


# =========================
# 4) SORT BEST FIRST
# =========================

df = df.sort_values(
    ["Avg_R2", "Avg_MSE"],
    ascending=[False, True]
).reset_index(drop=True)


# =========================
# 5) REORDER COLUMNS
# =========================

df = df[
    [
        "Method",

        "Avg_R2", "Min_R2", "Max_R2",
        "Avg_MSE", "Min_MSE", "Max_MSE",

        "R2_-3", "R2_-2", "R2_-1", "R2_Drift",
        "R2_+1", "R2_+2", "R2_+3",

        "MSE_-3", "MSE_-2", "MSE_-1", "MSE_Drift",
        "MSE_+1", "MSE_+2", "MSE_+3",
    ]
]


# =========================
# 6) PRINT AND SAVE
# =========================

print(df.round(6))

save_aggregate_results(df, gen_path)