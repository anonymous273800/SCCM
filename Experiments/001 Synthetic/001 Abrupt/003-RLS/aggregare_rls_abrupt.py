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


def safe_mean(x):
    valid_values = [v for v in x if not np.isnan(v)]
    if len(valid_values) == 0:
        return np.nan
    return float(np.mean(valid_values))


def safe_min(x):
    valid_values = [v for v in x if not np.isnan(v)]
    if len(valid_values) == 0:
        return np.nan
    return float(np.min(valid_values))


def safe_max(x):
    valid_values = [v for v in x if not np.isnan(v)]
    if len(valid_values) == 0:
        return np.nan
    return float(np.max(valid_values))


def get_idx(drift_location, increment_size):
    return int(drift_location // increment_size)


def get_val(arr, idx):
    if idx < 0 or idx >= len(arr):
        return np.nan
    return arr[idx]


# =========================
# 1) READ FILES
# =========================

gen_path = r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\001 Synthetic\001 Abrupt\003-RLS"

expr1 = read_file(gen_path + r"\001-RLS-DS01\001-RLS-ADS01\ADS01_abrupt_exp1_RLS_expr_data.txt")
expr2 = read_file(gen_path + r"\002-RLS-DS02\002-RLS-ADS02\ADS02_abrupt_exp2_RLS_expr_data.txt")
expr3 = read_file(gen_path + r"\003-RLS-DS03\003-RLS-ADS03\ADS03_abrupt_exp3_RLS_expr_data.txt")
expr4 = read_file(gen_path + r"\004-RLS-DS04\004-RLS-ADS04\ADS04_abrupt_exp4_RLS_expr_data.txt")
expr5 = read_file(gen_path + r"\005-RLS-DS05\005-RLS-ADS05\ADS05_abrupt_exp5_RLS_expr_data.txt")
expr6 = read_file(gen_path + r"\006-RLS-DS06\006-RLS-ADS06\ADS06_abrupt_exp6_RLS_expr_data.txt")

experiments = [expr1, expr2, expr3, expr4, expr5, expr6]


# =========================
# 2) AGGREGATION
# =========================

methods_data = {}

for exp in experiments:
    drift_loc = exp["drift_location"]
    inc = exp["increment_size"]
    drift_idx = get_idx(drift_loc, inc)

    for method, metrics in exp["methods"].items():

        if method not in methods_data:
            methods_data[method] = {
                "MSE": [],
                "MSE_pos": [[] for _ in range(7)]  # [-3, -2, -1, 0, +1, +2, +3]
            }

        mse = metrics["MSE"]

        methods_data[method]["MSE"].extend(mse)

        offsets = [-3, -2, -1, 0, 1, 2, 3]

        for i, off in enumerate(offsets):
            idx = drift_idx + off
            methods_data[method]["MSE_pos"][i].append(get_val(mse, idx))


# =========================
# 3) BUILD TABLE
# =========================

rows = []

for method, data in methods_data.items():

    row = {
        "Method": method,

        "Avg_MSE": safe_mean(data["MSE"]),
        "Min_MSE": safe_min(data["MSE"]),
        "Max_MSE": safe_max(data["MSE"]),
    }

    labels = ["-3", "-2", "-1", "Drift", "+1", "+2", "+3"]

    for i, label in enumerate(labels):
        row[f"MSE_{label}"] = safe_mean(data["MSE_pos"][i])

    rows.append(row)


df = pd.DataFrame(rows)


# =========================
# 4) SORT
# =========================
# For MSE, lower is better.
# So this sorts best methods first.

df = df.sort_values(
    ["Avg_MSE"],
    ascending=[True]
).reset_index(drop=True)


print(df.round(6))


# =========================
# 5) SAVE
# =========================

save_aggregate_results(df, gen_path)