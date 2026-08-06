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
    return [
        v for v in x
        if v is not None and not np.isnan(v)
    ]


def safe_mean(x):
    x = clean_values(x)
    return float(np.mean(x)) if x else np.nan


def safe_min(x):
    x = clean_values(x)
    return float(np.min(x)) if x else np.nan


def safe_max(x):
    x = clean_values(x)
    return float(np.max(x)) if x else np.nan


# =========================
# 1) READ FILES
# =========================

gen_path = Path(
    r"C:/PythonProjects/SCCM-StreamCruiseControlMethod/Experiments/002 Real/003-RLS-KCHSD"
)

expr1_path = gen_path / "003-RLS-KCHSD" / "003-KCHSD_realdataset_exp3_RLS_expr_data.txt"

expr1 = read_file(expr1_path)

experiments = [expr1]


# =========================
# 2) AGGREGATION
# =========================

methods_data = {}

for exp in experiments:

    for method, metrics in exp["methods"].items():

        if method not in methods_data:
            methods_data[method] = {
                "MSE": []
            }

        mse = metrics.get("MSE", [])

        methods_data[method]["MSE"].extend(mse)


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

    rows.append(row)


df = pd.DataFrame(rows)


# =========================
# 4) SORT BY LOWEST AVG MSE
# =========================

df = df.sort_values(
    by="Avg_MSE",
    ascending=True
).reset_index(drop=True)


# =========================
# 5) PRINT AND SAVE
# =========================

print(df.round(6))

save_aggregate_results(df, gen_path)