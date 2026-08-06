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
    return float(np.mean([v for v in x if not np.isnan(v)]))


def safe_min(x):
    return float(np.min(x))


def safe_max(x):
    return float(np.max(x))





def get_val(arr, idx):
    if idx < 0 or idx >= len(arr):
        return np.nan
    return arr[idx]


# =========================
# 1) READ FILES
# =========================
gen_path = r"C:/PythonProjects/SCCM-StreamCruiseControlMethod/Experiments/002 Real/001-OLR-WA-CCPP"
expr1 = read_file(gen_path + r"\001-OLR-WA-CCPP\001-CCPP_realdataset_exp1_expr_data.txt")

experiments = [expr1]


# =========================
# 2) AGGREGATION
# =========================

methods_data = {}

for exp in experiments:
    drift_loc = exp["drift_location"]
    inc = exp["increment_size"]


    for method, metrics in exp["methods"].items():

        if method not in methods_data:
            methods_data[method] = {
                "R2": [],
                "MSE": [],
                "R2_pos": [[] for _ in range(7)],   # [-3,-2,-1,0,+1,+2,+3]
                "MSE_pos": [[] for _ in range(7)]
            }

        r2 = metrics["R2"]
        mse = metrics["MSE"]

        methods_data[method]["R2"].extend(r2)
        methods_data[method]["MSE"].extend(mse)





# =========================
# 3) BUILD TABLE
# =========================

rows = []

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

    # labels = ["-3", "-2", "-1", "Drift", "+1", "+2", "+3"]
    #
    # for i, label in enumerate(labels):
    #     row[f"R2_{label}"] = safe_mean(data["R2_pos"][i])
    #     row[f"MSE_{label}"] = safe_mean(data["MSE_pos"][i])

    rows.append(row)


df = pd.DataFrame(rows)

# =========================
# 4) SORT (worst first)
# =========================

df = df.sort_values(
    ["Avg_R2", "Avg_MSE"],
    ascending=[False, True]
).reset_index(drop=True)

print(df.round(6))

# optional save
save_aggregate_results(df, gen_path)

