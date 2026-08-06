import ast
import numpy as np
import pandas as pd
from pathlib import Path


def save_aggregate_results(
        df,
        gen_path,
        filename="aggregated_results.csv"
):
    agg_dir = Path(gen_path) / "aggregate"

    agg_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = agg_dir / filename

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved to: {output_file}"
    )

    return output_file


def read_file(path):
    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        return ast.literal_eval(
            file.read()
        )


def clean_values(values):
    cleaned_values = []

    for value in values:

        if value is None:
            continue

        try:
            value = float(value)

        except (
            TypeError,
            ValueError
        ):
            continue

        if np.isfinite(value):
            cleaned_values.append(
                value
            )

    return cleaned_values


def safe_mean(values):
    values = clean_values(
        values
    )

    return (
        float(np.mean(values))
        if values
        else np.nan
    )


def safe_min(values):
    values = clean_values(
        values
    )

    return (
        float(np.min(values))
        if values
        else np.nan
    )


def safe_max(values):
    values = clean_values(
        values
    )

    return (
        float(np.max(values))
        if values
        else np.nan
    )


# =========================
# 1) READ FILES
# =========================

gen_path = Path(
    r"C:/New/003/SCCM-StreamCruiseControlMethod/"
    r"Experiments/002 Real/007-RLS-CALCOFI"
)

expr1_path = (
    gen_path
    / "007-RLS-CalCOFI"
    / "007-CalCOFI_realdataset_exp7_RLS_expr_data.txt"
)

if not expr1_path.is_file():
    raise FileNotFoundError(
        "The experiment result file was not found:\n"
        f"{expr1_path}"
    )

print(
    f"Reading from: {expr1_path}"
)

expr1 = read_file(
    expr1_path
)

experiments = [
    expr1
]


# =========================
# 2) AGGREGATION
# =========================

methods_data = {}

for experiment in experiments:

    for method, metrics in experiment["methods"].items():

        if method not in methods_data:

            methods_data[method] = {
                "MSE": []
            }

        mse_values = metrics.get(
            "MSE",
            []
        )

        methods_data[method]["MSE"].extend(
            mse_values
        )


# =========================
# 3) BUILD TABLE
# =========================

rows = []

for method, data in methods_data.items():

    cleaned_mse = clean_values(
        data["MSE"]
    )

    row = {
        "Method": method,

        "N_MSE": len(
            cleaned_mse
        ),

        "Avg_MSE": safe_mean(
            data["MSE"]
        ),

        "Min_MSE": safe_min(
            data["MSE"]
        ),

        "Max_MSE": safe_max(
            data["MSE"]
        )
    }

    rows.append(
        row
    )


df = pd.DataFrame(
    rows
)


# =========================
# 4) SORT BY LOWEST AVG MSE
# =========================

df = df.sort_values(
    by="Avg_MSE",
    ascending=True,
    na_position="last"
).reset_index(
    drop=True
)

df.insert(
    0,
    "Rank",
    range(
        1,
        len(df) + 1
    )
)


# =========================
# 5) PRINT AND SAVE
# =========================

print()
print(
    "RLS CALCOFI AGGREGATED RESULTS"
)

print(
    "=" * 90
)

print(
    df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

print(
    "=" * 90
)

save_aggregate_results(
    df=df,
    gen_path=gen_path,
    filename="aggregated_results.csv"
)