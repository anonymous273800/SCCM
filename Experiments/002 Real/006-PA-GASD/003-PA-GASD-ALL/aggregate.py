import ast
import numpy as np
import pandas as pd
from pathlib import Path


def save_aggregate_results(
        df,
        gen_path,
        filename="aggregated_pa_gasd_results.csv"
):
    """
    Save the aggregated results inside the aggregate folder.
    """

    agg_dir = Path(gen_path) / "aggregate"
    agg_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        agg_dir
        / filename
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved to: {output_file}"
    )

    return output_file


def read_file(path):
    """
    Read the experiment result dictionary.
    """

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as f:

        return ast.literal_eval(
            f.read()
        )


def clean_values(values):
    """
    Remove None, NaN, infinite, and invalid values.
    """

    cleaned = []

    for value in values:

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            cleaned.append(value)

    return cleaned


def safe_mean(values):
    values = clean_values(values)

    return (
        float(np.mean(values))
        if values
        else np.nan
    )


def safe_min(values):
    values = clean_values(values)

    return (
        float(np.min(values))
        if values
        else np.nan
    )


def safe_max(values):
    values = clean_values(values)

    return (
        float(np.max(values))
        if values
        else np.nan
    )


# =========================
# CONFIGURATION
# =========================

gen_path = Path(
    r"C:\New\003\SCCM-StreamCruiseControlMethod"
    r"\Experiments\002 Real\006-PA-GASD"
    r"\003-PA-GASD-ALL"
)

expr1_path = (
    gen_path
    / "003-PA-GASD-ALL_expr_data.txt"
)


# =========================
# 1) READ FILES
# =========================

if not expr1_path.is_file():
    raise FileNotFoundError(
        "The PA-GASD experiment result file "
        "was not found:\n"
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

    if "methods" not in experiment:
        raise KeyError(
            "The experiment result does not "
            "contain a 'methods' dictionary."
        )

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

    mse_values = data["MSE"]

    row = {
        "Method": method,

        "N_MSE": len(
            clean_values(mse_values)
        ),

        "Avg_MSE": safe_mean(
            mse_values
        ),

        "Min_MSE": safe_min(
            mse_values
        ),

        "Max_MSE": safe_max(
            mse_values
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
    ascending=True
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
# 5) PRINT RESULTS
# =========================

print()
print(
    "PA-GASD AGGREGATED RESULTS"
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


# =========================
# 6) SAVE RESULTS
# =========================

save_aggregate_results(
    df=df,
    gen_path=gen_path,
    filename="aggregated_pa_gasd_results.csv"
)