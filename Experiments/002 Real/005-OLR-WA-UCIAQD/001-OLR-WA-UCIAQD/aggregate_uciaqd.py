import ast
import numpy as np
import pandas as pd
from pathlib import Path


def save_aggregate_results(
        df,
        gen_path,
        filename="aggregated_uciaqd_results.csv"
):
    """
    Save the aggregated CSV in the same folder as this script.
    """

    output_file = Path(gen_path) / filename

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Saved to: {output_file}")

    return output_file


def read_file(path):
    """
    Read the experiment result dictionary from a text file.
    """

    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


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

    return float(np.mean(values)) if values else np.nan


def safe_median(values):
    values = clean_values(values)

    return float(np.median(values)) if values else np.nan


def safe_min(values):
    values = clean_values(values)

    return float(np.min(values)) if values else np.nan


def safe_max(values):
    values = clean_values(values)

    return float(np.max(values)) if values else np.nan


def safe_final(values):
    values = clean_values(values)

    return float(values[-1]) if values else np.nan


def find_experiment_file(gen_path):
    """
    Find the UCIAQD experiment result file.

    First, try the expected filename. If it does not exist,
    search for one file ending with '_expr_data.txt'.
    """

    expected_file = (
        Path(gen_path)
        / "001-OLR-WA-UCIAQD_expr_data.txt"
    )

    if expected_file.is_file():
        return expected_file

    matching_files = list(
        Path(gen_path).glob("*_expr_data.txt")
    )

    if len(matching_files) == 1:
        return matching_files[0]

    if len(matching_files) == 0:
        raise FileNotFoundError(
            "No experiment result file ending with "
            "'_expr_data.txt' was found in:\n"
            f"{gen_path}"
        )

    raise RuntimeError(
        "More than one experiment result file was found:\n"
        + "\n".join(str(path) for path in matching_files)
    )


# =========================
# CONFIGURATION
# =========================

gen_path = Path(
    r"C:\New\003\SCCM-StreamCruiseControlMethod"
    r"\Experiments\002 Real\005-OLR-WA-UCIAQD"
    r"\001-OLR-WA-UCIAQD"
)

# False:
# Include the base-model R2 and MSE values.
#
# True:
# Remove the first value and aggregate only online mini-batches.
EXCLUDE_BASE_MODEL = False


# =========================
# 1) READ EXPERIMENT FILE
# =========================

expr1_path = find_experiment_file(
    gen_path
)

print(f"Reading from: {expr1_path}")

expr1 = read_file(
    expr1_path
)

experiments = [
    expr1
]


# =========================
# 2) COLLECT METHOD VALUES
# =========================

methods_data = {}

for experiment in experiments:

    if "methods" not in experiment:
        raise KeyError(
            "The experiment result does not contain "
            "a 'methods' dictionary."
        )

    for method, metrics in experiment["methods"].items():

        if method not in methods_data:
            methods_data[method] = {
                "R2": [],
                "MSE": []
            }

        r2_values = list(
            metrics.get("R2", [])
        )

        mse_values = list(
            metrics.get("MSE", [])
        )

        if EXCLUDE_BASE_MODEL:
            r2_values = r2_values[1:]
            mse_values = mse_values[1:]

        methods_data[method]["R2"].extend(
            r2_values
        )

        methods_data[method]["MSE"].extend(
            mse_values
        )


# =========================
# 3) BUILD RESULTS TABLE
# =========================

rows = []

for method, data in methods_data.items():

    r2_values = data["R2"]
    mse_values = data["MSE"]

    row = {
        "Method": method,

        "N_R2": len(
            clean_values(r2_values)
        ),

        "Avg_R2": safe_mean(
            r2_values
        ),

        "Median_R2": safe_median(
            r2_values
        ),

        "Min_R2": safe_min(
            r2_values
        ),

        "Max_R2": safe_max(
            r2_values
        ),

        "Final_R2": safe_final(
            r2_values
        ),

        "N_MSE": len(
            clean_values(mse_values)
        ),

        "Avg_MSE": safe_mean(
            mse_values
        ),

        "Median_MSE": safe_median(
            mse_values
        ),

        "Min_MSE": safe_min(
            mse_values
        ),

        "Max_MSE": safe_max(
            mse_values
        ),

        "Final_MSE": safe_final(
            mse_values
        )
    }

    rows.append(row)


df = pd.DataFrame(
    rows
)


# =========================
# 4) SORT RESULTS
# =========================

# Lowest average MSE is best.
# If two methods have the same MSE, higher R2 is preferred.
df = df.sort_values(
    by=[
        "Avg_MSE",
        "Avg_R2"
    ],
    ascending=[
        True,
        False
    ]
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
print("UCIAQD AGGREGATED RESULTS")
print("=" * 160)

print(
    df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

print("=" * 160)


# =========================
# 6) SAVE CSV
# =========================

save_aggregate_results(
    df=df,
    gen_path=gen_path,
    filename="aggregated_uciaqd_results.csv"
)