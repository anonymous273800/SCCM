import ast
import numpy as np
from pathlib import Path


def read_experiment_file(input_path):
    """
    Reads one experiment .txt file.
    The file should contain a Python dictionary.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def average_every_k_values(values, k=10):
    """
    Average every k consecutive values.

    Example:
    170 values with k=10 -> 17 averaged values.

    Incomplete final chunks are ignored.
    """
    values = np.array(values, dtype=float)

    averaged_values = []

    for i in range(0, len(values), k):
        chunk = values[i:i + k]

        # Use only complete chunks of size k
        if len(chunk) == k:
            averaged_values.append(float(np.nanmean(chunk)))

    return averaged_values


def aggregate_multiple_experiment_files(input_paths, output_path, k=10):
    """
    Reads multiple experiment files and creates one output file.

    For each method:
    - R2 values from all files are concatenated
    - MSE values from all files are concatenated
    - every k values are averaged
    """

    aggregated_exp = {
        "methods": {}
    }

    for input_path in input_paths:
        exp = read_experiment_file(input_path)

        for method, metrics in exp["methods"].items():

            if method not in aggregated_exp["methods"]:
                aggregated_exp["methods"][method] = {
                    "R2": [],
                    "MSE": []
                }

            if "R2" in metrics:
                aggregated_exp["methods"][method]["R2"].extend(metrics["R2"])

            if "MSE" in metrics:
                aggregated_exp["methods"][method]["MSE"].extend(metrics["MSE"])

    # Average every k values after concatenating all files
    for method, metrics in aggregated_exp["methods"].items():

        if len(metrics["R2"]) > 0:
            metrics["R2"] = average_every_k_values(metrics["R2"], k)
        else:
            metrics.pop("R2")

        if len(metrics["MSE"]) > 0:
            metrics["MSE"] = average_every_k_values(metrics["MSE"], k)
        else:
            metrics.pop("MSE")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(aggregated_exp))

    print(f"Saved new aggregated file to: {output_path}")

    # Verification
    print("\nVerification:")
    for method, metrics in aggregated_exp["methods"].items():
        print(
            method,
            "R2 length:", len(metrics.get("R2", [])),
            "MSE length:", len(metrics.get("MSE", []))
        )


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    k = 10

    input_paths = [
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\001-OLR-WA-1KC\001-1KC_realdataset_exp1_expr_data.txt"
        ),
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\002-PA-MCPD\002-PA-MCPD\002-MCPD_realdataset_exp2_expr_data.txt"
        ),
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\003-RLS-KCHSD\003-RLS-KCHSD\003-KCHSD_realdataset_exp3_expr_data.txt"
        ),
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\004-WidrowHoff-1KC\004-WH-1KC\004-WH_realdataset_exp4_expr_data.txt"
        ),
    ]

    output_path = Path(
        r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\Aggregated_Real_Datasets_avg_every_10.txt"
    )

    aggregate_multiple_experiment_files(input_paths, output_path, k)