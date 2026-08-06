import ast
import numpy as np
from pathlib import Path


def read_experiment_file(input_path):
    """
    Reads the original experiment .txt file.
    The file should contain a Python dictionary.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def average_every_k_values(values, k=10):
    """
    Convert a list like 170 values into 17 values
    by averaging every k consecutive values.
    """
    values = np.array(values, dtype=float)

    averaged_values = []

    for i in range(0, len(values), k):
        chunk = values[i:i + k]

        # Use only complete chunks of size k
        if len(chunk) == k:
            averaged_values.append(float(np.nanmean(chunk)))

    return averaged_values


def aggregate_experiment_file(input_path, output_path, k=10):
    """
    Reads the original experiment file and creates a new file
    where every k R2/MSE values are averaged.
    """

    exp = read_experiment_file(input_path)

    for method, metrics in exp["methods"].items():

        if "R2" in metrics:
            metrics["R2"] = average_every_k_values(metrics["R2"], k)

        if "MSE" in metrics:
            metrics["MSE"] = average_every_k_values(metrics["MSE"], k)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(exp))

    print(f"Saved new aggregated file to: {output_path}")

    # Optional verification
    for method, metrics in exp["methods"].items():
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

    input_path = Path(
        r"C:/PythonProjects/SCCM-StreamCruiseControlMethod/Experiments/002 Real/001-OLR-WA-CCPP/001-OLR-WA-CCPP/001-CCPP_realdataset_exp1_expr_data.txt"
    )

    output_path = input_path.with_name(
        input_path.stem + f"_avg_every_{k}" + input_path.suffix
    )

    aggregate_experiment_file(input_path, output_path, k)