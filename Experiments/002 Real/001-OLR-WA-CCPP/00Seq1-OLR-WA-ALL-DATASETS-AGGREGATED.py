from Utils import RealDataMultiFileAveraging
from pathlib import Path

if __name__ == "__main__":

    k = 10

    input_paths = [
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\001-OLR-WA-CCPP\00A-OLR-WA-CCPP-data.txt"
        ),
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\001-OLR-WA-1KC\00D-OLR-WA-1KC_expr_data.txt"
        ),
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\001-OLR-WA-KCHSD\00C-OLR-WA-KCHSD_expr_data.txt"
        ),
        Path(
            r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\001-OLR-WA-MCPD\00B-OLR-WA-MCPD_expr_data.txt"
        ),
    ]

    output_path = Path(
        r"C:\PythonProjects\SCCM-StreamCruiseControlMethod\Experiments\002 Real\001-OLR-WA\Out-OLR-WA-Aggregated_Real_Datasets_Avg_Every_K.txt"
    )

    RealDataMultiFileAveraging.aggregate_multiple_experiment_files(input_paths, output_path, k)