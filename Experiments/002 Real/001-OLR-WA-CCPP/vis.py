import ast
import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIGURATION
# =========================

DATASET_NAME = "CCPP"

# This should point to the averaged file that has 17 R2/MSE values per model
INPUT_FILE = (
    r"C:/PythonProjects/SCCM-StreamCruiseControlMethod/"
    r"Experiments/002 Real/001-OLR-WA/"
    r"Out-OLR-WA-Aggregated_Real_Datasets_Avg_Every_K.txt"
)

PLOTTING_DIR = (
    r"C:/PythonProjects/SCCM-StreamCruiseControlMethod/"
    r"Experiments/002 Real/001-OLR-WA-CCPP/plots"
)


# =========================
# FILE READER
# =========================

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


# =========================
# PLOTTER
# =========================

class Plotter:

    @staticmethod
    def plot_results_ten_models_real_datasets(
        x_axis,
        olr_wa_r2_avg,
        olr_wa_mse_avg,
        olr_wa_sccm_r2_avg,
        olr_wa_sccm_mse_avg,
        adwin_reset_r2_avg,
        adwin_reset_mse_avg,
        adwin_window_r2_avg,
        adwin_window_mse_avg,
        adwin_sspt_r2_avg,
        adwin_sspt_mse_avg,
        adwin_ohl_r2_avg,
        adwin_ohl_mse_avg,
        kswin_reset_r2_avg,
        kswin_reset_mse_avg,
        kswin_window_r2_avg,
        kswin_window_mse_avg,
        kswin_sspt_r2_avg,
        kswin_sspt_mse_avg,
        kswin_ohl_r2_avg,
        kswin_ohl_mse_avg,
        kpi,
        label1,
        label2,
        label3,
        label4,
        label5,
        label6,
        label7,
        label8,
        label9,
        label10,
        log_enabled=False,
        legend_loc="upper left",
        save_path=None
    ):

        if kpi == "R2":
            y_axis1 = olr_wa_r2_avg
            y_axis2 = olr_wa_sccm_r2_avg
            y_axis3 = adwin_reset_r2_avg
            y_axis4 = adwin_window_r2_avg
            y_axis5 = adwin_sspt_r2_avg
            y_axis6 = adwin_ohl_r2_avg
            y_axis7 = kswin_reset_r2_avg
            y_axis8 = kswin_window_r2_avg
            y_axis9 = kswin_sspt_r2_avg
            y_axis10 = kswin_ohl_r2_avg
        else:
            y_axis1 = olr_wa_mse_avg
            y_axis2 = olr_wa_sccm_mse_avg
            y_axis3 = adwin_reset_mse_avg
            y_axis4 = adwin_window_mse_avg
            y_axis5 = adwin_sspt_mse_avg
            y_axis6 = adwin_ohl_mse_avg
            y_axis7 = kswin_reset_mse_avg
            y_axis8 = kswin_window_mse_avg
            y_axis9 = kswin_sspt_mse_avg
            y_axis10 = kswin_ohl_mse_avg

        length = np.min([
            len(arr) for arr in [
                x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5,
                y_axis6, y_axis7, y_axis8, y_axis9, y_axis10
            ]
        ])

        x_axis = np.array(x_axis[:length])
        y_axis1 = np.array(y_axis1[:length], dtype=float)
        y_axis2 = np.array(y_axis2[:length], dtype=float)
        y_axis3 = np.array(y_axis3[:length], dtype=float)
        y_axis4 = np.array(y_axis4[:length], dtype=float)
        y_axis5 = np.array(y_axis5[:length], dtype=float)
        y_axis6 = np.array(y_axis6[:length], dtype=float)
        y_axis7 = np.array(y_axis7[:length], dtype=float)
        y_axis8 = np.array(y_axis8[:length], dtype=float)
        y_axis9 = np.array(y_axis9[:length], dtype=float)
        y_axis10 = np.array(y_axis10[:length], dtype=float)

        if log_enabled:
            y_axis1 = np.log(y_axis1 + 1)
            y_axis2 = np.log(y_axis2 + 1)
            y_axis3 = np.log(y_axis3 + 1)
            y_axis4 = np.log(y_axis4 + 1)
            y_axis5 = np.log(y_axis5 + 1)
            y_axis6 = np.log(y_axis6 + 1)
            y_axis7 = np.log(y_axis7 + 1)
            y_axis8 = np.log(y_axis8 + 1)
            y_axis9 = np.log(y_axis9 + 1)
            y_axis10 = np.log(y_axis10 + 1)

        plt.figure(figsize=(10, 6))

        line1, = plt.plot(
            x_axis, y_axis1,
            linestyle="-", marker="o",
            markersize=2.5, linewidth=0.9,
            label=label1, color="#4c72b0"
        )

        line2, = plt.plot(
            x_axis, y_axis2,
            linestyle="-", marker="^",
            markersize=2.5, linewidth=1.1,
            label=label2, color="#dd8452"
        )

        line3, = plt.plot(
            x_axis, y_axis3,
            linestyle="--", marker="s",
            markersize=2.2, linewidth=0.9,
            label=label3, color="#55a868"
        )

        line4, = plt.plot(
            x_axis, y_axis4,
            linestyle=":", marker="D",
            markersize=2.2, linewidth=0.9,
            label=label4, color="#8172b3"
        )

        line5, = plt.plot(
            x_axis, y_axis5,
            linestyle=(0, (3, 1, 1, 1)), marker="P",
            markersize=2.3, linewidth=0.9,
            label=label5, color="#c44e52"
        )

        line6, = plt.plot(
            x_axis, y_axis6,
            linestyle=(0, (1, 1)), marker="*",
            markersize=3.0, linewidth=0.9,
            label=label6, color="#64b5cd"
        )

        line7, = plt.plot(
            x_axis, y_axis7,
            linestyle="-.", marker="v",
            markersize=2.3, linewidth=0.9,
            label=label7, color="#2a9d8f"
        )

        line8, = plt.plot(
            x_axis, y_axis8,
            linestyle=(0, (5, 1)), marker="X",
            markersize=2.3, linewidth=0.9,
            label=label8, color="#8c6d31"
        )

        line9, = plt.plot(
            x_axis, y_axis9,
            linestyle=(0, (3, 1, 1, 1, 1, 1)), marker="h",
            markersize=2.3, linewidth=0.9,
            label=label9, color="#e17c05"
        )

        line10, = plt.plot(
            x_axis, y_axis10,
            linestyle=(0, (7, 2)), marker="<",
            markersize=2.3, linewidth=0.9,
            label=label10, color="#76b7b2"
        )

        plt.xlabel("$N$", fontsize=8)

        if kpi == "R2":
            plt.ylabel("R$^2$", fontsize=8)
        elif kpi == "MSE":
            plt.ylabel("MSE", fontsize=8)

        plt.title(f"{DATASET_NAME} Performance Comparison", fontsize=8)

        plt.tick_params(axis="x", labelsize=7)
        plt.tick_params(axis="y", labelsize=7)

        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.grid(True, alpha=0.6)

        plt.legend(
            handles=[
                line1, line2, line3, line4, line5,
                line6, line7, line8, line9, line10
            ],
            fontsize=6,
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.15,
            facecolor="white",
            edgecolor="black"
        )

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Saved plot to: {save_path}")

        plt.show()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    exp = read_file(INPUT_FILE)
    methods = exp["methods"]

    # =========================
    # SHORT LABELS
    # =========================
    # OLR-WA-ADWIN-RESET  -> OLR-WA-AR
    # OLR-WA-ADWIN-WINDOW -> OLR-WA-AW
    # OLR-WA-ADWIN-SSPT   -> OLR-WA-AS
    # OLR-WA-ADWIN-OHL    -> OLR-WA-AO
    # OLR-WA-KSWIN-RESET  -> OLR-WA-KR
    # OLR-WA-KSWIN-WINDOW -> OLR-WA-KW
    # OLR-WA-KSWIN-SSPT   -> OLR-WA-KS
    # OLR-WA-KSWIN-OHL    -> OLR-WA-KO

    labels = [
        "OLR-WA",
        "OLR-WA$^*$",
        "OLR-WA-AR",
        "OLR-WA-AW",
        "OLR-WA-AS",
        "OLR-WA-AO",
        "OLR-WA-KR",
        "OLR-WA-KW",
        "OLR-WA-KS",
        "OLR-WA-KO",
    ]

    # =========================
    # R2 SERIES
    # =========================

    olr_wa_r2_avg = methods["OLR-WA"]["R2"]
    olr_wa_sccm_r2_avg = methods["OLR-WA-SCCM"]["R2"]

    adwin_reset_r2_avg = methods["OLR-WA-ADWIN-RESET"]["R2"]
    adwin_window_r2_avg = methods["OLR-WA-ADWIN-WINDOW"]["R2"]
    adwin_sspt_r2_avg = methods["OLR-WA-ADWIN-SSPT"]["R2"]
    adwin_ohl_r2_avg = methods["OLR-WA-ADWIN-OHL"]["R2"]

    kswin_reset_r2_avg = methods["OLR-WA-KSWIN-RESET"]["R2"]
    kswin_window_r2_avg = methods["OLR-WA-KSWIN-WINDOW"]["R2"]
    kswin_sspt_r2_avg = methods["OLR-WA-KSWIN-SSPT"]["R2"]
    kswin_ohl_r2_avg = methods["OLR-WA-KSWIN-OHL"]["R2"]

    # =========================
    # MSE SERIES
    # =========================

    olr_wa_mse_avg = methods["OLR-WA"]["MSE"]
    olr_wa_sccm_mse_avg = methods["OLR-WA-SCCM"]["MSE"]

    adwin_reset_mse_avg = methods["OLR-WA-ADWIN-RESET"]["MSE"]
    adwin_window_mse_avg = methods["OLR-WA-ADWIN-WINDOW"]["MSE"]
    adwin_sspt_mse_avg = methods["OLR-WA-ADWIN-SSPT"]["MSE"]
    adwin_ohl_mse_avg = methods["OLR-WA-ADWIN-OHL"]["MSE"]

    kswin_reset_mse_avg = methods["OLR-WA-KSWIN-RESET"]["MSE"]
    kswin_window_mse_avg = methods["OLR-WA-KSWIN-WINDOW"]["MSE"]
    kswin_sspt_mse_avg = methods["OLR-WA-KSWIN-SSPT"]["MSE"]
    kswin_ohl_mse_avg = methods["OLR-WA-KSWIN-OHL"]["MSE"]

    # Since every 10 readings were averaged, each model should have 17 values.
    x_axis = list(range(1, len(olr_wa_r2_avg) + 1))

    # =========================
    # R2 PLOT
    # =========================

    Plotter.plot_results_ten_models_real_datasets(
        x_axis,
        olr_wa_r2_avg,
        olr_wa_mse_avg,
        olr_wa_sccm_r2_avg,
        olr_wa_sccm_mse_avg,
        adwin_reset_r2_avg,
        adwin_reset_mse_avg,
        adwin_window_r2_avg,
        adwin_window_mse_avg,
        adwin_sspt_r2_avg,
        adwin_sspt_mse_avg,
        adwin_ohl_r2_avg,
        adwin_ohl_mse_avg,
        kswin_reset_r2_avg,
        kswin_reset_mse_avg,
        kswin_window_r2_avg,
        kswin_window_mse_avg,
        kswin_sspt_r2_avg,
        kswin_sspt_mse_avg,
        kswin_ohl_r2_avg,
        kswin_ohl_mse_avg,
        "R2",
        labels[0],
        labels[1],
        labels[2],
        labels[3],
        labels[4],
        labels[5],
        labels[6],
        labels[7],
        labels[8],
        labels[9],
        log_enabled=False,
        legend_loc="lower left",
        save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_R2_plot.pdf")
    )

    # =========================
    # MSE PLOT
    # =========================

    Plotter.plot_results_ten_models_real_datasets(
        x_axis,
        olr_wa_r2_avg,
        olr_wa_mse_avg,
        olr_wa_sccm_r2_avg,
        olr_wa_sccm_mse_avg,
        adwin_reset_r2_avg,
        adwin_reset_mse_avg,
        adwin_window_r2_avg,
        adwin_window_mse_avg,
        adwin_sspt_r2_avg,
        adwin_sspt_mse_avg,
        adwin_ohl_r2_avg,
        adwin_ohl_mse_avg,
        kswin_reset_r2_avg,
        kswin_reset_mse_avg,
        kswin_window_r2_avg,
        kswin_window_mse_avg,
        kswin_sspt_r2_avg,
        kswin_sspt_mse_avg,
        kswin_ohl_r2_avg,
        kswin_ohl_mse_avg,
        "MSE",
        labels[0],
        labels[1],
        labels[2],
        labels[3],
        labels[4],
        labels[5],
        labels[6],
        labels[7],
        labels[8],
        labels[9],
        log_enabled=False,
        legend_loc="upper left",
        save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_MSE_plot.pdf")
    )