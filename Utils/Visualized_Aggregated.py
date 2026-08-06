import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_visualized_aggregated(model_name, model_name_star, data, save_path, kpi, legend_loc):
    # =========================
    # Global Publication Settings
    # =========================
    plt.rcParams.update({
        "font.family": "serif",       # Matches standard academic serif fonts (e.g., Times New Roman)
        "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
        "font.size": 10,              # Standard journal font size
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,            # Higher display resolution
        "savefig.dpi": 600,           # Production-quality export resolution
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })



    df = pd.DataFrame(data)

    # =========================
    # Plot settings
    # =========================
    x = [-3, -2, -1, 0, 1, 2, 3]
    x_labels = [r"$t_d-3$", r"$t_d-2$", r"$t_d-1$", r"$t_d$", r"$t_d+1$", r"$t_d+2$", r"$t_d+3$"]
    drift_cols = ["td_minus_3", "td_minus_2", "td_minus_1", "td", "td_plus_1", "td_plus_2", "td_plus_3"]

    colors = {
        model_name: "#1f77b4", # Strong Blue
        model_name_star: "#ff7f0e", # Strong Orange
        f"{model_name}-KS": "#2ca02c", f"{model_name}-KO": "#d62728",
        f"{model_name}-AS": "#9467bd", f"{model_name}-AO": "#8c564b",
        f"{model_name}-KW": "#e377c2", f"{model_name}-AR": "#7f7f7f",
        f"{model_name}-AW": "#bcbd22", f"{model_name}-KR": "#17becf",
    }

    # =========================
    # Create figure
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5)) # Slightly more compact for single-column papers

    for _, row in df.iterrows():
        method = row["Method"]
        y_vals = row[drift_cols].values

        if method == model_name:
            ax.plot(x, y_vals, marker="s", lw=2.5, ms=6, color=colors[method], label=method, zorder=10)
        elif method == model_name_star:
            ax.plot(x, y_vals, marker="*", lw=2.5, ms=9, color=colors[method], label=method, zorder=9)
        else:
            ax.plot(x, y_vals, marker="o", lw=1.2, ms=4, alpha=0.6, color=colors[method], label=method, zorder=3)

    # =========================
    # Drift point line & Styling
    # =========================
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1.2)
    ax.text(0.1, 0.55, "Drift point", transform=ax.get_xaxis_transform(),
            rotation=90, va="bottom", fontsize=11, fontweight="bold", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(kpi)

    # Remove top and right spines for modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # =========================
    # Custom legend
    # =========================
    legend_order = [model_name, model_name_star] + [m for m in df["Method"] if m not in [model_name, model_name_star]]
    handles, labels = ax.get_legend_handles_labels()
    dict_handles = dict(zip(labels, handles))
    ordered_handles = [dict_handles[m] for m in legend_order]

    ax.legend(
        ordered_handles, legend_order,
        loc=legend_loc, ncol=2, frameon=True,
        edgecolor="black", framealpha=1
    )

    plt.tight_layout()

    # =========================
    # High-Res Export
    # =========================
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # PDF is vector-based (lossless) - best for LaTeX
    plt.savefig(str(save_path)+".pdf", bbox_inches="tight")
    # PNG at 600 DPI - best for Word/PowerPoint
    plt.savefig(str(save_path)+".png", dpi=600, bbox_inches="tight")

    plt.show()