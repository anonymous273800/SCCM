from Utils import Visualized_Aggregated as vis



if __name__ == "__main__":
    # =========================
    # Model name and Data
    # =========================
    model_name = "OLR-WA"
    model_name_star = rf"{model_name}$^{{\mathbf{{*}}}}$"
    save_path = "visualized/VIS_001_OLR_WA_Abrupt"
    kpi=r"$R^2$"
    legend_loc = "upper left"

    data = {
        "Method": [
            model_name_star, f"{model_name}-KS", f"{model_name}-KO", f"{model_name}-AS",
            f"{model_name}-AO", f"{model_name}-KW", f"{model_name}-AR", f"{model_name}-AW",
            f"{model_name}-KR", model_name,
        ],
        "td_minus_3": [0.912, 0.904, 0.900, 0.901, 0.900, 0.899, 0.899, 0.899, 0.899, 0.899],
        "td_minus_2": [0.922, 0.852, 0.850, 0.850, 0.849, 0.849, 0.849, 0.849, 0.849, 0.849],
        "td_minus_1": [0.932, 0.859, 0.844, 0.842, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841],
        "td": [0.928, 0.528, 0.342, 0.277, 0.273, 0.292, 0.272, 0.272, 0.300, 0.264],
        "td_plus_1": [0.927, 0.692, 0.600, 0.570, 0.572, 0.528, 0.534, 0.534, 0.508, 0.443],
        "td_plus_2": [0.927, 0.671, 0.641, 0.629, 0.628, 0.626, 0.612, 0.612, 0.614, 0.595],
        "td_plus_3": [0.831, 0.635, 0.625, 0.624, 0.624, 0.618, 0.620, 0.620, 0.616, 0.613],
    }

    vis.plot_visualized_aggregated(model_name, model_name_star, data, save_path, kpi, legend_loc)
