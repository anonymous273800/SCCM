from Datasets.Real import PublicDS
from Utils import Constants
import warnings
from Utils import Util

from Models.WidrowHoff import WidrowHoff
from Models.WidrowHoff import WidrowHoff_SCCM
from Models.WidrowHoff import WidrowHoff_ADWIN_RESET
from Models.WidrowHoff import WidrowHoff_ADWIN_WINDOW
from Models.WidrowHoff import WidrowHoff_ADWIN_SSPT
from Models.WidrowHoff import WidrowHoff_ADWIN_OHL
from Models.WidrowHoff import WidrowHoff_KSWIN_RESET
from Models.WidrowHoff import WidrowHoff_KSWIN_WINDOW
from Models.WidrowHoff import WidrowHoff_KSWIN_SSPT
from Models.WidrowHoff import WidrowHoff_KSWIN_OHL

from Utils import Plotter

import numpy as np
import os
import pprint


# ============================================================
# WIDROW-HOFF CONFIGURATION
# ============================================================

WH_LEARNING_RATE = 0.01

WH_SCCM_MULTIPLIER = 1.5
WH_SCCM_DS = "WSSF"

WH_ADWIN_DELTA = 0.002

WH_KSWIN_ALPHA = 0.005
WH_KSWIN_WINDOW_SIZE = 100
WH_KSWIN_STAT_SIZE = 30

WH_WINDOW_SIZE = 50

WH_SSPT_LR_CANDIDATES = (
    0.001,
    0.003,
    0.005,
    0.008,
    0.01,
    0.015,
    0.02,
    0.03
)

WH_OHL_ETA = 0.02
WH_OHL_EPS = 0.01

WH_LR_BOUNDS = (
    1e-4,
    0.05
)

# Widrow-Hoff still updates one observation at a time.
# This controls how often performance is stored.
REPORT_INTERVAL = 1


# ============================================================
# WSSF DATASET PATHS
# ============================================================

WSSF_TRAIN_RELATIVE_PATH = (
    r"WSSF\train.csv"
)

WSSF_FEATURES_RELATIVE_PATH = (
    r"WSSF\features.csv"
)

WSSF_STORES_RELATIVE_PATH = (
    r"WSSF\stores.csv"
)

WSSF_TEST_RELATIVE_PATH = (
    r"WSSF\test.csv"
)


def run_single_seed_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        seed,
        print_details=True
):
    """
    Run Widrow-Hoff and all adaptive variants for one seed.
    """

    np.random.seed(
        seed
    )

    # ========================================================
    # 1) PLAIN WIDROW-HOFF
    # ========================================================

    (
        wh_final_r2,
        wh_r2_list,
        wh_mse_list
    ) = WidrowHoff.widrow_hoff_generic(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 2) WIDROW-HOFF-SCCM
    # ========================================================

    (
        wh_sccm_final_r2,
        wh_sccm_r2_list,
        wh_sccm_mse_list
    ) = WidrowHoff_SCCM.ad_widrow_hoff_generic(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        kpi="MSE",
        multiplier=WH_SCCM_MULTIPLIER,
        DS=WH_SCCM_DS,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 3) WIDROW-HOFF + ADWIN-RESET
    # ========================================================

    (
        adwin_reset_final_r2,
        adwin_reset_r2_list,
        adwin_reset_mse_list
    ) = WidrowHoff_ADWIN_RESET.widrow_hoff_generic_adwin_reset(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        adwin_delta=WH_ADWIN_DELTA,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 4) WIDROW-HOFF + ADWIN-WINDOW
    # ========================================================

    (
        adwin_window_final_r2,
        adwin_window_r2_list,
        adwin_window_mse_list
    ) = WidrowHoff_ADWIN_WINDOW.widrow_hoff_generic_adwin_window(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        adwin_delta=WH_ADWIN_DELTA,
        window_size=WH_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 5) WIDROW-HOFF + ADWIN-SSPT
    # ========================================================

    (
        adwin_sspt_final_r2,
        adwin_sspt_r2_list,
        adwin_sspt_mse_list
    ) = WidrowHoff_ADWIN_SSPT.widrow_hoff_generic_adwin_sspt(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        adwin_delta=WH_ADWIN_DELTA,
        sspt_lr_candidates=WH_SSPT_LR_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 6) WIDROW-HOFF + ADWIN-OHL
    # ========================================================

    (
        adwin_ohl_final_r2,
        adwin_ohl_r2_list,
        adwin_ohl_mse_list
    ) = WidrowHoff_ADWIN_OHL.widrow_hoff_generic_adwin_ohl(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        adwin_delta=WH_ADWIN_DELTA,
        ohl_eta=WH_OHL_ETA,
        ohl_eps=WH_OHL_EPS,
        lr_bounds=WH_LR_BOUNDS,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 7) WIDROW-HOFF + KSWIN-RESET
    # ========================================================

    (
        kswin_reset_final_r2,
        kswin_reset_r2_list,
        kswin_reset_mse_list
    ) = WidrowHoff_KSWIN_RESET.widrow_hoff_generic_kswin_reset(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        kswin_alpha=WH_KSWIN_ALPHA,
        kswin_window_size=WH_KSWIN_WINDOW_SIZE,
        kswin_stat_size=WH_KSWIN_STAT_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 8) WIDROW-HOFF + KSWIN-WINDOW
    # ========================================================

    (
        kswin_window_final_r2,
        kswin_window_r2_list,
        kswin_window_mse_list
    ) = WidrowHoff_KSWIN_WINDOW.widrow_hoff_generic_kswin_window(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        kswin_alpha=WH_KSWIN_ALPHA,
        kswin_window_size=WH_KSWIN_WINDOW_SIZE,
        kswin_stat_size=WH_KSWIN_STAT_SIZE,
        window_size=WH_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 9) WIDROW-HOFF + KSWIN-SSPT
    # ========================================================

    (
        kswin_sspt_final_r2,
        kswin_sspt_r2_list,
        kswin_sspt_mse_list
    ) = WidrowHoff_KSWIN_SSPT.widrow_hoff_generic_kswin_sspt(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        kswin_alpha=WH_KSWIN_ALPHA,
        kswin_window_size=WH_KSWIN_WINDOW_SIZE,
        kswin_stat_size=WH_KSWIN_STAT_SIZE,
        sspt_lr_candidates=WH_SSPT_LR_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 10) WIDROW-HOFF + KSWIN-OHL
    # ========================================================

    (
        kswin_ohl_final_r2,
        kswin_ohl_r2_list,
        kswin_ohl_mse_list
    ) = WidrowHoff_KSWIN_OHL.widrow_hoff_generic_kswin_ohl(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        kswin_alpha=WH_KSWIN_ALPHA,
        kswin_window_size=WH_KSWIN_WINDOW_SIZE,
        kswin_stat_size=WH_KSWIN_STAT_SIZE,
        ohl_eta=WH_OHL_ETA,
        ohl_eps=WH_OHL_EPS,
        lr_bounds=WH_LR_BOUNDS,
        report_interval=REPORT_INTERVAL
    )

    return {
        "seed": seed,

        "Widrow-Hoff": {
            "MSE": wh_mse_list
        },

        "Widrow-Hoff-SCCM": {
            "MSE": wh_sccm_mse_list
        },

        "Widrow-Hoff-ADWIN-RESET": {
            "MSE": adwin_reset_mse_list
        },

        "Widrow-Hoff-ADWIN-WINDOW": {
            "MSE": adwin_window_mse_list
        },

        "Widrow-Hoff-ADWIN-SSPT": {
            "MSE": adwin_sspt_mse_list
        },

        "Widrow-Hoff-ADWIN-OHL": {
            "MSE": adwin_ohl_mse_list
        },

        "Widrow-Hoff-KSWIN-RESET": {
            "MSE": kswin_reset_mse_list
        },

        "Widrow-Hoff-KSWIN-WINDOW": {
            "MSE": kswin_window_mse_list
        },

        "Widrow-Hoff-KSWIN-SSPT": {
            "MSE": kswin_sspt_mse_list
        },

        "Widrow-Hoff-KSWIN-OHL": {
            "MSE": kswin_ohl_mse_list
        }
    }


def run_multi_seed_experiment(
        seeds,
        EXPERIMENT_NAME,
        DATASET_NAME,
        DRIFT_TYPE,
        DRIFT_LOCATION,
        PLOTTING_ENABLED,
        TRAIN_PERCENT,
        PLOTTING_DIR
):
    """
    Run Widrow-Hoff and all adaptive variants on WSSF.
    """

    # ========================================================
    # 1) LOAD WSSF
    # ========================================================

    train_path = Util.get_dataset_path_(
        WSSF_TRAIN_RELATIVE_PATH
    )

    features_path = Util.get_dataset_path_(
        WSSF_FEATURES_RELATIVE_PATH
    )

    stores_path = Util.get_dataset_path_(
        WSSF_STORES_RELATIVE_PATH
    )

    test_path = Util.get_dataset_path_(
        WSSF_TEST_RELATIVE_PATH
    )

    X, y, dataset_metadata = PublicDS.get_WSSF(
        train_path=train_path,
        features_path=features_path,
        stores_path=stores_path,
        test_path=test_path,
        train_percent=TRAIN_PERCENT,
        return_metadata=True
    )

    n_samples = int(
        X.shape[0]
    )

    train_count = int(
        TRAIN_PERCENT
        * n_samples
        / 100.0
    )

    X_train = X[
        :train_count
    ]

    y_train = y[
        :train_count
    ]

    X_test = X[
        train_count:
    ]

    y_test = y[
        train_count:
    ]

    # ========================================================
    # 2) DATASET VALIDATION
    # ========================================================

    print()
    print(
        "WSSF EXPERIMENT DATA"
    )
    print(
        "=" * 70
    )
    print(
        f"Train file: {train_path}"
    )
    print(
        f"Features file: {features_path}"
    )
    print(
        f"Stores file: {stores_path}"
    )
    print(
        f"Official test file: {test_path}"
    )
    print(
        f"Dataset: {DATASET_NAME}"
    )
    print(
        f"Target: "
        f"{dataset_metadata['target_name']}"
    )
    print(
        f"Prediction unit: "
        f"{dataset_metadata['prediction_unit']}"
    )
    print(
        f"Total samples: {n_samples}"
    )
    print(
        f"Number of features: {X.shape[1]}"
    )
    print(
        f"Training samples: {X_train.shape[0]}"
    )
    print(
        f"Test samples: {X_test.shape[0]}"
    )
    print(
        f"Report interval: {REPORT_INTERVAL}"
    )
    print(
        f"Learning rate: {WH_LEARNING_RATE}"
    )
    print(
        f"X minimum: {np.min(X):.6f}"
    )
    print(
        f"X maximum: {np.max(X):.6f}"
    )
    print(
        f"y minimum: {np.min(y):.6f}"
    )
    print(
        f"y maximum: {np.max(y):.6f}"
    )
    print(
        f"X contains NaN: {np.isnan(X).any()}"
    )
    print(
        f"y contains NaN: {np.isnan(y).any()}"
    )
    print(
        f"X contains infinity: {np.isinf(X).any()}"
    )
    print(
        f"y contains infinity: {np.isinf(y).any()}"
    )
    print(
        "=" * 70
    )
    print()

    # ========================================================
    # 3) RUN SEEDS
    # ========================================================

    all_runs = []

    for seed in seeds:

        print(
            f"**** Running seed = {seed}"
        )

        one_run = run_single_seed_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            seed=seed,
            print_details=False
        )

        all_runs.append(
            one_run
        )

    print(
        "Finished All Seeds now:"
    )

    # ========================================================
    # 4) AVERAGE RESULTS
    # ========================================================

    wh_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff"]["MSE"]
            for run in all_runs
        ]
    )

    wh_sccm_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-SCCM"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_reset_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-ADWIN-RESET"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_window_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-ADWIN-WINDOW"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_sspt_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-ADWIN-SSPT"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_ohl_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-ADWIN-OHL"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_reset_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-KSWIN-RESET"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_window_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-KSWIN-WINDOW"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_sspt_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-KSWIN-SSPT"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_ohl_mse_avg, _ = Util.average_lists(
        [
            run["Widrow-Hoff-KSWIN-OHL"]["MSE"]
            for run in all_runs
        ]
    )

    # ========================================================
    # 5) ALIGN RESULT LENGTHS
    # ========================================================

    min_len = min(
        len(wh_mse_avg),
        len(wh_sccm_mse_avg),
        len(adwin_reset_mse_avg),
        len(adwin_window_mse_avg),
        len(adwin_sspt_mse_avg),
        len(adwin_ohl_mse_avg),
        len(kswin_reset_mse_avg),
        len(kswin_window_mse_avg),
        len(kswin_sspt_mse_avg),
        len(kswin_ohl_mse_avg)
    )

    wh_mse_avg = (
        wh_mse_avg[:min_len]
    )

    wh_sccm_mse_avg = (
        wh_sccm_mse_avg[:min_len]
    )

    adwin_reset_mse_avg = (
        adwin_reset_mse_avg[:min_len]
    )

    adwin_window_mse_avg = (
        adwin_window_mse_avg[:min_len]
    )

    adwin_sspt_mse_avg = (
        adwin_sspt_mse_avg[:min_len]
    )

    adwin_ohl_mse_avg = (
        adwin_ohl_mse_avg[:min_len]
    )

    kswin_reset_mse_avg = (
        kswin_reset_mse_avg[:min_len]
    )

    kswin_window_mse_avg = (
        kswin_window_mse_avg[:min_len]
    )

    kswin_sspt_mse_avg = (
        kswin_sspt_mse_avg[:min_len]
    )

    kswin_ohl_mse_avg = (
        kswin_ohl_mse_avg[:min_len]
    )

    x_axis = [
        min(
            index * REPORT_INTERVAL,
            X_train.shape[0]
        )
        for index in range(
            1,
            min_len + 1
        )
    ]

    # ========================================================
    # 6) PRINT RESULTS
    # ========================================================

    Util.print_mse_lists_results(
        "WidrowHoff",
        wh_mse_avg,
        wh_sccm_mse_avg,
        adwin_reset_mse_avg,
        adwin_window_mse_avg,
        adwin_sspt_mse_avg,
        adwin_ohl_mse_avg,
        kswin_reset_mse_avg,
        kswin_window_mse_avg,
        kswin_sspt_mse_avg,
        kswin_ohl_mse_avg
    )

    # ========================================================
    # 7) PLOT RESULTS
    # ========================================================

    if PLOTTING_ENABLED:

        os.makedirs(
            PLOTTING_DIR,
            exist_ok=True
        )

        Plotter.plot_results_ten_models_only_mse(
            x_axis,
            wh_mse_avg,
            wh_sccm_mse_avg,
            adwin_reset_mse_avg,
            adwin_window_mse_avg,
            adwin_sspt_mse_avg,
            adwin_ohl_mse_avg,
            kswin_reset_mse_avg,
            kswin_window_mse_avg,
            kswin_sspt_mse_avg,
            kswin_ohl_mse_avg,
            "MSE",
            "WH",
            "WH$^*$",
            "WH$^†$",
            "WH$^‡$",
            "WH$^\\diamond$",
            "WH$^\\parallel$",
            "WH$^§$",
            "WH$^¶$",
            "WH$^\\#$",
            "WH$^\\triangle$",
            drift_location=DRIFT_LOCATION,
            log_enabled=False,
            legend_loc="upper right",
            drift_type=DRIFT_TYPE,
            gradual_drift_locations=None,
            gradual_drift_concepts=None,
            save_path=os.path.join(
                PLOTTING_DIR,
                f"{DATASET_NAME}_MSE_plot.pdf"
            )
        )

    # ========================================================
    # 8) PREPARE EXPERIMENT DATA
    # ========================================================

    expr_data = (
        Util.prepare_and_print_experiment_data_new_mse(
            "Widrow-Hoff",
            experiment_name=EXPERIMENT_NAME,
            dataset_name=DATASET_NAME,
            drift_type=DRIFT_TYPE,
            n_samples=n_samples,
            drift_location=DRIFT_LOCATION,
            increment_size=REPORT_INTERVAL,
            model_mse_list=wh_mse_avg,
            model_sccm_mse_list=wh_sccm_mse_avg,
            adwin_reset_mse_list=adwin_reset_mse_avg,
            adwin_window_mse_list=adwin_window_mse_avg,
            adwin_sspt_mse_list=adwin_sspt_mse_avg,
            adwin_ohl_mse_list=adwin_ohl_mse_avg,
            kswin_reset_mse_list=kswin_reset_mse_avg,
            kswin_window_mse_list=kswin_window_mse_avg,
            kswin_sspt_mse_list=kswin_sspt_mse_avg,
            kswin_ohl_mse_list=kswin_ohl_mse_avg
        )
    )

    # ========================================================
    # 9) ADD WSSF METADATA
    # ========================================================

    expr_data["seed"] = (
        "AVERAGED_OVER_SEEDS"
    )

    expr_data["seeds"] = list(
        seeds
    )

    expr_data["dataset_type"] = (
        "real"
    )

    expr_data["train_file"] = (
        dataset_metadata["train_file"]
    )

    expr_data["features_file"] = (
        dataset_metadata["features_file"]
    )

    expr_data["stores_file"] = (
        dataset_metadata["stores_file"]
    )

    expr_data["official_test_file"] = (
        dataset_metadata.get(
            "test_file",
            None
        )
    )

    expr_data["official_test_file_used"] = (
        dataset_metadata.get(
            "official_test_file_used",
            False
        )
    )

    expr_data["official_test_rows_not_used"] = (
        dataset_metadata.get(
            "official_test_rows_not_used",
            None
        )
    )

    expr_data["target_name"] = (
        dataset_metadata["target_name"]
    )

    expr_data["prediction_unit"] = (
        dataset_metadata["prediction_unit"]
    )

    expr_data["feature_names"] = (
        dataset_metadata["feature_names"]
    )

    expr_data["lag_weeks"] = (
        dataset_metadata["lag_weeks"]
    )

    expr_data["n_features"] = int(
        dataset_metadata["n_features"]
    )

    expr_data["training_percent"] = int(
        TRAIN_PERCENT
    )

    expr_data["training_samples"] = int(
        X_train.shape[0]
    )

    expr_data["test_samples"] = int(
        X_test.shape[0]
    )

    expr_data["original_train_rows"] = int(
        dataset_metadata[
            "original_train_rows"
        ]
    )

    expr_data["rows_removed_for_lags"] = int(
        dataset_metadata[
            "rows_removed_for_lags"
        ]
    )

    expr_data["start_date"] = (
        dataset_metadata["start_date"]
    )

    expr_data["end_date"] = (
        dataset_metadata["end_date"]
    )

    expr_data["normalization"] = (
        dataset_metadata["normalization"]
    )

    expr_data["normalization_range"] = (
        dataset_metadata[
            "normalization_range"
        ]
    )

    expr_data["scaler_fitted_on_training_only"] = (
        dataset_metadata[
            "scaler_fitted_on_training_only"
        ]
    )

    expr_data[
        "categorical_encoder_fitted_on_training_only"
    ] = dataset_metadata[
        "categorical_encoder_fitted_on_training_only"
    ]

    expr_data["report_interval"] = (
        REPORT_INTERVAL
    )

    expr_data["learning_rate"] = (
        WH_LEARNING_RATE
    )

    expr_data["sccm_multiplier"] = (
        WH_SCCM_MULTIPLIER
    )

    expr_data["adwin_delta"] = (
        WH_ADWIN_DELTA
    )

    expr_data["kswin_alpha"] = (
        WH_KSWIN_ALPHA
    )

    expr_data["kswin_window_size"] = (
        WH_KSWIN_WINDOW_SIZE
    )

    expr_data["kswin_stat_size"] = (
        WH_KSWIN_STAT_SIZE
    )

    expr_data["window_size"] = (
        WH_WINDOW_SIZE
    )

    expr_data["sspt_lr_candidates"] = list(
        WH_SSPT_LR_CANDIDATES
    )

    expr_data["ohl_eta"] = (
        WH_OHL_ETA
    )

    expr_data["ohl_eps"] = (
        WH_OHL_EPS
    )

    expr_data["lr_bounds"] = list(
        WH_LR_BOUNDS
    )

    expr_data["known_drift_location"] = (
        None
    )

    expr_data["known_drift_type"] = (
        None
    )

    expr_data["natural_nonstationarity"] = (
        True
    )

    # ========================================================
    # 10) SAVE RESULT FILE
    # ========================================================

    os.makedirs(
        PLOTTING_DIR,
        exist_ok=True
    )

    expr_txt_path = os.path.join(
        PLOTTING_DIR,
        f"{EXPERIMENT_NAME}_expr_data.txt"
    )

    with open(
            expr_txt_path,
            "w",
            encoding="utf-8"
    ) as file:

        pprint.pprint(
            expr_data,
            stream=file,
            width=120
        )

    print(
        f"Saved expr_data to: "
        f"{expr_txt_path}"
    )

    return expr_data


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    seeds = Constants.SEEDS5

    EXPERIMENT_NAME = (
        "008-WSSF_realdataset_exp8_WH"
    )

    DATASET_NAME = (
        "WSSF"
    )

    PLOTTING_ENABLED = True

    TRAIN_PERCENT = 90

    PLOTTING_DIR = (
        "008-WH-WSSF"
    )

    expr_data = run_multi_seed_experiment(
        seeds=seeds,
        EXPERIMENT_NAME=EXPERIMENT_NAME,
        DATASET_NAME=DATASET_NAME,
        DRIFT_TYPE=None,
        DRIFT_LOCATION=None,
        PLOTTING_ENABLED=PLOTTING_ENABLED,
        TRAIN_PERCENT=TRAIN_PERCENT,
        PLOTTING_DIR=PLOTTING_DIR
    )

    print(
        expr_data
    )