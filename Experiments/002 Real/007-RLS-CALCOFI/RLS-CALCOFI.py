from Datasets.Real import PublicDS
from Utils import Constants
import warnings
from Utils import Util

from Models.RLS import RLS
from Models.RLS import RLS_SCCM
from Models.RLS import RLS_ADWIN_RESET
from Models.RLS import RLS_ADWIN_WINDOW
from Models.RLS import RLS_ADWIN_SSPT
from Models.RLS import RLS_ADWIN_OHL
from Models.RLS import RLS_KSWIN_RESET
from Models.RLS import RLS_KSWIN_WINDOW
from Models.RLS import RLS_KSWIN_SSPT
from Models.RLS import RLS_KSWIN_OHL

from Utils import Plotter

import numpy as np
import os
import pprint


# ============================================================
# RLS CONFIGURATION
# ============================================================

# Use lambda=1.0 for numerical stability on the very long
# CalCOFI stream.
RLS_LAMBDA = 1.0

RLS_DELTA = 1.0

RLS_SCCM_MULTIPLIER = 1.5
RLS_SCCM_DS = "CalCOFI"

RLS_ADWIN_DELTA = 0.002

RLS_KSWIN_ALPHA = 0.005
RLS_KSWIN_WINDOW_SIZE = 100
RLS_KSWIN_STAT_SIZE = 30

RLS_WINDOW_SIZE = 50

# Keep all adaptive forgetting factors extremely close to 1.0.
RLS_SSPT_LAMBDA_CANDIDATES = (
    0.9999,
    0.99999,
    0.999999,
    1.0
)

RLS_OHL_ETA = 0.000001
RLS_OHL_EPS = 0.01

RLS_LAMBDA_BOUNDS = (
    0.99999,
    1.0
)

# RLS processes one observation at a time.
REPORT_INTERVAL = 1


# ============================================================
# CALCOFI DATASET PATHS
# ============================================================

CALCOFI_BOTTLE_RELATIVE_PATH = (
    r"CalCOFI\bottle.csv"
)

CALCOFI_CAST_RELATIVE_PATH = (
    r"CalCOFI\cast.csv"
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
    Run RLS and all drift-adaptation variants for one seed.
    """

    np.random.seed(
        seed
    )

    # ========================================================
    # 1) PLAIN RLS
    # ========================================================

    (
        rls_final_r2,
        rls_r2_list,
        rls_mse_list
    ) = RLS.rls_generic(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 2) RLS-SCCM
    # ========================================================

    (
        rls_sccm_final_r2,
        rls_sccm_mse_list
    ) = RLS_SCCM.ad_rls_generic(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        kpi="MSE",
        multiplier=RLS_SCCM_MULTIPLIER,
        DS=RLS_SCCM_DS,
        report_interval=REPORT_INTERVAL,
        lambda_bounds=RLS_LAMBDA_BOUNDS
    )

    # ========================================================
    # 3) RLS + ADWIN-RESET
    # ========================================================

    (
        adwin_reset_final_r2,
        adwin_reset_r2_list,
        adwin_reset_mse_list
    ) = RLS_ADWIN_RESET.rls_generic_adwin_reset(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        adwin_delta=RLS_ADWIN_DELTA,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 4) RLS + ADWIN-WINDOW
    # ========================================================

    (
        adwin_window_final_r2,
        adwin_window_r2_list,
        adwin_window_mse_list
    ) = RLS_ADWIN_WINDOW.rls_generic_adwin_window(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        adwin_delta=RLS_ADWIN_DELTA,
        window_size=RLS_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 5) RLS + ADWIN-SSPT
    # ========================================================

    (
        adwin_sspt_final_r2,
        adwin_sspt_r2_list,
        adwin_sspt_mse_list
    ) = RLS_ADWIN_SSPT.rls_generic_adwin_sspt(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        adwin_delta=RLS_ADWIN_DELTA,
        sspt_lambda_candidates=(
            RLS_SSPT_LAMBDA_CANDIDATES
        ),
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 6) RLS + ADWIN-OHL
    # ========================================================

    (
        adwin_ohl_final_r2,
        adwin_ohl_r2_list,
        adwin_ohl_mse_list
    ) = RLS_ADWIN_OHL.rls_generic_adwin_ohl(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        adwin_delta=RLS_ADWIN_DELTA,
        ohl_eta=RLS_OHL_ETA,
        ohl_eps=RLS_OHL_EPS,
        lambda_bounds=RLS_LAMBDA_BOUNDS,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 7) RLS + KSWIN-RESET
    # ========================================================

    (
        kswin_reset_final_r2,
        kswin_reset_r2_list,
        kswin_reset_mse_list
    ) = RLS_KSWIN_RESET.rls_generic_kswin_reset(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        kswin_alpha=RLS_KSWIN_ALPHA,
        kswin_window_size=RLS_KSWIN_WINDOW_SIZE,
        kswin_stat_size=RLS_KSWIN_STAT_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 8) RLS + KSWIN-WINDOW
    # ========================================================

    (
        kswin_window_final_r2,
        kswin_window_r2_list,
        kswin_window_mse_list
    ) = RLS_KSWIN_WINDOW.rls_generic_kswin_window(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        kswin_alpha=RLS_KSWIN_ALPHA,
        kswin_window_size=RLS_KSWIN_WINDOW_SIZE,
        kswin_stat_size=RLS_KSWIN_STAT_SIZE,
        window_size=RLS_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 9) RLS + KSWIN-SSPT
    # ========================================================

    (
        kswin_sspt_final_r2,
        kswin_sspt_r2_list,
        kswin_sspt_mse_list
    ) = RLS_KSWIN_SSPT.rls_generic_kswin_sspt(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        kswin_alpha=RLS_KSWIN_ALPHA,
        kswin_window_size=RLS_KSWIN_WINDOW_SIZE,
        kswin_stat_size=RLS_KSWIN_STAT_SIZE,
        sspt_lambda_candidates=(
            RLS_SSPT_LAMBDA_CANDIDATES
        ),
        report_interval=REPORT_INTERVAL
    )

    # ========================================================
    # 10) RLS + KSWIN-OHL
    # ========================================================

    (
        kswin_ohl_final_r2,
        kswin_ohl_r2_list,
        kswin_ohl_mse_list
    ) = RLS_KSWIN_OHL.rls_generic_kswin_ohl(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        kswin_alpha=RLS_KSWIN_ALPHA,
        kswin_window_size=RLS_KSWIN_WINDOW_SIZE,
        kswin_stat_size=RLS_KSWIN_STAT_SIZE,
        ohl_eta=RLS_OHL_ETA,
        ohl_eps=RLS_OHL_EPS,
        lambda_bounds=RLS_LAMBDA_BOUNDS,
        report_interval=REPORT_INTERVAL
    )

    return {
        "seed": seed,

        "RLS": {
            "MSE": rls_mse_list
        },

        "RLS-SCCM": {
            "MSE": rls_sccm_mse_list
        },

        "RLS-ADWIN-RESET": {
            "MSE": adwin_reset_mse_list
        },

        "RLS-ADWIN-WINDOW": {
            "MSE": adwin_window_mse_list
        },

        "RLS-ADWIN-SSPT": {
            "MSE": adwin_sspt_mse_list
        },

        "RLS-ADWIN-OHL": {
            "MSE": adwin_ohl_mse_list
        },

        "RLS-KSWIN-RESET": {
            "MSE": kswin_reset_mse_list
        },

        "RLS-KSWIN-WINDOW": {
            "MSE": kswin_window_mse_list
        },

        "RLS-KSWIN-SSPT": {
            "MSE": kswin_sspt_mse_list
        },

        "RLS-KSWIN-OHL": {
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
    Run all RLS methods on CalCOFI.
    """

    # ========================================================
    # 1) LOAD CALCOFI
    # ========================================================

    bottle_path = Util.get_dataset_path_(
        CALCOFI_BOTTLE_RELATIVE_PATH
    )

    cast_path = Util.get_dataset_path_(
        CALCOFI_CAST_RELATIVE_PATH
    )

    X, y, dataset_metadata = PublicDS.get_CALCOFI(
        bottle_path=bottle_path,
        cast_path=cast_path,
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
        "CALCOFI EXPERIMENT DATA"
    )
    print(
        "=" * 70
    )
    print(
        f"Bottle file: {bottle_path}"
    )
    print(
        f"Cast file: {cast_path}"
    )
    print(
        f"Dataset: {DATASET_NAME}"
    )
    print(
        f"Target: "
        f"{dataset_metadata['target_name']}"
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
        f"RLS lambda: {RLS_LAMBDA}"
    )
    print(
        f"RLS delta: {RLS_DELTA}"
    )
    print(
        f"Adaptive lambda bounds: "
        f"{RLS_LAMBDA_BOUNDS}"
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

    rls_mse_avg, _ = Util.average_lists(
        [
            run["RLS"]["MSE"]
            for run in all_runs
        ]
    )

    rls_sccm_mse_avg, _ = Util.average_lists(
        [
            run["RLS-SCCM"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_reset_mse_avg, _ = Util.average_lists(
        [
            run["RLS-ADWIN-RESET"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_window_mse_avg, _ = Util.average_lists(
        [
            run["RLS-ADWIN-WINDOW"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_sspt_mse_avg, _ = Util.average_lists(
        [
            run["RLS-ADWIN-SSPT"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_ohl_mse_avg, _ = Util.average_lists(
        [
            run["RLS-ADWIN-OHL"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_reset_mse_avg, _ = Util.average_lists(
        [
            run["RLS-KSWIN-RESET"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_window_mse_avg, _ = Util.average_lists(
        [
            run["RLS-KSWIN-WINDOW"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_sspt_mse_avg, _ = Util.average_lists(
        [
            run["RLS-KSWIN-SSPT"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_ohl_mse_avg, _ = Util.average_lists(
        [
            run["RLS-KSWIN-OHL"]["MSE"]
            for run in all_runs
        ]
    )

    # ========================================================
    # 5) ALIGN RESULT LENGTHS
    # ========================================================

    min_len = min(
        len(rls_mse_avg),
        len(rls_sccm_mse_avg),
        len(adwin_reset_mse_avg),
        len(adwin_window_mse_avg),
        len(adwin_sspt_mse_avg),
        len(adwin_ohl_mse_avg),
        len(kswin_reset_mse_avg),
        len(kswin_window_mse_avg),
        len(kswin_sspt_mse_avg),
        len(kswin_ohl_mse_avg)
    )

    rls_mse_avg = (
        rls_mse_avg[:min_len]
    )

    rls_sccm_mse_avg = (
        rls_sccm_mse_avg[:min_len]
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
        index * REPORT_INTERVAL
        for index in range(
            1,
            min_len + 1
        )
    ]

    # ========================================================
    # 6) PRINT RESULTS
    # ========================================================

    Util.print_mse_lists_results(
        "RLS",
        rls_mse_avg,
        rls_sccm_mse_avg,
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
            rls_mse_avg,
            rls_sccm_mse_avg,
            adwin_reset_mse_avg,
            adwin_window_mse_avg,
            adwin_sspt_mse_avg,
            adwin_ohl_mse_avg,
            kswin_reset_mse_avg,
            kswin_window_mse_avg,
            kswin_sspt_mse_avg,
            kswin_ohl_mse_avg,
            "MSE",
            "RLS",
            "RLS$^*$",
            "RLS$^†$",
            "RLS$^‡$",
            "RLS$^\\diamond$",
            "RLS$^\\parallel$",
            "RLS$^§$",
            "RLS$^¶$",
            "RLS$^\\#$",
            "RLS$^\\triangle$",
            drift_location=DRIFT_LOCATION,
            log_enabled=False,
            legend_loc="lower left",
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
            MODEL_NAME="RLS",
            experiment_name=EXPERIMENT_NAME,
            dataset_name=DATASET_NAME,
            drift_type=DRIFT_TYPE,
            n_samples=n_samples,
            drift_location=DRIFT_LOCATION,
            increment_size=REPORT_INTERVAL,
            model_mse_list=rls_mse_avg,
            model_sccm_mse_list=rls_sccm_mse_avg,
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
    # 9) ADD CALCOFI METADATA
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

    expr_data["bottle_file"] = (
        dataset_metadata["bottle_file"]
    )

    expr_data["cast_file"] = (
        dataset_metadata["cast_file"]
    )

    expr_data["target_name"] = (
        dataset_metadata["target_name"]
    )

    expr_data["target_unit"] = (
        dataset_metadata["target_unit"]
    )

    expr_data["feature_names"] = (
        dataset_metadata["feature_names"]
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

    expr_data["start_timestamp"] = (
        dataset_metadata["start_timestamp"]
    )

    expr_data["end_timestamp"] = (
        dataset_metadata["end_timestamp"]
    )

    expr_data["selected_date_format"] = (
        dataset_metadata[
            "selected_date_format"
        ]
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

    expr_data["merge_column"] = (
        dataset_metadata["merge_column"]
    )

    expr_data["report_interval"] = (
        REPORT_INTERVAL
    )

    expr_data["rls_lambda"] = (
        RLS_LAMBDA
    )

    expr_data["rls_delta"] = (
        RLS_DELTA
    )

    expr_data["sccm_multiplier"] = (
        RLS_SCCM_MULTIPLIER
    )

    expr_data["sspt_lambda_candidates"] = list(
        RLS_SSPT_LAMBDA_CANDIDATES
    )

    expr_data["lambda_bounds"] = list(
        RLS_LAMBDA_BOUNDS
    )

    expr_data["ohl_eta"] = (
        RLS_OHL_ETA
    )

    expr_data["ohl_eps"] = (
        RLS_OHL_EPS
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

    # First run only one seed to confirm that all ten
    # RLS methods remain numerically stable.
    seeds = [0]

    # After the first successful run, use:
    # seeds = Constants.SEEDS5

    EXPERIMENT_NAME = (
        "007-CalCOFI_realdataset_exp7_RLS"
    )

    DATASET_NAME = (
        "CalCOFI"
    )

    PLOTTING_ENABLED = True

    TRAIN_PERCENT = 90

    PLOTTING_DIR = (
        "007-RLS-CalCOFI"
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