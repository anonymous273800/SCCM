from Utils import Constants
from Utils import ExperimentRecorder
from Datasets.Synthetic.Gradual import GDS06
import warnings
from Utils import Util
from Hyperparameters import Hyperparameter
from Models.OLR_WA import OLR_WA
from Models.OLR_WA import OLR_WA_SCCM
from Models.OLR_WA import OLR_WA_ADWIN_RESET
from Models.OLR_WA import OLR_WA_ADWIN_WINDOW
from Models.OLR_WA import OLR_WA_ADWIN_SSPT
from Models.OLR_WA import OLR_WA_ADWIN_OHL
from Models.OLR_WA import OLR_WA_KSWIN_RESET
from Models.OLR_WA import OLR_WA_KSWIN_WINDOW
from Models.OLR_WA import OLR_WA_KSWIN_SSPT
from Models.OLR_WA import OLR_WA_KSWIN_OHL
from Utils import Plotter
import numpy as np
from Utils import QuantifyDrift
import os
import pprint


def run_single_seed_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        increment_size,
        seed,
        print_details=True
):
    """
    Run one experiment for one seed and return raw lists only.
    """

    # 1) OLR-WA
    olr_wa_final_r2, olr_wa_r2_list, olr_wa_mse_list = OLR_WA.olr_wa(
        X_train,
        y_train,
        Hyperparameter.olr_wa_w_base,
        Hyperparameter.olr_wa_w_inc,
        Hyperparameter.olr_wa_base_model_size0,
        increment_size,
        X_test,
        y_test
    )

    # 2) OLR-WA-SCCM
    multiplier_r2 = 1.5
    olr_wa_sccm_final_r2, olr_wa_sccm_r2_list, olr_wa_sccm_mse_list = OLR_WA_SCCM.olr_wa_sccm(
        X_train,
        y_train,
        Hyperparameter.olr_wa_w_base,
        Hyperparameter.olr_wa_w_inc,
        Hyperparameter.olr_wa_base_model_size0,
        increment_size,
        X_test,
        y_test,
        kpi='R2',
        multiplier=multiplier_r2
    )

    # 3) OLR-WA + ADWIN-RESET
    adwin_reset_final_r2, adwin_reset_r2_list, adwin_reset_mse_list = (
        OLR_WA_ADWIN_RESET.olr_wa_regression_adversarial_adwin_reset(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            adwin_delta=0.002
        )
    )

    # 4) OLR-WA + ADWIN-WINDOW
    adwin_window_final_r2, adwin_window_r2_list, adwin_window_mse_list = (
        OLR_WA_ADWIN_WINDOW.olr_wa_regression_adversarial_adwin_window(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            adwin_delta=0.002,
            window_size_in_batches=5
        )
    )

    # 5) OLR-WA + ADWIN-SSPT
    adwin_sspt_final_r2, adwin_sspt_r2_list, adwin_sspt_mse_list = (
        OLR_WA_ADWIN_SSPT.olr_wa_regression_adversarial_adwin_sspt(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            adwin_delta=0.002,
            sspt_w_inc_candidates=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            sspt_metric="r2"
        )
    )

    # 6) OLR-WA + ADWIN-OHL
    adwin_ohl_final_r2, adwin_ohl_r2_list, adwin_ohl_mse_list = (
        OLR_WA_ADWIN_OHL.olr_wa_regression_adversarial_adwin_ohl(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            adwin_delta=0.002,
            ohl_eta=0.1,
            ohl_eps=0.05,
            w_inc_bounds=(0.05, 0.95)
        )
    )

    # 7) OLR-WA + KSWIN-RESET
    kswin_reset_final_r2, kswin_reset_r2_list, kswin_reset_mse_list = (
        OLR_WA_KSWIN_RESET.olr_wa_regression_adversarial_kswin_reset(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            kswin_alpha=0.005,
            kswin_window_size=100,
            kswin_stat_size=30
        )
    )

    # 8) OLR-WA + KSWIN-WINDOW
    kswin_window_final_r2, kswin_window_r2_list, kswin_window_mse_list = (
        OLR_WA_KSWIN_WINDOW.olr_wa_regression_adversarial_kswin_window(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            kswin_alpha=0.005,
            kswin_window_size=100,
            kswin_stat_size=30,
            window_size_in_batches=5
        )
    )

    # 9) OLR-WA + KSWIN-SSPT
    kswin_sspt_final_r2, kswin_sspt_r2_list, kswin_sspt_mse_list = (
        OLR_WA_KSWIN_SSPT.olr_wa_regression_adversarial_kswin_sspt(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            kswin_alpha=0.005,
            kswin_window_size=100,
            kswin_stat_size=30,
            sspt_w_inc_candidates=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            sspt_metric="r2"
        )
    )

    # 10) OLR-WA + KSWIN-OHL
    kswin_ohl_final_r2, kswin_ohl_r2_list, kswin_ohl_mse_list = (
        OLR_WA_KSWIN_OHL.olr_wa_regression_adversarial_kswin_ohl(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter.olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            kswin_alpha=0.005,
            kswin_window_size=100,
            kswin_stat_size=30,
            ohl_eta=0.1,
            ohl_eps=0.05,
            w_inc_bounds=(0.05, 0.95)
        )
    )

    return {
        "seed":seed,
        "OLR-WA": {
            "R2": olr_wa_r2_list,
            "MSE": olr_wa_mse_list
        },
        "OLR-WA-SCCM": {
            "R2": olr_wa_sccm_r2_list,
            "MSE": olr_wa_sccm_mse_list
        },
        "OLR-WA-ADWIN-RESET": {
            "R2": adwin_reset_r2_list,
            "MSE": adwin_reset_mse_list
        },
        "OLR-WA-ADWIN-WINDOW": {
            "R2": adwin_window_r2_list,
            "MSE": adwin_window_mse_list
        },
        "OLR-WA-ADWIN-SSPT": {
            "R2": adwin_sspt_r2_list,
            "MSE": adwin_sspt_mse_list
        },
        "OLR-WA-ADWIN-OHL": {
            "R2": adwin_ohl_r2_list,
            "MSE": adwin_ohl_mse_list
        },
        "OLR-WA-KSWIN-RESET": {
            "R2": kswin_reset_r2_list,
            "MSE": kswin_reset_mse_list
        },
        "OLR-WA-KSWIN-WINDOW": {
            "R2": kswin_window_r2_list,
            "MSE": kswin_window_mse_list
        },
        "OLR-WA-KSWIN-SSPT": {
            "R2": kswin_sspt_r2_list,
            "MSE": kswin_sspt_mse_list
        },
        "OLR-WA-KSWIN-OHL": {
            "R2": kswin_ohl_r2_list,
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
        PLOTTING_DIR,
        GRADUAL_DRIFT_LOCATIONS,
        GRADUAL_DRIFT_CONCEPTS
):
    all_runs = []
    drift_metrics_all = []
    n_samples = None
    increment_size = None
    meta = None
    X = None

    for seed in seeds:
        print(f"**** Running seed = {seed}")
        X, y, X1, y1, X2, y2, meta = GDS06.get_GDS06(seed=seed, return_meta=True)

        drift_metrics_all.append({
            "y_ks_pvalue": meta["y_ks_pvalue"],
            "y_js_divergence": meta["y_js_divergence"],
            "y_wasserstein_distance": meta["y_wasserstein_distance"],
            "x_any_ks_significant": meta["x_any_ks_significant"],
            "x_min_ks_pvalue": meta["x_min_ks_pvalue"],
            "x_avg_js_divergence": meta["x_avg_js_divergence"],
            "euclidean_distance": meta["euclidean_distance_start_end"]
        })

        GDS06.print_GDS06_info(X, y, X1, y1, X2, y2, meta)

        n_samples = X.shape[0]
        train_percent = int(TRAIN_PERCENT * n_samples / 100)

        X_train = X[:train_percent]
        y_train = y[:train_percent]
        X_test = X[train_percent:]
        y_test = y[train_percent:]

        n_samples_trn, n_features_trn = X_train.shape
        increment_size = Hyperparameter.olr_wa_increment_size(
            n_features_trn,
            user_defined_val=10
        )
        print("INCREMENT_SIZE: ", increment_size)

        one_run = run_single_seed_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            increment_size,
            seed=seed,
            print_details=False
        )
        all_runs.append(one_run)

    ExperimentRecorder.save_seed_runs(
        experiment_name=EXPERIMENT_NAME,
        dataset_name=DATASET_NAME,
        drift_type=DRIFT_TYPE,
        runs=all_runs,
    )

    print("Finished All Seeds now:")

    # Convert to arrays
    y_ks_vals = [m["y_ks_pvalue"] for m in drift_metrics_all]
    y_js_vals = [m["y_js_divergence"] for m in drift_metrics_all]
    y_wass_vals = [m["y_wasserstein_distance"] for m in drift_metrics_all]

    x_any_ks_vals = [m["x_any_ks_significant"] for m in drift_metrics_all]
    x_min_ks_vals = [m["x_min_ks_pvalue"] for m in drift_metrics_all]
    x_js_vals = [m["x_avg_js_divergence"] for m in drift_metrics_all]

    dist_vals = [m["euclidean_distance"] for m in drift_metrics_all]

    # Averages
    avg_y_ks = np.mean(y_ks_vals)
    avg_y_js = np.mean(y_js_vals)
    avg_y_wass = np.mean(y_wass_vals)

    avg_x_any_ks = any(x_any_ks_vals)
    avg_x_min_ks = np.mean(x_min_ks_vals)
    avg_x_js = np.mean(x_js_vals)

    avg_distance = np.mean(dist_vals)

    # Average each method / metric across seeds
    olr_wa_r2_avg, _ = Util.average_lists([run["OLR-WA"]["R2"] for run in all_runs])
    olr_wa_mse_avg, _ = Util.average_lists([run["OLR-WA"]["MSE"] for run in all_runs])

    olr_wa_sccm_r2_avg, _ = Util.average_lists([run["OLR-WA-SCCM"]["R2"] for run in all_runs])
    olr_wa_sccm_mse_avg, _ = Util.average_lists([run["OLR-WA-SCCM"]["MSE"] for run in all_runs])

    adwin_reset_r2_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-RESET"]["R2"] for run in all_runs])
    adwin_reset_mse_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-RESET"]["MSE"] for run in all_runs])

    adwin_window_r2_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-WINDOW"]["R2"] for run in all_runs])
    adwin_window_mse_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-WINDOW"]["MSE"] for run in all_runs])

    adwin_sspt_r2_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-SSPT"]["R2"] for run in all_runs])
    adwin_sspt_mse_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-SSPT"]["MSE"] for run in all_runs])

    adwin_ohl_r2_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-OHL"]["R2"] for run in all_runs])
    adwin_ohl_mse_avg, _ = Util.average_lists([run["OLR-WA-ADWIN-OHL"]["MSE"] for run in all_runs])

    kswin_reset_r2_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-RESET"]["R2"] for run in all_runs])
    kswin_reset_mse_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-RESET"]["MSE"] for run in all_runs])

    kswin_window_r2_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-WINDOW"]["R2"] for run in all_runs])
    kswin_window_mse_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-WINDOW"]["MSE"] for run in all_runs])

    kswin_sspt_r2_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-SSPT"]["R2"] for run in all_runs])
    kswin_sspt_mse_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-SSPT"]["MSE"] for run in all_runs])

    kswin_ohl_r2_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-OHL"]["R2"] for run in all_runs])
    kswin_ohl_mse_avg, _ = Util.average_lists([run["OLR-WA-KSWIN-OHL"]["MSE"] for run in all_runs])

    # Final common alignment
    min_len = min(
        len(olr_wa_r2_avg), len(olr_wa_mse_avg),
        len(olr_wa_sccm_r2_avg), len(olr_wa_sccm_mse_avg),
        len(adwin_reset_r2_avg), len(adwin_reset_mse_avg),
        len(adwin_window_r2_avg), len(adwin_window_mse_avg),
        len(adwin_sspt_r2_avg), len(adwin_sspt_mse_avg),
        len(adwin_ohl_r2_avg), len(adwin_ohl_mse_avg),
        len(kswin_reset_r2_avg), len(kswin_reset_mse_avg),
        len(kswin_window_r2_avg), len(kswin_window_mse_avg),
        len(kswin_sspt_r2_avg), len(kswin_sspt_mse_avg),
        len(kswin_ohl_r2_avg), len(kswin_ohl_mse_avg)
    )

    olr_wa_r2_avg = olr_wa_r2_avg[:min_len]
    olr_wa_mse_avg = olr_wa_mse_avg[:min_len]

    olr_wa_sccm_r2_avg = olr_wa_sccm_r2_avg[:min_len]
    olr_wa_sccm_mse_avg = olr_wa_sccm_mse_avg[:min_len]

    adwin_reset_r2_avg = adwin_reset_r2_avg[:min_len]
    adwin_reset_mse_avg = adwin_reset_mse_avg[:min_len]

    adwin_window_r2_avg = adwin_window_r2_avg[:min_len]
    adwin_window_mse_avg = adwin_window_mse_avg[:min_len]

    adwin_sspt_r2_avg = adwin_sspt_r2_avg[:min_len]
    adwin_sspt_mse_avg = adwin_sspt_mse_avg[:min_len]

    adwin_ohl_r2_avg = adwin_ohl_r2_avg[:min_len]
    adwin_ohl_mse_avg = adwin_ohl_mse_avg[:min_len]

    kswin_reset_r2_avg = kswin_reset_r2_avg[:min_len]
    kswin_reset_mse_avg = kswin_reset_mse_avg[:min_len]

    kswin_window_r2_avg = kswin_window_r2_avg[:min_len]
    kswin_window_mse_avg = kswin_window_mse_avg[:min_len]

    kswin_sspt_r2_avg = kswin_sspt_r2_avg[:min_len]
    kswin_sspt_mse_avg = kswin_sspt_mse_avg[:min_len]

    kswin_ohl_r2_avg = kswin_ohl_r2_avg[:min_len]
    kswin_ohl_mse_avg = kswin_ohl_mse_avg[:min_len]

    x_axis = [i * increment_size for i in range(1, min_len + 1)]

    Util.print_acc_mse_lists_results(
        'OLR-WA',
        olr_wa_r2_avg, olr_wa_mse_avg,
        olr_wa_sccm_r2_avg, olr_wa_sccm_mse_avg,
        adwin_reset_r2_avg, adwin_reset_mse_avg,
        adwin_window_r2_avg, adwin_window_mse_avg,
        adwin_sspt_r2_avg, adwin_sspt_mse_avg,
        adwin_ohl_r2_avg, adwin_ohl_mse_avg,
        kswin_reset_r2_avg, kswin_reset_mse_avg,
        kswin_window_r2_avg, kswin_window_mse_avg,
        kswin_sspt_r2_avg, kswin_sspt_mse_avg,
        kswin_ohl_r2_avg, kswin_ohl_mse_avg
    )

    if PLOTTING_ENABLED:
        os.makedirs(PLOTTING_DIR, exist_ok=True)

        # R2 plot
        kpi = 'R2'
        Plotter.plot_results_ten_models(
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
            'OLR-WA',
            'OLR-WA$^*$',
            'OLR-WA$^†$',
            'OLR-WA$^‡$',
            'OLR-WA$^\\diamond$',
            'OLR-WA$^\\parallel$',
            'OLR-WA$^§$',
            'OLR-WA$^¶$',
            'OLR-WA$^\\#$',
            'OLR-WA$^\\triangle$',
            drift_location=DRIFT_LOCATION,
            log_enabled=False,
            legend_loc='lower left',
            drift_type=DRIFT_TYPE,
            gradual_drift_locations=GRADUAL_DRIFT_LOCATIONS,
            gradual_drift_concepts=GRADUAL_DRIFT_CONCEPTS,
            save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_R2_plot.pdf")
        )

        # MSE plot
        kpi = 'MSE'
        Plotter.plot_results_ten_models(
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
            'OLR-WA',
            'OLR-WA$^*$',
            'OLR-WA$^†$',
            'OLR-WA$^‡$',
            'OLR-WA$^\\diamond$',
            'OLR-WA$^\\parallel$',
            'OLR-WA$^§$',
            'OLR-WA$^¶$',
            'OLR-WA$^\\#$',
            'OLR-WA$^\\triangle$',
            drift_location=DRIFT_LOCATION,
            log_enabled=False,
            legend_loc='lower left',
            drift_type=DRIFT_TYPE,
            gradual_drift_locations=GRADUAL_DRIFT_LOCATIONS,
            gradual_drift_concepts=GRADUAL_DRIFT_CONCEPTS,
            save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_MSE_plot.pdf")
        )

    expr_data = Util.prepare_and_print_experiment_data(
        experiment_name=EXPERIMENT_NAME,
        dataset_name=DATASET_NAME,
        drift_type=DRIFT_TYPE,
        n_samples=n_samples,
        drift_location=DRIFT_LOCATION,
        increment_size=increment_size,
        olr_wa_r2_list=olr_wa_r2_avg,
        olr_wa_mse_list=olr_wa_mse_avg,
        olr_wa_sccm_r2_list=olr_wa_sccm_r2_avg,
        olr_wa_sccm_mse_list=olr_wa_sccm_mse_avg,
        adwin_reset_r2_list=adwin_reset_r2_avg,
        adwin_reset_mse_list=adwin_reset_mse_avg,
        adwin_window_r2_list=adwin_window_r2_avg,
        adwin_window_mse_list=adwin_window_mse_avg,
        adwin_sspt_r2_list=adwin_sspt_r2_avg,
        adwin_sspt_mse_list=adwin_sspt_mse_avg,
        adwin_ohl_r2_list=adwin_ohl_r2_avg,
        adwin_ohl_mse_list=adwin_ohl_mse_avg,
        kswin_reset_r2_list=kswin_reset_r2_avg,
        kswin_reset_mse_list=kswin_reset_mse_avg,
        kswin_window_r2_list=kswin_window_r2_avg,
        kswin_window_mse_list=kswin_window_mse_avg,
        kswin_sspt_r2_list=kswin_sspt_r2_avg,
        kswin_sspt_mse_list=kswin_sspt_mse_avg,
        kswin_ohl_r2_list=kswin_ohl_r2_avg,
        kswin_ohl_mse_list=kswin_ohl_mse_avg
    )
    expr_data["seed"] = "AVERAGED_OVER_SEEDS"
    expr_data["seeds"] = list(seeds)

    expr_data["drift_metrics"] = {
        "y_ks_pvalue": avg_y_ks,
        "y_js_divergence": avg_y_js,
        "y_wasserstein_distance": avg_y_wass,
        "x_any_ks_significant": avg_x_any_ks,
        "x_min_ks_pvalue": avg_x_min_ks,
        "x_avg_js_divergence": avg_x_js,
        "euclidean_distance": avg_distance
    }

    drift_row = QuantifyDrift.build_drift_metrics_row(
        dataset_name=DATASET_NAME,
        drift_type=DRIFT_TYPE,
        n_samples=n_samples,
        n_features=X.shape[1],
        noise_std=meta["noise_std"],
        drift_location=DRIFT_LOCATION,
        drift_metrics=expr_data["drift_metrics"]
    )

    QuantifyDrift.print_drift_metrics_table(drift_row)

    os.makedirs(PLOTTING_DIR, exist_ok=True)

    QuantifyDrift.save_drift_metrics_to_excel(
        drift_row,
        file_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_drift_metrics.xlsx")
    )

    # Save expr_data to txt file
    expr_txt_path = os.path.join(PLOTTING_DIR, f"{EXPERIMENT_NAME}_expr_data.txt")
    with open(expr_txt_path, "w", encoding="utf-8") as f:
        pprint.pprint(expr_data, stream=f, width=120)

    print(f"Saved expr_data to: {expr_txt_path}")

    return expr_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    seeds = Constants.SEEDS
    EXPERIMENT_NAME = "GDS06_gradual_exp6"
    DATASET_NAME = "GDS06"
    DRIFT_TYPE = "gradual"
    GRADUAL_DRIFT_LOCATIONS = [600, 800, 1000, 1200, 1400, 2000]
    GRADUAL_DRIFT_CONCEPTS = ['c1', 'c2', 'c1', 'c2', 'c1', 'c2']
    PLOTTING_ENABLED = True
    TRAIN_PERCENT = 90
    PLOTTING_DIR = '006-OLR-WA-GDS06'

    expr_data = run_multi_seed_experiment(
        seeds=seeds,
        EXPERIMENT_NAME=EXPERIMENT_NAME,
        DATASET_NAME=DATASET_NAME,
        DRIFT_TYPE=DRIFT_TYPE,
        DRIFT_LOCATION=None,
        PLOTTING_ENABLED=PLOTTING_ENABLED,
        TRAIN_PERCENT=TRAIN_PERCENT,
        PLOTTING_DIR=PLOTTING_DIR,
        GRADUAL_DRIFT_LOCATIONS=GRADUAL_DRIFT_LOCATIONS,
        GRADUAL_DRIFT_CONCEPTS=GRADUAL_DRIFT_CONCEPTS
    )

    print(expr_data)