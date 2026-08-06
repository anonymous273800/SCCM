from Utils import Constants
from Utils import ExperimentRecorder
from Datasets.Synthetic.Abrupt import ADS04
import warnings
from Utils import Util
from Models.PA import PA
from Models.PA import PA_SCCM
from Models.PA import PA_ADWIN_RESET
from Models.PA import PA_ADWIN_WINDOW
from Models.PA import PA_ADWIN_SSPT
from Models.PA import PA_ADWIN_OHL
from Models.PA import PA_KSWIN_RESET
from Models.PA import PA_KSWIN_WINDOW
from Models.PA import PA_KSWIN_SSPT
from Models.PA import PA_KSWIN_OHL
from Utils import Plotter
import numpy as np
from Utils import QuantifyDrift
import os
import pprint


PA_C = 1.0
PA_EPSILON = 0.1
PA_SCCM_MULTIPLIER = 1.5
PA_ADWIN_DELTA = 0.002
PA_KSWIN_ALPHA = 0.005
PA_KSWIN_WINDOW_SIZE = 100
PA_KSWIN_STAT_SIZE = 30
PA_WINDOW_SIZE = 50
PA_SSPT_C_CANDIDATES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
PA_OHL_ETA = 0.1
PA_OHL_EPS = 0.05
PA_C_BOUNDS = (0.05, 10.0)
REPORT_INTERVAL = 1


def run_single_seed_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        seed,
        print_details=True
):
    pa_final_r2, pa_r2_list, pa_mse_list = PA.pa_generic(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        report_interval=REPORT_INTERVAL
    )

    pa_sccm_final_r2, pa_sccm_mse_list = PA_SCCM.ad_pa_generic(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        kpi='MSE',
        multiplier=PA_SCCM_MULTIPLIER,
        report_interval=REPORT_INTERVAL,
        ds=DATASET_NAME
    )

    adwin_reset_final_r2, adwin_reset_r2_list, adwin_reset_mse_list = PA_ADWIN_RESET.pa_generic_adwin_reset(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        adwin_delta=PA_ADWIN_DELTA,
        report_interval=REPORT_INTERVAL
    )

    adwin_window_final_r2, adwin_window_r2_list, adwin_window_mse_list = PA_ADWIN_WINDOW.pa_generic_adwin_window(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        adwin_delta=PA_ADWIN_DELTA,
        window_size=PA_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    adwin_sspt_final_r2, adwin_sspt_r2_list, adwin_sspt_mse_list = PA_ADWIN_SSPT.pa_generic_adwin_sspt(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        adwin_delta=PA_ADWIN_DELTA,
        sspt_c_candidates=PA_SSPT_C_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    adwin_ohl_final_r2, adwin_ohl_r2_list, adwin_ohl_mse_list = PA_ADWIN_OHL.pa_generic_adwin_ohl(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        adwin_delta=PA_ADWIN_DELTA,
        ohl_eta=PA_OHL_ETA,
        ohl_eps=PA_OHL_EPS,
        c_bounds=PA_C_BOUNDS,
        report_interval=REPORT_INTERVAL
    )

    kswin_reset_final_r2, kswin_reset_r2_list, kswin_reset_mse_list = PA_KSWIN_RESET.pa_generic_kswin_reset(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        kswin_alpha=PA_KSWIN_ALPHA,
        kswin_window_size=PA_KSWIN_WINDOW_SIZE,
        kswin_stat_size=PA_KSWIN_STAT_SIZE,
        report_interval=REPORT_INTERVAL
    )

    kswin_window_final_r2, kswin_window_r2_list, kswin_window_mse_list = PA_KSWIN_WINDOW.pa_generic_kswin_window(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        kswin_alpha=PA_KSWIN_ALPHA,
        kswin_window_size=PA_KSWIN_WINDOW_SIZE,
        kswin_stat_size=PA_KSWIN_STAT_SIZE,
        window_size=PA_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    kswin_sspt_final_r2, kswin_sspt_r2_list, kswin_sspt_mse_list = PA_KSWIN_SSPT.pa_generic_kswin_sspt(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        kswin_alpha=PA_KSWIN_ALPHA,
        kswin_window_size=PA_KSWIN_WINDOW_SIZE,
        kswin_stat_size=PA_KSWIN_STAT_SIZE,
        sspt_c_candidates=PA_SSPT_C_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    kswin_ohl_final_r2, kswin_ohl_r2_list, kswin_ohl_mse_list = PA_KSWIN_OHL.pa_generic_kswin_ohl(
        X_train,
        y_train,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        kswin_alpha=PA_KSWIN_ALPHA,
        kswin_window_size=PA_KSWIN_WINDOW_SIZE,
        kswin_stat_size=PA_KSWIN_STAT_SIZE,
        ohl_eta=PA_OHL_ETA,
        ohl_eps=PA_OHL_EPS,
        c_bounds=PA_C_BOUNDS,
        report_interval=REPORT_INTERVAL
    )

    return {
        "seed": seed,
        "PA": {"MSE": pa_mse_list},
        "PA-SCCM": {"MSE": pa_sccm_mse_list},
        "PA-ADWIN-RESET": {"MSE": adwin_reset_mse_list},
        "PA-ADWIN-WINDOW": {"MSE": adwin_window_mse_list},
        "PA-ADWIN-SSPT": {"MSE": adwin_sspt_mse_list},
        "PA-ADWIN-OHL": {"MSE": adwin_ohl_mse_list},
        "PA-KSWIN-RESET": {"MSE": kswin_reset_mse_list},
        "PA-KSWIN-WINDOW": {"MSE": kswin_window_mse_list},
        "PA-KSWIN-SSPT": {"MSE": kswin_sspt_mse_list},
        "PA-KSWIN-OHL": {"MSE": kswin_ohl_mse_list}
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
    all_runs = []
    drift_metrics_all = []
    n_samples = None
    meta = None
    X = None

    for seed in seeds:
        print(f"**** Running seed = {seed}")
        X, y, X1, y1, X2, y2, meta = ADS04.get_DS04(seed=seed, return_meta=True)

        drift_metrics_all.append({
            "y_ks_pvalue": meta["y_ks_pvalue"],
            "y_js_divergence": meta["y_js_divergence"],
            "y_wasserstein_distance": meta["y_wasserstein_distance"],
            "x_any_ks_significant": meta["x_any_ks_significant"],
            "x_min_ks_pvalue": meta["x_min_ks_pvalue"],
            "x_avg_js_divergence": meta["x_avg_js_divergence"],
            "euclidean_distance": meta["euclidean_distance_start_end"]
        })

        ADS04.print_DS04_info(X, y, X1, y1, X2, y2, meta)

        n_samples = X.shape[0]
        train_percent = int(TRAIN_PERCENT * n_samples / 100)

        X_train = X[:train_percent]
        y_train = y[:train_percent]
        X_test = X[train_percent:]
        y_test = y[train_percent:]

        one_run = run_single_seed_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
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

    y_ks_vals = [m["y_ks_pvalue"] for m in drift_metrics_all]
    y_js_vals = [m["y_js_divergence"] for m in drift_metrics_all]
    y_wass_vals = [m["y_wasserstein_distance"] for m in drift_metrics_all]
    x_any_ks_vals = [m["x_any_ks_significant"] for m in drift_metrics_all]
    x_min_ks_vals = [m["x_min_ks_pvalue"] for m in drift_metrics_all]
    x_js_vals = [m["x_avg_js_divergence"] for m in drift_metrics_all]
    dist_vals = [m["euclidean_distance"] for m in drift_metrics_all]

    avg_y_ks = np.mean(y_ks_vals)
    avg_y_js = np.mean(y_js_vals)
    avg_y_wass = np.mean(y_wass_vals)
    avg_x_any_ks = any(x_any_ks_vals)
    avg_x_min_ks = np.mean(x_min_ks_vals)
    avg_x_js = np.mean(x_js_vals)
    avg_distance = np.mean(dist_vals)

    pa_mse_avg, _ = Util.average_lists([run["PA"]["MSE"] for run in all_runs])

    pa_sccm_mse_avg, _ = Util.average_lists([run["PA-SCCM"]["MSE"] for run in all_runs])

    adwin_reset_mse_avg, _ = Util.average_lists([run["PA-ADWIN-RESET"]["MSE"] for run in all_runs])

    adwin_window_mse_avg, _ = Util.average_lists([run["PA-ADWIN-WINDOW"]["MSE"] for run in all_runs])

    adwin_sspt_mse_avg, _ = Util.average_lists([run["PA-ADWIN-SSPT"]["MSE"] for run in all_runs])

    adwin_ohl_mse_avg, _ = Util.average_lists([run["PA-ADWIN-OHL"]["MSE"] for run in all_runs])

    kswin_reset_mse_avg, _ = Util.average_lists([run["PA-KSWIN-RESET"]["MSE"] for run in all_runs])

    kswin_window_mse_avg, _ = Util.average_lists([run["PA-KSWIN-WINDOW"]["MSE"] for run in all_runs])

    kswin_sspt_mse_avg, _ = Util.average_lists([run["PA-KSWIN-SSPT"]["MSE"] for run in all_runs])

    kswin_ohl_mse_avg, _ = Util.average_lists([run["PA-KSWIN-OHL"]["MSE"] for run in all_runs])

    min_len = min(
        len(pa_mse_avg),
        len(pa_sccm_mse_avg),
        len(adwin_reset_mse_avg),
        len(adwin_window_mse_avg),
        len(adwin_sspt_mse_avg),
        len(adwin_ohl_mse_avg),
        len(kswin_reset_mse_avg),
        len(kswin_window_mse_avg),
        len(kswin_sspt_mse_avg),
        len(kswin_ohl_mse_avg)
    )

    pa_mse_avg = pa_mse_avg[:min_len]

    pa_sccm_mse_avg = pa_sccm_mse_avg[:min_len]

    adwin_reset_mse_avg = adwin_reset_mse_avg[:min_len]

    adwin_window_mse_avg = adwin_window_mse_avg[:min_len]

    adwin_sspt_mse_avg = adwin_sspt_mse_avg[:min_len]

    adwin_ohl_mse_avg = adwin_ohl_mse_avg[:min_len]

    kswin_reset_mse_avg = kswin_reset_mse_avg[:min_len]

    kswin_window_mse_avg = kswin_window_mse_avg[:min_len]

    kswin_sspt_mse_avg = kswin_sspt_mse_avg[:min_len]

    kswin_ohl_mse_avg = kswin_ohl_mse_avg[:min_len]

    x_axis = [i * REPORT_INTERVAL for i in range(1, min_len + 1)]

    Util.print_mse_lists_results(
        'PA',
        pa_mse_avg,
        pa_sccm_mse_avg,
        adwin_reset_mse_avg,
        adwin_window_mse_avg,adwin_sspt_mse_avg,
        adwin_ohl_mse_avg,
        kswin_reset_mse_avg,
        kswin_window_mse_avg,
        kswin_sspt_mse_avg,
        kswin_ohl_mse_avg
    )

    if PLOTTING_ENABLED:
        os.makedirs(PLOTTING_DIR, exist_ok=True)

        Plotter.plot_results_ten_models_only_mse(
            x_axis,
            pa_mse_avg,
            pa_sccm_mse_avg,
            adwin_reset_mse_avg,
            adwin_window_mse_avg,
            adwin_sspt_mse_avg,
            adwin_ohl_mse_avg,
            kswin_reset_mse_avg,
            kswin_window_mse_avg,
            kswin_sspt_mse_avg,
            kswin_ohl_mse_avg,
            'MSE',
            'PA',
            'PA$^*$',
            'PA$^†$',
            'PA$^‡$',
            'PA$^\\diamond$',
            'PA$^\\parallel$',
            'PA$^§$',
            'PA$^¶$',
            'PA$^\\#$',
            'PA$^\\triangle$',
            drift_location=DRIFT_LOCATION,
            log_enabled=False,
            legend_loc='lower left',
            drift_type=DRIFT_TYPE,
            gradual_drift_locations=None,
            gradual_drift_concepts=None,
            save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_MSE_plot.pdf")
        )



    expr_data = Util.prepare_and_print_experiment_data_new_mse(
        MODEL_NAME='PA',
        experiment_name=EXPERIMENT_NAME,
        dataset_name=DATASET_NAME,
        drift_type=DRIFT_TYPE,
        n_samples=n_samples,
        drift_location=DRIFT_LOCATION,
        increment_size=REPORT_INTERVAL,
        model_mse_list=pa_mse_avg,
        model_sccm_mse_list=pa_sccm_mse_avg,
        adwin_reset_mse_list=adwin_reset_mse_avg,
        adwin_window_mse_list=adwin_window_mse_avg,
        adwin_sspt_mse_list=adwin_sspt_mse_avg,
        adwin_ohl_mse_list=adwin_ohl_mse_avg,
        kswin_reset_mse_list=kswin_reset_mse_avg,
        kswin_window_mse_list=kswin_window_mse_avg,
        kswin_sspt_mse_list=kswin_sspt_mse_avg,
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

    expr_txt_path = os.path.join(PLOTTING_DIR, f"{EXPERIMENT_NAME}_expr_data.txt")
    with open(expr_txt_path, "w", encoding="utf-8") as f:
        pprint.pprint(expr_data, stream=f, width=120)

    print(f"Saved expr_data to: {expr_txt_path}")

    return expr_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    seeds = Constants.SEEDS
    EXPERIMENT_NAME = "ADS04_abrupt_exp4_PA"
    DATASET_NAME = "ADS04"
    DRIFT_TYPE = "abrupt"
    DRIFT_LOCATION = 1000
    PLOTTING_ENABLED = True
    TRAIN_PERCENT = 90
    PLOTTING_DIR = '004-PA-ADS04'

    expr_data = run_multi_seed_experiment(
        seeds=seeds,
        EXPERIMENT_NAME=EXPERIMENT_NAME,
        DATASET_NAME=DATASET_NAME,
        DRIFT_TYPE=DRIFT_TYPE,
        DRIFT_LOCATION=DRIFT_LOCATION,
        PLOTTING_ENABLED=PLOTTING_ENABLED,
        TRAIN_PERCENT=TRAIN_PERCENT,
        PLOTTING_DIR=PLOTTING_DIR
    )

    print(expr_data)