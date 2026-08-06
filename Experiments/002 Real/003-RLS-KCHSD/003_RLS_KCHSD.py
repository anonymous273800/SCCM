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
import os
import pprint


RLS_LAMBDA = 0.99
RLS_DELTA = 1.0
RLS_SCCM_MULTIPLIER = 1.5
RLS_ADWIN_DELTA = 0.002
RLS_KSWIN_ALPHA = 0.005
RLS_KSWIN_WINDOW_SIZE = 100
RLS_KSWIN_STAT_SIZE = 30
RLS_WINDOW_SIZE = 50
RLS_SSPT_LAMBDA_CANDIDATES = (0.90, 0.93, 0.95, 0.97, 0.99, 0.995)
RLS_OHL_ETA = 0.1
RLS_OHL_EPS = 0.01
RLS_LAMBDA_BOUNDS = (0.85, 0.999)
REPORT_INTERVAL = 1


def run_single_seed_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        seed,
        print_details=True
):
    rls_final_r2, rls_r2_list, rls_mse_list = RLS.rls_generic(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        report_interval=REPORT_INTERVAL
    )

    rls_sccm_final_r2, rls_sccm_mse_list = RLS_SCCM.ad_rls_generic(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        kpi='MSE',
        multiplier=RLS_SCCM_MULTIPLIER,
        DS=DATASET_NAME,
        report_interval=REPORT_INTERVAL
    )

    adwin_reset_final_r2, adwin_reset_r2_list, adwin_reset_mse_list = RLS_ADWIN_RESET.rls_generic_adwin_reset(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        adwin_delta=RLS_ADWIN_DELTA,
        report_interval=REPORT_INTERVAL
    )

    adwin_window_final_r2, adwin_window_r2_list, adwin_window_mse_list = RLS_ADWIN_WINDOW.rls_generic_adwin_window(
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

    adwin_sspt_final_r2, adwin_sspt_r2_list, adwin_sspt_mse_list = RLS_ADWIN_SSPT.rls_generic_adwin_sspt(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        adwin_delta=RLS_ADWIN_DELTA,
        sspt_lambda_candidates=RLS_SSPT_LAMBDA_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    adwin_ohl_final_r2, adwin_ohl_r2_list, adwin_ohl_mse_list = RLS_ADWIN_OHL.rls_generic_adwin_ohl(
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

    kswin_reset_final_r2, kswin_reset_r2_list, kswin_reset_mse_list = RLS_KSWIN_RESET.rls_generic_kswin_reset(
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

    kswin_window_final_r2, kswin_window_r2_list, kswin_window_mse_list = RLS_KSWIN_WINDOW.rls_generic_kswin_window(
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

    kswin_sspt_final_r2, kswin_sspt_r2_list, kswin_sspt_mse_list = RLS_KSWIN_SSPT.rls_generic_kswin_sspt(
        X_train,
        y_train,
        RLS_LAMBDA,
        RLS_DELTA,
        X_test,
        y_test,
        kswin_alpha=RLS_KSWIN_ALPHA,
        kswin_window_size=RLS_KSWIN_WINDOW_SIZE,
        kswin_stat_size=RLS_KSWIN_STAT_SIZE,
        sspt_lambda_candidates=RLS_SSPT_LAMBDA_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    kswin_ohl_final_r2, kswin_ohl_r2_list, kswin_ohl_mse_list = RLS_KSWIN_OHL.rls_generic_kswin_ohl(
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
        "RLS": {"MSE": rls_mse_list},
        "RLS-SCCM": {"MSE": rls_sccm_mse_list},
        "RLS-ADWIN-RESET": {"MSE": adwin_reset_mse_list},
        "RLS-ADWIN-WINDOW": {"MSE": adwin_window_mse_list},
        "RLS-ADWIN-SSPT": {"MSE": adwin_sspt_mse_list},
        "RLS-ADWIN-OHL": {"MSE": adwin_ohl_mse_list},
        "RLS-KSWIN-RESET": {"MSE": kswin_reset_mse_list},
        "RLS-KSWIN-WINDOW": {"MSE": kswin_window_mse_list},
        "RLS-KSWIN-SSPT": {"MSE": kswin_sspt_mse_list},
        "RLS-KSWIN-OHL": {"MSE": kswin_ohl_mse_list}
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
    n_samples = None

    for seed in seeds:
        print(f"**** Running seed = {seed}")

        path = Util.get_dataset_path_('07_KCHSD\\007_kc_house_data.csv')
        X, y = PublicDS.get_king_county_house_sales_data(path)

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

    print("Finished All Seeds now:")

    rls_mse_avg, _ = Util.average_lists([run["RLS"]["MSE"] for run in all_runs])

    rls_sccm_mse_avg, _ = Util.average_lists([run["RLS-SCCM"]["MSE"] for run in all_runs])

    adwin_reset_mse_avg, _ = Util.average_lists([run["RLS-ADWIN-RESET"]["MSE"] for run in all_runs])

    adwin_window_mse_avg, _ = Util.average_lists([run["RLS-ADWIN-WINDOW"]["MSE"] for run in all_runs])

    adwin_sspt_mse_avg, _ = Util.average_lists([run["RLS-ADWIN-SSPT"]["MSE"] for run in all_runs])

    adwin_ohl_mse_avg, _ = Util.average_lists([run["RLS-ADWIN-OHL"]["MSE"] for run in all_runs])

    kswin_reset_mse_avg, _ = Util.average_lists([run["RLS-KSWIN-RESET"]["MSE"] for run in all_runs])

    kswin_window_mse_avg, _ = Util.average_lists([run["RLS-KSWIN-WINDOW"]["MSE"] for run in all_runs])

    kswin_sspt_mse_avg, _ = Util.average_lists([run["RLS-KSWIN-SSPT"]["MSE"] for run in all_runs])

    kswin_ohl_mse_avg, _ = Util.average_lists([run["RLS-KSWIN-OHL"]["MSE"] for run in all_runs])

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

    rls_mse_avg = rls_mse_avg[:min_len]
    rls_sccm_mse_avg = rls_sccm_mse_avg[:min_len]
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
        'RLS',
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

    if PLOTTING_ENABLED:
        os.makedirs(PLOTTING_DIR, exist_ok=True)

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
            'MSE',
            'RLS',
            'RLS$^*$',
            'RLS$^†$',
            'RLS$^‡$',
            'RLS$^\\diamond$',
            'RLS$^\\parallel$',
            'RLS$^§$',
            'RLS$^¶$',
            'RLS$^\\#$',
            'RLS$^\\triangle$',
            drift_location=DRIFT_LOCATION,
            log_enabled=False,
            legend_loc='lower left',
            drift_type=DRIFT_TYPE,
            gradual_drift_locations=None,
            gradual_drift_concepts=None,
            save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_MSE_plot.pdf")
        )

    expr_data = Util.prepare_and_print_experiment_data_new_mse(
        MODEL_NAME='RLS',
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

    expr_data["seed"] = "AVERAGED_OVER_SEEDS"
    expr_data["seeds"] = list(seeds)
    expr_data["dataset_type"] = "real"
    expr_data["known_drift_location"] = None
    expr_data["known_drift_type"] = None

    os.makedirs(PLOTTING_DIR, exist_ok=True)

    expr_txt_path = os.path.join(PLOTTING_DIR, f"{EXPERIMENT_NAME}_expr_data.txt")
    with open(expr_txt_path, "w", encoding="utf-8") as f:
        pprint.pprint(expr_data, stream=f, width=120)

    print(f"Saved expr_data to: {expr_txt_path}")

    return expr_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    seeds = Constants.SEEDS5

    EXPERIMENT_NAME = "003-KCHSD_realdataset_exp3_RLS"
    DATASET_NAME = "KCHSD"

    PLOTTING_ENABLED = True
    TRAIN_PERCENT = 90
    PLOTTING_DIR = '003-RLS-KCHSD'

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

    print(expr_data)

