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


WH_LEARNING_RATE = 0.01
WH_SCCM_MULTIPLIER = 1.5
WH_SCCM_DS = 'DS06'
WH_ADWIN_DELTA = 0.002
WH_KSWIN_ALPHA = 0.005
WH_KSWIN_WINDOW_SIZE = 100
WH_KSWIN_STAT_SIZE = 30
WH_WINDOW_SIZE = 50
WH_SSPT_LR_CANDIDATES = (0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03)
WH_OHL_ETA = 0.02
WH_OHL_EPS = 0.01
WH_LR_BOUNDS = (1e-4, 0.05)
REPORT_INTERVAL = 1


def run_single_seed_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        seed,
        print_details=True
):
    wh_final_r2, wh_r2_list, wh_mse_list = WidrowHoff.widrow_hoff_generic(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        report_interval=REPORT_INTERVAL
    )

    wh_sccm_final_r2, wh_sccm_r2_list, wh_sccm_mse_list = WidrowHoff_SCCM.ad_widrow_hoff_generic(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        kpi='MSE',
        multiplier=WH_SCCM_MULTIPLIER,
        DS=WH_SCCM_DS,
        report_interval=REPORT_INTERVAL
    )

    adwin_reset_final_r2, adwin_reset_r2_list, adwin_reset_mse_list = WidrowHoff_ADWIN_RESET.widrow_hoff_generic_adwin_reset(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        adwin_delta=WH_ADWIN_DELTA,
        report_interval=REPORT_INTERVAL
    )

    adwin_window_final_r2, adwin_window_r2_list, adwin_window_mse_list = WidrowHoff_ADWIN_WINDOW.widrow_hoff_generic_adwin_window(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        adwin_delta=WH_ADWIN_DELTA,
        window_size=WH_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    adwin_sspt_final_r2, adwin_sspt_r2_list, adwin_sspt_mse_list = WidrowHoff_ADWIN_SSPT.widrow_hoff_generic_adwin_sspt(
        X_train,
        y_train,
        WH_LEARNING_RATE,
        X_test,
        y_test,
        adwin_delta=WH_ADWIN_DELTA,
        sspt_lr_candidates=WH_SSPT_LR_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    adwin_ohl_final_r2, adwin_ohl_r2_list, adwin_ohl_mse_list = WidrowHoff_ADWIN_OHL.widrow_hoff_generic_adwin_ohl(
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

    kswin_reset_final_r2, kswin_reset_r2_list, kswin_reset_mse_list = WidrowHoff_KSWIN_RESET.widrow_hoff_generic_kswin_reset(
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

    kswin_window_final_r2, kswin_window_r2_list, kswin_window_mse_list = WidrowHoff_KSWIN_WINDOW.widrow_hoff_generic_kswin_window(
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

    kswin_sspt_final_r2, kswin_sspt_r2_list, kswin_sspt_mse_list = WidrowHoff_KSWIN_SSPT.widrow_hoff_generic_kswin_sspt(
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

    kswin_ohl_final_r2, kswin_ohl_r2_list, kswin_ohl_mse_list = WidrowHoff_KSWIN_OHL.widrow_hoff_generic_kswin_ohl(
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
        "Widrow-Hoff": {"MSE": wh_mse_list},
        "Widrow-Hoff-SCCM": {"MSE": wh_sccm_mse_list},
        "Widrow-Hoff-ADWIN-RESET": {"MSE": adwin_reset_mse_list},
        "Widrow-Hoff-ADWIN-WINDOW": {"MSE": adwin_window_mse_list},
        "Widrow-Hoff-ADWIN-SSPT": {"MSE": adwin_sspt_mse_list},
        "Widrow-Hoff-ADWIN-OHL": {"MSE": adwin_ohl_mse_list},
        "Widrow-Hoff-KSWIN-RESET": {"MSE": kswin_reset_mse_list},
        "Widrow-Hoff-KSWIN-WINDOW": {"MSE": kswin_window_mse_list},
        "Widrow-Hoff-KSWIN-SSPT": {"MSE": kswin_sspt_mse_list},
        "Widrow-Hoff-KSWIN-OHL": {"MSE": kswin_ohl_mse_list}
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
        path = Util.get_dataset_path_('06_1KC\\006_1000_Companies.csv')
        X, y = PublicDS.get_profit_estimation_for_companies_dataset(path)

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



    wh_mse_avg, _ = Util.average_lists([run["Widrow-Hoff"]["MSE"] for run in all_runs])

    wh_sccm_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-SCCM"]["MSE"] for run in all_runs])


    adwin_reset_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-ADWIN-RESET"]["MSE"] for run in all_runs])


    adwin_window_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-ADWIN-WINDOW"]["MSE"] for run in all_runs])


    adwin_sspt_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-ADWIN-SSPT"]["MSE"] for run in all_runs])

    adwin_ohl_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-ADWIN-OHL"]["MSE"] for run in all_runs])

    kswin_reset_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-KSWIN-RESET"]["MSE"] for run in all_runs])

    kswin_window_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-KSWIN-WINDOW"]["MSE"] for run in all_runs])

    kswin_sspt_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-KSWIN-SSPT"]["MSE"] for run in all_runs])

    kswin_ohl_mse_avg, _ = Util.average_lists([run["Widrow-Hoff-KSWIN-OHL"]["MSE"] for run in all_runs])

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
    wh_mse_avg = wh_mse_avg[:min_len]

    wh_sccm_mse_avg = wh_sccm_mse_avg[:min_len]

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
        'WidrowHoff',
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

    if PLOTTING_ENABLED:
        os.makedirs(PLOTTING_DIR, exist_ok=True)

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
            'MSE',
            'WH',
            'WH$^*$',
            'WH$^†$',
            'WH$^‡$',
            'WH$^\\diamond$',
            'WH$^\\parallel$',
            'WH$^§$',
            'WH$^¶$',
            'WH$^\\#$',
            'WH$^\\triangle$',
            drift_location=DRIFT_LOCATION,
            log_enabled=False,
            legend_loc='lower left',
            drift_type=DRIFT_TYPE,
            gradual_drift_locations=None,
            gradual_drift_concepts=None,
            save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_R2_plot.pdf")
        )


    expr_data = Util.prepare_and_print_experiment_data_new_mse(
        'Widrow-Hoff',
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

    EXPERIMENT_NAME = "004-WH_realdataset_exp4"
    DATASET_NAME = "1KC"

    PLOTTING_ENABLED = True
    TRAIN_PERCENT = 90
    PLOTTING_DIR = '004-WH-1KC'

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