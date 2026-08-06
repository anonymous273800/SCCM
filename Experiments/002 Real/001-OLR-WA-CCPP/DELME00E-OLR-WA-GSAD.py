from Utils import Constants
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
from Datasets.Real import PublicDS
from Utils import Plotter
import os
import pprint
import numpy as np



def run_single_seed_experiment(
        X_model,
        y_model,
        X_test,
        y_test,
        base_model_size,
        increment_size,
        seed,
        print_details=True
):
    """
    Run one experiment for one seed and return raw lists only.
    """

    # Keep any NumPy-based stochastic behavior reproducible.
    np.random.seed(seed)

    # 1) OLR-WA
    olr_wa_final_r2, olr_wa_r2_list, olr_wa_mse_list = OLR_WA.olr_wa(
        X_model,
        y_model,
        Hyperparameter.olr_wa_w_base,
        Hyperparameter.olr_wa_w_inc,
        base_model_size,
        increment_size,
        X_test,
        y_test
    )

    # 2) OLR-WA-SCCM
    multiplier_r2 = 1.5
    olr_wa_sccm_final_r2, olr_wa_sccm_r2_list, olr_wa_sccm_mse_list = OLR_WA_SCCM.olr_wa_sccm(
        X_model,
        y_model,
        Hyperparameter.olr_wa_w_base,
        Hyperparameter.olr_wa_w_inc,
        base_model_size,
        increment_size,
        X_test,
        y_test,
        kpi='R2',
        multiplier=multiplier_r2
    )

    # 3) OLR-WA + ADWIN-RESET
    adwin_reset_final_r2, adwin_reset_r2_list, adwin_reset_mse_list = (
        OLR_WA_ADWIN_RESET.olr_wa_regression_adversarial_adwin_reset(
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
            increment_size,
            X_test,
            y_test,
            adwin_delta=0.002
        )
    )

    # 4) OLR-WA + ADWIN-WINDOW
    adwin_window_final_r2, adwin_window_r2_list, adwin_window_mse_list = (
        OLR_WA_ADWIN_WINDOW.olr_wa_regression_adversarial_adwin_window(
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
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
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
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
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
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
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
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
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
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
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
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
            X_model,
            y_model,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            base_model_size,
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
        "seed": seed,

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
        PLOTTING_ENABLED,
        PLOTTING_DIR,
        GAS_ID
):
    """
    Run GASD as a real chronological regression stream.

    Batch 1 is used for initial training. Batches 2 through 10 are
    presented sequentially as the online test/adaptation stream.
    """

    if GAS_ID is not None and GAS_ID not in PublicDS.GASD_GAS_NAMES:
        raise ValueError(
            "GAS_ID must be between 1 and 6, or None to load all gases. "
            f"Received: {GAS_ID}"
        )

    # Util.get_dataset_path_ points to Datasets/Real in this project.
    dataset_path = Util.get_dataset_path_("GASD")

    # PublicDS loads and preprocesses the dataset in chronological order.
    X, y, batch_ids = PublicDS.get_GASD(
        dataset_path,
        gas_id=GAS_ID,
        return_batch_ids=True
    )

    n_samples = X.shape[0]
    n_features = X.shape[1]

    # PublicDS prepares the complete chronological sequence for the existing
    # model wrappers. Batch 1 remains the exact base-model segment, while
    # Batches 2-10 are retained as the final test stream.
    (
        X_model,
        y_model,
        X_test,
        y_test,
        stream_batch_ids,
        base_model_size,
        base_sample_count
    ) = PublicDS.prepare_GASD_for_existing_model_calls(
        X,
        y,
        batch_ids,
        base_batch=1
    )

    # These are temporal batch transitions, not confirmed ground-truth drifts.
    batch_transition_locations = (
        np.flatnonzero(stream_batch_ids[1:] != stream_batch_ids[:-1]) + 1
    ).tolist()

    observed_batches = np.unique(batch_ids).astype(int).tolist()
    stream_batches = np.unique(stream_batch_ids).astype(int).tolist()

    increment_size = 500

    selected_gas_name = "All gases" if GAS_ID is None else PublicDS.GASD_GAS_NAMES[GAS_ID]

    print(f"Dataset path: {dataset_path}")
    print(f"Dataset: GASD - {selected_gas_name} (gas_id={GAS_ID})")
    print(f"Total samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Observed batches: {observed_batches}")
    print(f"Training samples from Batch 1: {base_sample_count}")
    print(f"Streaming samples from Batches {stream_batches}: {X_test.shape[0]}")
    print(f"Stream-relative batch transitions: {batch_transition_locations}")
    print(f"INCREMENT_SIZE: {increment_size}")
    print(f"Base-model percentage passed to wrappers: {base_model_size:.10f}")
    print(
        "PublicDS stream preparation enabled: the complete chronological "
        "sequence is passed to the existing online loops."
    )

    all_runs = []

    for seed in seeds:
        print(f"**** Running seed = {seed}")

        one_run = run_single_seed_experiment(
            X_model,
            y_model,
            X_test,
            y_test,
            base_model_size,
            increment_size,
            seed=seed,
            print_details=False
        )

        all_runs.append(one_run)

    print("Finished All Seeds now:")

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

        labels = [
            'OLR-WA',
            'OLR-WA$^*$',
            'OLR-WA$^†$',
            'OLR-WA$^‡$',
            'OLR-WA$^\\diamond$',
            'OLR-WA$^\\parallel$',
            'OLR-WA$^§$',
            'OLR-WA$^¶$',
            'OLR-WA$^\\#$',
            'OLR-WA$^\\triangle$'
        ]

        # R2 plot
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
            'R2',
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
            legend_loc='lower left',
            save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_R2_plot.pdf")
        )

        # MSE plot
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
            'MSE',
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
            legend_loc='upper left',
            save_path=os.path.join(PLOTTING_DIR, f"{DATASET_NAME}_MSE_plot.pdf")
        )

    expr_data = Util.prepare_and_print_experiment_data(
        experiment_name=EXPERIMENT_NAME,
        dataset_name=DATASET_NAME,
        drift_type=None,
        n_samples=n_samples,
        drift_location=None,
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
    expr_data["dataset_type"] = "real"
    expr_data["gas_id"] = GAS_ID
    expr_data["gas_name"] = selected_gas_name
    expr_data["training_batch"] = 1
    expr_data["stream_batches"] = stream_batches
    expr_data["training_samples"] = int(base_sample_count)
    expr_data["stream_samples"] = int(X_test.shape[0])
    expr_data["model_input_samples"] = int(X_model.shape[0])
    expr_data["base_model_size_percent"] = float(base_model_size)
    expr_data["known_drift_location"] = None
    expr_data["known_drift_type"] = None
    expr_data["batch_transition_locations"] = batch_transition_locations
    expr_data["batch_transitions_are_proxy_drifts"] = True

    os.makedirs(PLOTTING_DIR, exist_ok=True)

    expr_txt_path = os.path.join(PLOTTING_DIR, f"{EXPERIMENT_NAME}_expr_data.txt")

    with open(expr_txt_path, "w", encoding="utf-8") as f:
        pprint.pprint(expr_data, stream=f, width=120)

    print(f"Saved expr_data to: {expr_txt_path}")

    return expr_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    seeds = Constants.SEEDS5

    # None loads all 13,910 observations from all six gases.
    # Use an integer from 1 through 6 to run one gas only.
    GAS_ID = None
    GAS_NAME = "All-Gases" if GAS_ID is None else PublicDS.GASD_GAS_NAMES[GAS_ID]
    GAS_SUFFIX = "ALL" if GAS_ID is None else f"GAS{GAS_ID}"

    EXPERIMENT_NAME = f"00E-OLR-WA-GASD-{GAS_SUFFIX}"
    DATASET_NAME = f"GASD-{GAS_NAME}"

    PLOTTING_ENABLED = True
    PLOTTING_DIR = f"001-OLR-WA-GASD-{GAS_SUFFIX}"

    expr_data = run_multi_seed_experiment(
        seeds=seeds,
        EXPERIMENT_NAME=EXPERIMENT_NAME,
        DATASET_NAME=DATASET_NAME,
        PLOTTING_ENABLED=PLOTTING_ENABLED,
        PLOTTING_DIR=PLOTTING_DIR,
        GAS_ID=GAS_ID
    )

    print(expr_data)