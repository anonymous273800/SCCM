from Datasets.Real import PublicDS
from Utils import Constants
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

# PA still updates one observation at a time.
# This controls only how often MSE is stored and reported.
REPORT_INTERVAL = 500


def run_single_seed_experiment(
        X_model,
        y_model,
        X_test,
        y_test,
        seed,
        print_details=True
):
    """
    Run one PA experiment for one seed.

    X_model and y_model contain the complete chronological GASD sequence.

    Batch 1 acts as the initial learning segment.
    Batches 2 through 10 continue as the online stream.

    X_test and y_test contain Batches 2 through 10 and are used for
    the final test R2 calculation performed by the existing wrappers.
    """

    np.random.seed(seed)

    # 1) PA
    pa_final_r2, pa_r2_list, pa_mse_list = PA.pa_generic(
        X_model,
        y_model,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        report_interval=REPORT_INTERVAL
    )

    # 2) PA-SCCM
    pa_sccm_final_r2, pa_sccm_mse_list = PA_SCCM.ad_pa_generic(
        X_model,
        y_model,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        kpi="MSE",
        multiplier=PA_SCCM_MULTIPLIER,
        report_interval=REPORT_INTERVAL,
        ds=DATASET_NAME
    )

    # 3) PA + ADWIN-RESET
    (
        adwin_reset_final_r2,
        adwin_reset_r2_list,
        adwin_reset_mse_list
    ) = PA_ADWIN_RESET.pa_generic_adwin_reset(
        X_model,
        y_model,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        adwin_delta=PA_ADWIN_DELTA,
        report_interval=REPORT_INTERVAL
    )

    # 4) PA + ADWIN-WINDOW
    (
        adwin_window_final_r2,
        adwin_window_r2_list,
        adwin_window_mse_list
    ) = PA_ADWIN_WINDOW.pa_generic_adwin_window(
        X_model,
        y_model,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        adwin_delta=PA_ADWIN_DELTA,
        window_size=PA_WINDOW_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # 5) PA + ADWIN-SSPT
    (
        adwin_sspt_final_r2,
        adwin_sspt_r2_list,
        adwin_sspt_mse_list
    ) = PA_ADWIN_SSPT.pa_generic_adwin_sspt(
        X_model,
        y_model,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        adwin_delta=PA_ADWIN_DELTA,
        sspt_c_candidates=PA_SSPT_C_CANDIDATES,
        report_interval=REPORT_INTERVAL
    )

    # 6) PA + ADWIN-OHL
    (
        adwin_ohl_final_r2,
        adwin_ohl_r2_list,
        adwin_ohl_mse_list
    ) = PA_ADWIN_OHL.pa_generic_adwin_ohl(
        X_model,
        y_model,
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

    # 7) PA + KSWIN-RESET
    (
        kswin_reset_final_r2,
        kswin_reset_r2_list,
        kswin_reset_mse_list
    ) = PA_KSWIN_RESET.pa_generic_kswin_reset(
        X_model,
        y_model,
        PA_C,
        PA_EPSILON,
        X_test,
        y_test,
        kswin_alpha=PA_KSWIN_ALPHA,
        kswin_window_size=PA_KSWIN_WINDOW_SIZE,
        kswin_stat_size=PA_KSWIN_STAT_SIZE,
        report_interval=REPORT_INTERVAL
    )

    # 8) PA + KSWIN-WINDOW
    (
        kswin_window_final_r2,
        kswin_window_r2_list,
        kswin_window_mse_list
    ) = PA_KSWIN_WINDOW.pa_generic_kswin_window(
        X_model,
        y_model,
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

    # 9) PA + KSWIN-SSPT
    (
        kswin_sspt_final_r2,
        kswin_sspt_r2_list,
        kswin_sspt_mse_list
    ) = PA_KSWIN_SSPT.pa_generic_kswin_sspt(
        X_model,
        y_model,
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

    # 10) PA + KSWIN-OHL
    (
        kswin_ohl_final_r2,
        kswin_ohl_r2_list,
        kswin_ohl_mse_list
    ) = PA_KSWIN_OHL.pa_generic_kswin_ohl(
        X_model,
        y_model,
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

        "PA": {
            "MSE": pa_mse_list
        },

        "PA-SCCM": {
            "MSE": pa_sccm_mse_list
        },

        "PA-ADWIN-RESET": {
            "MSE": adwin_reset_mse_list
        },

        "PA-ADWIN-WINDOW": {
            "MSE": adwin_window_mse_list
        },

        "PA-ADWIN-SSPT": {
            "MSE": adwin_sspt_mse_list
        },

        "PA-ADWIN-OHL": {
            "MSE": adwin_ohl_mse_list
        },

        "PA-KSWIN-RESET": {
            "MSE": kswin_reset_mse_list
        },

        "PA-KSWIN-WINDOW": {
            "MSE": kswin_window_mse_list
        },

        "PA-KSWIN-SSPT": {
            "MSE": kswin_sspt_mse_list
        },

        "PA-KSWIN-OHL": {
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
    Run PA and its adaptive variants on GASD.

    Batch 1 is the initial learning segment.
    Batches 2 through 10 form the chronological online stream.

    The complete chronological sequence is passed through the existing
    PA functions because those functions update their models using their
    first X and y arguments.
    """

    if (
            GAS_ID is not None
            and GAS_ID not in PublicDS.GASD_GAS_NAMES
    ):
        raise ValueError(
            "GAS_ID must be between 1 and 6, "
            "or None to load all gases. "
            f"Received: {GAS_ID}"
        )

    dataset_path = Util.get_dataset_path_(
        "GASD"
    )

    X, y, batch_ids = PublicDS.get_GASD(
        dataset_path,
        gas_id=GAS_ID,
        return_batch_ids=True
    )

    n_samples = int(
        X.shape[0]
    )

    n_features = int(
        X.shape[1]
    )

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

    observed_batches = (
        np.unique(batch_ids)
        .astype(int)
        .tolist()
    )

    stream_batches = (
        np.unique(stream_batch_ids)
        .astype(int)
        .tolist()
    )

    # Locations relative to the complete model input.
    model_batch_transition_locations = (
        np.flatnonzero(
            batch_ids[1:] != batch_ids[:-1]
        ) + 1
    ).astype(int).tolist()

    # Locations relative only to Batches 2 through 10.
    stream_batch_transition_locations = (
        np.flatnonzero(
            stream_batch_ids[1:]
            != stream_batch_ids[:-1]
        ) + 1
    ).astype(int).tolist()

    selected_gas_name = (
        "All gases"
        if GAS_ID is None
        else PublicDS.GASD_GAS_NAMES[GAS_ID]
    )

    print(f"Dataset path: {dataset_path}")

    print(
        f"Dataset: GASD - {selected_gas_name} "
        f"(gas_id={GAS_ID})"
    )

    print(
        f"Total samples: {n_samples}"
    )

    print(
        f"Number of features: {n_features}"
    )

    print(
        f"Observed batches: {observed_batches}"
    )

    print(
        "Training samples from Batch 1: "
        f"{base_sample_count}"
    )

    print(
        f"Streaming samples from Batches "
        f"{stream_batches}: {X_test.shape[0]}"
    )

    print(
        "Model-relative batch transitions: "
        f"{model_batch_transition_locations}"
    )

    print(
        "Stream-relative batch transitions: "
        f"{stream_batch_transition_locations}"
    )

    print(
        f"REPORT_INTERVAL: {REPORT_INTERVAL}"
    )

    print(
        f"Batch-1 percentage: "
        f"{base_model_size:.10f}"
    )

    print(
        "PA updates one observation at a time. "
        "REPORT_INTERVAL controls only the stored "
        "and plotted MSE intervals."
    )

    all_runs = []

    for seed in seeds:

        print(
            f"**** Running seed = {seed}"
        )

        one_run = run_single_seed_experiment(
            X_model,
            y_model,
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

    pa_mse_avg, _ = Util.average_lists(
        [
            run["PA"]["MSE"]
            for run in all_runs
        ]
    )

    pa_sccm_mse_avg, _ = Util.average_lists(
        [
            run["PA-SCCM"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_reset_mse_avg, _ = Util.average_lists(
        [
            run["PA-ADWIN-RESET"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_window_mse_avg, _ = Util.average_lists(
        [
            run["PA-ADWIN-WINDOW"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_sspt_mse_avg, _ = Util.average_lists(
        [
            run["PA-ADWIN-SSPT"]["MSE"]
            for run in all_runs
        ]
    )

    adwin_ohl_mse_avg, _ = Util.average_lists(
        [
            run["PA-ADWIN-OHL"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_reset_mse_avg, _ = Util.average_lists(
        [
            run["PA-KSWIN-RESET"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_window_mse_avg, _ = Util.average_lists(
        [
            run["PA-KSWIN-WINDOW"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_sspt_mse_avg, _ = Util.average_lists(
        [
            run["PA-KSWIN-SSPT"]["MSE"]
            for run in all_runs
        ]
    )

    kswin_ohl_mse_avg, _ = Util.average_lists(
        [
            run["PA-KSWIN-OHL"]["MSE"]
            for run in all_runs
        ]
    )

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

    pa_mse_avg = (
        pa_mse_avg[:min_len]
    )

    pa_sccm_mse_avg = (
        pa_sccm_mse_avg[:min_len]
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

    # The final reporting interval can contain fewer than 500 samples.
    x_axis = [
        min(
            i * REPORT_INTERVAL,
            n_samples
        )
        for i in range(
            1,
            min_len + 1
        )
    ]

    Util.print_mse_lists_results(
        "PA",
        pa_mse_avg,
        pa_sccm_mse_avg,
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

        os.makedirs(
            PLOTTING_DIR,
            exist_ok=True
        )

        Plotter.plot_results_ten_models_real_datasets_mse(
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
            "MSE",
            "PA",
            "PA$^*$",
            "PA$^†$",
            "PA$^‡$",
            "PA$^\\diamond$",
            "PA$^\\parallel$",
            "PA$^§$",
            "PA$^¶$",
            "PA$^\\#$",
            "PA$^\\triangle$",
            drift_location=None,
            log_enabled=False,
            legend_loc="upper right",
            drift_type=None,
            gradual_drift_locations=None,
            gradual_drift_concepts=None,
            save_path=os.path.join(
                PLOTTING_DIR,
                f"{DATASET_NAME}_MSE_plot.pdf"
            )
        )

    expr_data = Util.prepare_and_print_experiment_data_new_mse(
        MODEL_NAME="PA",
        experiment_name=EXPERIMENT_NAME,
        dataset_name=DATASET_NAME,
        drift_type=None,
        n_samples=n_samples,
        drift_location=None,
        increment_size=None,

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

    expr_data["seed"] = (
        "AVERAGED_OVER_SEEDS"
    )

    expr_data["seeds"] = (
        list(seeds)
    )

    expr_data["dataset_type"] = (
        "real"
    )

    expr_data["gas_id"] = (
        GAS_ID
    )

    expr_data["gas_name"] = (
        selected_gas_name
    )

    expr_data["training_batch"] = 1

    expr_data["observed_batches"] = (
        observed_batches
    )

    expr_data["stream_batches"] = (
        stream_batches
    )

    expr_data["training_samples"] = (
        int(base_sample_count)
    )

    expr_data["stream_samples"] = (
        int(X_test.shape[0])
    )

    expr_data["model_input_samples"] = (
        int(X_model.shape[0])
    )

    expr_data["n_features"] = (
        n_features
    )

    expr_data["report_interval"] = (
        REPORT_INTERVAL
    )

    expr_data["base_model_size_percent"] = (
        float(base_model_size)
    )

    expr_data["known_drift_location"] = (
        None
    )

    expr_data["known_drift_type"] = (
        None
    )

    expr_data["model_batch_transition_locations"] = (
        model_batch_transition_locations
    )

    expr_data["stream_batch_transition_locations"] = (
        stream_batch_transition_locations
    )

    expr_data["batch_transitions_are_proxy_drifts"] = (
        True
    )

    expr_data["documented_sensor_drift"] = (
        True
    )

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
    ) as f:

        pprint.pprint(
            expr_data,
            stream=f,
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

    # None loads all 13,910 observations from all six gases.
    # Use an integer from 1 through 6 to test one gas only.
    GAS_ID = None

    GAS_NAME = (
        "All-Gases"
        if GAS_ID is None
        else PublicDS.GASD_GAS_NAMES[GAS_ID]
    )

    GAS_SUFFIX = (
        "ALL"
        if GAS_ID is None
        else f"GAS{GAS_ID}"
    )

    EXPERIMENT_NAME = (
        f"003-PA-GASD-{GAS_SUFFIX}"
    )

    DATASET_NAME = (
        f"GASD-{GAS_NAME}"
    )

    PLOTTING_ENABLED = True

    PLOTTING_DIR = (
        f"003-PA-GASD-{GAS_SUFFIX}"
    )

    expr_data = run_multi_seed_experiment(
        seeds=seeds,
        EXPERIMENT_NAME=EXPERIMENT_NAME,
        DATASET_NAME=DATASET_NAME,
        PLOTTING_ENABLED=PLOTTING_ENABLED,
        PLOTTING_DIR=PLOTTING_DIR,
        GAS_ID=GAS_ID
    )

    print(
        expr_data
    )