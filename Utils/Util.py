import os

import numpy as np

from Models.BatchRegression import BatchRegression
from Utils import Printer


def get_dataset_path(file_name):
    """
    get the dataset path stored in the project directory.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets', 'Datasets_Generators_CSV', file_name)
    return path


def get_dataset_path_(file_name):
    """
    get the dataset path stored in the project directory.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets\\Real', 'Datasets_Generators_CSV', file_name)
    return path


def get_path_to_save_generated_dataset_file(directory):
    """
    returns the needed path to save the generated figure.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets', 'Datasets_Generators_CSV', directory)
    return path


def create_directory(path):
    """
    creates directory of the specified path if not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created at {path}")
    else:
        print(f"Folder already exists at {path}")


def sum_lists_element_wise(array_a, array_b):
    """
    Summing two arrays element wise
    """
    if len(array_a) == 0:
        array_a = array_b.copy()
        return array_a

    sum_values = np.sum([array_a, array_b], axis=0)
    return sum_values


def combine_two_datasets(xs, ys, xs_new, ys_new):
    """
    Combines two data sets, used to generate adversarial scenarios
    """
    temp1 = list(zip(xs, ys))
    temp2 = list(zip(xs_new, ys_new))
    temp = temp1 + temp2
    res1, res2 = zip(*temp)
    xn, yn = list(res1), list(res2)
    xs_new = np.array(xn)
    ys_new = np.array(yn, dtype=object)
    return xs_new, ys_new


def calculate_no_of_base_model_points(no_of_data_points, base_model_percent=10):
    """
    compute the number of base model points
    which is usually a percent of the total points like 1%, 10%
    """
    calculate_start_points = int(no_of_data_points * base_model_percent / 100)
    return calculate_start_points


def print_header(header_text):
    """
    For formatted printing, header like.
    """
    print(header_text)
    print("=" * len(header_text))


def sample_and_combine(Xj, yj, r_w_base):
    """
    generate sample data from the provided r_w_base which is the base model coefficients
    combines the generated data with the provided input feature matrix and output labels
    to generate the combined (current and sampled from the base model)

    Parameters:
    Xj (numpy.ndarray): Input feature matrix for the primary dataset.
    yj (numpy.ndarray): Output labels for the primary dataset.
    r_w_base (numpy.ndarray): The base model coefficients

    Returns:
    combinedXj (numpy.ndarray): Combined feature matrix
    combinedyj (numpy.ndarray): Combined output labels corresponding to the combined feature matrix.
    """
    d_b = r_w_base[-1]
    c_b = r_w_base[-2]

    base_yj = np.array([])
    for x in Xj:
        res = -1 * (np.dot(r_w_base[0:-2], x) + d_b) / c_b
        base_yj = np.append(base_yj, res)

    combinedXj, combinedyj = combine_two_datasets(Xj, yj, Xj, base_yj)
    return combinedXj, combinedyj


def prepare_and_print_experiment_data(
        experiment_name,
        dataset_name,
        drift_type,
        n_samples,
        drift_location,
        increment_size,
        olr_wa_r2_list,
        olr_wa_mse_list,
        olr_wa_sccm_r2_list,
        olr_wa_sccm_mse_list,
        adwin_reset_r2_list,
        adwin_reset_mse_list,
        adwin_window_r2_list,
        adwin_window_mse_list,
        adwin_sspt_r2_list,
        adwin_sspt_mse_list,
        adwin_ohl_r2_list,
        adwin_ohl_mse_list,
        kswin_reset_r2_list,
        kswin_reset_mse_list,
        kswin_window_r2_list,
        kswin_window_mse_list,
        kswin_sspt_r2_list,
        kswin_sspt_mse_list,
        kswin_ohl_r2_list,
        kswin_ohl_mse_list
):
    """
    Prepare one compact experiment dictionary and print it in a copy-paste friendly way.
    """

    expr_data = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "drift_type": drift_type,
        "n_samples": n_samples,
        "drift_location": drift_location,
        "increment_size": increment_size,
        "methods": {
            "OLR-WA": {
                "R2": list(olr_wa_r2_list),
                "MSE": list(olr_wa_mse_list)
            },
            "OLR-WA-SCCM": {
                "R2": list(olr_wa_sccm_r2_list),
                "MSE": list(olr_wa_sccm_mse_list)
            },
            "OLR-WA-ADWIN-RESET": {
                "R2": list(adwin_reset_r2_list),
                "MSE": list(adwin_reset_mse_list)
            },
            "OLR-WA-ADWIN-WINDOW": {
                "R2": list(adwin_window_r2_list),
                "MSE": list(adwin_window_mse_list)
            },
            "OLR-WA-ADWIN-SSPT": {
                "R2": list(adwin_sspt_r2_list),
                "MSE": list(adwin_sspt_mse_list)
            },
            "OLR-WA-ADWIN-OHL": {
                "R2": list(adwin_ohl_r2_list),
                "MSE": list(adwin_ohl_mse_list)
            },
            "OLR-WA-KSWIN-RESET": {
                "R2": list(kswin_reset_r2_list),
                "MSE": list(kswin_reset_mse_list)
            },
            "OLR-WA-KSWIN-WINDOW": {
                "R2": list(kswin_window_r2_list),
                "MSE": list(kswin_window_mse_list)
            },
            "OLR-WA-KSWIN-SSPT": {
                "R2": list(kswin_sspt_r2_list),
                "MSE": list(kswin_sspt_mse_list)
            },
            "OLR-WA-KSWIN-OHL": {
                "R2": list(kswin_ohl_r2_list),
                "MSE": list(kswin_ohl_mse_list)
            }
        }
    }

    print("\n" + "=" * 100)
    print("Copy the dictionary below:")
    print("=" * 100)
    print(f"expr_data = {expr_data}")
    print("=" * 100 + "\n")

    return expr_data


def average_lists(list_of_lists):
    """
    Element-wise average across multiple lists after trimming to min length.
    """
    min_len = min(len(lst) for lst in list_of_lists)
    trimmed = [lst[:min_len] for lst in list_of_lists]
    return np.mean(np.array(trimmed, dtype=float), axis=0).tolist(), min_len



# def print_acc_mse_lists_results(
#         MODEL_NAME,
#         olr_wa_mse_avg,
#         olr_wa_sccm_mse_avg,
#         adwin_reset_mse_avg,
#         adwin_window_mse_avg,
#         adwin_sspt_mse_avg,
#         adwin_ohl_mse_avg,
#         kswin_reset_mse_avg,
#         kswin_window_mse_avg,
#         kswin_sspt_mse_avg,
#         kswin_ohl_mse_avg
# ):
#     print("\n" + "=" * 100)
#     print("AVERAGED MSE RESULTS ACROSS SEEDS")
#     print("=" * 100)
#
#     print(MODEL_NAME, " Avg MSE list:")
#     Printer.print_list_tabulate(olr_wa_mse_avg)
#
#     print(MODEL_NAME, "SCCM Avg MSE list:")
#     Printer.print_list_tabulate(olr_wa_sccm_mse_avg)
#
#     print("ADWIN-RESET Avg MSE list:")
#     Printer.print_list_tabulate(adwin_reset_mse_avg)
#
#     print("ADWIN-WINDOW Avg MSE list:")
#     Printer.print_list_tabulate(adwin_window_mse_avg)
#
#     print("ADWIN-SSPT Avg MSE list:")
#     Printer.print_list_tabulate(adwin_sspt_mse_avg)
#
#     print("ADWIN-OHL Avg MSE list:")
#     Printer.print_list_tabulate(adwin_ohl_mse_avg)
#
#     print("KSWIN-RESET Avg MSE list:")
#     Printer.print_list_tabulate(kswin_reset_mse_avg)
#
#     print("KSWIN-WINDOW Avg MSE list:")
#     Printer.print_list_tabulate(kswin_window_mse_avg)
#
#     print("KSWIN-SSPT Avg MSE list:")
#     Printer.print_list_tabulate(kswin_sspt_mse_avg)
#
#     print("KSWIN-OHL Avg MSE list:")
#     Printer.print_list_tabulate(kswin_ohl_mse_avg)

def print_acc_mse_lists_results(MODEL_NAME,
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
):
    print("\n" + "=" * 100)
    print("AVERAGED RESULTS ACROSS SEEDS")
    print("=" * 100)

    print(MODEL_NAME, " Avg R2 list:")
    Printer.print_list_tabulate(olr_wa_r2_avg)
    print(MODEL_NAME, " Avg MSE list:")
    Printer.print_list_tabulate(olr_wa_mse_avg)

    print(MODEL_NAME,"SCCM Avg R2 list:")
    Printer.print_list_tabulate(olr_wa_sccm_r2_avg)
    print(MODEL_NAME,"SCCM Avg MSE list:")
    Printer.print_list_tabulate(olr_wa_sccm_mse_avg)

    print("ADWIN-RESET Avg R2 list:")
    Printer.print_list_tabulate(adwin_reset_r2_avg)
    print("ADWIN-RESET Avg MSE list:")
    Printer.print_list_tabulate(adwin_reset_mse_avg)

    print("ADWIN-WINDOW Avg R2 list:")
    Printer.print_list_tabulate(adwin_window_r2_avg)
    print("ADWIN-WINDOW Avg MSE list:")
    Printer.print_list_tabulate(adwin_window_mse_avg)

    print("ADWIN-SSPT Avg R2 list:")
    Printer.print_list_tabulate(adwin_sspt_r2_avg)
    print("ADWIN-SSPT Avg MSE list:")
    Printer.print_list_tabulate(adwin_sspt_mse_avg)

    print("ADWIN-OHL Avg R2 list:")
    Printer.print_list_tabulate(adwin_ohl_r2_avg)
    print("ADWIN-OHL Avg MSE list:")
    Printer.print_list_tabulate(adwin_ohl_mse_avg)

    print("KSWIN-RESET Avg R2 list:")
    Printer.print_list_tabulate(kswin_reset_r2_avg)
    print("KSWIN-RESET Avg MSE list:")
    Printer.print_list_tabulate(kswin_reset_mse_avg)

    print("KSWIN-WINDOW Avg R2 list:")
    Printer.print_list_tabulate(kswin_window_r2_avg)
    print("KSWIN-WINDOW Avg MSE list:")
    Printer.print_list_tabulate(kswin_window_mse_avg)

    print("KSWIN-SSPT Avg R2 list:")
    Printer.print_list_tabulate(kswin_sspt_r2_avg)
    print("KSWIN-SSPT Avg MSE list:")
    Printer.print_list_tabulate(kswin_sspt_mse_avg)

    print("KSWIN-OHL Avg R2 list:")
    Printer.print_list_tabulate(kswin_ohl_r2_avg)
    print("KSWIN-OHL Avg MSE list:")
    Printer.print_list_tabulate(kswin_ohl_mse_avg)


def print_mse_lists_results(MODEL_NAME,
        olr_wa_mse_avg,
        olr_wa_sccm_mse_avg,
        adwin_reset_mse_avg,
        adwin_window_mse_avg,
        adwin_sspt_mse_avg,
        adwin_ohl_mse_avg,
        kswin_reset_mse_avg,
        kswin_window_mse_avg,
        kswin_sspt_mse_avg,
        kswin_ohl_mse_avg
):
    print("\n" + "=" * 100)
    print("AVERAGED RESULTS ACROSS SEEDS")
    print("=" * 100)

    print(MODEL_NAME, " Avg MSE list:")
    Printer.print_list_tabulate(olr_wa_mse_avg)

    print(MODEL_NAME,"SCCM Avg MSE list:")
    Printer.print_list_tabulate(olr_wa_sccm_mse_avg)

    print("ADWIN-RESET Avg MSE list:")
    Printer.print_list_tabulate(adwin_reset_mse_avg)

    print("ADWIN-WINDOW Avg MSE list:")
    Printer.print_list_tabulate(adwin_window_mse_avg)

    print("ADWIN-SSPT Avg MSE list:")
    Printer.print_list_tabulate(adwin_sspt_mse_avg)

    print("ADWIN-OHL Avg MSE list:")
    Printer.print_list_tabulate(adwin_ohl_mse_avg)

    print("KSWIN-RESET Avg MSE list:")
    Printer.print_list_tabulate(kswin_reset_mse_avg)

    print("KSWIN-WINDOW Avg MSE list:")
    Printer.print_list_tabulate(kswin_window_mse_avg)

    print("KSWIN-SSPT Avg MSE list:")
    Printer.print_list_tabulate(kswin_sspt_mse_avg)

    print("KSWIN-OHL Avg MSE list:")
    Printer.print_list_tabulate(kswin_ohl_mse_avg)



def prepare_and_print_experiment_data_new(
        MODEL_NAME,
        experiment_name,
        dataset_name,
        drift_type,
        n_samples,
        drift_location,
        increment_size,
        olr_wa_r2_list,
        olr_wa_mse_list,
        olr_wa_sccm_r2_list,
        olr_wa_sccm_mse_list,
        adwin_reset_r2_list,
        adwin_reset_mse_list,
        adwin_window_r2_list,
        adwin_window_mse_list,
        adwin_sspt_r2_list,
        adwin_sspt_mse_list,
        adwin_ohl_r2_list,
        adwin_ohl_mse_list,
        kswin_reset_r2_list,
        kswin_reset_mse_list,
        kswin_window_r2_list,
        kswin_window_mse_list,
        kswin_sspt_r2_list,
        kswin_sspt_mse_list,
        kswin_ohl_r2_list,
        kswin_ohl_mse_list
):
    """
    Prepare one compact experiment dictionary and print it in a copy-paste friendly way.
    """

    expr_data = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "drift_type": drift_type,
        "n_samples": n_samples,
        "drift_location": drift_location,
        "increment_size": increment_size,
        "methods": {
            MODEL_NAME: {
                "R2": list(olr_wa_r2_list),
                "MSE": list(olr_wa_mse_list)
            },
            f"{MODEL_NAME}-SCCM": {
                "R2": list(olr_wa_sccm_r2_list),
                "MSE": list(olr_wa_sccm_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-RESET": {
                "R2": list(adwin_reset_r2_list),
                "MSE": list(adwin_reset_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-WINDOW": {
                "R2": list(adwin_window_r2_list),
                "MSE": list(adwin_window_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-SSPT": {
                "R2": list(adwin_sspt_r2_list),
                "MSE": list(adwin_sspt_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-OHL": {
                "R2": list(adwin_ohl_r2_list),
                "MSE": list(adwin_ohl_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-RESET": {
                "R2": list(kswin_reset_r2_list),
                "MSE": list(kswin_reset_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-WINDOW": {
                "R2": list(kswin_window_r2_list),
                "MSE": list(kswin_window_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-SSPT": {
                "R2": list(kswin_sspt_r2_list),
                "MSE": list(kswin_sspt_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-OHL": {
                "R2": list(kswin_ohl_r2_list),
                "MSE": list(kswin_ohl_mse_list)
            }
        }
    }

    print("\n" + "=" * 100)
    print("Copy the dictionary below:")
    print("=" * 100)
    print(f"expr_data = {expr_data}")
    print("=" * 100 + "\n")

    return expr_data

def prepare_and_print_experiment_data_new_mse(
        MODEL_NAME,
        experiment_name,
        dataset_name,
        drift_type,
        n_samples,
        drift_location,
        increment_size,
        model_mse_list,
        model_sccm_mse_list,
        adwin_reset_mse_list,
        adwin_window_mse_list,
        adwin_sspt_mse_list,
        adwin_ohl_mse_list,
        kswin_reset_mse_list,
        kswin_window_mse_list,
        kswin_sspt_mse_list,
        kswin_ohl_mse_list
):
    expr_data = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "drift_type": drift_type,
        "n_samples": n_samples,
        "drift_location": drift_location,
        "increment_size": increment_size,
        "methods": {
            MODEL_NAME: {
                "MSE": list(model_mse_list)
            },
            f"{MODEL_NAME}-SCCM": {
                "MSE": list(model_sccm_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-RESET": {
                "MSE": list(adwin_reset_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-WINDOW": {
                "MSE": list(adwin_window_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-SSPT": {
                "MSE": list(adwin_sspt_mse_list)
            },
            f"{MODEL_NAME}-ADWIN-OHL": {
                "MSE": list(adwin_ohl_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-RESET": {
                "MSE": list(kswin_reset_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-WINDOW": {
                "MSE": list(kswin_window_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-SSPT": {
                "MSE": list(kswin_sspt_mse_list)
            },
            f"{MODEL_NAME}-KSWIN-OHL": {
                "MSE": list(kswin_ohl_mse_list)
            }
        }
    }

    print("\n" + "=" * 100)
    print("Copy the MSE dictionary below:")
    print("=" * 100)
    print(f"expr_data = {expr_data}")
    print("=" * 100 + "\n")

    return expr_data

def prepare_and_print_experiment_data_new_r2(
        MODEL_NAME,
        experiment_name,
        dataset_name,
        drift_type,
        n_samples,
        drift_location,
        increment_size,
        model_r2_list,
        model_sccm_r2_list,
        adwin_reset_r2_list,
        adwin_window_r2_list,
        adwin_sspt_r2_list,
        adwin_ohl_r2_list,
        kswin_reset_r2_list,
        kswin_window_r2_list,
        kswin_sspt_r2_list,
        kswin_ohl_r2_list
):
    expr_data = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "drift_type": drift_type,
        "n_samples": n_samples,
        "drift_location": drift_location,
        "increment_size": increment_size,
        "methods": {
            MODEL_NAME: {
                "R2": list(model_r2_list)
            },
            f"{MODEL_NAME}-SCCM": {
                "R2": list(model_sccm_r2_list)
            },
            f"{MODEL_NAME}-ADWIN-RESET": {
                "R2": list(adwin_reset_r2_list)
            },
            f"{MODEL_NAME}-ADWIN-WINDOW": {
                "R2": list(adwin_window_r2_list)
            },
            f"{MODEL_NAME}-ADWIN-SSPT": {
                "R2": list(adwin_sspt_r2_list)
            },
            f"{MODEL_NAME}-ADWIN-OHL": {
                "R2": list(adwin_ohl_r2_list)
            },
            f"{MODEL_NAME}-KSWIN-RESET": {
                "R2": list(kswin_reset_r2_list)
            },
            f"{MODEL_NAME}-KSWIN-WINDOW": {
                "R2": list(kswin_window_r2_list)
            },
            f"{MODEL_NAME}-KSWIN-SSPT": {
                "R2": list(kswin_sspt_r2_list)
            },
            f"{MODEL_NAME}-KSWIN-OHL": {
                "R2": list(kswin_ohl_r2_list)
            }
        }
    }

    print("\n" + "=" * 100)
    print("Copy the R2 dictionary below:")
    print("=" * 100)
    print(f"expr_data = {expr_data}")
    print("=" * 100 + "\n")

    return expr_data


def fit_linear_model_as_plane(X_batch, y_batch):
    r_w = BatchRegression.linear_regression_(X_batch, y_batch)
    coeff = np.array(np.append(np.append(r_w[1:], -1), r_w[0]))
    return coeff