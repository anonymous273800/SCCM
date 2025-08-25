
import warnings
from Models.WidrowHoff import WidrowHoff, ADWidrowHoff

from Datasets.SyntheticDS.Incremental import DS05, DS06, DS07, DS08
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import Printer, Plotter
from Hyperparameters import Hyperparameter

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


def TB_Experiment01(X_train, y_train, X_test, y_test):
    print("1. WidrowHoff: Fixed Hyperparameters ")
    wh_learning_rate = 0.01  # fixed learning rate, chosen as the best in the last iteration
    # high learning rate, more aggressive update on the learning data, means more adaptation to new data and less to past data.
    wh_acc, wh_cost_list = WidrowHoff.widrow_hoff_generic(X_train, y_train, wh_learning_rate, X_test, y_test)


    Printer.print_list(wh_cost_list, kpi='MSE')
    Printer.print_list_tabulate(wh_cost_list)

    print('----------------------------------------------------------------------------------------')

    # 2. WidrowHoff: Dynamic Hyperparameters through LSTM-SCCM learning_rate variable from 0.01 to 1 (high fits new data more)
    print("2. WidrowHoff: Dynamic Hyperparameters ")
    ad_wh_learning_date = 0.01
    kpi = 'MSE'
    multiplier_mse = 1.5
    # learning_rate start with 0.01, then widrowhoff will dynamically adjust it
    adwidrowhoff_acc, adwidrowhoff_mse_list = ADWidrowHoff.ad_widrow_hoff_generic(X_train, y_train, ad_wh_learning_date, kpi, multiplier_mse, X_test, y_test, DS='DS05')

    Printer.print_list(adwidrowhoff_mse_list, kpi='MSE')
    Printer.print_list_tabulate(adwidrowhoff_mse_list)

    print('ADWidrowHoff DS01 Test Dataset ', adwidrowhoff_acc)

    print('*********************************')

    print('Plotting')
    plotting_enabled = True
    n_samples, n_features = X.shape
    if plotting_enabled:
        # take only every 10 values
        mini_batch_size = Hyperparameter.standard_mini_batch_size1(n_features, user_defined_val=10)
        # mini_batch_size = int(mini_batch_size / 100)

        x_axis = [i for i in range(mini_batch_size, n_samples + 1, mini_batch_size)]
        print('x_axis', x_axis)
        y_axis1 = wh_cost_list[::mini_batch_size]
        y_axis2 = adwidrowhoff_mse_list[::mini_batch_size]
        kpi = 'MSE'
        label1 = 'LMS'
        label2 = 'LMS$^*$'
        drift_location = 100
        Plotter.plot_results(x_axis, y_axis1, y_axis2, kpi, label1, label2, drift_location=drift_location, log_enabled=False,
                             legend_loc='upper left', drift_type='incremental',
                             gradual_drift_concepts=None, gradual_drift_locations=None)


import sys
if __name__ == "__main__":


    X, y = DS05.get_DS05()

    n_samples, n_features = X.shape
    train_percent = int(99 * n_samples / 100)
    X_train = X[:train_percent]
    y_train = y[:train_percent]
    X_test = X[train_percent:]
    y_test = y[train_percent:]

    # Specify the file path where you want to save the console output
    output_file_path = "C:\data\console_output2.txt"
    # Redirect stdout to the specified file
    sys.stdout = open(output_file_path, "w")

    TB_Experiment01(X_train, y_train, X_test, y_test)

    # Now, any print statements or console output will be written to the file
    print("Hello, this will be written to the file.")
    # Remember to close the file when you're done
    sys.stdout.close()