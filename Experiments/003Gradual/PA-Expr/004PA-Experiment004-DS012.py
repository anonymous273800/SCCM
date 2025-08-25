import warnings

import numpy as np
from Datasets.SyntheticDS.Gradual import DS12
from Hyperparameters import Hyperparameter
from Models.PA import PA, PA_LSTM_SCCM
from Utils import Printer, Plotter

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


def TB_Experiment04(X_train, y_train, X_test, y_test):
    print("1. PA: Fixed Hyperparameters equal weights")
    c = .1
    epsilon = .1
    pa_acc, pa_mse_list = PA.pa_generic(X_train, y_train, c, epsilon, X_test, y_test)

    Printer.print_list(pa_mse_list, kpi='MSE')
    Printer.print_list_tabulate(pa_mse_list)

    print('PA DS01 on Test Data', pa_acc)
    print('----------------------------------------------------------------------------------------')

    # 2. PA: Dynamic Hyperparameters through LSTM-SCCM
    print("2. PA: Dynamic Hyperparameter C")
    c = .1
    epsilon = .1
    kpi = 'MSE'
    multiplier_mse = 2.5
    adpa_acc, adpa_mse_list = PA_LSTM_SCCM.ad_pa_generic(X_train, y_train, c, epsilon,
                                                             X_test, y_test,
                                                             kpi=kpi,
                                                             multiplier=multiplier_mse)
    Printer.print_list(adpa_mse_list, kpi='MSE')
    Printer.print_list_tabulate(adpa_mse_list)

    print('RLS DS01 on Test Data', adpa_acc)


    print('Plotting')
    plotting_enabled = True
    n_samples, n_features = X.shape
    if (plotting_enabled):
        # take only every 10 values
        mini_batch_size = Hyperparameter.standard_mini_batch_size2(n_features, user_defined_val=10)
        x_axis = [i for i in range(mini_batch_size, n_samples + 1, mini_batch_size)]
        print('x_axis', x_axis)
        y_axis1 = pa_mse_list[::mini_batch_size]
        y_axis2 = adpa_mse_list[::mini_batch_size]
        kpi = 'MSE'
        label1 = 'PA'
        label2 = 'PA$^*$'
        Plotter.plot_results(x_axis, y_axis1, y_axis2, kpi, label1, label2, drift_location=100, log_enabled=False,
                             legend_loc='upper left',
                             drift_type='gradual',
                             gradual_drift_locations=[25000, 35000, 45000, 65000, 75000, 1000000], gradual_drift_concepts=['c2', 'c1', 'c2', 'c1', 'c2', 'c1'])


import sys
if __name__ == "__main__":
    X, y = DS12.get_DS12()

    n_samples, n_features = X.shape
    train_percent = int(99 * n_samples / 100)
    X_train = X[:train_percent]
    y_train = y[:train_percent]
    X_test = X[train_percent:]
    y_test = y[train_percent:]

    TB_Experiment04(X_train, y_train, X_test, y_test)
