import warnings

import numpy as np

from Datasets.SyntheticDS.Abrupt import DS02
from Hyperparameters import Hyperparameter
from Models.WidrowHoff import WidrowHoff, ADWidrowHoff
from Utils import Printer, Plotter

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


def TB_Experiment01(X_train, y_train, X_test, y_test):
    print("1. WidrowHoff: Fixed Hyperparameters ")
    wh_learning_rate = 0.01
    wh_acc, wh_cost_list = WidrowHoff.widrow_hoff_generic(X_train, y_train, wh_learning_rate, X_test, y_test)

    Printer.print_list(wh_cost_list, kpi='MSE')
    Printer.print_list_tabulate(wh_cost_list)

    print('----------------------------------------------------------------------------------------')

    # 2. WidrowHoff: Dynamic Hyperparameters through LSTM-SCCM learning_rate variable from 0.01 to 1 (high fits new data more)
    print("2. WidrowHoff: Dynamic Hyperparameters ")
    ad_wh_learning_date = 0.01
    kpi = 'MSE'
    multiplier_mse = 1.5
    adwidrowhoff_acc, adwidrowhoff_mse_list = ADWidrowHoff.ad_widrow_hoff_generic(X_train, y_train, ad_wh_learning_date,
                                                                                  kpi, multiplier_mse, X_test, y_test,
                                                                                  DS='DS02')

    Printer.print_list(adwidrowhoff_mse_list, kpi='MSE')
    Printer.print_list_tabulate(adwidrowhoff_mse_list)

    print('ADWidrowHoff DS01 Test Dataset ', adwidrowhoff_acc)

    print('*********************************')

    print('Plotting')
    plotting_enabled = True
    n_samples, n_features = X.shape
    if plotting_enabled:
        mini_batch_size = Hyperparameter.standard_mini_batch_size2(n_features, user_defined_val=10)

        x_axis = [i for i in range(mini_batch_size, n_samples + 1, mini_batch_size)]
        print('x_axis', x_axis)
        y_axis1 = wh_cost_list[::mini_batch_size]
        # y_axis2 = wh_cost_list[::mini_batch_size]
        y_axis2 = adwidrowhoff_mse_list[::mini_batch_size]
        kpi = 'MSE'
        label1 = 'LMS'
        label2 = 'LMS$^*$'
        drift_location = 5000
        Plotter.plot_results(x_axis, y_axis1, y_axis2, kpi, label1, label2, drift_location=drift_location,
                             log_enabled=False,
                             legend_loc='upper left', drift_type='abrupt', gradual_drift_locations=None,
                             gradual_drift_concepts=None)


import sys

if __name__ == "__main__":
    X, y = DS02.get_DS02()

    n_samples, n_features = X.shape
    train_percent = int(99 * n_samples / 100)
    X_train = X[:train_percent]
    y_train = y[:train_percent]
    X_test = X[train_percent:]
    y_test = y[train_percent:]

    TB_Experiment01(X_train, y_train, X_test, y_test)

