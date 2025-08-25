import warnings

import numpy as np

from Datasets.SyntheticDS.Gradual import DS12
from Hyperparameters import Hyperparameter
from Models.RLS import RLS, RLS_LSTM_SCCM
from Utils import Printer, Plotter

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


def TB_Experiment04(X_train, y_train, X_test, y_test):
    print("1. RLS: Fixed Hyperparameters equal weights past and new data rls_lambda_ = .99, rls_delta = .01")
    rls_lambda_1 = .99
    rls_delta = .01
    rls_acc, rls_acc_list, rls_mse_list = RLS.rls_generic(X_train, y_train, rls_lambda_1, rls_delta, X_test, y_test)
    print('RLS DS01 on Test Data', rls_acc)
    print()
    Printer.print_list(rls_mse_list, kpi='MSE')
    Printer.print_list_tabulate(rls_mse_list)
    print('----------------------------------------------------------------------------------------')

    # 2. RLS: Dynamic Hyperparameters through LSTM-SCCM rls_lambda_ = dynamic, rls_delta = .01
    print("2. RLS: Dynamic Hyperparameters rls_lambda_ = dynamic, rls_delta = .01")
    rls_lambda_1 = .99
    rls_delta = .01
    kpi = 'MSE'
    multiplier_mse = 2.5
    adrls_acc, adrls_mse_list = RLS_LSTM_SCCM.ad_rls_generic(X_train, y_train, rls_lambda_1, rls_delta,
                                                             X_test, y_test,
                                                             kpi=kpi,
                                                             multiplier=multiplier_mse, DS='DS12')
    Printer.print_list(adrls_mse_list, kpi='MSE')
    Printer.print_list_tabulate(adrls_mse_list)

    print('RLS DS01 on Test Data', adrls_acc)

    print('Plotting')
    plotting_enabled = True
    n_samples, n_features = X.shape
    if (plotting_enabled):
        mini_batch_size = Hyperparameter.standard_mini_batch_size2(n_features, user_defined_val=10)
        mini_batch_size = int(mini_batch_size / 5)
        x_axis = [i for i in range(mini_batch_size, n_samples + 1, mini_batch_size)]
        print('x_axis', x_axis)
        y_axis1 = rls_mse_list[::mini_batch_size]
        y_axis2 = adrls_mse_list[::mini_batch_size]
        kpi = 'MSE'
        label1 = 'RLS'
        label2 = 'RLS$^*$'
        Plotter.plot_results(x_axis, y_axis1, y_axis2, kpi, label1, label2, drift_location=100, log_enabled=False,
                             legend_loc='upper left',
                             drift_type='gradual',
                             gradual_drift_locations=[12500, 17500, 22500, 32500, 37500, 50000],
                             gradual_drift_concepts=['c2', 'c1', 'c2', 'c1', 'c2', 'c1'])


if __name__ == "__main__":
    X, y = DS12.get_DS12()
    n_samples, n_features = X.shape
    train_percent = int(99 * n_samples / 100)
    X_train = X[:train_percent]
    y_train = y[:train_percent]
    X_test = X[train_percent:]
    y_test = y[train_percent:]

    TB_Experiment04(X_train, y_train, X_test, y_test)
