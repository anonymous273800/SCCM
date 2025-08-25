import warnings

from Datasets.SyntheticDS.Incremental import DS08
from Hyperparameters import Hyperparameter
from Models.OLR_WA import OLR_WA
from Models.OLR_WA import OLR_WA_LSTM_SCCM
from Utils import Printer, Plotter

warnings.filterwarnings("ignore")


def TB_Experiment01(X_train, y_train, X_test, y_test):
    n_samples, n_features = X_train.shape
    print("1. Start of OLR-WA: Fixed Hyperparameters W_inc = 0.5, W_base = 0.5")
    olr_wa_acc, olr_wa_acc_list = OLR_WA.olr_wa_regression_adversarial_fixed_hyperparameters(X_train, y_train,
                                                                                             Hyperparameter.olr_wa_w_base,
                                                                                             Hyperparameter.olr_wa_w_inc,
                                                                                             Hyperparameter.olr_wa_base_model_size0,
                                                                                             Hyperparameter.olr_wa_increment_size(
                                                                                                 n_features,
                                                                                                 user_defined_val=10),
                                                                                             X_test,
                                                                                             y_test)

    Printer.print_list(olr_wa_acc_list, kpi='R2')
    Printer.print_list_tabulate(olr_wa_acc_list)
    print('Final R2 on Test Data', olr_wa_acc)
    print("End of OLR-WA: Fixed Hyperparameters W_inc = 0.5, W_base = 0.5")

    print('----------------------------------------------------------------------------------------')

    print("2. Start of OLR-WA/LSTM-SCCM: Dynamic Hyperparameter Tuning, start with W_inc = 0.5, W_base = 0.5")
    kpi = 'R2'
    multiplier_r2 = 1.5
    adolr_wa_acc, ad_olr_wa_acc_list = OLR_WA_LSTM_SCCM.olr_wa_regression_adversarial_dynamic_hyperparameters(X_train,
                                                                                                              y_train,
                                                                                                              Hyperparameter.olr_wa_w_base,
                                                                                                              Hyperparameter.olr_wa_w_inc,
                                                                                                              Hyperparameter.olr_wa_base_model_size0,
                                                                                                              Hyperparameter.olr_wa_increment_size(
                                                                                                                  n_features,
                                                                                                                  user_defined_val=10),
                                                                                                              X_test,
                                                                                                              y_test,
                                                                                                              kpi=kpi,
                                                                                                              multiplier=multiplier_r2)

    Printer.print_list(ad_olr_wa_acc_list, kpi='R2')
    Printer.print_list_tabulate(ad_olr_wa_acc_list)
    print('Final R2 on Test Data', adolr_wa_acc)

    print('Plotting')
    plotting_enabled = True
    if (plotting_enabled):
        mini_batch_size = Hyperparameter.olr_wa_increment_size2(n_features, user_defined_val=10)
        print("mini_batch_size", mini_batch_size)
        x_axis = [i for i in range(mini_batch_size, n_samples + 1, mini_batch_size)]
        print('x_axis', x_axis)
        y_axis1 = olr_wa_acc_list
        y_axis2 = ad_olr_wa_acc_list
        kpi = 'R2'
        label1 = 'OLR-WA'
        label2 = 'OLR-WA$^*$'
        Plotter.plot_results(x_axis, y_axis1, y_axis2, kpi, label1, label2, drift_location=10000, log_enabled=False,
                             legend_loc='lower left', drift_type='incremental')


if __name__ == "__main__":
    X, y = DS08.get_DS08()

    n_samples, n_features = X.shape
    train_percent = int(99 * n_samples / 100)
    X_train = X[:train_percent]
    y_train = y[:train_percent]
    X_test = X[train_percent:]
    y_test = y[train_percent:]

    TB_Experiment01(X_train, y_train, X_test, y_test)
