import numpy as np
from collections import deque
from Utils import DriftDetectors as drift

from .rls_family_common import (
    OnlineRegressionMetrics,
    rls_predict_one,
    rls_update_one,
    rls_train_from_scratch,
    finalize_rls_result,
    initial_rls_weights,
    initial_rls_covariance,
)



def rls_generic_adwin_window(
    X_train,
    y_train,
    lambda_,
    delta,
    X_test,
    y_test,
    adwin_delta=0.002,
    window_size=50,
    retrain_epochs=1,
    report_interval=1
):
    w, P, metrics = online_rls_adwin_window(
        X_train,
        y_train,
        lambda_,
        delta,
        adwin_delta=adwin_delta,
        window_size=window_size,
        retrain_epochs=retrain_epochs,
        report_interval=report_interval
    )
    return finalize_rls_result(w, metrics, X_test, y_test)


def online_rls_adwin_window(
    X,
    y,
    lambda_,
    delta,
    adwin_delta=0.002,
    window_size=50,
    retrain_epochs=1,
    report_interval=10
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n_samples, n_features = X.shape
    w = initial_rls_weights(n_features)
    P = initial_rls_covariance(n_features, delta)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    adwin = drift.ADWIN(delta=adwin_delta)
    retrain_count = 0

    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    print(f"ADWIN delta = {adwin_delta}")
    print(f"Sliding window size (samples) = {window_size}")
    print(f"Retrain epochs = {retrain_epochs}")

    for i in range(n_samples):
        x = np.array(X[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = rls_predict_one(w, x)
        sq_error = float((y_true - y_pred_before) ** 2)

        metrics.update(y_true, y_pred_before)
        adwin.update(sq_error)
        print(f"sample-{i} sq_error={sq_error:.5f}, drift={adwin.drift_detected}")

        recent_X.append(x.copy())
        recent_y.append(y_true)

        if adwin.drift_detected:
            print(f"ADWIN detected drift at global sample index: {i}")
            print("WINDOW RETRAIN ACTIVATED")
            retrain_count += 1

            # Important: do NOT clear the window. Retrain on the recent history.
            w, P = rls_train_from_scratch(
                np.array(recent_X),
                np.array(recent_y),
                lambda_,
                delta,
                epochs=retrain_epochs
            )
        else:
            w, P = rls_update_one(w, P, x, y_true, lambda_)

    print(f"Total ADWIN window retrains: {retrain_count}")
    return w, P, metrics
