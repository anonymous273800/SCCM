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



def rls_generic_kswin_reset(
    X_train,
    y_train,
    lambda_,
    delta,
    X_test,
    y_test,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    window_size=50,
    reset_mode="window",
    retrain_epochs=1,
    report_interval=1
):
    w, P, metrics = online_rls_kswin_reset(
        X_train,
        y_train,
        lambda_,
        delta,
        kswin_alpha=kswin_alpha,
        kswin_window_size=kswin_window_size,
        kswin_stat_size=kswin_stat_size,
        window_size=window_size,
        reset_mode=reset_mode,
        retrain_epochs=retrain_epochs,
        report_interval=report_interval
    )
    return finalize_rls_result(w, metrics, X_test, y_test)


def online_rls_kswin_reset(
    X,
    y,
    lambda_,
    delta,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    window_size=50,
    reset_mode="window",
    retrain_epochs=1,
    report_interval=10
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n_samples, n_features = X.shape
    w = initial_rls_weights(n_features)
    P = initial_rls_covariance(n_features, delta)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    kswin = drift.KSWIN(alpha=kswin_alpha, window_size=kswin_window_size, stat_size=kswin_stat_size)
    reset_count = 0

    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    print(f"KSWIN alpha = {kswin_alpha}, window_size = {kswin_window_size}, stat_size = {kswin_stat_size}")
    print(f"Reset mode = {reset_mode}")
    print(f"Window size = {window_size}")
    print(f"Retrain epochs = {retrain_epochs}")

    for i in range(n_samples):
        x = np.array(X[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = rls_predict_one(w, x)
        sq_error = float((y_true - y_pred_before) ** 2)

        metrics.update(y_true, y_pred_before)
        kswin.update(sq_error)
        print(f"sample-{i} sq_error={sq_error:.5f}, drift={kswin.drift_detected}")

        recent_X.append(x.copy())
        recent_y.append(y_true)

        if kswin.drift_detected:
            print(f"KSWIN detected drift at global sample index: {i}")
            print("RESET ACTIVATED")
            reset_count += 1

            if reset_mode == "zero":
                w = initial_rls_weights(n_features)
                P = initial_rls_covariance(n_features, delta)
                w, P = rls_update_one(w, P, x, y_true, lambda_)
            elif reset_mode == "soft":
                w = 0.5 * w
                P = initial_rls_covariance(n_features, delta)
                w, P = rls_update_one(w, P, x, y_true, lambda_)
            elif reset_mode == "window":
                w, P = rls_train_from_scratch(
                    np.array(recent_X),
                    np.array(recent_y),
                    lambda_,
                    delta,
                    epochs=retrain_epochs
                )
            else:
                raise ValueError("reset_mode must be one of: 'window', 'soft', 'zero'.")
        else:
            w, P = rls_update_one(w, P, x, y_true, lambda_)

    print(f"Total KSWIN resets: {reset_count}")
    return w, P, metrics
