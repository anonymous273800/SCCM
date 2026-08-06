import numpy as np
from collections import deque
from Utils import DriftDetectors as drift

from .widrowhoff_family_common import (
    OnlineRegressionMetrics,
    add_intercept_to_X,
    wh_predict_one,
    wh_update_one,
    wh_train_from_scratch,
    finalize_wh_result,
    safe_squared_error,
)


def widrow_hoff_generic_adwin_window(
    X_train,
    y_train,
    learning_rate,
    X_test,
    y_test,
    adwin_delta=0.002,
    window_size=50,
    retrain_epochs=2,
    report_interval=10
):
    w, metrics = online_widrow_hoff_adwin_window(
        X_train,
        y_train,
        learning_rate,
        adwin_delta=adwin_delta,
        window_size=window_size,
        retrain_epochs=retrain_epochs,
        report_interval=report_interval
    )
    return finalize_wh_result(w, metrics, X_test, y_test)


def online_widrow_hoff_adwin_window(
    X,
    y,
    learning_rate,
    adwin_delta=0.002,
    window_size=50,
    retrain_epochs=2,
    report_interval=10
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_aug = add_intercept_to_X(X)
    n_samples, n_features_aug = X_aug.shape

    w = np.zeros(n_features_aug, dtype=float)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    adwin = drift.ADWIN(delta=adwin_delta)
    retrain_count = 0

    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    print(f"ADWIN delta = {adwin_delta}")
    print(f"Sliding window size (samples) = {window_size}")
    print(f"Retrain epochs = {retrain_epochs}")

    for i in range(n_samples):
        x_aug = np.array(X_aug[i], dtype=float)
        x_raw = np.array(X[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = wh_predict_one(w, x_aug)
        # sq_error = float((y_true - y_pred_before) ** 2)
        sq_error = safe_squared_error(y_true, y_pred_before)

        metrics.update(y_true, y_pred_before)

        recent_X.append(x_raw.copy())
        recent_y.append(y_true)

        adwin.update(sq_error)
        print(f"sample-{i} sq_error={sq_error:.5f}, drift={adwin.drift_detected}")

        if adwin.drift_detected:
            print(f"ADWIN detected drift at global sample index: {i}")
            print("WINDOW RETRAIN ACTIVATED")
            retrain_count += 1

            # Do not clear the window. Retrain on the actual recent window.
            w = wh_train_from_scratch(
                np.array(recent_X),
                np.array(recent_y),
                learning_rate,
                epochs=retrain_epochs
            )
        else:
            w = wh_update_one(w, x_aug, y_true, learning_rate)

    print(f"Total ADWIN window retrains: {retrain_count}")
    return w, metrics
