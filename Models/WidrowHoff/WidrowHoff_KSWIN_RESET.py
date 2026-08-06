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

def widrow_hoff_generic_kswin_reset(
    X_train,
    y_train,
    learning_rate,
    X_test,
    y_test,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    window_size=50,
    reset_mode="window",
    retrain_epochs=2,
    report_interval=10
):
    w, metrics = online_widrow_hoff_kswin_reset(
        X_train,
        y_train,
        learning_rate,
        kswin_alpha=kswin_alpha,
        kswin_window_size=kswin_window_size,
        kswin_stat_size=kswin_stat_size,
        window_size=window_size,
        reset_mode=reset_mode,
        retrain_epochs=retrain_epochs,
        report_interval=report_interval
    )
    return finalize_wh_result(w, metrics, X_test, y_test)


def online_widrow_hoff_kswin_reset(
    X,
    y,
    learning_rate,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    window_size=50,
    reset_mode="window",
    retrain_epochs=2,
    report_interval=10
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_aug = add_intercept_to_X(X)
    n_samples, n_features_aug = X_aug.shape

    w = np.zeros(n_features_aug, dtype=float)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    kswin = drift.KSWIN(
        alpha=kswin_alpha,
        window_size=kswin_window_size,
        stat_size=kswin_stat_size
    )
    reset_count = 0

    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    print(
        f"KSWIN alpha = {kswin_alpha}, "
        f"window_size = {kswin_window_size}, stat_size = {kswin_stat_size}"
    )
    print(f"Reset mode = {reset_mode}")
    print(f"Window size = {window_size}")
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

        kswin.update(sq_error)
        drift_detected = bool(kswin.drift_detected)
        print(f"sample-{i} sq_error={sq_error:.5f}, drift={drift_detected}")

        if drift_detected:
            print(f"KSWIN detected drift at global sample index: {i}")
            print("RESET ACTIVATED")
            reset_count += 1

            if reset_mode == "zero":
                w = np.zeros(n_features_aug, dtype=float)
                w = wh_update_one(w, x_aug, y_true, learning_rate)
            elif reset_mode == "soft":
                w = 0.5 * w
                w = wh_update_one(w, x_aug, y_true, learning_rate)
            else:
                # Default: reset by retraining on the actual recent window.
                w = wh_train_from_scratch(
                    np.array(recent_X),
                    np.array(recent_y),
                    learning_rate,
                    epochs=retrain_epochs
                )
        else:
            w = wh_update_one(w, x_aug, y_true, learning_rate)

    print(f"Total KSWIN resets: {reset_count}")
    return w, metrics
