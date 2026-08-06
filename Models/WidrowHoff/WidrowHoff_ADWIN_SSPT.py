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

def widrow_hoff_generic_adwin_sspt(
    X_train,
    y_train,
    learning_rate,
    X_test,
    y_test,
    adwin_delta=0.002,
    sspt_lr_candidates=(0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03),
    window_size=50,
    tune_hold=20,
    report_interval=10
):
    w, metrics = online_widrow_hoff_adwin_sspt(
        X_train,
        y_train,
        learning_rate,
        adwin_delta=adwin_delta,
        sspt_lr_candidates=sspt_lr_candidates,
        window_size=window_size,
        tune_hold=tune_hold,
        report_interval=report_interval
    )
    return finalize_wh_result(w, metrics, X_test, y_test)


def _prequential_mse_after_continuing(w, X_window, y_window, lr):
    X_aug = add_intercept_to_X(X_window)
    y_window = np.asarray(y_window, dtype=float).reshape(-1)
    w_try = np.asarray(w, dtype=float).copy()

    errors = []
    for x_i, y_i in zip(X_aug, y_window):
        y_pred = wh_predict_one(w_try, x_i)
        # errors.append((float(y_i) - y_pred) ** 2)
        errors.append(safe_squared_error(y_i, y_pred))
        w_try = wh_update_one(w_try, x_i, float(y_i), lr)

    return float(np.mean(errors)) if errors else float("inf")


def _select_best_lr_on_window(w, X_window, y_window, lr_candidates):
    best_lr = float(lr_candidates[0])
    best_mse = float("inf")

    for lr in lr_candidates:
        mse = _prequential_mse_after_continuing(w, X_window, y_window, float(lr))
        if mse < best_mse:
            best_mse = mse
            best_lr = float(lr)

    return best_lr, best_mse


def online_widrow_hoff_adwin_sspt(
    X,
    y,
    learning_rate,
    adwin_delta=0.002,
    sspt_lr_candidates=(0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03),
    window_size=50,
    tune_hold=20,
    report_interval=10
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_aug = add_intercept_to_X(X)
    n_samples, n_features_aug = X_aug.shape

    w = np.zeros(n_features_aug, dtype=float)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    adwin = drift.ADWIN(delta=adwin_delta)
    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    current_lr = float(learning_rate)
    hold_counter = 0

    print(f"ADWIN delta = {adwin_delta}")
    print(f"SSPT candidates = {sspt_lr_candidates}")
    print(f"SSPT validation window = {window_size}, tune_hold = {tune_hold}")

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
            tuned_lr, tuned_mse = _select_best_lr_on_window(
                w,
                np.array(recent_X),
                np.array(recent_y),
                sspt_lr_candidates
            )
            current_lr = tuned_lr
            hold_counter = int(tune_hold)

            print(f"ADWIN detected drift at global sample index: {i}")
            print(f"SSPT tuned learning_rate = {current_lr}, validation_mse = {tuned_mse}")

        w = wh_update_one(w, x_aug, y_true, current_lr)

        if hold_counter > 0:
            hold_counter -= 1
            if hold_counter == 0:
                current_lr = float(learning_rate)

    return w, metrics
