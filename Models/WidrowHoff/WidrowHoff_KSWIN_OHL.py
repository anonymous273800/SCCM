import numpy as np
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


def widrow_hoff_generic_kswin_ohl(
    X_train,
    y_train,
    learning_rate,
    X_test,
    y_test,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    ohl_eta=0.02,
    ohl_eps=0.01,
    lr_bounds=(1e-4, 0.05),
    report_interval=10
):
    w, metrics = online_widrow_hoff_kswin_ohl(
        X_train,
        y_train,
        learning_rate,
        kswin_alpha=kswin_alpha,
        kswin_window_size=kswin_window_size,
        kswin_stat_size=kswin_stat_size,
        ohl_eta=ohl_eta,
        ohl_eps=ohl_eps,
        lr_bounds=lr_bounds,
        report_interval=report_interval
    )
    return finalize_wh_result(w, metrics, X_test, y_test)


def _clip(value, bounds):
    return float(np.clip(value, bounds[0], bounds[1]))


def online_widrow_hoff_kswin_ohl(
    X,
    y,
    learning_rate,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    ohl_eta=0.02,
    ohl_eps=0.01,
    lr_bounds=(1e-4, 0.05),
    report_interval=10
):
    X_aug = add_intercept_to_X(X)
    n_samples, n_features_aug = X_aug.shape

    w = np.zeros(n_features_aug, dtype=float)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    kswin = drift.KSWIN(
        alpha=kswin_alpha,
        window_size=kswin_window_size,
        stat_size=kswin_stat_size
    )
    current_lr = _clip(float(learning_rate), lr_bounds)

    print(
        f"KSWIN alpha = {kswin_alpha}, "
        f"window_size = {kswin_window_size}, stat_size = {kswin_stat_size}"
    )
    print(f"OHL eta = {ohl_eta}, lr_bounds = {lr_bounds}")

    for i in range(n_samples):
        x = np.array(X_aug[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = wh_predict_one(w, x)
        # sq_error = float((y_true - y_pred_before) ** 2)
        sq_error = safe_squared_error(y_true, y_pred_before)

        metrics.update(y_true, y_pred_before)

        kswin.update(sq_error)
        drift_detected = bool(kswin.drift_detected)
        print(f"sample-{i} sq_error={sq_error:.5f}, drift={drift_detected}")

        if drift_detected:
            grad_sign = np.sign(sq_error - ohl_eps)
            current_lr = _clip(current_lr * (1.0 + ohl_eta * grad_sign), lr_bounds)

            print(f"KSWIN detected drift at global sample index: {i}")
            print(f"OHL tuned learning_rate = {current_lr:.6f}")

        w = wh_update_one(w, x, y_true, current_lr)

    return w, metrics
