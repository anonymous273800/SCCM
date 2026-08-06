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


def widrow_hoff_generic_adwin_ohl(
    X_train,
    y_train,
    learning_rate,
    X_test,
    y_test,
    adwin_delta=0.002,
    ohl_eta=0.02,
    ohl_eps=0.01,
    lr_bounds=(1e-4, 0.05),
    report_interval=10
):
    w, metrics = online_widrow_hoff_adwin_ohl(
        X_train,
        y_train,
        learning_rate,
        adwin_delta=adwin_delta,
        ohl_eta=ohl_eta,
        ohl_eps=ohl_eps,
        lr_bounds=lr_bounds,
        report_interval=report_interval
    )
    return finalize_wh_result(w, metrics, X_test, y_test)


def _clip(value, bounds):
    return float(np.clip(value, bounds[0], bounds[1]))


def online_widrow_hoff_adwin_ohl(
    X,
    y,
    learning_rate,
    adwin_delta=0.002,
    ohl_eta=0.02,
    ohl_eps=0.01,
    lr_bounds=(1e-4, 0.05),
    report_interval=10
):
    X_aug = add_intercept_to_X(X)
    n_samples, n_features_aug = X_aug.shape

    w = np.zeros(n_features_aug, dtype=float)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    adwin = drift.ADWIN(delta=adwin_delta)
    current_lr = _clip(float(learning_rate), lr_bounds)

    print(f"ADWIN delta = {adwin_delta}")
    print(f"OHL eta = {ohl_eta}, lr_bounds = {lr_bounds}")

    for i in range(n_samples):
        x = np.array(X_aug[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = wh_predict_one(w, x)
        # sq_error = float((y_true - y_pred_before) ** 2)
        sq_error = safe_squared_error(y_true, y_pred_before)

        metrics.update(y_true, y_pred_before)

        adwin.update(sq_error)
        print(f"sample-{i} sq_error={sq_error:.5f}, drift={adwin.drift_detected}")

        if adwin.drift_detected:
            grad_sign = np.sign(sq_error - ohl_eps)
            current_lr = _clip(current_lr * (1.0 + ohl_eta * grad_sign), lr_bounds)

            print(f"ADWIN detected drift at global sample index: {i}")
            print(f"OHL tuned learning_rate = {current_lr:.6f}")

        w = wh_update_one(w, x, y_true, current_lr)

    return w, metrics
