import numpy as np
from collections import deque
from Utils import DriftDetectors as drift

from .rls_family_common import (
    OnlineRegressionMetrics,
    rls_predict_one,
    rls_update_one,
    finalize_rls_result,
    initial_rls_weights,
    initial_rls_covariance,
)



def _clip(value, bounds):
    return float(np.clip(value, bounds[0], bounds[1]))


def rls_generic_adwin_ohl(
    X_train,
    y_train,
    lambda_,
    delta,
    X_test,
    y_test,
    adwin_delta=0.002,
    ohl_eta=0.02,
    ohl_eps=0.01,
    lambda_bounds=(0.90, 0.999),
    tune_hold=20,
    report_interval=1
):
    w, P, metrics = online_rls_adwin_ohl(
        X_train,
        y_train,
        lambda_,
        delta,
        adwin_delta=adwin_delta,
        ohl_eta=ohl_eta,
        ohl_eps=ohl_eps,
        lambda_bounds=lambda_bounds,
        tune_hold=tune_hold,
        report_interval=report_interval
    )
    return finalize_rls_result(w, metrics, X_test, y_test)


def online_rls_adwin_ohl(
    X,
    y,
    lambda_,
    delta,
    adwin_delta=0.002,
    ohl_eta=0.02,
    ohl_eps=0.01,
    lambda_bounds=(0.90, 0.999),
    tune_hold=20,
    report_interval=10
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n_samples, n_features = X.shape
    w = initial_rls_weights(n_features)
    P = initial_rls_covariance(n_features, delta)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    adwin = drift.ADWIN(delta=adwin_delta)
    current_lambda = float(lambda_)
    hold_counter = 0

    print(f"ADWIN delta = {adwin_delta}")
    print(f"OHL lambda bounds = {lambda_bounds}")
    print(f"OHL tune_hold = {tune_hold}")

    for i in range(n_samples):
        x = np.array(X[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = rls_predict_one(w, x)
        sq_error = float((y_true - y_pred_before) ** 2)

        metrics.update(y_true, y_pred_before)
        adwin.update(sq_error)
        print(f"sample-{i} sq_error={sq_error:.5f}, drift={adwin.drift_detected}")

        if adwin.drift_detected:
            # Larger error => lower lambda => faster forgetting.
            grad_sign = np.sign(sq_error - ohl_eps)
            current_lambda = _clip(current_lambda - ohl_eta * grad_sign, lambda_bounds)
            hold_counter = int(tune_hold)

            print(f"ADWIN detected drift at global sample index: {i}")
            print(f"OHL tuned lambda = {current_lambda:.6f}")

        update_lambda = current_lambda if hold_counter > 0 else float(lambda_)
        w, P = rls_update_one(w, P, x, y_true, update_lambda)

        if hold_counter > 0:
            hold_counter -= 1
            if hold_counter == 0:
                current_lambda = float(lambda_)

    return w, P, metrics
