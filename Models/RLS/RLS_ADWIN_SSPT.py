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



def _evaluate_lambda_on_window(w, P, X_window, y_window, lambda_, delta, score_on="post_update"):
    X_window = np.asarray(X_window, dtype=float)
    y_window = np.asarray(y_window, dtype=float).ravel()

    if len(y_window) == 0:
        return float("inf")

    w_try = w.copy()
    P_try = P.copy()
    losses = []

    for x_i, y_i in zip(X_window, y_window):
        if score_on == "pre_update":
            y_pred = rls_predict_one(w_try, x_i)
            losses.append((float(y_i) - y_pred) ** 2)
            w_try, P_try = rls_update_one(w_try, P_try, x_i, float(y_i), lambda_)
        else:
            w_try, P_try = rls_update_one(w_try, P_try, x_i, float(y_i), lambda_)
            y_pred = rls_predict_one(w_try, x_i)
            losses.append((float(y_i) - y_pred) ** 2)

    return float(np.mean(losses))


def _select_best_lambda_on_window(w, P, X_window, y_window, lambda_candidates, delta):
    best_lambda = float(lambda_candidates[0])
    best_loss = float("inf")

    for lambda_val in lambda_candidates:
        lambda_val = float(lambda_val)
        loss = _evaluate_lambda_on_window(w, P, X_window, y_window, lambda_val, delta)
        if loss < best_loss:
            best_loss = loss
            best_lambda = lambda_val

    return best_lambda, best_loss


def rls_generic_adwin_sspt(
    X_train,
    y_train,
    lambda_,
    delta,
    X_test,
    y_test,
    adwin_delta=0.002,
    sspt_lambda_candidates=(0.90, 0.93, 0.95, 0.97, 0.99, 0.995),
    window_size=50,
    tune_hold=20,
    report_interval=1
):
    w, P, metrics = online_rls_adwin_sspt(
        X_train,
        y_train,
        lambda_,
        delta,
        adwin_delta=adwin_delta,
        sspt_lambda_candidates=sspt_lambda_candidates,
        window_size=window_size,
        tune_hold=tune_hold,
        report_interval=report_interval
    )
    return finalize_rls_result(w, metrics, X_test, y_test)


def online_rls_adwin_sspt(
    X,
    y,
    lambda_,
    delta,
    adwin_delta=0.002,
    sspt_lambda_candidates=(0.90, 0.93, 0.95, 0.97, 0.99, 0.995),
    window_size=50,
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
    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    current_lambda = float(lambda_)
    hold_counter = 0

    print(f"ADWIN delta = {adwin_delta}")
    print(f"SSPT window size = {window_size}")
    print(f"SSPT tune_hold = {tune_hold}")

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
            if len(recent_y) >= 2:
                current_lambda, best_loss = _select_best_lambda_on_window(
                    w,
                    P,
                    np.array(recent_X),
                    np.array(recent_y),
                    sspt_lambda_candidates,
                    delta
                )
            else:
                current_lambda = float(lambda_)
                best_loss = sq_error

            hold_counter = int(tune_hold)
            print(f"ADWIN detected drift at global sample index: {i}")
            print(f"SSPT tuned lambda = {current_lambda}, validation_mse = {best_loss:.6f}")

        update_lambda = current_lambda if hold_counter > 0 else float(lambda_)
        w, P = rls_update_one(w, P, x, y_true, update_lambda)

        if hold_counter > 0:
            hold_counter -= 1
            if hold_counter == 0:
                current_lambda = float(lambda_)

    return w, P, metrics
