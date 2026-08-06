from collections import deque
from Utils import DriftDetectors as drift

from .pa_family_common import (
    OnlineRegressionMetrics,
    initial_pa_weights,
    pa_predict_one,
    pa_update_one,
    select_best_c_on_window,
    finalize_pa_result,
)


def pa_generic_kswin_sspt(
    X_train,
    y_train,
    c,
    epsilon,
    X_test,
    y_test,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    sspt_c_candidates=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    window_size=50,
    tune_hold=20,
    n_inner_updates=1,
    report_interval=10,
    metric_mode="pre"
):
    w, metrics = online_passive_aggressive_kswin_sspt(
        X_train,
        y_train,
        c,
        epsilon,
        kswin_alpha=kswin_alpha,
        kswin_window_size=kswin_window_size,
        kswin_stat_size=kswin_stat_size,
        sspt_c_candidates=sspt_c_candidates,
        window_size=window_size,
        tune_hold=tune_hold,
        n_inner_updates=n_inner_updates,
        report_interval=report_interval
    )
    return finalize_pa_result(w, metrics, X_test, y_test, metric_mode=metric_mode)


def online_passive_aggressive_kswin_sspt(
    X,
    y,
    C,
    epsilon,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    sspt_c_candidates=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    window_size=50,
    tune_hold=20,
    n_inner_updates=1,
    report_interval=10
):
    n_samples, n_features = X.shape
    w = initial_pa_weights(n_features)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    kswin = drift.KSWIN(alpha=kswin_alpha, window_size=kswin_window_size, stat_size=kswin_stat_size)
    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    current_c = float(C)
    hold_left = 0

    print(f"KSWIN alpha = {kswin_alpha}, window_size = {kswin_window_size}, stat_size = {kswin_stat_size}")
    print(f"SSPT candidates = {sspt_c_candidates}")
    print(f"SSPT window size = {window_size}")
    print(f"SSPT tune hold = {tune_hold}")

    for i in range(n_samples):
        x = X[i]
        y_true = float(y[i])

        y_pred_pre = pa_predict_one(w, x)
        sq_error_pre = float((y_true - y_pred_pre) ** 2)

        recent_X.append(x.copy())
        recent_y.append(y_true)

        kswin.update(sq_error_pre)
        drift_detected = bool(kswin.drift_detected)
        print(f"sample-{i} sq_error_before={sq_error_pre:.5f}, drift={drift_detected}")

        if drift_detected:
            current_c = select_best_c_on_window(
                w,
                recent_X,
                recent_y,
                epsilon,
                sspt_c_candidates,
                n_inner_updates=n_inner_updates
            )
            hold_left = int(tune_hold)
            print(f"KSWIN detected drift at global sample index: {i}")
            print(f"SSPT tuned C = {current_c}")

        c_for_update = current_c if hold_left > 0 else float(C)
        w = pa_update_one(w, x, y_true, c_for_update, epsilon)

        if hold_left > 0:
            hold_left -= 1

        y_pred_post = pa_predict_one(w, x)
        metrics.update(y_true, y_pred_pre, y_pred_post)

    return w, metrics
