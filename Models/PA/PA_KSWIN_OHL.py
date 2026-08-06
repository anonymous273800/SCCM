from Utils import DriftDetectors as drift

from .pa_family_common import (
    OnlineRegressionMetrics,
    initial_pa_weights,
    pa_predict_one,
    pa_update_one,
    clip_c,
    finalize_pa_result,
)


def pa_generic_kswin_ohl(
    X_train,
    y_train,
    c,
    epsilon,
    X_test,
    y_test,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    ohl_eta=0.5,
    ohl_eps=0.05,
    c_bounds=(0.1, 10.0),
    tune_hold=20,
    report_interval=10,
    metric_mode="pre"
):
    w, metrics = online_passive_aggressive_kswin_ohl(
        X_train,
        y_train,
        c,
        epsilon,
        kswin_alpha=kswin_alpha,
        kswin_window_size=kswin_window_size,
        kswin_stat_size=kswin_stat_size,
        ohl_eta=ohl_eta,
        ohl_eps=ohl_eps,
        c_bounds=c_bounds,
        tune_hold=tune_hold,
        report_interval=report_interval
    )
    return finalize_pa_result(w, metrics, X_test, y_test, metric_mode=metric_mode)


def online_passive_aggressive_kswin_ohl(
    X,
    y,
    C,
    epsilon,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    ohl_eta=0.5,
    ohl_eps=0.05,
    c_bounds=(0.1, 10.0),
    tune_hold=20,
    report_interval=10
):
    n_samples, n_features = X.shape
    w = initial_pa_weights(n_features)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    kswin = drift.KSWIN(alpha=kswin_alpha, window_size=kswin_window_size, stat_size=kswin_stat_size)
    current_c = clip_c(C, c_bounds)
    hold_left = 0

    print(f"KSWIN alpha = {kswin_alpha}, window_size = {kswin_window_size}, stat_size = {kswin_stat_size}")
    print(f"OHL eta = {ohl_eta}, eps = {ohl_eps}, C bounds = {c_bounds}")
    print(f"OHL tune hold = {tune_hold}")

    for i in range(n_samples):
        x = X[i]
        y_true = float(y[i])

        y_pred_pre = pa_predict_one(w, x)
        abs_error = abs(y_true - y_pred_pre)
        sq_error_pre = float((y_true - y_pred_pre) ** 2)

        kswin.update(sq_error_pre)
        drift_detected = bool(kswin.drift_detected)
        print(f"sample-{i} sq_error_before={sq_error_pre:.5f}, drift={drift_detected}")

        if drift_detected:
            grad_sign = 1.0 if abs_error > (float(epsilon) + float(ohl_eps)) else -1.0
            current_c = clip_c(current_c + float(ohl_eta) * grad_sign, c_bounds)
            hold_left = int(tune_hold)

            print(f"KSWIN detected drift at global sample index: {i}")
            print(f"OHL tuned C = {current_c:.6f}")

        c_for_update = current_c if hold_left > 0 else clip_c(C, c_bounds)
        w = pa_update_one(w, x, y_true, c_for_update, epsilon)

        if hold_left > 0:
            hold_left -= 1

        y_pred_post = pa_predict_one(w, x)
        metrics.update(y_true, y_pred_pre, y_pred_post)

    return w, metrics
