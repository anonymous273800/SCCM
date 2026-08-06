from collections import deque
from Utils import DriftDetectors as drift

from .pa_family_common import (
    OnlineRegressionMetrics,
    initial_pa_weights,
    pa_predict_one,
    pa_update_one,
    pa_retrain_from_window,
    finalize_pa_result,
)


def pa_generic_adwin_window(
    X_train,
    y_train,
    c,
    epsilon,
    X_test,
    y_test,
    adwin_delta=0.0020,
    window_size=50,
    retrain_epochs=2,
    report_interval=10,
    metric_mode="pre"
):
    w, metrics = online_passive_aggressive_adwin_window(
        X_train,
        y_train,
        c,
        epsilon,
        adwin_delta=adwin_delta,
        window_size=window_size,
        retrain_epochs=retrain_epochs,
        report_interval=report_interval
    )
    return finalize_pa_result(w, metrics, X_test, y_test, metric_mode=metric_mode)


def online_passive_aggressive_adwin_window(
    X,
    y,
    C,
    epsilon,
    adwin_delta=0.0020,
    window_size=50,
    retrain_epochs=2,
    report_interval=10
):
    n_samples, n_features = X.shape
    w = initial_pa_weights(n_features)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    adwin = drift.ADWIN(delta=adwin_delta)
    retrain_count = 0

    recent_X = deque(maxlen=window_size)
    recent_y = deque(maxlen=window_size)

    print(f"ADWIN delta = {adwin_delta}")
    print(f"Sliding window size (samples) = {window_size}")
    print(f"Retrain epochs = {retrain_epochs}")

    for i in range(n_samples):
        x = X[i]
        y_true = float(y[i])

        y_pred_pre = pa_predict_one(w, x)
        sq_error_pre = float((y_true - y_pred_pre) ** 2)

        recent_X.append(x.copy())
        recent_y.append(y_true)

        adwin.update(sq_error_pre)
        drift_detected = bool(adwin.drift_detected)
        print(f"sample-{i} sq_error_before={sq_error_pre:.5f}, drift={drift_detected}")

        if drift_detected:
            print(f"ADWIN detected drift at global sample index: {i}")
            print("WINDOW RETRAIN ACTIVATED")
            retrain_count += 1

            w_new = pa_retrain_from_window(
                recent_X,
                recent_y,
                C,
                epsilon,
                n_epochs=retrain_epochs
            )
            if w_new is not None:
                w = w_new
        else:
            w = pa_update_one(w, x, y_true, C, epsilon)

        y_pred_post = pa_predict_one(w, x)
        metrics.update(y_true, y_pred_pre, y_pred_post)

    print(f"Total ADWIN window retrains: {retrain_count}")
    return w, metrics
