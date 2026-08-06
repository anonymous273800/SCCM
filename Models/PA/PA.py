from .pa_family_common import (
    OnlineRegressionMetrics,
    initial_pa_weights,
    pa_predict_one,
    pa_update_one,
    finalize_pa_result,
)


def pa_generic(
    X_train,
    y_train,
    c,
    epsilon,
    X_test,
    y_test,
    report_interval=10,
    metric_mode="pre"
):
    w, metrics = online_passive_aggressive(
        X_train,
        y_train,
        c,
        epsilon,
        report_interval=report_interval
    )
    return finalize_pa_result(w, metrics, X_test, y_test, metric_mode=metric_mode)


def online_passive_aggressive(X, y, C, epsilon, report_interval=10):
    n_samples, n_features = X.shape
    w = initial_pa_weights(n_features)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    for i in range(n_samples):
        x = X[i]
        y_true = float(y[i])

        y_pred_pre = pa_predict_one(w, x)
        w = pa_update_one(w, x, y_true, C, epsilon)
        y_pred_post = pa_predict_one(w, x)

        metrics.update(y_true, y_pred_pre, y_pred_post)

    return w, metrics
