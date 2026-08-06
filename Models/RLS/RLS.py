import numpy as np

from .rls_family_common import (
    OnlineRegressionMetrics,
    rls_predict_one,
    rls_update_one,
    finalize_rls_result,
    initial_rls_weights,
    initial_rls_covariance,
)


def rls_generic(
    X_train,
    y_train,
    lambda_,
    delta,
    X_test,
    y_test,
    report_interval=1
):
    w, P, metrics = online_rls(
        X_train,
        y_train,
        lambda_,
        delta,
        report_interval=report_interval
    )
    return finalize_rls_result(w, metrics, X_test, y_test)


def online_rls(X, y, lambda_, delta, report_interval=10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n_samples, n_features = X.shape
    w = initial_rls_weights(n_features)
    P = initial_rls_covariance(n_features, delta)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    for i in range(n_samples):
        x = np.array(X[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = rls_predict_one(w, x)
        metrics.update(y_true, y_pred_before)

        w, P = rls_update_one(w, P, x, y_true, lambda_)

    return w, P, metrics
