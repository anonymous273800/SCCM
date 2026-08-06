import numpy as np

from .widrowhoff_family_common import (
    OnlineRegressionMetrics,
    add_intercept_to_X,
    wh_predict_one,
    wh_update_one,
    finalize_wh_result,
)


def widrow_hoff_generic(
    X_train,
    y_train,
    learning_rate,
    X_test,
    y_test,
    report_interval=10
):
    w, metrics = online_widrow_hoff(
        X_train,
        y_train,
        learning_rate,
        report_interval=report_interval
    )
    return finalize_wh_result(w, metrics, X_test, y_test)


def online_widrow_hoff(X, y, learning_rate, report_interval=10):
    X_aug = add_intercept_to_X(X)
    n_samples, n_features_aug = X_aug.shape

    w = np.zeros(n_features_aug, dtype=float)
    metrics = OnlineRegressionMetrics(report_interval=report_interval)

    for i in range(n_samples):
        x = np.array(X_aug[i], dtype=float)
        y_true = float(y[i])

        y_pred_before = wh_predict_one(w, x)
        metrics.update(y_true, y_pred_before)

        w = wh_update_one(w, x, y_true, learning_rate)

    return w, metrics
