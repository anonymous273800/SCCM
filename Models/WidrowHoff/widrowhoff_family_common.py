import numpy as np
from Utils import Measures


class OnlineRegressionMetrics:
    """
    Collects online/prequential predictions and reports chunked R2 + MSE.
    R2 is computed only when at least 2 points exist in the chunk.
    """

    def __init__(self, report_interval=10):
        self.report_interval = int(report_interval)
        self._y_true_chunk = []
        self._y_pred_chunk = []
        self.r2_list = []
        self.mse_list = []

    def update(self, y_true, y_pred):
        y_true = float(y_true)
        y_pred = float(y_pred)

        self._y_true_chunk.append(y_true)
        self._y_pred_chunk.append(y_pred)

        if len(self._y_true_chunk) >= self.report_interval:
            self._flush_chunk()

    def finalize(self):
        if len(self._y_true_chunk) > 0:
            self._flush_chunk()

    def _flush_chunk(self):
        y_true = np.array(self._y_true_chunk, dtype=float)
        y_pred = np.array(self._y_pred_chunk, dtype=float)

        mse = float(np.mean((y_true - y_pred) ** 2))

        if len(y_true) >= 2 and np.var(y_true) > 0:
            r2 = float(Measures.r2_score_(y_true, y_pred))
        else:
            r2 = 0.0

        self.mse_list.append(mse)
        self.r2_list.append(r2)

        self._y_true_chunk = []
        self._y_pred_chunk = []


def add_intercept_to_X(X):
    """
    Adds a leading bias/intercept column of 1s.

    Input:
        X shape = (n_samples, n_features)

    Output:
        X_aug shape = (n_samples, n_features + 1)
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    intercept = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack((intercept, X))


def initialize_wh_weights(X):
    """
    Initializes weights including the bias/intercept weight.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        n_features = X.shape[0]
    else:
        n_features = X.shape[1]

    return np.zeros(n_features + 1, dtype=float)


def wh_predict_one(w, x):
    """
    Predicts one sample.

    Assumes x already includes the intercept term when len(x) == len(w).
    If x does not include the intercept term, it is added automatically.
    """
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)

    if len(x) == len(w) - 1:
        x = np.insert(x, 0, 1.0)

    if len(x) != len(w):
        raise ValueError(
            f"Shape mismatch in wh_predict_one: len(w)={len(w)}, len(x)={len(x)}"
        )

    return float(np.dot(w, x))


def wh_update_one(w, x, y_true, learning_rate):
    """
    Widrow-Hoff / LMS update with intercept support.

    This matches your old implementation:

        w = w - (2 * learning_rate * (((dot(w.T, x)) - y) * x))

    Equivalent form:

        error = y_true - y_pred
        w_new = w + 2 * learning_rate * error * x
    """
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)

    if len(x) == len(w) - 1:
        x = np.insert(x, 0, 1.0)

    if len(x) != len(w):
        raise ValueError(
            f"Shape mismatch in wh_update_one: len(w)={len(w)}, len(x)={len(x)}"
        )

    y_pred = wh_predict_one(w, x)
    error = float(y_true - y_pred)

    with np.errstate(over="ignore", invalid="ignore"):
        w_new = w + (2.0 * learning_rate * error * x)

    # Reject only numerically divergent updates. Standard finite updates are
    # unchanged. Targets are standardized, so weights above this guard are
    # unequivocal divergence rather than a meaningful solution.
    if not np.all(np.isfinite(w_new)) or np.max(np.abs(w_new)) > 1e6:
        return w.copy()
    return w_new


def wh_train_from_scratch(X, y, learning_rate):
    """
    Trains Widrow-Hoff from scratch using bias/intercept.
    """
    X_aug = add_intercept_to_X(X)
    w = np.zeros(X_aug.shape[1], dtype=float)

    for i in range(X_aug.shape[0]):
        x = np.array(X_aug[i], dtype=float)
        y_true = float(y[i])
        w = wh_update_one(w, x, y_true, learning_rate)

    return w


def wh_predict_many(X, w):
    """
    Predicts multiple samples using bias/intercept.
    """
    X_aug = add_intercept_to_X(X)
    w = np.asarray(w, dtype=float)

    if X_aug.shape[1] != len(w):
        raise ValueError(
            f"Shape mismatch in wh_predict_many: X_aug.shape[1]={X_aug.shape[1]}, len(w)={len(w)}"
        )

    return X_aug @ w


def finalize_wh_result(w, metrics, X_test, y_test):
    """
    Finalizes online metrics and computes final test R2 using intercept-aware prediction.
    """
    metrics.finalize()
    predicted_y_test = wh_predict_many(X_test, w)
    final_r2 = float(Measures.r2_score_(y_test, predicted_y_test))
    return final_r2, metrics.r2_list, metrics.mse_list