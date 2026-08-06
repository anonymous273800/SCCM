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


def _as_1d_x(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)


def add_bias(x):
    """Append intercept feature 1.0 to one sample."""
    x = _as_1d_x(x)
    return np.append(x, 1.0)


def initial_rls_weights(n_features):
    """Weights include one additional intercept/bias coefficient."""
    return np.zeros(int(n_features) + 1, dtype=float)


def initial_rls_covariance(n_features, delta):
    """Covariance matrix includes one additional intercept/bias dimension."""
    return np.eye(int(n_features) + 1, dtype=float) * float(delta)


def rls_predict_one(w, x):
    x_aug = add_bias(x)
    value = float(np.dot(np.asarray(w, dtype=float), x_aug))
    if not np.isfinite(value):
        raise FloatingPointError("RLS produced a non-finite prediction.")
    return value


def rls_update_one(w, P, x, y_true, lambda_):
    """
    One-step Recursive Least Squares update with intercept support.
    x is augmented internally with a constant 1.0.
    """
    if not (0.0 < float(lambda_) <= 1.0):
        raise ValueError("lambda_ must be in (0, 1] for RLS.")

    x_aug = add_bias(x).reshape(-1, 1)
    y_pred = float(w @ x_aug.flatten())
    error = float(y_true - y_pred)

    Px = P @ x_aug
    denom = float(lambda_ + (x_aug.T @ Px)[0, 0])

    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        K = Px / denom
        w_new = w + (K.flatten() * error)
        P_new = (P - K @ x_aug.T @ P) / float(lambda_)
        P_new = 0.5 * (P_new + P_new.T)

    # Numerical safeguard only: reject an update that leaves the finite state.
    # Successful ordinary updates are unchanged.
    if (
        not np.all(np.isfinite(w_new))
        or not np.all(np.isfinite(P_new))
        or np.max(np.abs(w_new)) > 1e8
        or np.max(np.abs(P_new)) > 1e12
    ):
        return np.asarray(w, dtype=float).copy(), np.asarray(P, dtype=float).copy()

    return w_new, P_new


def rls_train_from_scratch(X, y, lambda_, delta, epochs=1):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_samples, n_features = X.shape
    w = initial_rls_weights(n_features)
    P = initial_rls_covariance(n_features, delta)

    for _ in range(int(epochs)):
        for i in range(n_samples):
            w, P = rls_update_one(w, P, X[i], float(y[i]), lambda_)

    return w, P


def rls_predict_many(w, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.array([rls_predict_one(w, x) for x in X], dtype=float)


def finalize_rls_result(w, metrics, X_test, y_test):
    metrics.finalize()
    predicted_y_test = rls_predict_many(w, X_test)
    final_r2 = float(Measures.r2_score_(y_test, predicted_y_test))
    return final_r2, metrics.r2_list, metrics.mse_list
