import numpy as np
from Utils import Measures


class OnlineRegressionMetrics:
    """
    Collects online/prequential predictions in chunks.

    PRE metrics:
        Prediction is evaluated before the model sees/updates on the current sample.
        Use this for the official online-learning comparison.

    POST metrics:
        Prediction is evaluated after the model updates/adapts on the current sample.
        Use this only for debugging adaptation behavior.
    """

    def __init__(self, report_interval=10):
        self.report_interval = int(report_interval)

        self._y_true_chunk = []
        self._y_pred_pre_chunk = []
        self._y_pred_post_chunk = []

        self.r2_pre_list = []
        self.mse_pre_list = []

        self.r2_post_list = []
        self.mse_post_list = []

        # Backward-compatible names.
        # These are assigned to PRE metrics in finalize().
        self.r2_list = self.r2_pre_list
        self.mse_list = self.mse_pre_list

    def update(self, y_true, y_pred_pre, y_pred_post=None):
        y_true = float(y_true)
        y_pred_pre = float(y_pred_pre)

        if y_pred_post is None:
            y_pred_post = y_pred_pre
        y_pred_post = float(y_pred_post)

        self._y_true_chunk.append(y_true)
        self._y_pred_pre_chunk.append(y_pred_pre)
        self._y_pred_post_chunk.append(y_pred_post)

        if len(self._y_true_chunk) >= self.report_interval:
            self._flush_chunk()

    def finalize(self):
        if len(self._y_true_chunk) > 0:
            self._flush_chunk()

        # Keep backward compatibility with previous code.
        self.r2_list = self.r2_pre_list
        self.mse_list = self.mse_pre_list

    def _compute_r2_mse(self, y_true, y_pred):
        mse = float(np.mean((y_true - y_pred) ** 2))

        if len(y_true) >= 2 and np.var(y_true) > 0:
            r2 = float(Measures.r2_score_(y_true, y_pred))
        else:
            r2 = 0.0

        return r2, mse

    def _flush_chunk(self):
        y_true = np.array(self._y_true_chunk, dtype=float)
        y_pred_pre = np.array(self._y_pred_pre_chunk, dtype=float)
        y_pred_post = np.array(self._y_pred_post_chunk, dtype=float)

        r2_pre, mse_pre = self._compute_r2_mse(y_true, y_pred_pre)
        r2_post, mse_post = self._compute_r2_mse(y_true, y_pred_post)

        self.r2_pre_list.append(r2_pre)
        self.mse_pre_list.append(mse_pre)

        self.r2_post_list.append(r2_post)
        self.mse_post_list.append(mse_post)

        self._y_true_chunk = []
        self._y_pred_pre_chunk = []
        self._y_pred_post_chunk = []


def add_bias_feature(x):
    """
    Add a constant 1.0 feature so PA learns an intercept/bias term.
    """
    x = np.asarray(x, dtype=float)
    return np.append(x, 1.0)


def initial_pa_weights(n_features):
    """
    Return weights with one extra coefficient for the intercept.
    """
    return np.zeros(int(n_features) + 1, dtype=float)


def pa_predict_one(w, x):
    x_aug = add_bias_feature(x)
    return float(np.dot(w, x_aug))


def pa_update_one(w, x, y_true, C, epsilon):
    """
    PA-II-style update for regression with intercept support.

    Larger C => more aggressive update.
    Smaller C => more conservative update.
    """
    C = float(C)
    if C <= 0:
        raise ValueError("C must be positive.")

    x_aug = add_bias_feature(x)
    y_true = float(y_true)
    y_pred = float(np.dot(w, x_aug))

    loss = max(0.0, abs(y_pred - y_true) - float(epsilon))
    denom = float(np.linalg.norm(x_aug) ** 2 + 1.0 / (2.0 * C))
    tau = loss / denom if denom > 0 else 0.0

    return w + tau * np.sign(y_true - y_pred) * x_aug


def pa_train_from_scratch(X, y, C, epsilon, n_epochs=1):
    """
    Re-train PA from scratch over a recent window.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_samples, n_features = X.shape
    w = initial_pa_weights(n_features)

    for _ in range(int(n_epochs)):
        for i in range(n_samples):
            w = pa_update_one(w, X[i], float(y[i]), C, epsilon)

    return w


def pa_retrain_from_window(recent_X, recent_y, C, epsilon, n_epochs=1):
    """
    Safe helper for retraining from a deque/list window.
    """
    if len(recent_X) == 0:
        return None

    X_win = np.array(recent_X, dtype=float)
    y_win = np.array(recent_y, dtype=float)
    return pa_train_from_scratch(X_win, y_win, C, epsilon, n_epochs=n_epochs)


def select_best_c_on_window(
    w,
    recent_X,
    recent_y,
    epsilon,
    c_candidates,
    n_inner_updates=1
):
    """
    Select C using recent samples.
    """
    if len(recent_X) == 0:
        return float(c_candidates[0])

    X_win = np.array(recent_X, dtype=float)
    y_win = np.array(recent_y, dtype=float)

    best_c = float(c_candidates[0])
    best_mse = float("inf")

    for c_val in c_candidates:
        c_val = float(c_val)
        w_try = w.copy()

        for _ in range(int(n_inner_updates)):
            for x_i, y_i in zip(X_win, y_win):
                w_try = pa_update_one(w_try, x_i, float(y_i), c_val, epsilon)

        preds = np.array([pa_predict_one(w_try, x_i) for x_i in X_win], dtype=float)
        mse = float(np.mean((y_win - preds) ** 2))

        if mse < best_mse:
            best_mse = mse
            best_c = c_val

    return best_c


def clip_c(value, bounds):
    """
    Clip C into a safe interval.
    """
    return float(np.clip(float(value), float(bounds[0]), float(bounds[1])))


def finalize_pa_result(w, metrics, X_test, y_test, metric_mode="pre"):
    """
    Final evaluation using pa_predict_one so the learned intercept is included.

    metric_mode:
        "pre"  -> return pre-update online MSE/R2 lists. Recommended for fair comparison.
        "post" -> return post-update online MSE/R2 lists. Useful only for debugging.
    """
    metrics.finalize()

    predicted_y_test = np.array([pa_predict_one(w, x) for x in X_test], dtype=float)
    final_r2 = float(Measures.r2_score_(y_test, predicted_y_test))

    if metric_mode == "post":
        return final_r2, metrics.r2_post_list, metrics.mse_post_list

    return final_r2, metrics.r2_pre_list, metrics.mse_pre_list
