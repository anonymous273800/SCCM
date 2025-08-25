import numpy as np

from Utils import Measures, Predictions


def rls_generic(X_train, y_train, lambda_, delta, X_test, y_test):
    w, epoch_list, cost_list, acc_list = rls(X_train, y_train, lambda_, delta)
    predicted_y_test = Predictions.compute_predictions(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, acc_list, cost_list


def rls(X, y, lambda_, delta):
    """
    Implement the Recursive Least Squares algorithm for optimizing linear regression coefficients.

    Args:
        X (array-like): Input feature matrix.
        y (array-like): Target values.
        lambda_ (float): Forgetting factor, controls the influence of previous data.
        delta (float): Initial value for the covariance matrix P.

    Returns:
        w (array): Optimized coefficient vector.
        epoch_list (array): List of epoch indices.
        cost_list (array): List of corresponding costs.

    """
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    P = delta * np.eye(num_features)

    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    for t in range(len(X)):
        x_t = X[t, :].reshape(-1, 1)
        y_t = y[t]

        error = y_t - np.dot(x_t.T, w)
        K = np.dot(P, x_t) / (lambda_ + np.dot(np.dot(x_t.T, P), x_t))
        w += np.dot(K, error)
        P = (P - np.dot(np.dot(K, x_t.T), P)) / lambda_

        y_predicted = np.dot(w, x_t)

        cost = np.square(y_t - y_predicted)
        cost_list = np.append(cost_list, cost)

    return w, epoch_list, cost_list, acc_list
