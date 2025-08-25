import numpy as np

from Utils import Predictions, Measures


def pa_generic(X_train, y_train, c, epsilon, X_test, y_test):
    w, cost_list = online_passive_aggressive(X_train, y_train, c, epsilon)
    predicted_y_test = Predictions.compute_predictions(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, cost_list


def online_passive_aggressive(X, y, C, epsilon):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    cost_list = np.array([])

    for i in range(n_samples):
        x = X[i]
        y_true = y[i]

        y_pred = np.dot(w, x)

        # epsilon_insensitive_hinge_loss
        loss = max(0, abs(y_pred - y_true) - epsilon)

        # Calculate lagrange multiplier T
        # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
        # T = min(C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
        T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))  # for PA3

        # Update weights
        w += T * np.sign(y_true - y_pred) * x

        cost = np.square(y_true - y_pred)  # make sure this is correct cost.

        cost_list = np.append(cost_list, cost)

    return w, cost_list
