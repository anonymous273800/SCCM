import numpy as np

from Utils import Measures, Predictions


def widrow_hoff_generic(X_train, y_train, learning_rate, X_test, y_test):
    w, cost_list = widrow_hoff(X_train, y_train, learning_rate)

    predicted_y_test = Predictions._compute_predictions_(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, cost_list


def widrow_hoff(X, y, learning_rate):
    cost_list = np.array([])
    epoch_list = np.array([])

    y_list = np.array([])
    acc_list = np.array([])
    y_pred_list = np.array([])

    n_samples, n_features = X.shape
    w = np.zeros(n_features + 1)
    x0 = np.ones(len(X))
    X = np.concatenate((np.matrix(x0).T, X), axis=1)
    i = 0
    for xs, ys in zip(X, y):
        xs = np.squeeze(np.asarray(xs))
        w = w - (2 * learning_rate * (((np.dot(w.T, xs)) - ys) * xs))

        y_predicted = np.dot(w, xs)
        cost = np.square(ys - y_predicted)  # make sure this is correct cost.
        cost_list = np.append(cost_list, cost)

        y_list = np.append(y_list, ys)
        y_pred_list = np.append(y_pred_list, y_predicted)

    return w, cost_list
