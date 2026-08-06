"""
Batch Linear Regression using Pseudo-Inverse

This script implements Batch Linear Regression using the Pseudo-Inverse equation. It includes functions to perform linear
regression, calculate the Pseudo-Inverse, and evaluate the performance of the regression model using R-squared.

Functions:
- `_psudo_inverse_linear_regression`: Calculate the Pseudo-Inverse using the direct method.
- `_psudo_inverse_linear_regression_`: Calculate the Pseudo-Inverse using the Singular Value Decomposition (SVD) method.
- `linear_regression`: Perform Batch Linear Regression using the Pseudo-Inverse (direct method).
- `linear_regression_`: Perform Batch Linear Regression using the Pseudo-Inverse (SVD method).
- `batch_regression_KFold`: Perform K-Fold cross-validation with Batch Linear Regression.
- `batch_regression_adversarial`: Evaluate Batch Linear Regression on adversarial test data.

Dependencies:
- `numpy`: Numerical library for array manipulation and calculations.
- `sklearn.model_selection.KFold`: K-Fold cross-validation for splitting data into training and test sets.
- `Utils`: Custom utility module providing measures, predictions, and other functionalities.

"""

import numpy as np
from sklearn.model_selection import KFold
from Utils import Measures, Predictions


def _pseudo_inverse_linear_regression(X, y):
    """
        Calculate the Pseudo-Inverse using the direct method.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.

        Returns:
            w (array): Optimized coefficient vector.
        """

    XT = np.transpose(X)
    x_pseudo_inv = np.matmul(np.linalg.inv(np.matmul(XT, X)), XT)
    w = np.matmul(x_pseudo_inv, y)
    return w


def _psudo_inverse_linear_regression_(X, y):
    """
        Calculate the Pseudo-Inverse using the Singular Value Decomposition (SVD) method.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.

        Returns:
            w (array): Optimized coefficient vector.
        """

    U, S, VT = np.linalg.svd(X, full_matrices=False)
    tol = np.max(X.shape) * np.finfo(float).eps  # Set tolerance based on machine precision
    S_inv = np.divide(1, S, where=S > tol, out=np.zeros_like(S))
    x_pseudo_inv = np.matmul(VT.T, np.matmul(np.diag(S_inv), U.T))
    w = np.matmul(x_pseudo_inv, y)
    return w


def linear_regression(xs, ys):
    """
        Perform Batch Linear Regression using the Pseudo-Inverse (direct method).

        Args:
            xs (array-like): Input feature matrix.
            ys (array-like): Target values.

        Returns:
            w (array): Optimized coefficient vector.
        """

    x0 = np.ones(len(xs))
    xs = np.concatenate((np.matrix(x0).T, xs), axis=1)

    w = _pseudo_inverse_linear_regression(xs, ys)
    w = np.asarray(w).reshape(-1)  # convert to one 1 array
    return w


def linear_regression_(xs, ys):
    """
       Perform Batch Linear Regression using the Pseudo-Inverse (SVD method).

       Args:
           xs (array-like): Input feature matrix.
           ys (array-like): Target values.

       Returns:
           w (array): Optimized coefficient vector.
       """
    x0 = np.ones(len(xs))
    xs = np.concatenate((np.matrix(x0).T, xs), axis=1)

    w = _psudo_inverse_linear_regression_(xs, ys)
    w = np.asarray(w).reshape(-1)  # convert to one 1 array
    return w


def batch_regression_KFold(X, y, seed):
    """
        Perform K-Fold cross-validation with Batch Linear Regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            seed (int): Random seed for reproducibility.

        Returns:
            acc (float): Mean accuracy (R-squared) across K-Fold splits.
        """
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w = linear_regression(X_train, y_train)
        predicted_y_test = Predictions._compute_predictions_(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()

    return acc


def batch_regression_KFold_(X, y, seed):
    """
        Perform K-Fold cross-validation with Batch Linear Regression.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            seed (int): Random seed for reproducibility.

        Returns:
            acc (float): Mean accuracy (R-squared) across K-Fold splits.
        """
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w = linear_regression_(X_train, y_train)
        predicted_y_test = Predictions._compute_predictions_(X_test, w)
        acc_per_split_for_same_seed = Measures.r2_score_(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()

    return acc

def batch_regression_adversarial(X, y, X_test, y_test):
    """
        Evaluate Batch Linear Regression on adversarial test data.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.
        """

    w = linear_regression(X, y)
    predicted_y_test = Predictions._compute_predictions_(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)
    return acc


