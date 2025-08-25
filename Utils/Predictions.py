import numpy as np


def compute_predictions(xs, w):
    """
        Compute the predicted target values based on the input features and weights.

        Args:
            xs (array-like): Input features.
            w (array-like): Model's weight parameters.

        Returns:
            array-like: Predicted target values.
        """
    y_predicted = [(np.dot(x, w)) for x in xs]
    return y_predicted


def compute_predictions_(xs, w, b):
    """
        Compute the predicted target values based on the input features, weights, and bias.

        Args:
            xs (array-like): Input features.
            w (array-like): Model's weight parameters.
            b (float): Model's bias parameter.

        Returns:
            array-like: Predicted target values.
        """
    y_predicted = np.array([(np.dot(x, w) + b) for x in xs]).flatten()
    return y_predicted


def compute_predictions_XX(xs, w, b):
    """
        Compute the predicted target values based on the input features, weights, and bias.

        Args:
            xs (array-like): Input features.
            w (array-like): Model's weight parameters.
            b (float): Model's bias parameter.

        Returns:
            array-like: Predicted target values.
        """
    y_predicted = np.array([(np.dot(x, w.T) + b) for x in xs]).flatten()
    return y_predicted


def _compute_predictions_(xs, w):
    """
        Compute the predicted target values based on the input features and weights,
        considering that the first element of `w` is the bias term.

        Args:
            xs (array-like): Input features.
            w (array-like): Model's weight parameters.

        Returns:
            array-like: Predicted target values.
        """
    b = w[0]
    w = w[1:]
    return compute_predictions_(xs, w, b)


def _compute_predictions__(xs, w):
    """
        Compute predicted target values using adjusted weights and parameters.

        This function calculates predicted target values based on input features and a set of adjusted
        weights and parameters. The adjustment involves using specific elements from the weight array `w`
        to enhance the prediction process.

        Args:
            xs (array-like): Input features for which predictions will be computed.
            w (array-like): Model's weight parameters including additional adjustment parameters.

        Returns:
            array-like: Predicted target values for the input features.

        Notes:
            The function utilizes specific elements from the `w` array for adjustment:
            - The last element (w[-1]) is assigned to `d_b` for further calculations.
            - The second-to-last element (w[-2]) is assigned to `c_b` for further calculations.
            - The weight array `w` is then updated to exclude the last two elements, removing adjustment parameters.

            The predictions are computed using a linear combination of the adjusted weights and the input features.
            The result is adjusted using `d_b` and `c_b` to fine-tune the prediction.

            This function is particularly useful for specialized prediction scenarios where adjustments based on
            specific parameters are required.
        """
    d_b = w[-1]
    c_b = w[-2]
    w = w[0:-2]
    predicted_test_y = np.array([(-1 * (np.dot(w, x) + d_b) / c_b) for x in xs]).flatten()
    return predicted_test_y
