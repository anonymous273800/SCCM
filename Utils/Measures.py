import math
import numpy as np
from sklearn.metrics import r2_score
from Utils import Util


def r2_score_(y, y_pred):
    """
    Calculate the R-squared score between the observed and predicted values.

    Args:
        y (array-like): Observed target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: R-squared score.
    """
    r2 = r2_score(y, y_pred)
    return r2


def MSE(Xj, yj, avg_plane, r_w_base):
    """
        Computes the MSE against the combined data (current and sampled from the base model)

        Args:
            Xj (array-like): Input features for dataset Xj.
            yj (array-like): Actual target values for dataset yj.
            avg_plane (array-like): Coefficients of the average hyperplane.
            r_w_base (array-like): Coefficients of the weighted hyperplane.

        Returns:
            float: Mean Squared Error (MSE)
        """

    combinedXj, combinedyj = Util.sample_and_combine(Xj, yj, r_w_base)

    d_avg_plane = avg_plane[-1]
    c_avg_plane = avg_plane[-2]
    avg_plane = avg_plane[0:-2]

    sum = 0
    for x, y in zip(combinedXj, combinedyj):
        sum += (- (np.dot(avg_plane, x) + d_avg_plane) / c_avg_plane - y) ** 2

    return math.sqrt(sum) / len(combinedXj)


