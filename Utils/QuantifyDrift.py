import numpy as np
from sklearn.linear_model import LinearRegression


def quantify_drift(X1, y1, X2, y2):
    # Fit linear regression models
    model1 = LinearRegression().fit(X1, y1)
    model2 = LinearRegression().fit(X2, y2)

    # Get coefficients
    intercept1, coef1 = model1.intercept_, model1.coef_
    intercept2, coef2 = model2.intercept_, model2.coef_

    # Compare coefficients using Euclidean distance
    coef_distance1 = np.linalg.norm(coef1 - coef2)
    print(f"Euclidean distance between coefficients - Method 1: {coef_distance1}")

    delta_intercept = intercept2 - intercept1
    delta_coef = coef2 - coef1
    coef_distance2 = np.sqrt(delta_intercept ** 2 + np.sum(delta_coef ** 2))
    print(f"Euclidean distance between coefficients - Method 2: {coef_distance2}")
