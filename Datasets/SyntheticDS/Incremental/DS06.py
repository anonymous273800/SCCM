import matplotlib.pyplot as plt
from sklearn import datasets

from Utils import Util, QuantifyDrift

'''
Incremental Drift
'''


def get_DS06():
    n_samples = 1000
    n_features = 10
    noise = 20
    seed = 42

    X0, y0 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)

    X1, y1 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X1 *= .9

    X, y = Util.combine_two_datasets(X0, y0, X1, y1)
    QuantifyDrift.quantify_drift(X0, y0, X1, y1)

    X2, y2 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X2 *= .8

    X, y = Util.combine_two_datasets(X, y, X2, y2)
    QuantifyDrift.quantify_drift(X1, y1, X2, y2)

    X3, y3 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X3 *= .7

    X, y = Util.combine_two_datasets(X, y, X3, y3)
    QuantifyDrift.quantify_drift(X2, y2, X3, y3)

    X4, y4 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X4 *= .6

    X, y = Util.combine_two_datasets(X, y, X4, y4)
    QuantifyDrift.quantify_drift(X3, y3, X4, y4)

    X5, y5 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X5 *= .5
    X, y = Util.combine_two_datasets(X, y, X5, y5)
    QuantifyDrift.quantify_drift(X4, y4, X5, y5)

    X6, y6 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X6 *= .4
    X, y = Util.combine_two_datasets(X, y, X6, y6)
    QuantifyDrift.quantify_drift(X5, y5, X6, y6)

    X7, y7 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X7 *= .3
    X, y = Util.combine_two_datasets(X, y, X7, y7)
    QuantifyDrift.quantify_drift(X6, y6, X7, y7)

    X8, y8 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X8 *= .2
    X, y = Util.combine_two_datasets(X, y, X8, y8)
    QuantifyDrift.quantify_drift(X7, y7, X8, y8)

    X9, y9 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X9 *= .1
    X, y = Util.combine_two_datasets(X, y, X9, y9)
    QuantifyDrift.quantify_drift(X8, y8, X9, y9)

    print("Final Quantiying Drift Between Start Concept and End Concept")
    QuantifyDrift.quantify_drift(X0, y0, X9, y9)

    return X, y


if __name__ == "__main__":

    X, y = get_DS06()
    n_samples, n_features = X.shape

    if n_features == 1:
        # Filter data where X is positive
        X_positive = X[X[:, 0] > 0]
        y_positive = y[X[:, 0] > 0]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(X_positive, y_positive, color='royalblue', s=15, alpha=0.7,
                   edgecolors='w')  # Increase marker size and transparency
        ax.grid(True)  # Add grid lines
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2-dimensional dataset (Positive X)')
        plt.show()

    if n_features == 2:
        # Filter data where X[:, 0] is positive
        X_positive = X[X[:, 0] > 0]
        y_positive = y[X[:, 0] > 0]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_positive[:, 0], X_positive[:, 1], y_positive, color='blue', s=20)  # Plot only positive X values
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Rotated Points in 3D (Positive X)')
        ax.view_init(elev=20, azim=30)  # Set view angle
        ax.grid(True)  # Add grid lines
        plt.show()
