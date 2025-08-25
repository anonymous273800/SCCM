import matplotlib.pyplot as plt
from sklearn import datasets

from Utils import Util, QuantifyDrift

'''
Gradual Drift
'''


def get_DS09():
    n_samples = 100
    n_features = 1
    noise = 10
    seed = 42

    X1, y1 = datasets.make_regression(n_samples=int(2.5 * n_samples), n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)

    X2, y2 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X2 *= .1

    X, y = Util.combine_two_datasets(X1, y1, X2, y2)
    QuantifyDrift.quantify_drift(X1, y1, X2, y2)

    X3, y3 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X, y = Util.combine_two_datasets(X, y, X3, y3)
    QuantifyDrift.quantify_drift(X2, y2, X3, y3)

    X4, y4 = datasets.make_regression(n_samples=int(2 * n_samples), n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X4 *= .1
    X, y = Util.combine_two_datasets(X, y, X4, y4)
    QuantifyDrift.quantify_drift(X3, y3, X4, y4)

    X5, y5 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X, y = Util.combine_two_datasets(X, y, X5, y5)
    QuantifyDrift.quantify_drift(X4, y4, X5, y5)

    X6, y6 = datasets.make_regression(n_samples=int(2.5 * n_samples), n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X6 *= .1
    X, y = Util.combine_two_datasets(X, y, X6, y6)
    QuantifyDrift.quantify_drift(X5, y5, X6, y6)

    print("Final Quantiying Drift Between Start Concept and End Concept")
    QuantifyDrift.quantify_drift(X1, y1, X6, y6)

    return X, y


if __name__ == "__main__":

    X, y = get_DS09()
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
        # plt.axis('equal')  # Set equal scaling for x and y axes
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
