import matplotlib.pyplot as plt
from sklearn import datasets

from Utils import Util, QuantifyDrift

'''
DS02
Abrupt (Sudden) Drift
'''


def get_DS02():
    n_samples = 5000
    n_features = 10
    noise = 20
    seed = 42

    X1, y1 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X2, y2 = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                      random_state=seed)
    X2 *= -1

    X, y = Util.combine_two_datasets(X1, y1, X2, y2)
    QuantifyDrift.quantify_drift(X1, y1, X2, y2)

    return X, y


if __name__ == "__main__":
    X, y = get_DS02()

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
