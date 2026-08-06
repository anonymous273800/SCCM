import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from Utils import Util, QuantifyDrift


def compute_r2_fit_on_same_data(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return r2_score(y, y_pred), model


def print_dataset_info(X, y, X1, y1, X2, y2, meta):
    print(meta)
    print("Total samples:", X.shape[0])

    r2_c1, _ = compute_r2_fit_on_same_data(X1, y1)
    print(f"R² on Concept 1 only: {r2_c1:.4f}")

    r2_c2, _ = compute_r2_fit_on_same_data(X2, y2)
    print(f"R² on Concept 2 only: {r2_c2:.4f}")

    r2_all, _ = compute_r2_fit_on_same_data(X, y)
    print(f"R² on whole dataset: {r2_all:.4f}")


def generate_abrupt_dataset(
    *,
    dataset_id,
    template_id,
    dimension_level,
    magnitude_level,
    n_samples_per_concept=500,
    n_features=1,
    seed=42,
    noise_std=1.5,
    beta1=None,
    bias1=5.0,
    mu1=0.0,
    sigma1=1.0,
    beta2=None,
    bias2=6.0,
    mu2=0.5,
    sigma2=1.0,
    return_meta=False,
    print_drift_results=True
):
    rng = np.random.default_rng(seed)

    # Normalize beta format
    if n_features == 1:
        if beta1 is None:
            beta1 = 3.0
        if beta2 is None:
            beta2 = 2.2

        beta1_arr = np.array([beta1], dtype=float)
        beta2_arr = np.array([beta2], dtype=float)

        # Concept 1
        X1 = rng.normal(loc=mu1, scale=sigma1, size=(n_samples_per_concept, 1))
        eps1 = rng.normal(loc=0.0, scale=noise_std, size=n_samples_per_concept)
        y1 = beta1_arr[0] * X1[:, 0] + bias1 + eps1

        # Concept 2
        X2 = rng.normal(loc=mu2, scale=sigma2, size=(n_samples_per_concept, 1))
        eps2 = rng.normal(loc=0.0, scale=noise_std, size=n_samples_per_concept)
        y2 = beta2_arr[0] * X2[:, 0] + bias2 + eps2

    else:
        if beta1 is None or beta2 is None:
            raise ValueError("For multi-dimensional datasets, beta1 and beta2 must be provided.")

        beta1_arr = np.asarray(beta1, dtype=float)
        beta2_arr = np.asarray(beta2, dtype=float)

        if len(beta1_arr) != n_features or len(beta2_arr) != n_features:
            raise ValueError("Length of beta1 and beta2 must match n_features.")

        # Concept 1
        X1 = rng.normal(loc=mu1, scale=sigma1, size=(n_samples_per_concept, n_features))
        eps1 = rng.normal(loc=0.0, scale=noise_std, size=n_samples_per_concept)
        y1 = X1 @ beta1_arr + bias1 + eps1

        # Concept 2
        X2 = rng.normal(loc=mu2, scale=sigma2, size=(n_samples_per_concept, n_features))
        eps2 = rng.normal(loc=0.0, scale=noise_std, size=n_samples_per_concept)
        y2 = X2 @ beta2_arr + bias2 + eps2

    # Combine into one abrupt stream
    X, y = Util.combine_two_datasets(X1, y1, X2, y2)

    # Drift metrics
    drift_metrics = QuantifyDrift.quantify_drift(
        X1, y1, X2, y2,
        true_coef1=beta1_arr.tolist(),
        true_intercept1=bias1,
        true_coef2=beta2_arr.tolist(),
        true_intercept2=bias2,
        print_results=print_drift_results
    )

    euclidean_distance = (
        drift_metrics["theoretical_distance"]
        if drift_metrics["theoretical_distance"] is not None
        else drift_metrics["method_2_empirical"]
    )

    if return_meta:
        meta = {
            "dataset_id": dataset_id,
            "template_id": template_id,
            "drift_type": "abrupt",
            "dimension_level": dimension_level,
            "magnitude_level": magnitude_level,
            "drift_point": n_samples_per_concept,
            "n_samples_total": 2 * n_samples_per_concept,
            "n_features": n_features,
            "noise_std": noise_std,
            "seed": seed,
            "concept_1_params": {
                "mu": mu1,
                "sigma": sigma1,
                "beta": beta1_arr.tolist() if n_features > 1 else beta1_arr[0],
                "bias": bias1
            },
            "concept_2_params": {
                "mu": mu2,
                "sigma": sigma2,
                "beta": beta2_arr.tolist() if n_features > 1 else beta2_arr[0],
                "bias": bias2
            },
            "y_ks_pvalue": drift_metrics["y_ks_pvalue"],
            "y_js_divergence": drift_metrics["y_js_divergence"],
            "y_wasserstein_distance": drift_metrics["y_wasserstein_distance"],
            "x_any_ks_significant": drift_metrics["x_any_ks_significant"],
            "x_min_ks_pvalue": drift_metrics["x_min_ks_pvalue"],
            "x_avg_js_divergence": drift_metrics["x_avg_js_divergence"],
            "euclidean_distance_start_end": euclidean_distance,
            "euclidean_distance_consecutive": euclidean_distance,
            "empirical_distance_coef_only": drift_metrics["method_1_empirical"],
            "empirical_distance_full_params": drift_metrics["method_2_empirical"]
        }
        return X, y, X1, y1, X2, y2, meta

    return X, y


def fit_linear_model_and_r2(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return r2_score(y, y_pred), model


def print_dataset_info(X, y, X1, y1, X2, y2, meta):
    print(meta)
    print("Total samples:", X.shape[0])

    r2_c1, _ = fit_linear_model_and_r2(X1, y1)
    print(f"R² on Concept 1 only: {r2_c1:.4f}")

    r2_c2, _ = fit_linear_model_and_r2(X2, y2)
    print(f"R² on Concept 2 only: {r2_c2:.4f}")

    r2_all, _ = fit_linear_model_and_r2(X, y)
    print(f"R² on whole dataset: {r2_all:.4f}")


def plot_abrupt_dataset(X1, y1, X2, y2, title="Abrupt Drift Dataset"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.scatter(X1[:, 0], y1, color='royalblue', s=15, alpha=0.7, edgecolors='w', label='Concept 1')
    ax.scatter(X2[:, 0], y2, color='darkorange', s=15, alpha=0.7, edgecolors='w', label='Concept 2')

    if X1.shape[1] == 1:
        _, model_c1 = fit_linear_model_and_r2(X1, y1)
        _, model_c2 = fit_linear_model_and_r2(X2, y2)

        x1_line = np.linspace(X1[:, 0].min(), X1[:, 0].max(), 200).reshape(-1, 1)
        y1_line = model_c1.predict(x1_line)
        ax.plot(x1_line, y1_line, color='blue', linewidth=2, label='Fit Concept 1')

        x2_line = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 200).reshape(-1, 1)
        y2_line = model_c2.predict(x2_line)
        ax.plot(x2_line, y2_line, color='red', linewidth=2, label='Fit Concept 2')

        plt.xlabel('X')
    else:
        plt.xlabel('First feature (X[:, 0])')

    plt.ylabel('Y')
    plt.title(title)
    ax.grid(True)
    plt.legend()
    plt.show()