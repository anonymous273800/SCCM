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


def fit_linear_model_and_r2(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return r2_score(y, y_pred), model


def print_incremental_dataset_info(X, y, X_steps, y_steps, meta):
    print(meta)
    print("Total samples:", X.shape[0])
    print("Number of concepts:", len(X_steps))

    for i, (Xi, yi) in enumerate(zip(X_steps, y_steps), start=1):
        r2_i, _ = fit_linear_model_and_r2(Xi, yi)
        print(f"R² on Concept {i}: {r2_i:.4f}")

    r2_all, _ = fit_linear_model_and_r2(X, y)
    print(f"R² on whole dataset: {r2_all:.4f}")


def generate_incremental_dataset(
    *,
    dataset_id,
    template_id,
    dimension_level,
    magnitude_level,
    n_samples_total=1000,
    n_features=1,
    seed=42,
    noise_std=1.5,
    beta_start=None,
    beta_end=None,
    bias_start=5.0,
    bias_end=6.0,
    mu_start=0.0,
    mu_end=0.5,
    sigma=1.0,
    n_steps=10,
    return_meta=False,
    print_drift_results=True
):
    rng = np.random.default_rng(seed)

    if n_features == 1:
        beta_start = 3.0 if beta_start is None else beta_start
        beta_end = 2.0 if beta_end is None else beta_end

        beta_start_arr = np.array([beta_start], dtype=float)
        beta_end_arr = np.array([beta_end], dtype=float)
    else:
        if beta_start is None or beta_end is None:
            raise ValueError("For multi-dimensional datasets, beta_start and beta_end must be provided.")

        beta_start_arr = np.asarray(beta_start, dtype=float)
        beta_end_arr = np.asarray(beta_end, dtype=float)

        if len(beta_start_arr) != n_features or len(beta_end_arr) != n_features:
            raise ValueError("Length of beta_start and beta_end must match n_features.")

    betas = np.linspace(beta_start_arr, beta_end_arr, n_steps)
    biases = np.linspace(bias_start, bias_end, n_steps)
    mus = np.linspace(mu_start, mu_end, n_steps)

    samples_per_step = n_samples_total // n_steps

    X_steps = []
    y_steps = []

    for i in range(n_steps):
        X_i = rng.normal(loc=mus[i], scale=sigma, size=(samples_per_step, n_features))
        eps_i = rng.normal(loc=0.0, scale=noise_std, size=samples_per_step)

        if n_features == 1:
            y_i = betas[i][0] * X_i[:, 0] + biases[i] + eps_i
        else:
            y_i = X_i @ betas[i] + biases[i] + eps_i

        X_steps.append(X_i)
        y_steps.append(y_i)

    X = np.vstack(X_steps)
    y = np.concatenate(y_steps)

    drift_metrics = QuantifyDrift.quantify_drift(
        X_steps[0], y_steps[0],
        X_steps[-1], y_steps[-1],
        true_coef1=beta_start_arr.tolist(),
        true_intercept1=bias_start,
        true_coef2=beta_end_arr.tolist(),
        true_intercept2=bias_end,
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
            "drift_type": "incremental",
            "dimension_level": dimension_level,
            "magnitude_level": magnitude_level,
            "n_samples_total": X.shape[0],
            "n_features": n_features,
            "n_steps": n_steps,
            "samples_per_step": samples_per_step,
            "noise_std": noise_std,
            "seed": seed,
            "concept_start_params": {
                "mu": mu_start,
                "sigma": sigma,
                "beta": beta_start_arr.tolist() if n_features > 1 else beta_start_arr[0],
                "bias": bias_start
            },
            "concept_end_params": {
                "mu": mu_end,
                "sigma": sigma,
                "beta": beta_end_arr.tolist() if n_features > 1 else beta_end_arr[0],
                "bias": bias_end
            },
            "y_ks_pvalue": drift_metrics["y_ks_pvalue"],
            "y_js_divergence": drift_metrics["y_js_divergence"],
            "y_wasserstein_distance": drift_metrics["y_wasserstein_distance"],
            "x_any_ks_significant": drift_metrics["x_any_ks_significant"],
            "x_min_ks_pvalue": drift_metrics["x_min_ks_pvalue"],
            "x_avg_js_divergence": drift_metrics["x_avg_js_divergence"],
            "euclidean_distance_start_end": euclidean_distance,
            "empirical_distance_coef_only": drift_metrics["method_1_empirical"],
            "empirical_distance_full_params": drift_metrics["method_2_empirical"]
        }
        return X, y, X_steps, y_steps, meta

    return X, y


def plot_incremental_dataset(X_steps, y_steps, title="Incremental Drift Dataset"):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    n_steps = len(X_steps)
    cmap = plt.cm.viridis

    for i, (Xi, yi) in enumerate(zip(X_steps, y_steps)):
        color = cmap(i / max(n_steps - 1, 1))

        ax.scatter(
            Xi[:, 0],
            yi,
            color=color,
            s=15,
            alpha=0.7,
            edgecolors='w',
            label=f'Concept {i + 1}'
        )

        if Xi.shape[1] == 1:
            _, model_i = fit_linear_model_and_r2(Xi, yi)
            x_line = np.linspace(Xi[:, 0].min(), Xi[:, 0].max(), 200).reshape(-1, 1)
            y_line = model_i.predict(x_line)
            ax.plot(x_line, y_line, color=color, linewidth=2)

    if X_steps[0].shape[1] == 1:
        plt.xlabel('X')
    else:
        plt.xlabel('First feature (X[:, 0])')

    plt.ylabel('Y')
    plt.title(title)
    ax.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.show()