import numpy as np

from Datasets.Synthetic.Incremental.IncrementalGeneric import (
    generate_incremental_dataset,
    print_incremental_dataset_info,
    plot_incremental_dataset
)


def get_IDS05(
    n_samples_total=2000,
    n_features=10,
    seed=42,
    noise_std=1.5,
    beta_start=None,
    beta_end=None,
    return_meta=False
):
    if beta_start is None:
        beta_start = np.array([3.0, 1.5, -1.0, 0.5, 2.0, -2.5, 1.0, -0.5, 0.8, -1.2])

    if beta_end is None:
        beta_end = np.array([1.8, 0.8, -0.2, 1.2, 1.0, -1.2, 0.3, 0.2, 1.4, -0.3])

    return generate_incremental_dataset(
        dataset_id="IDS05",
        template_id="I_H_M",
        dimension_level="high",
        magnitude_level="medium",
        n_samples_total=n_samples_total,
        n_features=n_features,
        seed=seed,
        noise_std=noise_std,
        beta_start=beta_start,
        beta_end=beta_end,
        bias_start=5.0,
        bias_end=8.0,
        mu_start=0.0,
        mu_end=0.8,
        n_steps=10,
        return_meta=return_meta
    )


def print_IDS05_info(X, y, X_steps, y_steps, meta):
    print_incremental_dataset_info(X, y, X_steps, y_steps, meta)


if __name__ == "__main__":
    X, y, X_steps, y_steps, meta = get_IDS05(return_meta=True)
    print_IDS05_info(X, y, X_steps, y_steps, meta)
    plot_incremental_dataset(
        X_steps,
        y_steps,
        title="IDS05: Incremental Drift in High-D Regression (High, Medium)"
    )