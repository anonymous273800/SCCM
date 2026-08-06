import numpy as np

from Datasets.Synthetic2.Incremental.IncrementalGeneric import (
    generate_incremental_dataset,
    print_incremental_dataset_info,
    plot_incremental_dataset
)


def get_IDS06(
    n_samples_total=5000,
    n_features=10,
    seed=42,
    noise_std=1.0,
    beta_start=None,
    beta_end=None,
    return_meta=False
):
    if beta_start is None:
        beta_start = np.array([
            3.0,
            1.5,
            -1.0,
            0.5,
            2.0,
            -2.5,
            1.0,
            -0.5,
            0.8,
            -1.2
        ])

    if beta_end is None:
        beta_end = np.array([
            -6.0,
            -4.0,
            5.0,
            -4.0,
            -5.0,
            6.0,
            -5.0,
            4.0,
            -5.0,
            5.0
        ])

    return generate_incremental_dataset(
        dataset_id="IDS06",
        template_id="I_H_L",
        dimension_level="high",
        magnitude_level="large",
        n_samples_total=n_samples_total,
        n_features=n_features,
        seed=seed,
        noise_std=noise_std,
        beta_start=beta_start,
        beta_end=beta_end,
        bias_start=5.0,
        bias_end=25.0,
        mu_start=0.0,
        mu_end=4.0,
        n_steps=10,
        return_meta=return_meta
    )


def print_IDS06_info(X, y, X_steps, y_steps, meta):
    print_incremental_dataset_info(X, y, X_steps, y_steps, meta)


if __name__ == "__main__":
    X, y, X_steps, y_steps, meta = get_IDS06(return_meta=True)
    print_IDS06_info(X, y, X_steps, y_steps, meta)
    plot_incremental_dataset(
        X_steps,
        y_steps,
        title="IDS06: Incremental Drift in High-D Regression (High, Large)"
    )
