from Datasets.Synthetic2.Incremental.IncrementalGeneric import (
    generate_incremental_dataset,
    print_incremental_dataset_info,
    plot_incremental_dataset
)


def get_IDS03(
    n_samples_total=5000,
    seed=42,
    noise_std=1.0,
    return_meta=False
):
    return generate_incremental_dataset(
        dataset_id="IDS03",
        template_id="I_L_L",
        dimension_level="low",
        magnitude_level="large",
        n_samples_total=n_samples_total,
        n_features=1,
        seed=seed,
        noise_std=noise_std,
        beta_start=3.0,
        beta_end=-8.0,
        bias_start=5.0,
        bias_end=28.0,
        mu_start=0.0,
        mu_end=4.0,
        n_steps=10,
        return_meta=return_meta
    )


def print_IDS03_info(X, y, X_steps, y_steps, meta):
    print_incremental_dataset_info(X, y, X_steps, y_steps, meta)


if __name__ == "__main__":
    X, y, X_steps, y_steps, meta = get_IDS03(return_meta=True)
    print_IDS03_info(X, y, X_steps, y_steps, meta)
    plot_incremental_dataset(
        X_steps,
        y_steps,
        title="IDS03: Incremental Drift in 1D Regression (Low, Large)"
    )
