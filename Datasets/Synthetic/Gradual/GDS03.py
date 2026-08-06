from Datasets.Synthetic.Gradual.GradualGeneric import (
    generate_gradual_dataset,
    print_gradual_dataset_info,
    plot_gradual_dataset
)


def get_GDS03(
    n_samples_total=1000,
    seed=42,
    noise_std=1.5,
    return_meta=False
):
    return generate_gradual_dataset(
        dataset_id="GDS03",
        template_id="G_L_L",
        dimension_level="low",
        magnitude_level="large",
        n_samples_total=n_samples_total,
        n_features=1,
        seed=seed,
        noise_std=noise_std,
        beta1=3.0,
        bias1=5.0,
        mu1=0.0,
        sigma1=1.0,
        beta2=-2.5,
        bias2=12.0,
        mu2=1.5,
        sigma2=1.0,
        return_meta=return_meta
    )


def print_GDS03_info(X, y, X1, y1, X2, y2, meta):
    print_gradual_dataset_info(X, y, X1, y1, X2, y2, meta)


if __name__ == "__main__":
    X, y, X1, y1, X2, y2, meta = get_GDS03(return_meta=True)
    print_GDS03_info(X, y, X1, y1, X2, y2, meta)
    plot_gradual_dataset(
        X1,
        y1,
        X2,
        y2,
        title="GDS03: Gradual Drift in 1D Regression (Low, Large)"
    )