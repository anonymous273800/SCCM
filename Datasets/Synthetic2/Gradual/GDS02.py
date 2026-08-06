from Datasets.Synthetic2.Gradual.GradualGeneric import (
    generate_gradual_dataset,
    print_gradual_dataset_info,
    plot_gradual_dataset
)

def get_GDS02(n_samples_total=5000, seed=42, noise_std=1.0, return_meta=False):
    return generate_gradual_dataset(
        dataset_id="GDS02",
        template_id="G_L_M",
        dimension_level="low",
        magnitude_level="medium",
        n_samples_total=n_samples_total,
        n_features=1,
        seed=seed,
        noise_std=noise_std,
        beta1=3.0,
        bias1=5.0,
        mu1=0.0,
        sigma1=1.0,
        beta2=-3.5,
        bias2=17.0,
        mu2=2.5,
        sigma2=1.2,
        return_meta=return_meta
    )

def print_GDS02_info(X, y, X1, y1, X2, y2, meta):
    print_gradual_dataset_info(X, y, X1, y1, X2, y2, meta)

if __name__ == "__main__":
    X, y, X1, y1, X2, y2, meta = get_GDS02(return_meta=True)
    print_GDS02_info(X, y, X1, y1, X2, y2, meta)
    plot_gradual_dataset(
        X1, y1, X2, y2,
        title="GDS02: Gradual Drift in 1D Regression (Low, Medium)"
    )
