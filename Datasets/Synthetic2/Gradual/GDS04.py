import numpy as np

from Datasets.Synthetic2.Gradual.GradualGeneric import (
    generate_gradual_dataset,
    print_gradual_dataset_info,
    plot_gradual_dataset
)

def get_GDS04(
    n_samples_total=5000,
    n_features=10,
    seed=42,
    noise_std=1.0,
    beta1=None,
    beta2=None,
    return_meta=False
):
    if beta1 is None:
        beta1 = np.array(
            [3.0, 1.5, -1.0, 0.5, 2.0, -2.5, 1.0, -0.5, 0.8, -1.2],
            dtype=float
        )
    if beta2 is None:
        beta2 = np.array([-1.5, -0.8, 0.8, -0.5, -1.0, 1.5, -0.8, 0.5, -0.6, 0.8], dtype=float)

    return generate_gradual_dataset(
        dataset_id="GDS04",
        template_id="G_H_S",
        dimension_level="high",
        magnitude_level="small",
        n_samples_total=n_samples_total,
        n_features=n_features,
        seed=seed,
        noise_std=noise_std,
        beta1=beta1,
        bias1=5.0,
        mu1=0.0,
        sigma1=1.0,
        beta2=beta2,
        bias2=12.0,
        mu2=1.5,
        sigma2=1.0,
        return_meta=return_meta
    )

def print_GDS04_info(X, y, X1, y1, X2, y2, meta):
    print_gradual_dataset_info(X, y, X1, y1, X2, y2, meta)

if __name__ == "__main__":
    X, y, X1, y1, X2, y2, meta = get_GDS04(return_meta=True)
    print_GDS04_info(X, y, X1, y1, X2, y2, meta)
    plot_gradual_dataset(
        X1, y1, X2, y2,
        title="GDS04: Gradual Drift in High-D Regression (High, Small)"
    )
