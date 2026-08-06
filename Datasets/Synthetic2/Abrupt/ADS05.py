import numpy as np

from Datasets.Synthetic2.Abrupt.AbruptGeneric import (
    generate_abrupt_dataset,
    print_dataset_info,
    plot_abrupt_dataset
)


def get_DS05(
    n_samples_per_concept=1000,
    n_features=10,
    seed=42,
    noise_std=1.0,
    beta1=None,
    bias1=5.0,
    mu1=0.0,
    sigma1=1.0,
    beta2=None,
    bias2=20.0,
    mu2=3.0,
    sigma2=1.2,
    return_meta=False
):
    if beta1 is None:
        beta1 = np.array(
            [3.0, 1.5, -1.0, 0.5, 2.0, -2.5, 1.0, -0.5, 0.8, -1.2],
            dtype=float
        )

    if beta2 is None:
        beta2 = np.array(
            [-4.0, -2.5, 2.5, -1.8, -3.0, 3.5, -2.0, 1.8, -2.2, 2.8],
            dtype=float
        )

    return generate_abrupt_dataset(
        dataset_id="ADS05",
        template_id="A_H_M",
        dimension_level="high",
        magnitude_level="medium",
        n_samples_per_concept=n_samples_per_concept,
        n_features=n_features,
        seed=seed,
        noise_std=noise_std,
        beta1=beta1,
        bias1=bias1,
        mu1=mu1,
        sigma1=sigma1,
        beta2=beta2,
        bias2=bias2,
        mu2=mu2,
        sigma2=sigma2,
        return_meta=return_meta
    )


def print_DS05_info(X, y, X1, y1, X2, y2, meta):
    print_dataset_info(X, y, X1, y1, X2, y2, meta)


if __name__ == "__main__":
    X, y, X1, y1, X2, y2, meta = get_DS05(return_meta=True)
    print_DS05_info(X, y, X1, y1, X2, y2, meta)
    plot_abrupt_dataset(
        X1,
        y1,
        X2,
        y2,
        title="ADS05: Abrupt Drift in High-D Regression (High, Medium)"
    )
