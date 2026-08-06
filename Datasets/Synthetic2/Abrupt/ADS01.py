import numpy as np
from Datasets.Synthetic2.Abrupt.AbruptGeneric import (
    generate_abrupt_dataset,
    print_dataset_info,
    plot_abrupt_dataset
)


def get_DS01(
    n_samples_per_concept=500,
    seed=42,
    noise_std=1.0,
    beta1=3.0,
    bias1=5.0,
    mu1=0.0,
    sigma1=1.0,
    beta2=-2.0,
    bias2=12.0,
    mu2=2.0,
    sigma2=1.0,
    return_meta=False
):
    return generate_abrupt_dataset(
        dataset_id="ADS01",
        template_id="A_L_S",
        dimension_level="low",
        magnitude_level="small",
        n_samples_per_concept=n_samples_per_concept,
        n_features=1,
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


def print_DS01_info(X, y, X1, y1, X2, y2, meta):
    print_dataset_info(X, y, X1, y1, X2, y2, meta)


if __name__ == "__main__":
    X, y, X1, y1, X2, y2, meta = get_DS01(return_meta=True)
    print_DS01_info(X, y, X1, y1, X2, y2, meta)
    plot_abrupt_dataset(
        X1,
        y1,
        X2,
        y2,
        title="ADS01: Abrupt Drift in 1D Regression (Low, Small)"
    )
