import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import pandas as pd

def _safe_hist_probs(a, b, bins=30):
    all_vals = np.concatenate([a, b])
    hist_range = (np.min(all_vals), np.max(all_vals))

    p, _ = np.histogram(a, bins=bins, range=hist_range, density=False)
    q, _ = np.histogram(b, bins=bins, range=hist_range, density=False)

    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12

    p /= p.sum()
    q /= q.sum()
    return p, q


def quantify_drift(
    X1, y1, X2, y2,
    true_coef1=None, true_intercept1=None,
    true_coef2=None, true_intercept2=None,
    bins=30,
    ks_alpha=0.05,
    print_results=True
):
    """
    Quantify drift between two regression concepts.

    Returns
    -------
    results : dict
        Empirical distances:
            - coefficients only
            - full fitted parameters (intercept + coefficients)

        Theoretical distance:
            - distance between true generating parameters, if provided

        Distributional drift metrics:
            - Y KS p-value
            - Y JS divergence
            - Y Wasserstein distance
            - X any KS significant
            - X min KS p-value
            - X avg JS divergence
    """

    # -------------------------------------------------
    # 1) Fit linear models on both concepts
    # -------------------------------------------------
    model1 = LinearRegression().fit(X1, y1)
    model2 = LinearRegression().fit(X2, y2)

    intercept1, coef1 = model1.intercept_, np.atleast_1d(model1.coef_)
    intercept2, coef2 = model2.intercept_, np.atleast_1d(model2.coef_)

    # -------------------------------------------------
    # 2) Empirical parameter-space drift
    # -------------------------------------------------
    method_1_empirical = float(np.linalg.norm(coef1 - coef2))

    delta_intercept = intercept2 - intercept1
    delta_coef = coef2 - coef1
    method_2_empirical = float(np.sqrt(delta_intercept ** 2 + np.sum(delta_coef ** 2)))

    # -------------------------------------------------
    # 3) Theoretical parameter-space drift
    # -------------------------------------------------
    theoretical_distance = None
    if (
        true_coef1 is not None and true_intercept1 is not None and
        true_coef2 is not None and true_intercept2 is not None
    ):
        true_coef1 = np.atleast_1d(np.array(true_coef1, dtype=float))
        true_coef2 = np.atleast_1d(np.array(true_coef2, dtype=float))

        if true_coef1.shape != true_coef2.shape:
            raise ValueError("true_coef1 and true_coef2 must have the same shape.")

        theoretical_distance = float(np.sqrt(
            (true_intercept2 - true_intercept1) ** 2 +
            np.sum((true_coef2 - true_coef1) ** 2)
        ))

    # -------------------------------------------------
    # 4) Y-space distributional drift
    # -------------------------------------------------
    y_ks_pvalue = float(ks_2samp(y1, y2).pvalue)

    py, qy = _safe_hist_probs(y1, y2, bins=bins)
    y_js_divergence = float(jensenshannon(py, qy, base=2.0) ** 2)

    y_wasserstein_distance = float(wasserstein_distance(y1, y2))

    # -------------------------------------------------
    # 5) X-space distributional drift
    # -------------------------------------------------
    x_ks_pvalues = []
    x_js_divs = []

    for j in range(X1.shape[1]):
        x1j = X1[:, j]
        x2j = X2[:, j]

        ks_p = float(ks_2samp(x1j, x2j).pvalue)
        x_ks_pvalues.append(ks_p)

        px, qx = _safe_hist_probs(x1j, x2j, bins=bins)
        js_x = float(jensenshannon(px, qx, base=2.0) ** 2)
        x_js_divs.append(js_x)

    x_any_ks_significant = bool(any(p < ks_alpha for p in x_ks_pvalues))
    x_min_ks_pvalue = float(np.min(x_ks_pvalues))
    x_avg_js_divergence = float(np.mean(x_js_divs))

    # -------------------------------------------------
    # 6) Final results dictionary
    # -------------------------------------------------
    results = {
        "method_1_empirical": method_1_empirical,
        "method_2_empirical": method_2_empirical,
        "theoretical_distance": theoretical_distance,
        "fitted_concept_1": {
            "intercept": float(intercept1),
            "coef": coef1.tolist()
        },
        "fitted_concept_2": {
            "intercept": float(intercept2),
            "coef": coef2.tolist()
        },
        "y_ks_pvalue": y_ks_pvalue,
        "y_js_divergence": y_js_divergence,
        "y_wasserstein_distance": y_wasserstein_distance,
        "x_any_ks_significant": x_any_ks_significant,
        "x_min_ks_pvalue": x_min_ks_pvalue,
        "x_avg_js_divergence": x_avg_js_divergence
    }

    # -------------------------------------------------
    # 7) Optional printing
    # -------------------------------------------------
    if print_results:
        print(f"Euclidean distance between coefficients - Method 1 (empirical): {method_1_empirical:.6f}")
        print(f"Euclidean distance between coefficients - Method 2 (empirical): {method_2_empirical:.6f}")
        if theoretical_distance is not None:
            print(f"Euclidean distance between true parameters (theoretical): {theoretical_distance:.6f}")

        print(f"Y KS P-Value: {y_ks_pvalue:.6f}")
        print(f"Y JS Divergence: {y_js_divergence:.6f}")
        print(f"Y Wasserstein Distance: {y_wasserstein_distance:.6f}")
        print(f"X Any KS Significant: {x_any_ks_significant}")
        print(f"X Min KS P-Value: {x_min_ks_pvalue:.6f}")
        print(f"X Avg JS Divergence: {x_avg_js_divergence:.6f}")

    return results


def build_drift_metrics_row(
    dataset_name,
    drift_type,
    n_samples,
    n_features,
    noise_std,
    drift_location,
    drift_metrics
):
    """
    Build one row for reporting averaged drift metrics.
    """
    return {
        "Dataset": dataset_name,
        "Drift Type": drift_type,
        "Data Points": n_samples,
        "Dimensions": n_features,
        "Noise": noise_std,
        "Concept Loc.": drift_location,
        "Y KS P-Value": drift_metrics["y_ks_pvalue"],
        "Y JS Div": drift_metrics["y_js_divergence"],
        "Y Wasserstein Dist": drift_metrics["y_wasserstein_distance"],
        "X Any KS Sig": drift_metrics["x_any_ks_significant"],
        "X Min KS P-Value": drift_metrics["x_min_ks_pvalue"],
        "X Avg JS Div": drift_metrics["x_avg_js_divergence"],
        "Euclidean Distance": drift_metrics["euclidean_distance"]
    }


def print_drift_metrics_table(row_or_rows):
    """
    Print one row or multiple rows as a clean table.
    """
    if isinstance(row_or_rows, dict):
        row_or_rows = [row_or_rows]

    df = pd.DataFrame(row_or_rows)

    # nicer printing
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.colheader_justify", "center")

    print("\n===== DRIFT METRICS TABLE =====")
    print(df.to_string(index=False))


def save_drift_metrics_to_excel(row_or_rows, file_path="drift_metrics_summary.xlsx", sheet_name="DriftMetrics"):
    """
    Save one row or multiple rows to Excel.
    """
    if isinstance(row_or_rows, dict):
        row_or_rows = [row_or_rows]

    df = pd.DataFrame(row_or_rows)
    df.to_excel(file_path, index=False, sheet_name=sheet_name)
    print(f"Saved drift metrics table to: {file_path}")