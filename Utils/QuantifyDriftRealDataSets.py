import numpy as np
import pandas as pd
from Datasets.Real import PublicDS
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LinearRegression, Ridge


# ============================================================
# Utility functions
# ============================================================

def _safe_hist_probs(a, b, bins=30):
    """
    Build safe histogram probability distributions for JS divergence.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    all_vals = np.concatenate([a, b])

    if np.min(all_vals) == np.max(all_vals):
        # Avoid zero-width histogram range
        hist_range = (np.min(all_vals) - 1e-6, np.max(all_vals) + 1e-6)
    else:
        hist_range = (np.min(all_vals), np.max(all_vals))

    p, _ = np.histogram(a, bins=bins, range=hist_range, density=False)
    q, _ = np.histogram(b, bins=bins, range=hist_range, density=False)

    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12

    p /= p.sum()
    q /= q.sum()

    return p, q


def _to_numpy(X, y):
    """
    Convert X and y to clean numpy arrays.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X, y


def _remove_missing_rows(X, y):
    """
    Remove rows containing NaN or infinite values.
    """
    X, y = _to_numpy(X, y)

    valid_X = np.all(np.isfinite(X), axis=1)
    valid_y = np.isfinite(y)

    valid = valid_X & valid_y

    return X[valid], y[valid]


def _standardize_segments(X1, X2):
    """
    Standardize X segments using only the first segment as reference.
    This avoids using future information when comparing stream segments.
    """
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    X2_scaled = scaler.transform(X2)

    return X1_scaled, X2_scaled


# ============================================================
# Core real-dataset drift quantification
# ============================================================

def _fit_segment_regression(
    X,
    y,
    regression_model="linear",
    ridge_alpha=1.0
):
    """
    Fit the regression model used to estimate the relationship
    between X and y in one stream segment.

    Parameters
    ----------
    X : numpy.ndarray
        Segment features.

    y : numpy.ndarray
        Segment target.

    regression_model : str
        "linear" uses ordinary least squares.
        "ridge" uses L2-regularized linear regression.

    ridge_alpha : float
        Ridge regularization strength. Used only when
        regression_model="ridge".
    """

    regression_model = regression_model.lower()

    if regression_model == "linear":
        model = LinearRegression()

    elif regression_model == "ridge":

        if ridge_alpha <= 0:
            raise ValueError(
                "ridge_alpha must be greater than zero."
            )

        model = Ridge(
            alpha=ridge_alpha,
            solver="svd"
        )

    else:
        raise ValueError(
            "regression_model must be 'linear' or 'ridge'. "
            f"Received: {regression_model}"
        )

    model.fit(X, y)

    return model

def quantify_real_drift(
    X1,
    y1,
    X2,
    y2,
    bins=30,
    ks_alpha=0.05,
    standardize_X=True,
    regression_model="linear",
    ridge_alpha=1.0,
    print_results=True
):
    """
    Quantify empirical drift between two real-world stream segments.

    Parameters
    ----------
    X1, y1 : array-like
        First stream segment.

    X2, y2 : array-like
        Second stream segment.

    bins : int
        Number of bins used for histogram-based JS divergence.

    ks_alpha : float
        Significance level for the KS test.

    standardize_X : bool
        If True, standardize X1 and X2 using statistics fitted
        only on X1.

    regression_model : str
        "linear" uses ordinary least squares.
        "ridge" uses Ridge regression for more stable coefficients.

    ridge_alpha : float
        Ridge regularization strength. Used only when
        regression_model="ridge".

    print_results : bool
        If True, print the calculated metrics.

    Returns
    -------
    results : dict
        Drift metrics between the two stream segments.
    """

    # --------------------------------------------------------
    # 1) Clean inputs
    # --------------------------------------------------------
    X1, y1 = _remove_missing_rows(X1, y1)
    X2, y2 = _remove_missing_rows(X2, y2)

    if len(X1) < 2 or len(X2) < 2:
        raise ValueError(
            "Each segment must contain at least two valid samples."
        )

    if X1.shape[1] != X2.shape[1]:
        raise ValueError(
            "X1 and X2 must have the same number of features."
        )

    # --------------------------------------------------------
    # 2) Standardize using the first segment only
    # --------------------------------------------------------
    if standardize_X:
        X1_model, X2_model = _standardize_segments(
            X1,
            X2
        )
    else:
        X1_model = X1
        X2_model = X2

    # --------------------------------------------------------
    # 3) Fit one model for each segment
    # --------------------------------------------------------
    model1 = _fit_segment_regression(
        X=X1_model,
        y=y1,
        regression_model=regression_model,
        ridge_alpha=ridge_alpha
    )

    model2 = _fit_segment_regression(
        X=X2_model,
        y=y2,
        regression_model=regression_model,
        ridge_alpha=ridge_alpha
    )

    intercept1 = float(model1.intercept_)
    intercept2 = float(model2.intercept_)

    coef1 = np.atleast_1d(
        np.asarray(model1.coef_, dtype=float)
    )

    coef2 = np.atleast_1d(
        np.asarray(model2.coef_, dtype=float)
    )

    # --------------------------------------------------------
    # 4) Parameter distances
    # --------------------------------------------------------
    coef_distance = float(
        np.linalg.norm(
            coef1 - coef2
        )
    )

    delta_intercept = (
        intercept2 - intercept1
    )

    delta_coef = (
        coef2 - coef1
    )

    full_param_distance = float(
        np.sqrt(
            delta_intercept ** 2
            + np.sum(
                delta_coef ** 2
            )
        )
    )

    # --------------------------------------------------------
    # 5) Target-distribution drift
    # --------------------------------------------------------
    y_ks_pvalue = float(
        ks_2samp(
            y1,
            y2,
            method="asymp"
        ).pvalue
    )

    py, qy = _safe_hist_probs(
        y1,
        y2,
        bins=bins
    )

    y_js_divergence = float(
        jensenshannon(
            py,
            qy,
            base=2.0
        ) ** 2
    )

    y_wasserstein_distance = float(
        wasserstein_distance(
            y1,
            y2
        )
    )

    # --------------------------------------------------------
    # 6) Feature-distribution drift
    # --------------------------------------------------------
    x_ks_pvalues = []
    x_js_divs = []

    for feature_index in range(
        X1.shape[1]
    ):
        x1_feature = X1[:, feature_index]
        x2_feature = X2[:, feature_index]

        ks_pvalue = float(
            ks_2samp(
                x1_feature,
                x2_feature,
                method="asymp"
            ).pvalue
        )

        x_ks_pvalues.append(
            ks_pvalue
        )

        px, qx = _safe_hist_probs(
            x1_feature,
            x2_feature,
            bins=bins
        )

        js_divergence = float(
            jensenshannon(
                px,
                qx,
                base=2.0
            ) ** 2
        )

        x_js_divs.append(
            js_divergence
        )

    x_any_ks_significant = bool(
        any(
            pvalue < ks_alpha
            for pvalue in x_ks_pvalues
        )
    )

    x_min_ks_pvalue = float(
        np.min(
            x_ks_pvalues
        )
    )

    x_avg_js_divergence = float(
        np.mean(
            x_js_divs
        )
    )

    # --------------------------------------------------------
    # 7) Final results
    # --------------------------------------------------------
    results = {
        "regression_model": regression_model,
        "ridge_alpha": (
            float(ridge_alpha)
            if regression_model.lower() == "ridge"
            else None
        ),
        "coef_distance": coef_distance,
        "full_param_distance": full_param_distance,
        "fitted_segment_1": {
            "intercept": intercept1,
            "coef": coef1.tolist()
        },
        "fitted_segment_2": {
            "intercept": intercept2,
            "coef": coef2.tolist()
        },
        "y_ks_pvalue": y_ks_pvalue,
        "y_js_divergence": y_js_divergence,
        "y_wasserstein_distance": y_wasserstein_distance,
        "x_any_ks_significant": x_any_ks_significant,
        "x_min_ks_pvalue": x_min_ks_pvalue,
        "x_avg_js_divergence": x_avg_js_divergence
    }

    if print_results:

        model_description = regression_model.upper()

        if regression_model.lower() == "ridge":
            model_description += (
                f" (alpha={ridge_alpha})"
            )

        print(
            f"Parameter Model: {model_description}"
        )

        print(
            f"Coefficient Distance: "
            f"{coef_distance:.6f}"
        )

        print(
            f"Full Parameter Distance: "
            f"{full_param_distance:.6f}"
        )

        print(
            f"Y KS P-Value: "
            f"{y_ks_pvalue:.6e}"
        )

        print(
            f"Y JS Divergence: "
            f"{y_js_divergence:.6f}"
        )

        print(
            f"Y Wasserstein Distance: "
            f"{y_wasserstein_distance:.6f}"
        )

        print(
            f"X Any KS Significant: "
            f"{x_any_ks_significant}"
        )

        print(
            f"X Min KS P-Value: "
            f"{x_min_ks_pvalue:.6e}"
        )

        print(
            f"X Avg JS Divergence: "
            f"{x_avg_js_divergence:.6f}"
        )

    return results


# ============================================================
# Segmenting real datasets
# ============================================================

def split_real_dataset_half(X, y):
    """
    Split a real stream into first half and second half.
    """
    X, y = _remove_missing_rows(X, y)

    split_idx = len(X) // 2

    X1, y1 = X[:split_idx], y[:split_idx]
    X2, y2 = X[split_idx:], y[split_idx:]

    return X1, y1, X2, y2


def split_real_dataset_consecutive_windows(X, y, n_windows=4):
    """
    Split a real stream into consecutive windows.

    Example:
        n_windows = 4 gives:
        W1, W2, W3, W4

    Drift can then be computed as:
        W1 vs W2
        W2 vs W3
        W3 vs W4
    """
    X, y = _remove_missing_rows(X, y)

    if n_windows < 2:
        raise ValueError("n_windows must be at least 2.")

    indices = np.array_split(np.arange(len(X)), n_windows)

    windows = []
    for idx in indices:
        windows.append((X[idx], y[idx]))

    return windows


# ============================================================
# Row builders
# ============================================================

def build_real_drift_metrics_row(
    dataset_name,
    domain,
    n_samples,
    n_features,
    segment_definition,
    drift_metrics
):
    """
    Build one row for the real-dataset drift table.
    """
    return {
        "Dataset": dataset_name,
        "Type": "Real",
        "Domain": domain,
        "Data Points": n_samples,
        "Dimensions": n_features,
        "Segment Definition": segment_definition,
        "Y KS P-Value": drift_metrics["y_ks_pvalue"],
        "Y JS Div": drift_metrics["y_js_divergence"],
        "Y Wasserstein Dist": drift_metrics["y_wasserstein_distance"],
        "X Any KS Sig": drift_metrics["x_any_ks_significant"],
        "X Min KS P-Value": drift_metrics["x_min_ks_pvalue"],
        "X Avg JS Div": drift_metrics["x_avg_js_divergence"],
        "Coef Distance": drift_metrics["coef_distance"],
        "Full Param Distance": drift_metrics["full_param_distance"]
    }


def print_real_drift_metrics_table(rows):
    """
    Print real-dataset drift metrics as a clean table.
    """
    if isinstance(rows, dict):
        rows = [rows]

    df = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.colheader_justify", "center")

    print("\n===== REAL DATASET DRIFT METRICS TABLE =====")
    print(df.to_string(index=False))


def save_real_drift_metrics_to_excel(
    rows,
    file_path="real_drift_metrics_summary.xlsx",
    sheet_name="RealDriftMetrics"
):
    """
    Save real-dataset drift metrics to Excel.
    """
    if isinstance(rows, dict):
        rows = [rows]

    df = pd.DataFrame(rows)
    df.to_excel(file_path, index=False, sheet_name=sheet_name)

    print(f"Saved real drift metrics table to: {file_path}")


def save_real_drift_metrics_to_csv(
    rows,
    file_path="real_drift_metrics_summary.csv"
):
    """
    Save real-dataset drift metrics to CSV.
    """
    if isinstance(rows, dict):
        rows = [rows]

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)

    print(f"Saved real drift metrics table to: {file_path}")


# ============================================================
# Dataset loading helper
# ============================================================

def load_real_dataset_from_csv(file_path, target_column, drop_columns=None):
    """
    Load a real regression dataset from CSV.

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    target_column : str
        Name of the target column.

    drop_columns : list or None
        Columns to remove before modeling.

    Returns
    -------
    X : np.ndarray
        Feature matrix.

    y : np.ndarray
        Target vector.
    """
    df = pd.read_csv(file_path)

    if drop_columns is not None:
        df = df.drop(columns=drop_columns, errors="ignore")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {file_path}")

    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    X, y = _remove_missing_rows(X, y)

    return X, y


# ============================================================
# Main function for one real dataset
# ============================================================

def quantify_one_real_dataset_half_split(
    dataset_name,
    domain,
    X,
    y,
    bins=30,
    ks_alpha=0.05,
    standardize_X=True,
    regression_model="linear",
    ridge_alpha=1.0,
    print_results=True
):
    """
    Quantify drift for one real dataset using the first half
    versus the second half.
    """

    X, y = _remove_missing_rows(
        X,
        y
    )

    X1, y1, X2, y2 = split_real_dataset_half(
        X,
        y
    )

    metrics = quantify_real_drift(
        X1=X1,
        y1=y1,
        X2=X2,
        y2=y2,
        bins=bins,
        ks_alpha=ks_alpha,
        standardize_X=standardize_X,
        regression_model=regression_model,
        ridge_alpha=ridge_alpha,
        print_results=print_results
    )

    row = build_real_drift_metrics_row(
        dataset_name=dataset_name,
        domain=domain,
        n_samples=len(X),
        n_features=X.shape[1],
        segment_definition=(
            "First half vs second half"
        ),
        drift_metrics=metrics
    )

    return row


def quantify_one_real_dataset_windowed(
    dataset_name,
    domain,
    X,
    y,
    n_windows=4,
    bins=30,
    ks_alpha=0.05,
    standardize_X=True,
    regression_model="linear",
    ridge_alpha=1.0,
    print_results=False
):
    """
    Quantify drift using consecutive windows.

    Comparisons:
        Window 1 vs Window 2
        Window 2 vs Window 3
        ...
    """

    X, y = _remove_missing_rows(
        X,
        y
    )

    windows = split_real_dataset_consecutive_windows(
        X,
        y,
        n_windows=n_windows
    )

    rows = []

    for window_index in range(
        len(windows) - 1
    ):
        X1, y1 = windows[
            window_index
        ]

        X2, y2 = windows[
            window_index + 1
        ]

        metrics = quantify_real_drift(
            X1=X1,
            y1=y1,
            X2=X2,
            y2=y2,
            bins=bins,
            ks_alpha=ks_alpha,
            standardize_X=standardize_X,
            regression_model=regression_model,
            ridge_alpha=ridge_alpha,
            print_results=print_results
        )

        row = build_real_drift_metrics_row(
            dataset_name=dataset_name,
            domain=domain,
            n_samples=len(X),
            n_features=X.shape[1],
            segment_definition=(
                f"Window {window_index + 1} "
                f"vs Window {window_index + 2}"
            ),
            drift_metrics=metrics
        )

        rows.append(
            row
        )

    return rows


# ============================================================
# Example usage
# ============================================================
# ============================================================
# Main execution for real datasets
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Import your project dataset utilities
    # --------------------------------------------------------
    import Util


    all_rows = []

    # ========================================================
    # 1) CCPP: Combined Cycle Power Plant
    # ========================================================
    print("\nProcessing CCPP...")

    path = Util.get_dataset_path_('08_CCPP\\008_Folds5x2_pp.csv')
    X, y = PublicDS.get_CCPP(path)

    row = quantify_one_real_dataset_half_split(
        dataset_name="CCPP",
        domain="Energy Systems",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        print_results=True
    )

    all_rows.append(row)

    # ========================================================
    # 2) MCPD: Medical Cost Personal Dataset
    # ========================================================
    print("\nProcessing MCPD...")

    path = Util.get_dataset_path_('05_MCPD\\005_insurance.csv')
    X, y = PublicDS.get_medical_cost_personal_dataset(path)

    row = quantify_one_real_dataset_half_split(
        dataset_name="MCPD",
        domain="Healthcare",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        print_results=True
    )

    all_rows.append(row)

    # ========================================================
    # 3) KCHSD: King County House Sales Dataset
    # ========================================================
    print("\nProcessing KCHSD...")

    path = Util.get_dataset_path_('07_KCHSD\\007_kc_house_data.csv')
    X, y = PublicDS.get_king_county_house_sales_data(path)

    row = quantify_one_real_dataset_half_split(
        dataset_name="KCHSD",
        domain="Housing/Real Estate",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        print_results=True
    )

    all_rows.append(row)

    # ========================================================
    # 4) 1KC: 1000 Companies Dataset
    # ========================================================
    print("\nProcessing 1KC...")

    path = Util.get_dataset_path_('06_1KC\\006_1000_Companies.csv')
    X, y = PublicDS.get_profit_estimation_for_companies_dataset(path)

    row = quantify_one_real_dataset_half_split(
        dataset_name="1KC",
        domain="Business/Finance",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        print_results=True
    )

    all_rows.append(row)


    # ========================================================
    # 5) UCIAQD: Air Quality UCI Dataset
    # ========================================================
    print("\nProcessing UCIAQD...")

    path = Path(
        r"C:\New\003\SCCM-StreamCruiseControlMethod"
        r"\Datasets\Real\Datasets_Generators_CSV"
        r"\UCIAQD\AirQualityUCI.csv"
    )

    X, y = PublicDS.get_UCIAQD(
        path=path,
        train_percent=10
    )

    row = quantify_one_real_dataset_half_split(
        dataset_name="UCIAQD",
        domain="Environmental/Air Quality",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        print_results=True
    )

    all_rows.append(row)


    # ========================================================
    # 6) GASD: Gas Sensor Array Drift Dataset
    # ========================================================
    print("\nProcessing GASD...")

    path = Path(
        r"C:\New\003\SCCM-StreamCruiseControlMethod"
        r"\Datasets\Real\Datasets_Generators_CSV\GASD"
    )

    X, y = PublicDS.get_GASD(
        path,
        gas_id=None
    )

    row = quantify_one_real_dataset_half_split(
        dataset_name="GASD",
        domain="Chemical Sensing",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        regression_model="ridge",
        ridge_alpha=1.0,
        print_results=True
    )

    all_rows.append(row)

    # ========================================================
    # 7) CalCOFI: Oceanographic Dataset
    # ========================================================
    print("\nProcessing CalCOFI...")

    dataset_dir = Path(
        r"C:\New\003\SCCM-StreamCruiseControlMethod"
        r"\Datasets\Real\Datasets_Generators_CSV\CalCOFI"
    )

    bottle_path = dataset_dir / "bottle.csv"
    cast_path = dataset_dir / "cast.csv"

    X, y = PublicDS.get_CALCOFI(
        bottle_path=bottle_path,
        cast_path=cast_path,
        train_percent=90
    )

    row = quantify_one_real_dataset_half_split(
        dataset_name="CalCOFI",
        domain="Oceanography",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        print_results=True
    )

    all_rows.append(row)


    # ========================================================
    # 8) WSSF: Walmart Store Sales Forecasting
    # ========================================================
    print("\nProcessing WSSF...")

    dataset_dir = Path(
        r"C:\New\003\SCCM-StreamCruiseControlMethod"
        r"\Datasets\Real\Datasets_Generators_CSV\WSSF"
    )

    train_path = dataset_dir / "train.csv"
    features_path = dataset_dir / "features.csv"
    stores_path = dataset_dir / "stores.csv"
    test_path = dataset_dir / "test.csv"

    X, y = PublicDS.get_WSSF(
        train_path=train_path,
        features_path=features_path,
        stores_path=stores_path,
        test_path=test_path,
        train_percent=90
    )

    row = quantify_one_real_dataset_half_split(
        dataset_name="WSSF",
        domain="Retail/Sales Forecasting",
        X=X,
        y=y,
        bins=30,
        ks_alpha=0.05,
        standardize_X=True,
        regression_model="ridge",
        ridge_alpha=1.0,
        print_results=True
    )

    all_rows.append(row)
    # ========================================================
    # ========================================================
    # ========================================================
    # Print and save final table
    # ========================================================
    print_real_drift_metrics_table(all_rows)

    output_dir = Path("Results/DriftMetrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_real_drift_metrics_to_csv(
        all_rows,
        file_path=output_dir / "real_drift_metrics_summary.csv"
    )

    save_real_drift_metrics_to_excel(
        all_rows,
        file_path=output_dir / "real_drift_metrics_summary.xlsx"
    )