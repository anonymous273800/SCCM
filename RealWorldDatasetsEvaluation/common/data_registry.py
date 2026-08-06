from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from RealWorldDatasetsEvaluation.config import DATASET_SETTINGS
from RealWorldDatasetsEvaluation.common.project import ensure_project_importable


@dataclass
class DataBundle:
    dataset: str
    fit_X: np.ndarray
    fit_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    total_samples: int
    train_samples: int
    stream_samples: int
    n_features: int
    increment_size: int
    report_interval: int
    olr_base_model_size: float
    metadata: dict[str, Any] = field(default_factory=dict)


def dataset_root() -> Path:
    root = ensure_project_importable()
    return root / "Datasets" / "Real" / "Datasets_Generators_CSV"


def required_paths() -> dict[str, list[Path]]:
    base = dataset_root()
    return {
        "CCPP": [base / "08_CCPP" / "008_Folds5x2_pp.csv"],
        "MCPD": [base / "05_MCPD" / "005_insurance.csv"],
        "KCHSD": [base / "07_KCHSD" / "007_kc_house_data.csv"],
        "1KC": [base / "06_1KC" / "006_1000_Companies.csv"],
        "UCIAQD": [base / "UCIAQD" / "AirQualityUCI.csv"],
        "GASD": [base / "GASD" / f"batch{i}.dat" for i in range(1, 11)],
        "CalCOFI": [base / "CalCOFI" / "bottle.csv", base / "CalCOFI" / "cast.csv"],
        "WSSF": [
            base / "WSSF" / "train.csv",
            base / "WSSF" / "features.csv",
            base / "WSSF" / "stores.csv",
            base / "WSSF" / "test.csv",
        ],
    }


def _ordinary_bundle(dataset: str, X, y, metadata=None) -> DataBundle:
    settings = DATASET_SETTINGS[dataset]
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    train_count = int(float(settings["train_percent"]) * len(y) / 100.0)
    return DataBundle(
        dataset=dataset,
        fit_X=X[:train_count],
        fit_y=y[:train_count],
        test_X=X[train_count:],
        test_y=y[train_count:],
        total_samples=int(len(y)),
        train_samples=int(train_count),
        stream_samples=int(len(y) - train_count),
        n_features=int(X.shape[1]),
        increment_size=int(settings["increment_size"]),
        report_interval=int(settings["report_interval"]),
        olr_base_model_size=1.0,
        metadata=metadata or {},
    )



def _load_uciaqd_locally(path: Path, train_percent: float) -> DataBundle:
    """Load AirQualityUCI.csv without depending on the repository PublicDS version.

    The parser supports the original semicolon/decimal-comma format and common
    comma-delimited copies. All imputation and scaling statistics are fitted only
    on the initial chronological training segment.
    """
    target = "CO(GT)"
    features = [
        "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
        "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    ]

    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    required = ["Date", "Time", target] + features
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "AirQualityUCI.csv is missing required columns: "
            f"{missing}. Found columns: {list(df.columns)}"
        )

    date_text = df["Date"].astype(str).str.strip()
    time_text = df["Time"].astype(str).str.strip().str.replace(".", ":", regex=False)
    timestamp_text = date_text + " " + time_text
    candidates = []
    for fmt, name in [
        ("%d/%m/%Y %H:%M:%S", "day/month/year"),
        ("%m/%d/%Y %H:%M:%S", "month/day/year"),
        ("%Y-%m-%d %H:%M:%S", "year-month-day"),
    ]:
        ts = pd.to_datetime(timestamp_text, format=fmt, errors="coerce")
        candidates.append((int(ts.notna().sum()), name, ts))
    valid_count, selected_format, timestamps = max(candidates, key=lambda x: x[0])
    if valid_count == 0:
        raise ValueError("Could not parse any Date/Time values from AirQualityUCI.csv.")

    numeric = [target] + features
    for column in numeric:
        values = (
            df[column].astype(str).str.strip().str.replace(",", ".", regex=False)
        )
        df[column] = pd.to_numeric(values, errors="coerce")
    df[numeric] = df[numeric].replace(-200, np.nan)

    prepared = df[numeric].copy()
    prepared.insert(0, "Timestamp", timestamps)
    invalid_timestamp = prepared["Timestamp"].isna()
    missing_target = prepared[target].isna()
    rows_invalid_timestamp = int(invalid_timestamp.sum())
    rows_missing_target = int(((~invalid_timestamp) & missing_target).sum())
    prepared = prepared.dropna(subset=["Timestamp", target])
    prepared = prepared.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("No valid UCIAQD observations remained after cleaning.")

    n_samples = int(len(prepared))
    train_size = int(float(train_percent) * n_samples / 100.0)
    if train_size < 2 or train_size >= n_samples:
        raise ValueError(
            f"Invalid UCIAQD training segment: train_size={train_size}, total={n_samples}"
        )

    X_df = prepared[features].copy()
    missing_features_before = int(X_df.isna().sum().sum())
    X_df = X_df.ffill()
    base_medians = X_df.iloc[:train_size].median(numeric_only=True)
    X_df = X_df.fillna(base_medians)
    if X_df.isna().any().any():
        unresolved = X_df.columns[X_df.isna().any()].tolist()
        raise ValueError(f"Missing UCIAQD features remain after imputation: {unresolved}")

    X = X_df.to_numpy(dtype=np.float64)
    y = prepared[target].to_numpy(dtype=np.float64)
    X_scaler = StandardScaler().fit(X[:train_size])
    y_scaler = StandardScaler().fit(y[:train_size].reshape(-1, 1))
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

    if not np.all(np.isfinite(X_scaled)) or not np.all(np.isfinite(y_scaled)):
        raise ValueError("UCIAQD preprocessing produced non-finite values.")

    olr_base_pct = 100.0 * (train_size + 0.5) / n_samples
    metadata = {
        "dataset_name": "Air Quality UCI",
        "target_name": target,
        "feature_names": features,
        "selected_date_format": selected_format,
        "rows_with_invalid_timestamp_removed": rows_invalid_timestamp,
        "rows_with_missing_target_removed": rows_missing_target,
        "missing_feature_values_imputed": missing_features_before,
        "preprocessing": "chronological imputation and base-segment standardization",
    }
    print(
        f"UCIAQD loaded locally | samples={n_samples} | features={X_scaled.shape[1]} | "
        f"base_samples={train_size} | stream_samples={n_samples-train_size} | "
        f"date_format={selected_format}"
    )
    return DataBundle(
        dataset="UCIAQD", fit_X=X_scaled, fit_y=y_scaled,
        test_X=X_scaled[train_size:], test_y=y_scaled[train_size:],
        total_samples=n_samples, train_samples=train_size,
        stream_samples=n_samples-train_size, n_features=int(X_scaled.shape[1]),
        increment_size=int(DATASET_SETTINGS["UCIAQD"]["increment_size"]),
        report_interval=int(DATASET_SETTINGS["UCIAQD"]["report_interval"]),
        olr_base_model_size=float(olr_base_pct), metadata=metadata,
    )

def load_dataset(dataset: str) -> DataBundle:
    ensure_project_importable()
    from Datasets.Real import PublicDS

    paths = required_paths()[dataset]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s): " + "; ".join(missing))

    if dataset == "CCPP":
        return _ordinary_bundle(dataset, *PublicDS.get_CCPP(paths[0]))
    if dataset == "MCPD":
        return _ordinary_bundle(dataset, *PublicDS.get_medical_cost_personal_dataset(paths[0]))
    if dataset == "KCHSD":
        return _ordinary_bundle(dataset, *PublicDS.get_king_county_house_sales_data(paths[0]))
    if dataset == "1KC":
        return _ordinary_bundle(dataset, *PublicDS.get_profit_estimation_for_companies_dataset(paths[0]))

    settings = DATASET_SETTINGS[dataset]
    if dataset == "UCIAQD":
        # Use a local robust loader so the evaluation does not depend on which
        # historical PublicDS.get_UCIAQD implementation is installed.
        return _load_uciaqd_locally(paths[0], settings["train_percent"])

    if dataset == "GASD":
        X, y, batch_ids = PublicDS.get_GASD(
            dataset_root() / "GASD", gas_id=None, return_batch_ids=True
        )
        X_model, y_model, X_test, y_test, stream_batch_ids, base_pct, base_count = (
            PublicDS.prepare_GASD_for_existing_model_calls(X, y, batch_ids, base_batch=1)
        )
        olr_wrapper_total = int(len(y_model) + len(y_test))
        olr_base_pct = 100.0 * (int(base_count) + 0.5) / olr_wrapper_total
        return DataBundle(
            dataset=dataset, fit_X=np.asarray(X_model), fit_y=np.asarray(y_model),
            test_X=np.asarray(X_test), test_y=np.asarray(y_test),
            total_samples=int(len(y_model)), train_samples=int(base_count),
            stream_samples=int(len(y_test)), n_features=int(X_model.shape[1]),
            increment_size=int(settings["increment_size"]),
            report_interval=int(settings["report_interval"]),
            olr_base_model_size=float(olr_base_pct),
            metadata={"stream_batches": np.unique(stream_batch_ids).astype(int).tolist()},
        )

    if dataset == "CalCOFI":
        X, y, metadata = PublicDS.get_CALCOFI(
            bottle_path=paths[0], cast_path=paths[1],
            train_percent=settings["train_percent"], return_metadata=True,
        )
        return _ordinary_bundle(dataset, X, y, metadata)

    if dataset == "WSSF":
        X, y, metadata = PublicDS.get_WSSF(
            train_path=paths[0], features_path=paths[1], stores_path=paths[2],
            test_path=paths[3], train_percent=settings["train_percent"],
            return_metadata=True,
        )
        return _ordinary_bundle(dataset, X, y, metadata)

    raise KeyError(f"Unknown dataset: {dataset}")
