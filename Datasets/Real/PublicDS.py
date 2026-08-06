import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, Normalizer
import numpy as np
from pathlib import Path
import re


def get_medical_cost_personal_dataset(path):
    """
        Dataset: Medical Cost Personal Datasets
        source: https://www.kaggle.com/datasets/mirichoi0218/insurance
        Purpose: Predicts charges "Individual medical costs billed by health insurance"

        Load and preprocess the Medical Cost Personal Datasets.

        This function reads the CSV dataset file, performs one-hot encoding on categorical variables, and returns
        the feature matrix (X) and target vector (y) for predicting individual medical costs billed by health insurance.

        Parameters:
            path (str): Path to the CSV dataset file.

        Returns:
            X_scaled (numpy.ndarray): Scaled feature matrix (X) after preprocessing.
            y_scaled (numpy.ndarray): Scaled target vector (y) after preprocessing.
        """

    data = pd.read_csv(path)
    data = data.dropna()

    # Perform one-hot encoding using pd.get_dummies()
    data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    X = data_encoded.drop('charges', axis=1)
    y = data_encoded['charges'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled.flatten()



def get_profit_estimation_for_companies_dataset(path):
    """
        Dataset: 1000 Companies Profit
        source: https://www.kaggle.com/datasets/rupakroy/1000-companies-profit
        Purpose: Predicts the profit of these companies based on their operating expenses
        and other factors.

        Load and preprocess the 1000 Companies Profit dataset.

        This function reads the CSV dataset file, performs one-hot encoding on the categorical variable 'State',
        scales numerical features using Min-Max scaling, and returns the feature matrix (X) and target vector (y)
        for predicting the profit of companies based on their operating expenses and other factors.

        Parameters:
            path (str): Path to the CSV dataset file.

        Returns:
            X (numpy.ndarray): Feature matrix (X) after preprocessing.
            y (numpy.ndarray): Target vector (y) after preprocessing.
        """

    # Load the dataset from the given path
    df = pd.read_csv(path)
    # Encode categorical variable 'State' using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=["State"])
    # Scale numerical features using Min-Max scaling to normalize the data
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    # Separate features (X) and target variable (y)
    X = df_normalized.drop(["Profit"], axis=1).values
    y = df_normalized["Profit"].values

    return X, y


def get_king_county_house_sales_data(path):
    """
    Dataset: Kind County House Sales Dataset
    source: https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa
    Purpose: Predicts the sales price of houses in King County, Seattle.

    Load and preprocess the Kind County House Sales Dataset.

    This function reads the CSV dataset file, performs necessary preprocessing steps, and returns the feature matrix (X)
    and target vector (y) for predicting the sales price of houses in King County, Seattle.

    Parameters:
        path (str): Path to the CSV dataset file.

    Returns:
        X_scaled (numpy.ndarray): Scaled feature matrix (X) after preprocessing.
        y_scaled (numpy.ndarray): Scaled target vector (y) after preprocessing.
    """

    # Load the dataset
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis=1)

    # Just keep those features
    columns_to_keep = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15',
                       'bedrooms', 'bathrooms', 'view', 'price']
    df = df[columns_to_keep]
    # Handling missing values (if any)
    df = df.dropna()
    # Create feature matrix X
    X = df.drop('price', axis=1)
    # Create target vector y
    y = df['price']

    # Scaling and normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled


def get_CCPP(path):
    """
        Dataset: combined cycle powerplant
        source: https://www.kaggle.com/datasets/gova26/airpressure
        Purpose: Predicts the net hourly electrical energy output (EP) of the plant.

        Load and preprocess the Combined Cycle Power Plant dataset.

        This function reads the CSV dataset file, normalizes the features using Min-Max scaling,
        and returns the feature matrix (X) and target vector (y) for predicting the net hourly
        electrical energy output (EP) of the power plant.

        Parameters:
            path (str): Path to the CSV dataset file.

        Returns:
            X (numpy.ndarray): Feature matrix (X) after preprocessing.
            y (numpy.ndarray): Target vector (y) after preprocessing.
        """
    # Load the dataset from the given path
    df = pd.read_csv(path)
    # Normalize the features using Min-Max scaling
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    # Separate features (X) and target variable (y)
    X = df_normalized.drop(["PE"], axis=1).values
    y = df_normalized["PE"].values

    return X, y


GASD_GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene"
}








def get_GASD(path, gas_id=None, return_batch_ids=False):
    """
    Dataset: Gas Sensor Array Drift at Different Concentrations
    Source:
    https://archive.ics.uci.edu/dataset/270/
    gas+sensor+array+drift+dataset+at+different+concentrations

    Purpose:
        Predict gas concentration using 128 gas-sensor features while
        preserving the chronological order of batch1.dat through batch10.dat.

    Gas IDs:
        1 = Ethanol
        2 = Ethylene
        3 = Ammonia
        4 = Acetaldehyde
        5 = Acetone
        6 = Toluene

    Parameters:
        path (str):
            Folder containing batch1.dat through batch10.dat.

        gas_id (int or None):
            Use 1 through 6 to load one gas. Use None to load all gases.

        return_batch_ids (bool):
            When True, also return the chronological batch number associated
            with every observation.

    Returns:
        X_scaled (numpy.ndarray):
            Standardized sensor features. When gas_id is None, six one-hot
            gas-identity columns are appended to the 128 sensor features,
            producing 134 features.

        y_scaled (numpy.ndarray):
            Standardized gas-concentration regression target.

        batch_ids (numpy.ndarray), optional:
            Batch number for every observation.

    Notes:
        The feature and target scalers are fitted only on Batch 1, then applied
        unchanged to Batches 2 through 10. This prevents future-data leakage.
    """

    if gas_id is not None and gas_id not in GASD_GAS_NAMES:
        raise ValueError(
            "gas_id must be between 1 and 6, or None to load all gases. "
            f"Received: {gas_id}"
        )

    dataset_path = Path(path)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"GASD folder was not found: {dataset_path}"
        )

    batch_files = []

    for file_path in dataset_path.glob("*.dat"):
        match = re.fullmatch(
            r"batch(\d+)",
            file_path.stem,
            flags=re.IGNORECASE
        )

        if match:
            batch_number = int(match.group(1))
            batch_files.append((batch_number, file_path))

    batch_files.sort(key=lambda item: item[0])

    available_batches = [batch_number for batch_number, _ in batch_files]
    expected_batches = list(range(1, 11))

    if available_batches != expected_batches:
        raise FileNotFoundError(
            "Expected batch1.dat through batch10.dat.\n"
            f"Found batches: {available_batches}"
        )

    X_rows = []
    y_values = []
    batch_ids = []
    gas_ids = []

    for batch_number, file_path in batch_files:
        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    parts = line.split()
                    gas_text, concentration_text = parts[0].split(";", 1)

                    current_gas_id = int(gas_text)
                    concentration = float(concentration_text)

                    if current_gas_id not in GASD_GAS_NAMES:
                        raise ValueError(
                            f"Invalid gas ID: {current_gas_id}"
                        )

                    if gas_id is not None and current_gas_id != gas_id:
                        continue

                    features = np.zeros(128, dtype=np.float64)

                    for feature_item in parts[1:]:
                        index_text, value_text = feature_item.split(":", 1)

                        feature_index = int(index_text)
                        feature_value = float(value_text)

                        if not 1 <= feature_index <= 128:
                            raise ValueError(
                                f"Invalid feature index: {feature_index}"
                            )

                        features[feature_index - 1] = feature_value

                    X_rows.append(features)
                    y_values.append(concentration)
                    batch_ids.append(batch_number)
                    gas_ids.append(current_gas_id)

                except (ValueError, IndexError) as error:
                    raise ValueError(
                        f"Could not parse {file_path.name}, "
                        f"line {line_number}: {error}"
                    ) from error

    if not X_rows:
        selected_name = (
            "all gases"
            if gas_id is None
            else GASD_GAS_NAMES[gas_id]
        )
        raise ValueError(
            f"No observations were found for {selected_name}."
        )

    X = np.vstack(X_rows)
    y = np.asarray(y_values, dtype=np.float64)
    batch_ids = np.asarray(batch_ids, dtype=np.int32)
    gas_ids = np.asarray(gas_ids, dtype=np.int32)

    initial_batch_mask = batch_ids == 1

    if not np.any(initial_batch_mask):
        raise ValueError(
            "Batch 1 has no observations for the selected gas setting."
        )

    X_scaler = StandardScaler()
    X_scaler.fit(X[initial_batch_mask])
    X_scaled = X_scaler.transform(X)

    # A combined model needs the gas identity because each gas can have a
    # different sensor-response-to-concentration relationship.
    if gas_id is None:
        gas_one_hot = np.eye(6, dtype=np.float64)[gas_ids - 1]
        X_scaled = np.hstack((X_scaled, gas_one_hot))

    y_scaler = StandardScaler()
    y_scaler.fit(y[initial_batch_mask].reshape(-1, 1))
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).flatten()

    selected_gas_name = (
        "All gases"
        if gas_id is None
        else GASD_GAS_NAMES[gas_id]
    )

    print(
        f"GASD loaded: {selected_gas_name} | "
        f"samples={len(y_scaled)} | "
        f"features={X_scaled.shape[1]} | "
        f"batches={available_batches}"
    )

    if return_batch_ids:
        return X_scaled, y_scaled, batch_ids

    return X_scaled, y_scaled


def prepare_GASD_for_existing_model_calls(X, y, batch_ids, base_batch=1):
    """
    Prepare GASD for the project's existing OLR-WA model wrappers.

    The existing wrappers run their internal online loop over the first X/y
    arguments and use X_test/y_test only for the final test prediction. This
    function therefore returns the complete chronological sequence as the
    model input, while also returning Batches 2-10 as the final test stream.

    The returned base_model_size is calculated so the existing project helper
    resolves to exactly the number of samples in Batch 1. No model loop is
    changed by this preparation.

    Returns:
        X_model (numpy.ndarray):
            Complete chronological sequence beginning with Batch 1.

        y_model (numpy.ndarray):
            Target values corresponding to X_model.

        X_test (numpy.ndarray):
            Chronological samples after the base batch.

        y_test (numpy.ndarray):
            Targets corresponding to X_test.

        stream_batch_ids (numpy.ndarray):
            Batch IDs corresponding to X_test/y_test.

        base_model_size (float):
            Exact base-model percentage to pass to the existing wrappers.

        base_sample_count (int):
            Number of samples in the base batch.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    batch_ids = np.asarray(batch_ids)

    if X.ndim != 2:
        raise ValueError(
            f"X must be two-dimensional. Received shape: {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"y must be one-dimensional. Received shape: {y.shape}"
        )

    if batch_ids.ndim != 1:
        raise ValueError(
            "batch_ids must be one-dimensional. "
            f"Received shape: {batch_ids.shape}"
        )

    if not (X.shape[0] == y.shape[0] == batch_ids.shape[0]):
        raise ValueError(
            "X, y, and batch_ids must contain the same number of samples."
        )

    base_mask = batch_ids == base_batch
    stream_mask = batch_ids > base_batch

    if not np.any(base_mask):
        raise ValueError(
            f"Base batch {base_batch} contains no samples."
        )

    if not np.any(stream_mask):
        raise ValueError(
            f"No stream samples were found after base batch {base_batch}."
        )

    # Reconstruct explicitly to guarantee that the complete sequence starts
    # with the exact base segment expected by the existing wrappers.
    X_base = X[base_mask]
    y_base = y[base_mask]
    X_test = X[stream_mask]
    y_test = y[stream_mask]
    stream_batch_ids = batch_ids[stream_mask]

    X_model = np.vstack((X_base, X_test))
    y_model = np.concatenate((y_base, y_test))

    base_sample_count = int(X_base.shape[0])
    total_sample_count = int(X_model.shape[0])

    # The project's base-size helper uses:
    # int(total_samples * percentage / 100).
    # The midpoint below resolves to the exact Batch-1 sample count.
    base_model_size = (
        100.0 * (base_sample_count + 0.5) / total_sample_count
    )

    resolved_base_count = int(
        total_sample_count * base_model_size / 100.0
    )

    if resolved_base_count != base_sample_count:
        raise RuntimeError(
            "Could not preserve the exact Batch-1 base-model size. "
            f"Expected {base_sample_count}, resolved {resolved_base_count}."
        )

    return (
        X_model,
        y_model,
        X_test,
        y_test,
        stream_batch_ids,
        base_model_size,
        base_sample_count
    )


UCIAQD_TARGET = "CO(GT)"

UCIAQD_FEATURES = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]


def get_UCIAQD(path, train_percent=10, return_metadata=False):
    """
    Load and prepare the Air Quality UCI regression dataset.

    Target:
        CO(GT)

    Features:
        PT08.S1(CO)
        PT08.S2(NMHC)
        PT08.S3(NOx)
        PT08.S4(NO2)
        PT08.S5(O3)
        T
        RH
        AH

    The chronological order is preserved.

    Missing-value handling:
        -200 is treated as missing.
        Rows with missing target values are removed.
        Missing feature values are forward-filled.
        Remaining initial missing values are filled using medians calculated
        only from the initial training segment.

    Scaling:
        Feature and target scalers are fitted only on the initial training
        segment to avoid using future stream information.
    """

    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"AirQualityUCI.csv was not found: {dataset_path}"
        )

    if not 0 < train_percent < 100:
        raise ValueError(
            "train_percent must be between 0 and 100. "
            f"Received: {train_percent}"
        )

    # Original UCI format:
    # semicolon separator and comma decimal notation.
    df = pd.read_csv(
        dataset_path,
        sep=";",
        decimal=",",
        low_memory=False
    )

    # Support Kaggle copies that use ordinary comma-separated formatting.
    if df.shape[1] == 1:
        df = pd.read_csv(
            dataset_path,
            sep=",",
            low_memory=False
        )

    # Clean column names.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Remove empty trailing columns.
    df = df.dropna(axis=1, how="all")
    df = df.loc[
        :,
        ~df.columns.str.startswith("Unnamed")
    ]

    required_columns = (
        ["Date", "Time", UCIAQD_TARGET]
        + UCIAQD_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "AirQualityUCI.csv is missing required columns: "
            f"{missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    # Prepare the Date and Time text.
    date_text = (
        df["Date"]
        .astype(str)
        .str.strip()
    )

    time_text = (
        df["Time"]
        .astype(str)
        .str.strip()
        .str.replace(".", ":", regex=False)
    )

    timestamp_text = date_text + " " + time_text

    # Different Kaggle/UCI copies can use different date formats.
    # Try all supported formats and keep the one that parses the most rows.
    timestamp_formats = [
        ("%d/%m/%Y %H:%M:%S", "day/month/year"),
        ("%m/%d/%Y %H:%M:%S", "month/day/year"),
        ("%Y-%m-%d %H:%M:%S", "year-month-day")
    ]

    timestamp_candidates = []

    for timestamp_format, format_name in timestamp_formats:
        candidate = pd.to_datetime(
            timestamp_text,
            format=timestamp_format,
            errors="coerce"
        )

        timestamp_candidates.append(
            (
                int(candidate.notna().sum()),
                format_name,
                candidate
            )
        )

    (
        valid_timestamp_count,
        selected_date_format,
        timestamps
    ) = max(
        timestamp_candidates,
        key=lambda item: item[0]
    )

    if valid_timestamp_count == 0:
        raise ValueError(
            "Could not parse any Date/Time values "
            "from AirQualityUCI.csv."
        )

    numeric_columns = [
        UCIAQD_TARGET
    ] + UCIAQD_FEATURES

    # Convert target and features to numeric values.
    for column in numeric_columns:
        values = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        df[column] = pd.to_numeric(
            values,
            errors="coerce"
        )

    # The dataset uses -200 to represent missing values.
    df[numeric_columns] = df[numeric_columns].replace(
        -200,
        np.nan
    )

    prepared = df[numeric_columns].copy()

    prepared.insert(
        0,
        "Timestamp",
        timestamps
    )

    invalid_timestamp_mask = prepared["Timestamp"].isna()
    missing_target_mask = prepared[UCIAQD_TARGET].isna()

    rows_with_invalid_timestamp = int(
        invalid_timestamp_mask.sum()
    )

    rows_with_missing_target = int(
        (
            (~invalid_timestamp_mask)
            & missing_target_mask
        ).sum()
    )

    # The regression target must be known for evaluation.
    prepared = prepared.dropna(
        subset=[
            "Timestamp",
            UCIAQD_TARGET
        ]
    )

    # Preserve chronological stream order.
    prepared = (
        prepared
        .sort_values(
            "Timestamp",
            kind="stable"
        )
        .reset_index(drop=True)
    )

    if prepared.empty:
        raise ValueError(
            "No valid observations remained after cleaning."
        )

    n_samples = int(prepared.shape[0])

    train_size = int(
        train_percent
        * n_samples
        / 100.0
    )

    if train_size < 2:
        raise ValueError(
            "The initial training segment is too small. "
            f"train_size={train_size}, "
            f"total_samples={n_samples}"
        )

    if train_size >= n_samples:
        raise ValueError(
            "The initial training segment leaves no "
            "observations for the stream."
        )

    X_df = prepared[
        UCIAQD_FEATURES
    ].copy()

    y = prepared[
        UCIAQD_TARGET
    ].to_numpy(
        dtype=np.float64
    )

    missing_feature_values_before = int(
        X_df.isna().sum().sum()
    )

    # Forward fill uses only earlier chronological observations.
    X_df = X_df.ffill()

    # Fill any missing values at the beginning of the stream using
    # medians calculated only from the initial training segment.
    training_medians = (
        X_df
        .iloc[:train_size]
        .median(numeric_only=True)
    )

    X_df = X_df.fillna(
        training_medians
    )

    if X_df.isna().any().any():
        unresolved_columns = (
            X_df
            .columns[
                X_df.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Missing feature values remain after imputation: "
            f"{unresolved_columns}"
        )

    X = X_df.to_numpy(
        dtype=np.float64
    )

    # Fit the feature scaler only on the initial training segment.
    X_scaler = StandardScaler()

    X_scaler.fit(
        X[:train_size]
    )

    X_scaled = X_scaler.transform(
        X
    )

    # Fit the target scaler only on the initial training segment.
    y_scaler = StandardScaler()

    y_scaler.fit(
        y[:train_size].reshape(-1, 1)
    )

    y_scaled = (
        y_scaler
        .transform(
            y.reshape(-1, 1)
        )
        .flatten()
    )

    metadata = {
        "dataset_name": "Air Quality UCI",
        "target_name": UCIAQD_TARGET,
        "target_unit": "mg/m^3",
        "feature_names": list(UCIAQD_FEATURES),

        "n_samples": n_samples,
        "n_features": int(X_scaled.shape[1]),

        "train_percent": float(train_percent),
        "training_samples": train_size,
        "stream_samples": n_samples - train_size,

        "start_timestamp": (
            prepared["Timestamp"]
            .iloc[0]
            .isoformat()
        ),

        "end_timestamp": (
            prepared["Timestamp"]
            .iloc[-1]
            .isoformat()
        ),

        "selected_date_format": selected_date_format,

        "valid_timestamp_count": valid_timestamp_count,

        "rows_with_invalid_timestamp_removed": (
            rows_with_invalid_timestamp
        ),

        "rows_with_missing_target_removed": (
            rows_with_missing_target
        ),

        "missing_feature_values_imputed": (
            missing_feature_values_before
        )
    }

    print(
        f"UCIAQD loaded | "
        f"samples={n_samples} | "
        f"features={X_scaled.shape[1]} | "
        f"target={UCIAQD_TARGET} | "
        f"base_samples={train_size} | "
        f"stream_samples={n_samples - train_size} | "
        f"date_format={selected_date_format}"
    )

    if return_metadata:
        return (
            X_scaled,
            y_scaled,
            metadata
        )

    return (
        X_scaled,
        y_scaled
    )


def prepare_UCIAQD_for_existing_model_calls(
        X,
        y,
        train_percent=10
):
    """
    Prepare UCIAQD for the existing OLR-WA model functions.

    The complete chronological dataset is passed as X_model/y_model because
    the existing model implementations perform their online mini-batch loop
    over the first X/y arguments.

    X_test/y_test contain the observations after the initial base segment.

    No existing model loop is changed.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional. "
            f"Received shape: {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be one-dimensional. "
            f"Received shape: {y.shape}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if not 0 < train_percent < 100:
        raise ValueError(
            "train_percent must be between 0 and 100. "
            f"Received: {train_percent}"
        )

    total_sample_count = int(
        X.shape[0]
    )

    base_sample_count = int(
        train_percent
        * total_sample_count
        / 100.0
    )

    if base_sample_count < 2:
        raise ValueError(
            "The base segment is too small: "
            f"{base_sample_count} samples."
        )

    if base_sample_count >= total_sample_count:
        raise ValueError(
            "The base segment leaves no observations "
            "for the online stream."
        )

    # The existing models need the complete chronological sequence.
    X_model = X
    y_model = y

    # Final post-training stream.
    X_test = X[base_sample_count:]
    y_test = y[base_sample_count:]

    # The existing project helper calculates:
    #
    # int(total_samples * percentage / 100)
    #
    # Adding 0.5 ensures the percentage resolves to exactly the desired
    # integer base sample count.
    base_model_size = (
        100.0
        * (base_sample_count + 0.5)
        / total_sample_count
    )

    resolved_base_count = int(
        total_sample_count
        * base_model_size
        / 100.0
    )

    if resolved_base_count != base_sample_count:
        raise RuntimeError(
            "Could not preserve the exact UCIAQD "
            "base-model size. "
            f"Expected {base_sample_count}, "
            f"resolved {resolved_base_count}."
        )

    return (
        X_model,
        y_model,
        X_test,
        y_test,
        base_model_size,
        base_sample_count
    )




# ============================================================
# ELECTRIC POWER CONSUMPTION OF TETOUAN CITY
# ============================================================

EPC_TARGET_COLUMNS = {
    1: "PowerConsumption_Zone1",
    2: "PowerConsumption_Zone2",
    3: "PowerConsumption_Zone3"
}


EPC_BASE_FEATURES = [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "GeneralDiffuseFlows",
    "DiffuseFlows"
]


def _normalize_epc_column_name(column_name):
    """
    Normalize a column name for matching different Kaggle versions.
    """

    return "".join(
        character.lower()
        for character in str(column_name).strip()
        if character.isalnum()
    )


def _resolve_epc_columns(df):
    """
    Resolve the column names used by different copies of
    the Tetouan power-consumption dataset.
    """

    normalized_to_original = {
        _normalize_epc_column_name(column): column
        for column in df.columns
    }

    aliases = {
        "Datetime": [
            "datetime",
            "dateandtime",
            "date_time",
            "date"
        ],

        "Temperature": [
            "temperature",
            "temp"
        ],

        "Humidity": [
            "humidity"
        ],

        "WindSpeed": [
            "windspeed",
            "wind_speed",
            "wind"
        ],

        "GeneralDiffuseFlows": [
            "generaldiffuseflows",
            "generaldiffuseflow",
            "general_diffuse_flows"
        ],

        "DiffuseFlows": [
            "diffuseflows",
            "diffuseflow",
            "diffuse_flows"
        ],

        "PowerConsumption_Zone1": [
            "powerconsumptionzone1",
            "zone1powerconsumption",
            "zone1"
        ],

        "PowerConsumption_Zone2": [
            "powerconsumptionzone2",
            "zone2powerconsumption",
            "zone2"
        ],

        "PowerConsumption_Zone3": [
            "powerconsumptionzone3",
            "zone3powerconsumption",
            "zone3"
        ]
    }

    resolved_columns = {}

    for canonical_name, possible_names in aliases.items():

        matched_column = None

        for possible_name in possible_names:

            normalized_alias = _normalize_epc_column_name(
                possible_name
            )

            if normalized_alias in normalized_to_original:

                matched_column = normalized_to_original[
                    normalized_alias
                ]

                break

        if matched_column is None:

            raise ValueError(
                f"Could not find the EPC column "
                f"'{canonical_name}'. "
                f"Available columns: {list(df.columns)}"
            )

        resolved_columns[canonical_name] = matched_column

    return resolved_columns


def _parse_epc_datetime(datetime_values):
    """
    Parse the EPC DateTime column.

    Different copies may use month/day/year,
    day/month/year, or year-month-day.
    """

    datetime_text = (
        datetime_values
        .astype(str)
        .str.strip()
    )

    timestamp_formats = [
        ("%m/%d/%Y %H:%M", "month/day/year"),
        ("%d/%m/%Y %H:%M", "day/month/year"),
        ("%m/%d/%Y %H:%M:%S", "month/day/year"),
        ("%d/%m/%Y %H:%M:%S", "day/month/year"),
        ("%Y-%m-%d %H:%M", "year-month-day"),
        ("%Y-%m-%d %H:%M:%S", "year-month-day")
    ]

    candidates = []

    for timestamp_format, format_name in timestamp_formats:

        parsed_timestamps = pd.to_datetime(
            datetime_text,
            format=timestamp_format,
            errors="coerce"
        )

        valid_timestamp_count = int(
            parsed_timestamps.notna().sum()
        )

        valid_timestamps = (
            parsed_timestamps
            .dropna()
            .reset_index(drop=True)
        )

        if len(valid_timestamps) > 1:

            timestamp_differences = (
                valid_timestamps
                .diff()
                .dropna()
            )

            backward_steps = int(
                (
                    timestamp_differences
                    < pd.Timedelta(0)
                ).sum()
            )

            median_gap_seconds = float(
                timestamp_differences
                .abs()
                .median()
                .total_seconds()
            )

        else:

            backward_steps = 10 ** 9
            median_gap_seconds = float("inf")

        # EPC is normally recorded every 10 minutes.
        gap_distance = abs(
            median_gap_seconds - 600.0
        )

        candidates.append(
            (
                valid_timestamp_count,
                -backward_steps,
                -gap_distance,
                format_name,
                parsed_timestamps
            )
        )

    (
        valid_timestamp_count,
        _,
        _,
        selected_date_format,
        timestamps
    ) = max(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[2]
        )
    )

    if valid_timestamp_count == 0:

        raise ValueError(
            "Could not parse any EPC DateTime values."
        )

    return (
        timestamps,
        selected_date_format,
        valid_timestamp_count
    )





########################################CALCOFI
# ============================================================
# CALCOFI OCEANOGRAPHIC DATA
# ============================================================

CALCOFI_TARGET = "Salnty"

CALCOFI_BOTTLE_FEATURES = [
    "Depthm",
    "T_degC",
    "O2ml_L"
]

CALCOFI_CAST_FEATURES = [
    "Lat_Dec",
    "Lon_Dec"
]


def _normalize_calcofi_column_name(column_name):
    """
    Normalize a CalCOFI column name for robust matching.
    """

    return "".join(
        character.lower()
        for character in str(column_name).strip()
        if character.isalnum()
    )


def _resolve_calcofi_columns(
        available_columns,
        required_columns,
        file_name
):
    """
    Resolve CalCOFI columns even when the CSV contains
    leading or trailing spaces in its headers.
    """

    normalized_to_original = {
        _normalize_calcofi_column_name(column): column
        for column in available_columns
    }

    resolved_columns = {}

    for required_column in required_columns:

        normalized_required = (
            _normalize_calcofi_column_name(
                required_column
            )
        )

        if normalized_required not in normalized_to_original:

            raise ValueError(
                f"Could not find the CalCOFI column "
                f"'{required_column}' in {file_name}. "
                f"Available columns: "
                f"{list(available_columns)}"
            )

        resolved_columns[required_column] = (
            normalized_to_original[
                normalized_required
            ]
        )

    return resolved_columns


def _read_calcofi_required_columns(
        path,
        required_columns
):
    """
    Read only the required columns from a CalCOFI CSV file.

    Reading only the required columns is important because
    bottle.csv is very large.
    """

    dataset_path = Path(
        path
    )

    if not dataset_path.is_file():

        raise FileNotFoundError(
            f"CalCOFI file was not found: "
            f"{dataset_path}"
        )

    header = pd.read_csv(
        dataset_path,
        nrows=0,
        skipinitialspace=True
    )

    header.columns = [
        str(column).strip()
        for column in header.columns
    ]

    resolved_columns = _resolve_calcofi_columns(
        available_columns=header.columns,
        required_columns=required_columns,
        file_name=dataset_path.name
    )

    original_columns = [
        resolved_columns[column]
        for column in required_columns
    ]

    data = pd.read_csv(
        dataset_path,
        usecols=original_columns,
        skipinitialspace=True,
        low_memory=False
    )

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    rename_map = {
        resolved_columns[canonical_name]: canonical_name
        for canonical_name in required_columns
    }

    data = data.rename(
        columns=rename_map
    )

    return data


def _parse_calcofi_timestamp(
        cast_data
):
    """
    Create a chronological timestamp from the CalCOFI
    Date and Time columns.

    Falls back to Date only when Time is missing or invalid.
    """

    date_text = (
        cast_data["Date"]
        .astype(str)
        .str.strip()
    )

    if "Time" in cast_data.columns:

        time_text = (
            cast_data["Time"]
            .astype(str)
            .str.strip()
        )

        invalid_time_values = {
            "",
            "nan",
            "none",
            "nat"
        }

        time_text = time_text.where(
            ~time_text.str.lower().isin(
                invalid_time_values
            ),
            "00:00:00"
        )

        timestamp_text = (
            date_text
            + " "
            + time_text
        )

    else:

        timestamp_text = date_text

    timestamp_formats = [
        (
            "%m/%d/%Y %H:%M:%S",
            "month/day/year with time"
        ),
        (
            "%m/%d/%Y %H:%M",
            "month/day/year with time"
        ),
        (
            "%Y-%m-%d %H:%M:%S",
            "year-month-day with time"
        ),
        (
            "%Y-%m-%d %H:%M",
            "year-month-day with time"
        ),
        (
            "%m/%d/%Y",
            "month/day/year"
        ),
        (
            "%Y-%m-%d",
            "year-month-day"
        )
    ]

    timestamp_candidates = []

    for timestamp_format, format_name in timestamp_formats:

        candidate = pd.to_datetime(
            timestamp_text,
            format=timestamp_format,
            errors="coerce"
        )

        timestamp_candidates.append(
            (
                int(candidate.notna().sum()),
                format_name,
                candidate
            )
        )

    (
        valid_timestamp_count,
        selected_date_format,
        timestamps
    ) = max(
        timestamp_candidates,
        key=lambda item: item[0]
    )

    # Try automatic parsing as a final fallback.
    if valid_timestamp_count == 0:

        timestamps = pd.to_datetime(
            timestamp_text,
            errors="coerce"
        )

        valid_timestamp_count = int(
            timestamps.notna().sum()
        )

        selected_date_format = "automatic"

    if valid_timestamp_count == 0:

        raise ValueError(
            "Could not parse any CalCOFI cast timestamps."
        )

    return (
        timestamps,
        selected_date_format,
        valid_timestamp_count
    )


def get_CALCOFI(
        bottle_path,
        cast_path,
        train_percent=90,
        return_metadata=False
):
    """
    Load and prepare the CalCOFI Oceanographic dataset.

    Regression target:
        Salnty

    Features:
        Depthm
        T_degC
        O2ml_L
        Lat_Dec
        Lon_Dec
        Year
        Month_Sin
        Month_Cos
        DayOfYear_Sin
        DayOfYear_Cos

    The bottle and cast tables are merged using Cst_Cnt.

    Chronological order is preserved using the cast date,
    cast time, cast identifier, and bottle depth.

    MinMaxScaler objects are fitted only on the initial
    chronological training segment.

    Parameters
    ----------
    bottle_path : str or Path
        Path to bottle.csv.

    cast_path : str or Path
        Path to cast.csv.

    train_percent : float
        Initial chronological percentage used to fit
        the feature and target scalers.

    return_metadata : bool
        When True, also return preparation metadata.

    Returns
    -------
    X_normalized : numpy.ndarray
        Normalized feature matrix.

    y_normalized : numpy.ndarray
        Normalized salinity target.

    metadata : dict, optional
        Dataset preparation information.
    """

    bottle_dataset_path = Path(
        bottle_path
    )

    cast_dataset_path = Path(
        cast_path
    )

    if not 0 < train_percent < 100:

        raise ValueError(
            "train_percent must be between 0 and 100. "
            f"Received: {train_percent}"
        )

    # ========================================================
    # 1) READ BOTTLE DATA
    # ========================================================

    bottle_required_columns = [
        "Cst_Cnt",
        "Btl_Cnt",
        "Depthm",
        "T_degC",
        "Salnty",
        "O2ml_L"
    ]

    bottle_data = _read_calcofi_required_columns(
        path=bottle_dataset_path,
        required_columns=bottle_required_columns
    )

    original_bottle_rows = int(
        bottle_data.shape[0]
    )

    # ========================================================
    # 2) READ CAST DATA
    # ========================================================

    # Time is available in the standard cast.csv file.
    # Read the header first so the loader can also support
    # copies where Time is absent.
    cast_header = pd.read_csv(
        cast_dataset_path,
        nrows=0,
        skipinitialspace=True
    )

    cast_header.columns = [
        str(column).strip()
        for column in cast_header.columns
    ]

    normalized_cast_columns = {
        _normalize_calcofi_column_name(column)
        for column in cast_header.columns
    }

    cast_required_columns = [
        "Cst_Cnt",
        "Date",
        "Lat_Dec",
        "Lon_Dec"
    ]

    if (
        _normalize_calcofi_column_name("Time")
        in normalized_cast_columns
    ):

        cast_required_columns.append(
            "Time"
        )

    cast_data = _read_calcofi_required_columns(
        path=cast_dataset_path,
        required_columns=cast_required_columns
    )

    original_cast_rows = int(
        cast_data.shape[0]
    )

    # ========================================================
    # 3) CONVERT REQUIRED VALUES
    # ========================================================

    bottle_numeric_columns = [
        "Cst_Cnt",
        "Btl_Cnt",
        "Depthm",
        "T_degC",
        "Salnty",
        "O2ml_L"
    ]

    for column in bottle_numeric_columns:

        bottle_data[column] = pd.to_numeric(
            bottle_data[column],
            errors="coerce"
        )

    cast_numeric_columns = [
        "Cst_Cnt",
        "Lat_Dec",
        "Lon_Dec"
    ]

    for column in cast_numeric_columns:

        cast_data[column] = pd.to_numeric(
            cast_data[column],
            errors="coerce"
        )

    # ========================================================
    # 4) CREATE CAST TIMESTAMP
    # ========================================================

    (
        cast_timestamps,
        selected_date_format,
        valid_timestamp_count
    ) = _parse_calcofi_timestamp(
        cast_data
    )

    cast_data["Timestamp"] = (
        cast_timestamps
    )

    invalid_cast_timestamp_rows = int(
        cast_data["Timestamp"]
        .isna()
        .sum()
    )

    # Keep one metadata row for each cast.
    cast_data = cast_data.dropna(
        subset=[
            "Cst_Cnt",
            "Timestamp",
            "Lat_Dec",
            "Lon_Dec"
        ]
    )

    cast_data = (
        cast_data
        .sort_values(
            by=[
                "Timestamp",
                "Cst_Cnt"
            ],
            kind="stable"
        )
        .drop_duplicates(
            subset="Cst_Cnt",
            keep="first"
        )
        .reset_index(drop=True)
    )

    valid_cast_rows = int(
        cast_data.shape[0]
    )

    # ========================================================
    # 5) MERGE BOTTLE AND CAST TABLES
    # ========================================================

    prepared = bottle_data.merge(
        cast_data[
            [
                "Cst_Cnt",
                "Timestamp",
                "Lat_Dec",
                "Lon_Dec"
            ]
        ],
        on="Cst_Cnt",
        how="inner",
        validate="many_to_one"
    )

    merged_rows_before_cleaning = int(
        prepared.shape[0]
    )

    # ========================================================
    # 6) REMOVE INVALID ESSENTIAL VALUES
    # ========================================================

    essential_columns = [
        "Timestamp",
        "Cst_Cnt",
        "Btl_Cnt",
        "Depthm",
        "T_degC",
        "Salnty",
        "Lat_Dec",
        "Lon_Dec"
    ]

    rows_with_missing_target = int(
        prepared["Salnty"]
        .isna()
        .sum()
    )

    rows_with_missing_temperature = int(
        prepared["T_degC"]
        .isna()
        .sum()
    )

    rows_with_missing_depth = int(
        prepared["Depthm"]
        .isna()
        .sum()
    )

    prepared = prepared.dropna(
        subset=essential_columns
    )

    # Remove physically invalid measurements.
    prepared = prepared[
        (prepared["Depthm"] >= 0.0)
        & prepared["T_degC"].between(
            -5.0,
            45.0
        )
        & prepared["Salnty"].between(
            0.0,
            50.0
        )
        & prepared["Lat_Dec"].between(
            -90.0,
            90.0
        )
        & prepared["Lon_Dec"].between(
            -180.0,
            180.0
        )
    ]

    # Preserve the actual stream order.
    prepared = (
        prepared
        .sort_values(
            by=[
                "Timestamp",
                "Cst_Cnt",
                "Depthm",
                "Btl_Cnt"
            ],
            kind="stable"
        )
        .reset_index(drop=True)
    )

    if prepared.empty:

        raise ValueError(
            "No valid CalCOFI observations remained "
            "after cleaning and merging."
        )

    # ========================================================
    # 7) DETERMINE TRAINING SIZE
    # ========================================================

    n_samples = int(
        prepared.shape[0]
    )

    train_size = int(
        train_percent
        * n_samples
        / 100.0
    )

    if train_size < 2:

        raise ValueError(
            "The CalCOFI training segment is too small. "
            f"train_size={train_size}"
        )

    if train_size >= n_samples:

        raise ValueError(
            "The CalCOFI training segment leaves no "
            "test observations."
        )

    # ========================================================
    # 8) HANDLE OPTIONAL OXYGEN VALUES
    # ========================================================

    missing_oxygen_values_before = int(
        prepared["O2ml_L"]
        .isna()
        .sum()
    )

    # First use earlier observations from the same cast.
    prepared["O2ml_L"] = (
        prepared
        .groupby(
            "Cst_Cnt",
            sort=False
        )["O2ml_L"]
        .ffill()
    )

    # Then use previous chronological observations.
    prepared["O2ml_L"] = (
        prepared["O2ml_L"]
        .ffill()
    )

    # Fill unresolved initial values using only the training
    # segment median.
    training_oxygen_median = (
        prepared["O2ml_L"]
        .iloc[:train_size]
        .median()
    )

    if pd.isna(
            training_oxygen_median
    ):

        raise ValueError(
            "The initial CalCOFI training segment "
            "contains no valid O2ml_L measurements."
        )

    prepared["O2ml_L"] = (
        prepared["O2ml_L"]
        .fillna(
            training_oxygen_median
        )
    )

    # Reasonable physical validation for dissolved oxygen.
    prepared["O2ml_L"] = (
        prepared["O2ml_L"]
        .clip(
            lower=0.0
        )
    )

    # ========================================================
    # 9) CREATE CHRONOLOGICAL FEATURES
    # ========================================================

    timestamp_series = prepared[
        "Timestamp"
    ]

    month = (
        timestamp_series.dt.month
    )

    day_of_year = (
        timestamp_series.dt.dayofyear
    )

    prepared["Year"] = (
        timestamp_series.dt.year
        .astype(np.float64)
    )

    prepared["Month_Sin"] = np.sin(
        2.0
        * np.pi
        * (month - 1)
        / 12.0
    )

    prepared["Month_Cos"] = np.cos(
        2.0
        * np.pi
        * (month - 1)
        / 12.0
    )

    prepared["DayOfYear_Sin"] = np.sin(
        2.0
        * np.pi
        * (day_of_year - 1)
        / 365.25
    )

    prepared["DayOfYear_Cos"] = np.cos(
        2.0
        * np.pi
        * (day_of_year - 1)
        / 365.25
    )

    feature_names = [
        "Depthm",
        "T_degC",
        "O2ml_L",
        "Lat_Dec",
        "Lon_Dec",
        "Year",
        "Month_Sin",
        "Month_Cos",
        "DayOfYear_Sin",
        "DayOfYear_Cos"
    ]

    # ========================================================
    # 10) BUILD X AND y
    # ========================================================

    X = (
        prepared[feature_names]
        .to_numpy(
            dtype=np.float64
        )
    )

    y = (
        prepared[CALCOFI_TARGET]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.all(
            np.isfinite(X)
    ):

        raise ValueError(
            "CalCOFI features contain NaN or infinity "
            "before normalization."
        )

    if not np.all(
            np.isfinite(y)
    ):

        raise ValueError(
            "CalCOFI target contains NaN or infinity "
            "before normalization."
        )

    # ========================================================
    # 11) NORMALIZE USING TRAINING SEGMENT ONLY
    # ========================================================

    X_scaler = MinMaxScaler(
        feature_range=(0.0, 1.0),
        clip=True
    )

    X_scaler.fit(
        X[:train_size]
    )

    X_normalized = X_scaler.transform(
        X
    )

    y_scaler = MinMaxScaler(
        feature_range=(0.0, 1.0),
        clip=True
    )

    y_scaler.fit(
        y[:train_size].reshape(-1, 1)
    )

    y_normalized = (
        y_scaler
        .transform(
            y.reshape(-1, 1)
        )
        .flatten()
    )

    # ========================================================
    # 12) FINAL VALIDATION
    # ========================================================

    if not np.all(
            np.isfinite(X_normalized)
    ):

        raise ValueError(
            "Normalized CalCOFI features contain "
            "NaN or infinity."
        )

    if not np.all(
            np.isfinite(y_normalized)
    ):

        raise ValueError(
            "Normalized CalCOFI target contains "
            "NaN or infinity."
        )

    # ========================================================
    # 13) METADATA
    # ========================================================

    metadata = {
        "dataset_name": (
            "CalCOFI Oceanographic Data"
        ),

        "dataset_short_name": "CalCOFI",

        "bottle_file": (
            bottle_dataset_path.name
        ),

        "cast_file": (
            cast_dataset_path.name
        ),

        "target_name": CALCOFI_TARGET,

        "target_unit": "Practical Salinity",

        "feature_names": list(
            feature_names
        ),

        "n_samples": int(
            n_samples
        ),

        "n_features": int(
            X_normalized.shape[1]
        ),

        "training_percent": float(
            train_percent
        ),

        "training_samples": int(
            train_size
        ),

        "test_samples": int(
            n_samples - train_size
        ),

        "original_bottle_rows": int(
            original_bottle_rows
        ),

        "original_cast_rows": int(
            original_cast_rows
        ),

        "valid_cast_rows": int(
            valid_cast_rows
        ),

        "merged_rows_before_cleaning": int(
            merged_rows_before_cleaning
        ),

        "invalid_cast_timestamp_rows": int(
            invalid_cast_timestamp_rows
        ),

        "rows_with_missing_target_removed": int(
            rows_with_missing_target
        ),

        "rows_with_missing_temperature_removed": int(
            rows_with_missing_temperature
        ),

        "rows_with_missing_depth_removed": int(
            rows_with_missing_depth
        ),

        "missing_oxygen_values_imputed": int(
            missing_oxygen_values_before
        ),

        "selected_date_format": (
            selected_date_format
        ),

        "valid_timestamp_count": int(
            valid_timestamp_count
        ),

        "start_timestamp": (
            prepared["Timestamp"]
            .iloc[0]
            .isoformat()
        ),

        "end_timestamp": (
            prepared["Timestamp"]
            .iloc[-1]
            .isoformat()
        ),

        "normalization": "MinMaxScaler",

        "normalization_range": [
            0.0,
            1.0
        ],

        "scaler_fitted_on_training_only": True,

        "chronological_order": True,

        "merge_column": "Cst_Cnt",

        "known_drift_location": None,

        "known_drift_type": None,

        "natural_nonstationarity": True
    }

    # ========================================================
    # 14) PRINT SUMMARY
    # ========================================================

    print()
    print(
        "CalCOFI loaded and normalized"
    )
    print(
        "========================================"
    )
    print(
        f"Bottle file: {bottle_dataset_path}"
    )
    print(
        f"Cast file: {cast_dataset_path}"
    )
    print(
        f"Original bottle rows: "
        f"{original_bottle_rows}"
    )
    print(
        f"Original cast rows: "
        f"{original_cast_rows}"
    )
    print(
        f"Final samples: {n_samples}"
    )
    print(
        f"Features: {X_normalized.shape[1]}"
    )
    print(
        f"Target: {CALCOFI_TARGET}"
    )
    print(
        f"Training samples: {train_size}"
    )
    print(
        f"Test samples: {n_samples - train_size}"
    )
    print(
        f"Date format: {selected_date_format}"
    )
    print(
        f"Start timestamp: "
        f"{metadata['start_timestamp']}"
    )
    print(
        f"End timestamp: "
        f"{metadata['end_timestamp']}"
    )
    print(
        f"X range: "
        f"[{np.min(X_normalized):.6f}, "
        f"{np.max(X_normalized):.6f}]"
    )
    print(
        f"y range: "
        f"[{np.min(y_normalized):.6f}, "
        f"{np.max(y_normalized):.6f}]"
    )
    print(
        "Contains NaN:",
        bool(
            np.isnan(X_normalized).any()
            or np.isnan(y_normalized).any()
        )
    )
    print(
        "Contains infinity:",
        bool(
            np.isinf(X_normalized).any()
            or np.isinf(y_normalized).any()
        )
    )
    print(
        "========================================"
    )
    print()

    if return_metadata:

        return (
            X_normalized,
            y_normalized,
            metadata
        )

    return (
        X_normalized,
        y_normalized
    )


###########################WSSF
# ============================================================
# WALMART STORE SALES FORECASTING
# ============================================================
from sklearn.preprocessing import OneHotEncoder
WSSF_TARGET = "Weekly_Sales"

WSSF_LAG_WEEKS = (
    1,      # Previous week
    4,      # Previous month
    52      # Previous year
)


def _create_wssf_encoder():
    """
    Create a dense OneHotEncoder compatible with different
    scikit-learn versions.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64
        )


def _add_wssf_lag(
        data,
        lag_weeks,
        column_name
):
    """
    Add an exact Weekly_Sales lag for each Store and Dept.
    """

    lag_data = data[
        [
            "Store",
            "Dept",
            "Date",
            WSSF_TARGET
        ]
    ].copy()

    lag_data["Date"] = (
        lag_data["Date"]
        + pd.Timedelta(
            weeks=lag_weeks
        )
    )

    lag_data = lag_data.rename(
        columns={
            WSSF_TARGET: column_name
        }
    )

    return data.merge(
        lag_data,
        on=[
            "Store",
            "Dept",
            "Date"
        ],
        how="left",
        validate="one_to_one"
    )


def get_WSSF(
        train_path,
        features_path,
        stores_path,
        test_path=None,
        train_percent=90,
        return_metadata=False
):
    """
    Load Walmart Store Sales Forecasting.

    Target:
        Weekly_Sales for each Store, Department, and week.

    Files used:
        train.csv
        features.csv
        stores.csv

    test.csv is validated but is not used because it does not
    contain Weekly_Sales labels.

    Data remain in chronological order.

    Scaling and categorical encoding are fitted only on the
    initial chronological training segment.
    """

    train_path = Path(
        train_path
    )

    features_path = Path(
        features_path
    )

    stores_path = Path(
        stores_path
    )

    test_path = (
        Path(test_path)
        if test_path is not None
        else None
    )

    for path in (
        train_path,
        features_path,
        stores_path
    ):
        if not path.is_file():

            raise FileNotFoundError(
                f"WSSF file was not found: {path}"
            )

    if (
            test_path is not None
            and not test_path.is_file()
    ):
        raise FileNotFoundError(
            f"WSSF test file was not found: {test_path}"
        )

    if not 0 < train_percent < 100:

        raise ValueError(
            "train_percent must be between 0 and 100."
        )

    # =========================
    # 1) READ FILES
    # =========================

    train = pd.read_csv(
        train_path,
        low_memory=False
    )

    features = pd.read_csv(
        features_path,
        low_memory=False
    )

    stores = pd.read_csv(
        stores_path,
        low_memory=False
    )

    original_train_rows = int(
        train.shape[0]
    )

    official_test_rows = None

    if test_path is not None:

        official_test_rows = int(
            pd.read_csv(
                test_path,
                usecols=["Store"]
            ).shape[0]
        )

    for dataframe in (
        train,
        features,
        stores
    ):
        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

    required_train_columns = [
        "Store",
        "Dept",
        "Date",
        "Weekly_Sales",
        "IsHoliday"
    ]

    required_feature_columns = [
        "Store",
        "Date",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment"
    ]

    required_store_columns = [
        "Store",
        "Type",
        "Size"
    ]

    for (
            file_name,
            dataframe,
            required_columns
    ) in (
        (
            train_path.name,
            train,
            required_train_columns
        ),
        (
            features_path.name,
            features,
            required_feature_columns
        ),
        (
            stores_path.name,
            stores,
            required_store_columns
        )
    ):

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{file_name} is missing columns: "
                f"{missing_columns}"
            )

    # =========================
    # 2) CLEAN TRAIN DATA
    # =========================

    train["Date"] = pd.to_datetime(
        train["Date"],
        errors="coerce"
    )

    features["Date"] = pd.to_datetime(
        features["Date"],
        errors="coerce"
    )

    for column in [
        "Store",
        "Dept",
        "Weekly_Sales"
    ]:

        train[column] = pd.to_numeric(
            train[column],
            errors="coerce"
        )

    holiday_text = (
        train["IsHoliday"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train["IsHoliday"] = holiday_text.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0
        }
    )

    train = train.dropna(
        subset=[
            "Store",
            "Dept",
            "Date",
            "Weekly_Sales",
            "IsHoliday"
        ]
    )

    train = (
        train
        .sort_values(
            by=[
                "Date",
                "Store",
                "Dept"
            ],
            kind="stable"
        )
        .drop_duplicates(
            subset=[
                "Store",
                "Dept",
                "Date"
            ],
            keep="last"
        )
    )

    # =========================
    # 3) CLEAN FEATURES DATA
    # =========================

    for column in required_feature_columns:

        if column == "Date":
            continue

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    features = (
        features[
            required_feature_columns
        ]
        .sort_values(
            by=[
                "Date",
                "Store"
            ],
            kind="stable"
        )
        .drop_duplicates(
            subset=[
                "Store",
                "Date"
            ],
            keep="last"
        )
    )

    # =========================
    # 4) CLEAN STORES DATA
    # =========================

    stores["Store"] = pd.to_numeric(
        stores["Store"],
        errors="coerce"
    )

    stores["Size"] = pd.to_numeric(
        stores["Size"],
        errors="coerce"
    )

    stores = (
        stores[
            required_store_columns
        ]
        .drop_duplicates(
            subset="Store",
            keep="last"
        )
    )

    # =========================
    # 5) MERGE FILES
    # =========================

    prepared = train.merge(
        features,
        on=[
            "Store",
            "Date"
        ],
        how="left",
        validate="many_to_one"
    )

    prepared = prepared.merge(
        stores,
        on="Store",
        how="left",
        validate="many_to_one"
    )

    prepared = (
        prepared
        .sort_values(
            by=[
                "Date",
                "Store",
                "Dept"
            ],
            kind="stable"
        )
        .reset_index(drop=True)
    )

    # =========================
    # 6) ADD SALES LAGS
    # =========================

    lag_feature_names = []

    for lag_weeks in WSSF_LAG_WEEKS:

        lag_column = (
            f"WeeklySales_Lag_{lag_weeks}W"
        )

        prepared = _add_wssf_lag(
            prepared,
            lag_weeks=lag_weeks,
            column_name=lag_column
        )

        lag_feature_names.append(
            lag_column
        )

    rows_before_lag_removal = int(
        prepared.shape[0]
    )

    prepared = prepared.dropna(
        subset=lag_feature_names
    )

    rows_removed_for_lags = int(
        rows_before_lag_removal
        - prepared.shape[0]
    )

    prepared = (
        prepared
        .sort_values(
            by=[
                "Date",
                "Store",
                "Dept"
            ],
            kind="stable"
        )
        .reset_index(drop=True)
    )

    if prepared.empty:

        raise ValueError(
            "No WSSF rows remained after creating lags."
        )

    # =========================
    # 7) TIME FEATURES
    # =========================

    week_of_year = (
        prepared["Date"]
        .dt.isocalendar()
        .week
        .astype(np.float64)
    )

    month = (
        prepared["Date"]
        .dt.month
        .astype(np.float64)
    )

    prepared["Year"] = (
        prepared["Date"]
        .dt.year
        .astype(np.float64)
    )

    prepared["Week_Sin"] = np.sin(
        2.0
        * np.pi
        * (week_of_year - 1.0)
        / 52.0
    )

    prepared["Week_Cos"] = np.cos(
        2.0
        * np.pi
        * (week_of_year - 1.0)
        / 52.0
    )

    prepared["Month_Sin"] = np.sin(
        2.0
        * np.pi
        * (month - 1.0)
        / 12.0
    )

    prepared["Month_Cos"] = np.cos(
        2.0
        * np.pi
        * (month - 1.0)
        / 12.0
    )

    prepared["WeeksSinceStart"] = (
        (
            prepared["Date"]
            - prepared["Date"].min()
        )
        .dt.days
        / 7.0
    )

    # =========================
    # 8) TRAINING SIZE
    # =========================

    n_samples = int(
        prepared.shape[0]
    )

    train_size = int(
        train_percent
        * n_samples
        / 100.0
    )

    if (
            train_size < 2
            or train_size >= n_samples
    ):
        raise ValueError(
            f"Invalid WSSF train_size: {train_size}"
        )

    # =========================
    # 9) MISSING VALUES
    # =========================

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    prepared[markdown_columns] = (
        prepared[markdown_columns]
        .fillna(0.0)
    )

    external_columns = [
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Size"
    ]

    # Use only previous observations from the same store.
    prepared[external_columns] = (
        prepared
        .groupby(
            "Store",
            sort=False
        )[external_columns]
        .ffill()
    )

    training_medians = (
        prepared[external_columns]
        .iloc[:train_size]
        .median(numeric_only=True)
    )

    prepared[external_columns] = (
        prepared[external_columns]
        .fillna(training_medians)
    )

    if (
        prepared[external_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing WSSF external values remain."
        )

    prepared["Type"] = (
        prepared["Type"]
        .fillna("Unknown")
        .astype(str)
    )

    # =========================
    # 10) NUMERIC FEATURES
    # =========================

    numeric_feature_names = [
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "Size",
        "IsHoliday",
        "Year",
        "Week_Sin",
        "Week_Cos",
        "Month_Sin",
        "Month_Cos",
        "WeeksSinceStart"
    ] + lag_feature_names

    X_numeric = (
        prepared[
            numeric_feature_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_scaler = MinMaxScaler(
        feature_range=(0.0, 1.0),
        clip=True
    )

    X_scaler.fit(
        X_numeric[:train_size]
    )

    X_numeric = X_scaler.transform(
        X_numeric
    )

    # =========================
    # 11) CATEGORICAL FEATURES
    # =========================

    categorical_columns = [
        "Store",
        "Dept",
        "Type"
    ]

    categorical_data = (
        prepared[
            categorical_columns
        ]
        .astype(str)
    )

    encoder = _create_wssf_encoder()

    encoder.fit(
        categorical_data.iloc[:train_size]
    )

    X_categorical = encoder.transform(
        categorical_data
    )

    categorical_feature_names = (
        encoder
        .get_feature_names_out(
            categorical_columns
        )
        .tolist()
    )

    X = np.hstack(
        [
            X_numeric,
            X_categorical
        ]
    ).astype(
        np.float64
    )

    feature_names = (
        numeric_feature_names
        + categorical_feature_names
    )

    # =========================
    # 12) TARGET
    # =========================

    y = prepared[
        WSSF_TARGET
    ].to_numpy(
        dtype=np.float64
    )

    y_scaler = MinMaxScaler(
        feature_range=(0.0, 1.0),
        clip=True
    )

    y_scaler.fit(
        y[:train_size].reshape(-1, 1)
    )

    y = (
        y_scaler
        .transform(
            y.reshape(-1, 1)
        )
        .flatten()
    )

    if not np.all(
            np.isfinite(X)
    ):
        raise ValueError(
            "WSSF X contains NaN or infinity."
        )

    if not np.all(
            np.isfinite(y)
    ):
        raise ValueError(
            "WSSF y contains NaN or infinity."
        )

    # =========================
    # 13) METADATA
    # =========================

    metadata = {
        "dataset_name": (
            "Walmart Store Sales Forecasting"
        ),

        "dataset_short_name": "WSSF",

        "train_file": train_path.name,

        "features_file": features_path.name,

        "stores_file": stores_path.name,

        "test_file": (
            test_path.name
            if test_path is not None
            else None
        ),

        "official_test_rows_not_used": (
            official_test_rows
        ),

        "target_name": WSSF_TARGET,

        "prediction_unit": (
            "Store-Department-Week"
        ),

        "feature_names": list(
            feature_names
        ),

        "lag_weeks": list(
            WSSF_LAG_WEEKS
        ),

        "n_samples": int(
            n_samples
        ),

        "n_features": int(
            X.shape[1]
        ),

        "training_percent": float(
            train_percent
        ),

        "training_samples": int(
            train_size
        ),

        "test_samples": int(
            n_samples - train_size
        ),

        "original_train_rows": int(
            original_train_rows
        ),

        "rows_removed_for_lags": int(
            rows_removed_for_lags
        ),

        "start_date": (
            prepared["Date"]
            .iloc[0]
            .isoformat()
        ),

        "end_date": (
            prepared["Date"]
            .iloc[-1]
            .isoformat()
        ),

        "normalization": "MinMaxScaler",

        "normalization_range": [
            0.0,
            1.0
        ],

        "scaler_fitted_on_training_only": True,

        "categorical_encoder_fitted_on_training_only": True,

        "chronological_order": True,

        "official_test_file_used": False,

        "known_drift_location": None,

        "known_drift_type": None,

        "natural_nonstationarity": True
    }

    # =========================
    # 14) PRINT SUMMARY
    # =========================

    print()
    print(
        "WSSF loaded and normalized"
    )
    print(
        "========================================"
    )
    print(
        f"Original train rows: "
        f"{original_train_rows}"
    )
    print(
        f"Rows removed for lags: "
        f"{rows_removed_for_lags}"
    )
    print(
        f"Final samples: {n_samples}"
    )
    print(
        f"Features: {X.shape[1]}"
    )
    print(
        f"Training samples: {train_size}"
    )
    print(
        f"Test samples: {n_samples - train_size}"
    )
    print(
        f"X range: "
        f"[{np.min(X):.6f}, {np.max(X):.6f}]"
    )
    print(
        f"y range: "
        f"[{np.min(y):.6f}, {np.max(y):.6f}]"
    )
    print(
        "Contains NaN:",
        bool(
            np.isnan(X).any()
            or np.isnan(y).any()
        )
    )
    print(
        "Contains infinity:",
        bool(
            np.isinf(X).any()
            or np.isinf(y).any()
        )
    )
    print(
        "========================================"
    )
    print()

    if return_metadata:

        return (
            X,
            y,
            metadata
        )

    return (
        X,
        y
    )