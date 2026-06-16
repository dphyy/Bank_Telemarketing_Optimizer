# Importing necessary libraries for data preprocessing
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "Data" / "bank_telemarketing.csv"
RANDOM_STATE = 42
TARGET_COLUMN = "y"
# Drop 'pdays' as it is accounted for in 'contacted'.
# Drop 'duration' as including it introduces data leakage.
# Drop 'emp.var.rate' it has a strong correlation with 'euribor3m' and 'nr.employed',
# and is redundant when 'euribor3m' and 'nr.employed' which are more interpretable are included.
# Drop 'default' column as there is only 3 positive cases, which is not enough for modeling
DROP_COLUMNS = ["duration", "pdays", "emp.var.rate", "default"]

RAW_INPUT_DEFAULTS = {
    "age": 0,
    "job": "unknown",
    "marital": "unknown",
    "education": "unknown",
    "default": 0,
    "housing": "unknown",
    "loan": "unknown",
    "contact": "unknown",
    "month": "unknown",
    "day_of_week": "unknown",
    "duration": 0,
    "campaign": 0,
    "pdays": 999,
    "previous": 0,
    "emp.var.rate": 0,
    "cons.price.idx": 0,
    "cons.conf.idx": 0,
    "euribor3m": 0,
    "nr.employed": 0,
}


# Resolve relative paths from this module so calls are stable from any cwd.
def load_data(file_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(file_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return pd.read_csv(path)


# Handles different input formats for the target column 'y', normalizing to binary integers.
def _normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    mapped = lowered.map(mapping)
    if mapped.isna().any():
        raise ValueError("Target column 'y' contains unsupported labels")
    return mapped.astype(int)


# Helper function to compute default values for missing columns based on the reference dataset.
@lru_cache(maxsize=1)
def _reference_raw_defaults() -> dict[str, object]:
    reference = load_data()
    defaults: dict[str, object] = {}
    for col, fallback in RAW_INPUT_DEFAULTS.items():
        if col not in reference.columns:
            defaults[col] = fallback
            continue
        series = reference[col]
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            defaults[col] = float(clean_series.median()) if not clean_series.empty else fallback
        else:
            mode = series.dropna().astype(str).mode()
            defaults[col] = mode.iloc[0] if not mode.empty else fallback
    return defaults


# Ensures the raw input DataFrame has all expected columns, filling in missing ones with defaults.
def _ensure_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    completed = df.copy()
    defaults = _reference_raw_defaults()
    for col, fallback in defaults.items():
        if col not in completed.columns:
            completed[col] = fallback
        elif completed[col].isna().any():
            completed[col] = completed[col].fillna(fallback)
    return completed


# Cleans raw data by imputing missing values, encoding education, creating a 'contacted' feature, and dropping unnecessary columns.
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid mutating caller-owned DataFrames.
    cleaned = _ensure_raw_schema(df)
    edu_order = {
        "illiterate": 1,
        "basic.4y": 1,
        "basic.6y": 2,
        "basic.9y": 3,
        "unknown": 4,
        "high.school": 5,
        "professional.course": 6,
        "university.degree": 7,
    }

    # Impute categorical fields with their mode
    for col in ["job", "marital", "housing", "loan"]:
        if col in cleaned.columns:
            mode = cleaned[col].dropna().mode()
            fallback = mode.iloc[0] if not mode.empty else "unknown"
            cleaned[col] = cleaned[col].fillna(fallback).astype(str)

    # Impute numeric fields with the median so downstream preprocessing and SHAP
    # can still run when an input row contains blanks.
    for col in ["age", "campaign", "previous", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "emp.var.rate", "pdays"]:
        if col in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[col]):
            median_value = cleaned[col].dropna().median()
            if pd.notna(median_value):
                cleaned[col] = cleaned[col].fillna(median_value)

    # Ordinal encoding for education reflects domain ordering and handles missing values as 'unknown'.
    if "education" in cleaned.columns:
        cleaned["education"] = cleaned["education"].fillna("unknown").astype(str).map(edu_order)
        cleaned["education"] = cleaned["education"].fillna(edu_order["unknown"])

    # Add a new column 'contacted' to indicate whether the client has been contacted before.
    if "pdays" in cleaned.columns:
        cleaned["contacted"] = np.where(cleaned["pdays"].eq(999), 0, 1)
    elif "contacted" not in cleaned.columns:
        cleaned["contacted"] = 0

    # Drop columns before scaling to avoid unnecessary processing.
    cleaned = cleaned.drop(columns=DROP_COLUMNS, errors="ignore")
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = _normalize_target(cleaned[TARGET_COLUMN])
    return cleaned


# Builds a ColumnTransformer that applies appropriate transformations to different feature types, including scaling, encoding, and passthroughs.
def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# Constructs a ColumnTransformer that applies RobustScaler to skewed numeric features, StandardScaler to more symmetric numeric features, OneHotEncoder to categorical features, and passthrough for the 'contacted' feature.
def build_preprocessor() -> ColumnTransformer:
    # 'age', 'campaign' and 'previous' chosen as robust features due to high skewness and extreme outlier.
    # 'education', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed' chosen as standard features
    # as they are either bounded or relatively symmetric and do not have extreme outlier.
    robust_features = ["age", "campaign", "previous"]
    standard_features = ["education", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    categorical_features = ["job", "marital", "housing", "loan", "contact", "month", "day_of_week"]
    passthrough_features = ["contacted"]

    return ColumnTransformer(
        transformers=[
            ("robust", RobustScaler(), robust_features),
            ("standard", StandardScaler(), standard_features),
            ("categorical", _one_hot_encoder(), categorical_features),
            ("passthrough", "passthrough", passthrough_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# Ingests raw bank telemarketing data, splits the raw source first, then applies preprocessing.
def preprocess_data(file_path="../Data/bank_telemarketing.csv", df=None):
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame")
        source = df.copy()
    else:
        source = load_data(file_path)

    if TARGET_COLUMN not in source.columns:
        raise ValueError(f"Source data must include target column '{TARGET_COLUMN}'")

    X_source = source.drop(columns=[TARGET_COLUMN]).copy()
    y_source = _normalize_target(source[TARGET_COLUMN])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_source,
        y_source,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_source,
    )

    raw_train_frame = X_train_raw.copy()
    raw_test_frame = X_test_raw.copy()

    X_train_raw = clean_raw_data(X_train_raw)
    X_test_raw = clean_raw_data(X_test_raw)

    # Instantiate ColumnTransformer
    preprocessor = build_preprocessor()
    
    # Transform the data natively (Handles encoding, scaling, and passthroughs simultaneously)
    # We use fit_transform on training data, and only transform on test data.
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    # Convert arrays back to DataFrames using the preserved feature names
    feature_names = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)

    # Apply SMOTE to training data only.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_train_res, y_train_res, X_test, y_test, raw_train_frame, raw_test_frame