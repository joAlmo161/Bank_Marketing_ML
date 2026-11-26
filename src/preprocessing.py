"""
preprocessing.py
================
Everything that turns raw customer records into numbers a model can use.

KEY IDEA - avoiding data leakage
--------------------------------
1. We split the data into train / test FIRST (split_data).
2. All transformations that LEARN something from data (the median used by the
   imputer, the mean/std used by the scaler, the list of categories used by the
   one-hot encoder) are placed inside a scikit-learn Pipeline.
3. A Pipeline is fitted only on training data (or only on the training folds
   during cross-validation), so no information from the test set can leak
   into the model.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES, ENGINEERED_NUMERIC_FEATURES, LEAKAGE_FEATURE,
    NUMERIC_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE,
)
from src.data_loader import encode_target


# ---------------------------------------------------------------------------
# 1. Train / test split (done BEFORE any fitting)
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, include_duration: bool = False):
    """Split the dataset into training (80 %) and test (20 %) sets.

    * stratify=y keeps the same share of 'yes' (about 11.7 %) in both sets.
      Without it, an unlucky split could put too few positives in the test set.
    * random_state fixes the shuffle so the split is identical every run.
    * duration is dropped by default because it is only known after the
      call has ended (target leakage - see the report, Section 6).
    """
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if include_duration:
        features = features + [LEAKAGE_FEATURE]

    X = df[features].copy()
    y = encode_target(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add simple, leakage-free features computed row by row.

    previously_contacted: pdays = -1 is a special code meaning 'never contacted
    in a previous campaign' (about 82 % of rows). For a linear model the value
    -1 looks like 'one day ago', which is misleading. An explicit 0/1 flag lets
    every model see the difference between 'never contacted' and 'contacted
    N days ago'.

    This function uses only the row itself (no statistics from other rows), so
    it cannot leak information between train and test.
    """
    X = X.copy()
    X["previously_contacted"] = (X["pdays"] != -1).astype(int)
    return X


def _engineered_feature_names(transformer, input_features):
    """Tell scikit-learn the output column names of add_engineered_features."""
    return list(input_features) + ENGINEERED_NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline (ColumnTransformer)
# ---------------------------------------------------------------------------
def build_preprocessor(include_duration: bool = False) -> Pipeline:
    """Build the full preprocessing pipeline.

    Numeric columns:
        SimpleImputer(median) -> fills a missing number with the TRAINING median
                                 (the raw data has no NaN, but the Streamlit app
                                 may receive an empty field).
        StandardScaler        -> rescales to mean 0 / std 1. Required for
                                 distance-based models (KNN)
                                 and helps Logistic Regression converge. Tree
                                 models ignore scaling, so it does no harm.
    Categorical columns:
        SimpleImputer(constant='unknown') -> a missing category becomes
                                 'unknown', the same placeholder the bank used.
        OneHotEncoder         -> one 0/1 column per category. We use one-hot
                                 (not label numbers) because categories such as
                                 job have no natural order.
                                 handle_unknown='ignore' means a category never
                                 seen in training produces all zeros instead of
                                 an error.
    """
    numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    if include_duration:
        numeric = numeric + [LEAKAGE_FEATURE]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    # Step 1 adds the engineered flag, step 2 encodes/scales all columns.
    return Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(
            add_engineered_features, feature_names_out=_engineered_feature_names)),
        ("columns", column_transformer),
    ])


def build_model_pipeline(model, include_duration: bool = False) -> Pipeline:
    """Attach a classifier to the preprocessing steps.

    Calling .fit() on this object fits the imputers, scaler, encoder AND the
    model on the same (training) data, in the right order.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(include_duration)),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# 4. Outlier analysis helper (used for reporting only, nothing is removed)
# ---------------------------------------------------------------------------
def iqr_outlier_table(X: pd.DataFrame, columns) -> pd.DataFrame:
    """Count values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for each column.

    We report these counts but DO NOT delete the rows: large balances or many
    campaign calls are real customers, not data-entry errors.
    """
    rows = []
    for col in columns:
        q1, q3 = X[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((X[col] < low) | (X[col] > high)).sum())
        rows.append({
            "feature": col, "Q1": q1, "Q3": q3, "lower_fence": low,
            "upper_fence": high, "min": X[col].min(), "max": X[col].max(),
            "n_outliers": n_out, "pct_outliers": round(100 * n_out / len(X), 2),
            "skewness": round(float(X[col].skew()), 2),
        })
    return pd.DataFrame(rows)
