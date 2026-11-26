"""
predict.py
==========
Load the saved model and make predictions for NEW customers.

The saved file (models/final_model.joblib) is a complete Pipeline:
    raw customer data -> feature engineering -> imputing -> scaling /
    one-hot encoding -> trained classifier
so we can pass the raw values exactly as a bank employee would enter them.

Example (from the project root):
    python -m src.predict
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES, FINAL_MODEL_FILE, MODEL_METADATA_FILE, NUMERIC_FEATURES,
)


def load_model():
    """Load the trained pipeline and its metadata (threshold, valid values)."""
    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{FINAL_MODEL_FILE} not found. Train the model first by running "
            "notebook 04 or `python -m src.train_models`.")
    model = joblib.load(FINAL_MODEL_FILE)
    metadata = json.loads(MODEL_METADATA_FILE.read_text())
    return model, metadata


def validate_customer(customer: dict, metadata: dict) -> tuple:
    """Check one customer's inputs before predicting.

    Returns (clean_row, errors, warnings):
      * errors   -> problems that make a prediction impossible
                    (e.g. text in a numeric field, negative age).
      * warnings -> unusual but allowed values (outside the training range,
                    or a missing value that will be imputed).
    Missing values are allowed: the pipeline's SimpleImputer fills them.
    """
    errors, warnings, clean = [], [], {}

    for col in NUMERIC_FEATURES:
        value = customer.get(col)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            clean[col] = np.nan
            warnings.append(f"'{col}' is missing -> the training median will be used.")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a number (got {value!r}).")
            continue
        low, high, _ = metadata["numeric_ranges"][col]
        if col in ("age", "campaign") and value < 1:
            errors.append(f"'{col}' must be at least 1.")
        elif col == "day" and not 1 <= value <= 31:
            errors.append("'day' must be between 1 and 31.")
        elif col == "previous" and value < 0:
            errors.append("'previous' cannot be negative.")
        elif col == "pdays" and value < -1:
            errors.append("'pdays' must be -1 (never contacted) or >= 0.")
        elif not low <= value <= high:
            warnings.append(f"'{col}' = {value:g} is outside the training range "
                            f"[{low}, {high}]; the prediction may be less reliable.")
        clean[col] = value

    for col in CATEGORICAL_FEATURES:
        value = customer.get(col)
        if value is None or str(value).strip() == "":
            clean[col] = np.nan
            warnings.append(f"'{col}' is missing -> treated as 'unknown'.")
        elif value not in metadata["categories"][col]:
            errors.append(f"'{col}' has an invalid value {value!r}. "
                          f"Allowed: {metadata['categories'][col]}")
        else:
            clean[col] = value

    # Consistency rule from the data definition: pdays = -1 <=> previous = 0.
    if not errors and clean.get("pdays") == -1 and (clean.get("previous") or 0) > 0:
        warnings.append("pdays = -1 means 'never contacted before', but 'previous' > 0.")

    return clean, errors, warnings


def predict_customer(customer: dict, model=None, metadata=None) -> dict:
    """Predict YES / NO for one customer using the real trained model."""
    if model is None or metadata is None:
        model, metadata = load_model()

    clean, errors, warnings = validate_customer(customer, metadata)
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    # Columns must be in the same order and with the same names as in training.
    row = pd.DataFrame([clean])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    probability = float(model.predict_proba(row)[0, 1])
    threshold = metadata["decision_threshold"]
    return {
        "ok": True,
        "prediction": "YES" if probability >= threshold else "NO",
        "probability_yes": probability,
        "threshold": threshold,
        "warnings": warnings,
    }


if __name__ == "__main__":
    example = {
        "age": 35, "job": "management", "marital": "single", "education": "tertiary",
        "default": "no", "balance": 2500, "housing": "no", "loan": "no",
        "contact": "cellular", "day": 15, "month": "mar", "campaign": 1,
        "pdays": 90, "previous": 2, "poutcome": "success",
    }
    print(predict_customer(example))
