"""
data_loader.py
==============
Functions to DOWNLOAD the UCI Bank Marketing dataset (ID 222), LOAD it into a
pandas DataFrame and produce a basic DATA-QUALITY report.

Run directly to download the data:
    python -m src.data_loader
"""

import io
import zipfile
import urllib.request

import pandas as pd

from src.config import RAW_DATA_FILE, DATA_RAW_DIR, UCI_DATASET_URL, TARGET


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_dataset(force: bool = False) -> None:
    """Download bank-full.csv from the official UCI repository.

    The UCI archive is a ZIP that contains another ZIP (bank.zip). We open both
    in memory and only extract the two files we need:
      * bank-full.csv   -> the full dataset (45,211 rows)
      * bank-names.txt  -> the official description of every column
    """
    if RAW_DATA_FILE.exists() and not force:
        print(f"Dataset already present: {RAW_DATA_FILE}")
        return

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {UCI_DATASET_URL} ...")
    with urllib.request.urlopen(UCI_DATASET_URL, timeout=60) as response:
        outer_zip = zipfile.ZipFile(io.BytesIO(response.read()))

    inner_zip = zipfile.ZipFile(io.BytesIO(outer_zip.read("bank.zip")))
    for name in ["bank-full.csv", "bank-names.txt"]:
        (DATA_RAW_DIR / name).write_bytes(inner_zip.read(name))
    print(f"Saved to {DATA_RAW_DIR}")


# ---------------------------------------------------------------------------
# 2. Load
# ---------------------------------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    """Load bank-full.csv into a DataFrame.

    NOTE: the UCI file uses ';' (semicolon) as separator, not ','.
    If the file is missing we try to download it automatically.
    """
    if not RAW_DATA_FILE.exists():
        download_dataset()
    df = pd.read_csv(RAW_DATA_FILE, sep=";")
    return df


def encode_target(y: pd.Series) -> pd.Series:
    """Convert the target from text ('yes'/'no') to numbers (1/0).

    scikit-learn metrics such as precision/recall treat label 1 as the
    'positive' class, which here is the interesting class: a subscription.
    """
    return (y == "yes").astype(int)


# ---------------------------------------------------------------------------
# 3. Data-quality report
# ---------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column with type, missing values, 'unknown' counts
    and number of distinct values.

    Important: this dataset has NO empty (NaN) cells, but several categorical
    columns use the word 'unknown' as a placeholder for missing information.
    We therefore count both.
    """
    rows = []
    for col in df.columns:
        is_text = not pd.api.types.is_numeric_dtype(df[col])
        n_unknown = int((df[col] == "unknown").sum()) if is_text else 0
        rows.append({
            "column": col,
            "dtype": "categorical" if is_text else str(df[col].dtype),
            "n_missing_NaN": int(df[col].isna().sum()),
            "n_unknown": n_unknown,
            "pct_unknown": round(100 * n_unknown / len(df), 2),
            "n_distinct": int(df[col].nunique()),
        })
    return pd.DataFrame(rows)


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Small dictionary with the headline facts about the dataset."""
    counts = df[TARGET].value_counts()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_input_features": int(df.shape[1] - 1),
        "n_duplicates": int(df.duplicated().sum()),
        "n_missing_NaN": int(df.isna().sum().sum()),
        "n_yes": int(counts.get("yes", 0)),
        "n_no": int(counts.get("no", 0)),
        "pct_yes": round(100 * counts.get("yes", 0) / len(df), 2),
    }


if __name__ == "__main__":
    download_dataset()
    data = load_raw_data()
    print(summarize_dataset(data))
