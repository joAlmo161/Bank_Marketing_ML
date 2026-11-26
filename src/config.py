"""
config.py
=========
Central place for every constant used in the project (paths, random seed,
feature lists). Keeping them in ONE file means every notebook, script and the
Streamlit app use exactly the same settings, which is essential for
reproducibility.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains README.md. We compute it from the
# location of this file so the code works no matter where it is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_FILE = DATA_RAW_DIR / "bank-full.csv"
TRAIN_FILE = DATA_PROCESSED_DIR / "train.csv"
TEST_FILE = DATA_PROCESSED_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_FILE = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Official UCI download link (dataset ID 222). The ZIP contains bank.zip,
# which in turn contains bank-full.csv (45,211 rows).
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------
# The same seed is passed to EVERY random operation (split, CV folds, models,
# search) so anybody re-running the project gets identical numbers.
RANDOM_STATE = 42
TEST_SIZE = 0.20      # 20 % of the rows are kept aside as the final test set
CV_FOLDS = 5          # number of folds for Stratified K-Fold cross-validation

# ---------------------------------------------------------------------------
# 3. Columns
# ---------------------------------------------------------------------------
TARGET = "y"                     # "yes" / "no" -> converted to 1 / 0
LEAKAGE_FEATURE = "duration"     # only known AFTER the call -> excluded

NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]

# Feature we create ourselves (see preprocessing.add_engineered_features).
ENGINEERED_NUMERIC_FEATURES = ["previously_contacted"]

# All raw input columns expected by the final model (duration NOT included).
MODEL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Main metric used for model selection and tuning (explained in the report).
PRIMARY_METRIC = "roc_auc"
