"""
train_models.py
===============
Defines the models, compares them with Stratified K-Fold cross-validation,
tunes the best ones with GridSearchCV and saves the final model.

The notebooks 03 and 04 call these functions step by step and explain the
results. Running this file directly reproduces the whole training process:
    python -m src.train_models

The TEST SET IS NEVER USED HERE. Everything in this file only sees the
training data. The single final test evaluation happens in notebook 05.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CATEGORICAL_FEATURES, CV_FOLDS, FINAL_MODEL_FILE, MODEL_METADATA_FILE,
    MODELS_DIR, NUMERIC_FEATURES, PRIMARY_METRIC, PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR,
)
from src.data_loader import load_raw_data
from src.evaluate import find_best_threshold, summarize_cv
from src.preprocessing import build_model_pipeline, split_data

# ---------------------------------------------------------------------------
# 1. Validation scheme and metrics (identical for every model = fair comparison)
# ---------------------------------------------------------------------------
# Stratified K-Fold: the training set is cut into 5 folds that each keep the
# 11.7 % 'yes' rate. Each model is trained 5 times on 4 folds and validated on
# the remaining fold. shuffle + random_state -> the SAME folds for all models.
CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# zero_division=0 avoids warnings when a model never predicts 'yes'
# (this happens with the majority-class baseline).
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "accuracy": "accuracy",
}


# ---------------------------------------------------------------------------
# 2. Model definitions
# ---------------------------------------------------------------------------
def get_baseline_and_core_models() -> dict:
    """Baseline + three core algorithms (Units 4-5).

    All models use their DEFAULT hyperparameters at this stage, so the
    comparison is not biased by tuning one model more than the others.
    class_weight='balanced' gives the rare 'yes' class a larger weight in the
    loss (about 7.5x here), so models do not simply ignore it.
    """
    return {
        # Always predicts the most frequent class ('no'). Any useful model
        # must beat it; it shows why accuracy alone is misleading.
        "Dummy (majority class)": DummyClassifier(strategy="most_frequent"),

        # Linear model of the log-odds of subscribing. max_iter is raised so
        # the solver has enough iterations to converge on 50+ features.
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),

        # A single tree of if/else rules. With default settings it grows until
        # leaves are pure, which usually over-fits (see notebook 03).
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),

        # Classifies a customer by majority vote of the k=5 most similar
        # customers (Euclidean distance on the scaled features).
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


def get_ensemble_models() -> dict:
    """Two ensemble methods (Unit 6)."""
    return {
        # Bagging of many decorrelated trees -> mainly REDUCES VARIANCE.
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),

        # Boosting: each new shallow tree corrects the errors of the previous
        # ones -> mainly REDUCES BIAS. (GradientBoostingClassifier has no
        # class_weight option; because we select models by ROC-AUC, which does
        # not depend on the threshold, and later tune the threshold, this is
        # not a problem.)
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


# Hyperparameter grids. The 'model__' prefix tells GridSearchCV that the
# parameter belongs to the step named 'model' inside the Pipeline.
PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators": [200, 400],       # more trees = more stable, slower
        "model__max_depth": [10, 20, None],      # limits tree complexity
        "model__min_samples_leaf": [1, 5, 10],   # bigger leaves = smoother model
        "model__max_features": ["sqrt", 0.3],    # features tried at each split
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 300],  # number of boosting stages
        "model__learning_rate": [0.05, 0.1],     # shrinkage of each tree's update
        "model__max_depth": [3, 4, 5],           # depth of each weak learner
        "model__subsample": [0.8, 1.0],          # <1.0 = stochastic boosting
    },
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],          # inverse regularisation strength
    },
    "Decision Tree": {
        "model__max_depth": [4, 6, 8, 10, 12],
        "model__min_samples_leaf": [1, 10, 50],
    },
    "K-Nearest Neighbors": {
        "model__n_neighbors": [15, 51, 101],
        "model__weights": ["uniform", "distance"],
    },
}


# ---------------------------------------------------------------------------
# 3. Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_models(models: dict, X_train, y_train, n_jobs=-1) -> pd.DataFrame:
    """Evaluate each model with the SAME 5 stratified folds.

    return_train_score=True also records scores on the training folds, which
    lets us compare training vs validation performance (over-fitting check).
    """
    rows = []
    for name, model in models.items():
        start = time.time()
        pipe = build_model_pipeline(model)
        result = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING,
                                return_train_score=True, n_jobs=n_jobs)
        rows.append(summarize_cv(result, name))
        print(f"{name:<25} ROC-AUC = {np.mean(result['test_roc_auc']):.4f} "
              f"+/- {np.std(result['test_roc_auc']):.4f}   ({time.time() - start:.0f} s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_model(name: str, model, X_train, y_train, n_jobs=-1) -> GridSearchCV:
    """Grid search with the SAME 5-fold CV, scored on ROC-AUC, training data only.

    refit=True: after the search, the best combination is re-trained on the
    whole training set, so search.best_estimator_ is ready to use.
    """
    search = GridSearchCV(
        estimator=build_model_pipeline(model),
        param_grid=PARAM_GRIDS[name],
        scoring=SCORING,
        refit=PRIMARY_METRIC,
        cv=CV,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    start = time.time()
    search.fit(X_train, y_train)
    print(f"{name}: {len(search.cv_results_['params'])} combinations in "
          f"{time.time() - start:.0f} s -> best CV ROC-AUC = {search.best_score_:.4f}")
    return search


def tuning_results_table(search: GridSearchCV, top: int = 10) -> pd.DataFrame:
    """Most useful columns of GridSearchCV.cv_results_, best first."""
    res = pd.DataFrame(search.cv_results_)
    cols = (["params", "mean_test_roc_auc", "std_test_roc_auc", "mean_train_roc_auc",
             "mean_test_average_precision", "mean_test_f1"])
    return res.sort_values("rank_test_roc_auc")[cols].head(top).reset_index(drop=True)


def best_params_clean(search: GridSearchCV) -> dict:
    """Remove the 'model__' prefix for display."""
    return {k.replace("model__", ""): v for k, v in search.best_params_.items()}


# ---------------------------------------------------------------------------
# 5. Threshold selection and saving
# ---------------------------------------------------------------------------
def out_of_fold_scores(estimator, X_train, y_train, n_jobs=-1):
    """Scores for every training row produced by a model that did NOT see that
    row during fitting (cross_val_predict). Used to choose the threshold."""
    scores = cross_val_predict(estimator, X_train, y_train, cv=CV,
                               method="predict_proba", n_jobs=n_jobs)
    return scores[:, 1]          # column 1 = probability of class 1 ('yes')


def save_final_model(estimator, model_name, best_params, threshold, cv_summary,
                     X_train) -> None:
    """Save the fitted Pipeline (preprocessing + model) with joblib, plus a
    JSON file of metadata the Streamlit app needs (threshold, valid inputs)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, FINAL_MODEL_FILE, compress=3)

    metadata = {
        "model_name": model_name,
        "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                        for k, v in best_params.items()},
        "decision_threshold": threshold,
        "cv_summary": cv_summary,
        "features_used": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "excluded_features": ["duration"],
        # Valid choices and ranges for every input, taken from TRAINING data,
        # so the app only offers values the model has seen.
        "categories": {c: sorted(X_train[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        "numeric_ranges": {c: [int(X_train[c].min()), int(X_train[c].max()),
                               int(X_train[c].median())] for c in NUMERIC_FEATURES},
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    # Print a project-relative path (no local user folders in public notebook outputs)
    print(f"Saved model to {FINAL_MODEL_FILE.relative_to(PROJECT_ROOT).as_posix()}")


# ---------------------------------------------------------------------------
# 6. Full training run (same steps as notebooks 03 + 04)
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)   # X_test is NOT used here

    print("\n=== Cross-validation of all models (training data only) ===")
    all_models = {**get_baseline_and_core_models(), **get_ensemble_models()}
    cv_table = cross_validate_models(all_models, X_train, y_train)
    cv_table.to_csv(RESULTS_DIR / "cv_results_all_models.csv", index=False)

    # Tune the two best non-baseline models according to CV ROC-AUC.
    ranked = cv_table[~cv_table["model"].str.startswith("Dummy")] \
        .sort_values("roc_auc_mean", ascending=False)["model"].tolist()
    to_tune = ranked[:2]
    print(f"\n=== Tuning the two best models: {to_tune} ===")
    searches = {name: tune_model(name, all_models[name], X_train, y_train)
                for name in to_tune}

    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best_search = searches[best_name]
    best_idx = best_search.best_index_
    cv_summary = {m: float(best_search.cv_results_[f"mean_test_{m}"][best_idx])
                  for m in SCORING}

    oof = out_of_fold_scores(best_search.best_estimator_, X_train, y_train)
    threshold, f1_at_threshold = find_best_threshold(y_train, oof)
    print(f"\nFinal model: {best_name}, threshold = {threshold:.3f} "
          f"(out-of-fold F1 = {f1_at_threshold:.3f})")

    save_final_model(best_search.best_estimator_, best_name,
                     best_params_clean(best_search), threshold, cv_summary, X_train)


if __name__ == "__main__":
    main()
