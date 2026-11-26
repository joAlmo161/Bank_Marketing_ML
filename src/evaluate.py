"""
evaluate.py
===========
Metrics and plots used to judge the models.

Why not accuracy alone?
-----------------------
Only about 11.7 % of customers say 'yes'. A useless model that ALWAYS answers
'no' is already about 88 % accurate. We therefore also report:

* Precision  = of the customers we predict as 'yes', how many really subscribe?
               (high precision -> few wasted calls)
* Recall     = of the customers who really subscribe, how many did we find?
               (high recall -> few missed opportunities)
* F1-score   = harmonic mean of precision and recall (balances both).
* ROC-AUC    = probability that a random subscriber gets a higher score than a
               random non-subscriber. 0.5 = random guessing, 1.0 = perfect.
               It does not depend on a decision threshold -> our MAIN metric
               for comparing and tuning models.
* Average precision (PR-AUC) = area under the precision-recall curve; more
               sensitive to performance on the rare positive class.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR
from src.plot_style import BLUE


# ---------------------------------------------------------------------------
# 1. Scores
# ---------------------------------------------------------------------------
def get_scores(model, X):
    """Return the predicted probability of the positive class ('yes').

    predict_proba returns one column per class: [:, 0] = P(no), [:, 1] = P(yes).
    All six models in this project provide predict_proba. These probabilities
    are used for ROC-AUC, PR curves and the decision threshold.
    """
    return model.predict_proba(X)[:, 1]


def classification_metrics(y_true, y_pred, y_score) -> dict:
    """Compute all metrics for one set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


# ---------------------------------------------------------------------------
# 2. Decision threshold
# ---------------------------------------------------------------------------
def find_best_threshold(y_true, y_score) -> tuple:
    """Find the probability cut-off that maximises F1.

    By default a classifier says 'yes' when probability >= 0.5. With rare
    positives, a lower cut-off often gives a better balance of precision and
    recall. IMPORTANT: we call this function on OUT-OF-FOLD predictions of the
    TRAINING set (cross_val_predict), never on the test set, so choosing the
    threshold is part of honest model selection.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # The last precision/recall pair has no threshold, so we drop it.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
def _save(fig, filename):
    """Save a figure into figures/ at print quality."""
    if filename:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")


def plot_confusion_matrix(y_true, y_pred, title, filename=None):
    """Show counts AND row percentages.

    Rows = actual class, columns = predicted class.
    Top-right = false positives (called but did not subscribe).
    Bottom-left = false negatives (missed subscribers).
    """
    cm = confusion_matrix(y_true, y_pred)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1f}% of row)" for j in range(2)]
                       for i in range(2)])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted: no", "Predicted: yes"],
                yticklabels=["Actual: no", "Actual: yes"])
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    _save(fig, filename)
    return fig


def plot_roc_curve(y_true, y_score, label, title, filename=None):
    """ROC curve: True-Positive Rate vs False-Positive Rate for all thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, color=BLUE, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (non-subscribers wrongly flagged)")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, filename)
    return fig


def plot_precision_recall_curve(y_true, y_score, label, title,
                                threshold=None, filename=None):
    """Precision-Recall curve. The horizontal line is the 'yes' rate, which is
    the precision a random classifier would get."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, lw=2, color=BLUE, label=f"{label} (AP = {ap:.3f})")
    ax.axhline(np.mean(y_true), ls="--", color="grey",
               label=f"Random guess (AP = {np.mean(y_true):.3f})")
    if threshold is not None:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(recall[idx], precision[idx], s=60, color="red", zorder=3,
                   label=f"Chosen threshold = {threshold:.3f}")
    ax.set_xlabel("Recall (share of subscribers found)")
    ax.set_ylabel("Precision (share of predicted 'yes' that are correct)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, filename)
    return fig


def summarize_cv(cv_result: dict, model_name: str) -> dict:
    """Turn the dictionary from sklearn.cross_validate into mean / std values
    for every metric, for both the validation folds and the training folds."""
    row = {"model": model_name}
    for key, values in cv_result.items():
        if key.startswith("test_"):
            metric = key[len("test_"):]
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)
        elif key.startswith("train_"):
            metric = key[len("train_"):]
            row[f"train_{metric}_mean"] = np.mean(values)
        elif key == "fit_time":
            row["fit_time_s"] = np.mean(values)
    return row


def format_cv_table(df: pd.DataFrame, metrics=("roc_auc", "average_precision", "f1",
                                                "precision", "recall", "accuracy")):
    """Readable 'mean ± std' table for notebooks and the report."""
    out = pd.DataFrame({"Model": df["model"]})
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
    return out
