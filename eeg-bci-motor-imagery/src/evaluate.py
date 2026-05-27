"""
Evaluation metrics for BCI motor imagery classification.
Computes accuracy, Cohen's kappa, confusion matrix, and per-subject stats.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_predictions(y_true, y_pred, class_names=None):
    """Full evaluation report for a set of predictions."""
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    return {
        "accuracy": acc,
        "kappa": kappa,
        "confusion_matrix": cm,
        "report": report,
    }


def cross_validate_sklearn(pipeline, X, y, n_splits=5):
    """
    Stratified k-fold cross-validation for sklearn pipelines (CSP+LDA etc).
    Returns per-fold accuracy and mean ± std.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return scores


def per_subject_summary(results: list[dict]) -> pd.DataFrame:
    """
    Build a summary DataFrame from a list of per-subject result dicts.
    Each dict must have keys: subject_id, accuracy, kappa, model.
    """
    df = pd.DataFrame(results)
    summary = df.groupby("model").agg(
        mean_acc=("accuracy", "mean"),
        std_acc=("accuracy", "std"),
        mean_kappa=("kappa", "mean"),
        n_subjects=("subject_id", "count"),
    ).reset_index()
    return summary


def chance_level(n_classes: int) -> float:
    return 1.0 / n_classes


def print_results(results: dict, model_name: str):
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Kappa    : {results['kappa']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(results["confusion_matrix"])
