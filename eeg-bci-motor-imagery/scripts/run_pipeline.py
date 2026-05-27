"""
Main pipeline: download data, preprocess, train all models, evaluate, visualize.
Usage:
    python scripts/run_pipeline.py --n_subjects 10 --task LR --device cpu
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from sklearn.pipeline import Pipeline

from data.download import load_subject_raw
from src.preprocessing import preprocess_raw, extract_epochs, get_epochs_data
from src.features import build_csp_lda_pipeline, build_logpower_lda_pipeline
from src.models.eegnet import EEGNet
from src.models.shallow_cnn import ShallowConvNet
from src.train import train_model, cross_validate_model
from src.evaluate import (
    evaluate_predictions,
    cross_validate_sklearn,
    per_subject_summary,
    print_results,
    chance_level,
)
from src.visualize import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_model_comparison,
    plot_csp_patterns,
)


DATA_PATH = ROOT / "data" / "physionet"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {"LR": ["Left Fist", "Right Fist"], "FB": ["Both Fists", "Both Feet"]}
PHYS_EVENT_ID = {"T1": 1, "T2": 2}


def run_subject(subject_id, task, device, n_epochs_nn=150):
    print(f"\n[Subject {subject_id:03d}] Loading & preprocessing...")
    raw = load_subject_raw(subject_id, DATA_PATH, task=task)
    raw = preprocess_raw(raw)
    epochs = extract_epochs(raw, PHYS_EVENT_ID)

    if len(epochs) < 10:
        print(f"  Too few epochs ({len(epochs)}), skipping.")
        return None

    X, y = get_epochs_data(epochs)
    n_trials, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    class_names = CLASS_NAMES[task]
    print(f"  Epochs: {n_trials} | Channels: {n_channels} | Times: {n_times} | Classes: {n_classes}")

    subject_results = []

    # ── CSP + LDA ──────────────────────────────────────────────────────────────
    print("  Training CSP+LDA...")
    csp_lda = build_csp_lda_pipeline(n_components=6)
    scores = cross_validate_sklearn(csp_lda, X, y, n_splits=5)
    csp_lda.fit(X, y)
    y_pred_csp = csp_lda.predict(X)
    res_csp = evaluate_predictions(y, y_pred_csp, class_names)
    res_csp["accuracy"] = scores.mean()
    res_csp["subject_id"] = subject_id
    res_csp["model"] = "CSP+LDA"
    subject_results.append(res_csp)
    print(f"    CV Acc: {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Log-Power + LDA ────────────────────────────────────────────────────────
    print("  Training LogPower+LDA...")
    lp_lda = build_logpower_lda_pipeline()
    scores_lp = cross_validate_sklearn(lp_lda, X, y, n_splits=5)
    print(f"    CV Acc: {scores_lp.mean():.4f} ± {scores_lp.std():.4f}")
    res_lp = {"accuracy": scores_lp.mean(), "kappa": 0.0, "subject_id": subject_id, "model": "LogPower+LDA"}
    subject_results.append(res_lp)

    # ── EEGNet ─────────────────────────────────────────────────────────────────
    print("  Training EEGNet...")
    def make_eegnet():
        return EEGNet(n_classes=n_classes, n_channels=n_channels, n_times=n_times)

    fold_accs, preds_en, labels_en = cross_validate_model(
        make_eegnet, X, y, n_splits=5, device=device,
        n_epochs=n_epochs_nn, batch_size=16, patience=20,
    )
    res_en = evaluate_predictions(labels_en, preds_en, class_names)
    res_en["accuracy"] = np.mean(fold_accs)
    res_en["subject_id"] = subject_id
    res_en["model"] = "EEGNet"
    subject_results.append(res_en)
    print(f"    CV Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    # ── ShallowConvNet ─────────────────────────────────────────────────────────
    print("  Training ShallowConvNet...")
    def make_shallow():
        return ShallowConvNet(n_classes=n_classes, n_channels=n_channels, n_times=n_times)

    fold_accs_s, preds_s, labels_s = cross_validate_model(
        make_shallow, X, y, n_splits=5, device=device,
        n_epochs=n_epochs_nn, batch_size=16, patience=20,
    )
    res_s = evaluate_predictions(labels_s, preds_s, class_names)
    res_s["accuracy"] = np.mean(fold_accs_s)
    res_s["subject_id"] = subject_id
    res_s["model"] = "ShallowConvNet"
    subject_results.append(res_s)
    print(f"    CV Acc: {np.mean(fold_accs_s):.4f} ± {np.std(fold_accs_s):.4f}")

    # ── Per-subject plots (for first subject only to save time) ────────────────
    if subject_id == 1:
        info = epochs.info
        try:
            plot_csp_patterns(csp_lda.named_steps["csp"], info, save=True)
        except Exception:
            pass
        plot_confusion_matrix(res_en["confusion_matrix"], class_names, "EEGNet", save=True)
        plot_confusion_matrix(res_csp["confusion_matrix"], class_names, "CSP+LDA", save=True)

    return subject_results


def main():
    parser = argparse.ArgumentParser(description="EEG Motor Imagery BCI Pipeline")
    parser.add_argument("--n_subjects", type=int, default=10, help="Number of subjects to process")
    parser.add_argument("--task", choices=["LR", "FB"], default="LR", help="LR=left/right fist, FB=fists/feet")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--n_epochs_nn", type=int, default=150, help="Max training epochs for neural models")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  EEG Motor Imagery BCI — PhysioNet Dataset")
    print(f"  Task: {args.task} | Subjects: {args.n_subjects} | Device: {args.device}")
    print(f"{'='*60}")

    all_results = []
    for sid in range(1, args.n_subjects + 1):
        try:
            results = run_subject(sid, args.task, args.device, args.n_epochs_nn)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"  [Subject {sid:03d}] Error: {e}")
            continue

    if not all_results:
        print("No results to aggregate.")
        return

    # Aggregate and display summary
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / f"results_{args.task}.csv", index=False)

    summary = per_subject_summary(all_results)
    print(f"\n{'='*60}")
    print("  FINAL RESULTS (across subjects)")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / f"summary_{args.task}.csv", index=False)

    # Model comparison plot
    plot_model_comparison(summary, save=True)

    print(f"\nChance level: {chance_level(2):.2f}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {ROOT / 'figures'}")


if __name__ == "__main__":
    main()
