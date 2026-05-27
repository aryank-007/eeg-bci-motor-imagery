"""
Quick demo: downloads data for 3 subjects, trains CSP+LDA, prints results.
No GPU needed. Runs in ~5 minutes.
Usage: python scripts/quick_demo.py
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.download import load_subject_raw
from src.preprocessing import preprocess_raw, extract_epochs, get_epochs_data
from src.features import build_csp_lda_pipeline
from src.evaluate import cross_validate_sklearn, chance_level

DATA_PATH = ROOT / "data" / "physionet"
PHYS_EVENT_ID = {"T1": 1, "T2": 2}


def demo():
    print("EEG BCI Quick Demo — CSP+LDA on PhysioNet Dataset")
    print("Downloading data for 3 subjects (auto via MNE)...\n")

    all_scores = []
    for sid in [1, 2, 3]:
        raw = load_subject_raw(sid, DATA_PATH, task="LR")
        raw = preprocess_raw(raw)
        epochs = extract_epochs(raw, PHYS_EVENT_ID)
        X, y = get_epochs_data(epochs)

        pipeline = build_csp_lda_pipeline(n_components=6)
        scores = cross_validate_sklearn(pipeline, X, y, n_splits=5)
        all_scores.extend(scores)
        print(f"Subject {sid:02d} | {len(epochs)} epochs | CV Acc: {scores.mean():.4f} ± {scores.std():.4f}")

    print(f"\nMean across subjects: {np.mean(all_scores):.4f}")
    print(f"Chance level:         {chance_level(2):.4f}")
    print(f"\nAbove-chance by:      {np.mean(all_scores) - chance_level(2):.4f}")


if __name__ == "__main__":
    demo()
