"""
Downloads the PhysioNet EEG Motor Movement/Imagery Dataset via MNE.
109 subjects, 64 channels, 160 Hz sampling rate.
Tasks: left fist, right fist, both fists, both feet (motor imagery + execution).
"""

import mne
import numpy as np
from pathlib import Path

# Motor imagery runs: imagined left fist vs right fist (runs 4, 8, 12)
# Motor imagery runs: imagined both fists vs both feet (runs 6, 10, 14)
IMAGERY_RUNS_LR = [4, 8, 12]
IMAGERY_RUNS_FB = [6, 10, 14]

EVENT_ID = {
    "left_fist": 2,
    "right_fist": 3,
    "both_fists": 2,
    "both_feet": 3,
}


def download_subject(subject_id: int, data_path: Path) -> list:
    """Download all motor imagery runs for one subject."""
    runs = IMAGERY_RUNS_LR + IMAGERY_RUNS_FB
    raw_files = mne.datasets.eegbci.load_data(
        subject=subject_id,
        runs=runs,
        path=str(data_path),
        verbose=False,
    )
    return raw_files


def load_subject_raw(subject_id: int, data_path: Path, task: str = "LR"):
    """
    Load and concatenate raw EEG for one subject.
    task: 'LR' = left vs right fist, 'FB' = both fists vs both feet
    """
    runs = IMAGERY_RUNS_LR if task == "LR" else IMAGERY_RUNS_FB
    raw_files = mne.datasets.eegbci.load_data(
        subject=subject_id,
        runs=runs,
        path=str(data_path),
        verbose=False,
    )
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    return raw


def download_all(n_subjects: int = 20, data_path: Path = None):
    """Download dataset for the first n_subjects subjects."""
    if data_path is None:
        data_path = Path(__file__).parent / "physionet"
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PhysioNet EEG dataset for {n_subjects} subjects...")
    for sid in range(1, n_subjects + 1):
        print(f"  Subject {sid:03d}/{n_subjects:03d}", end="\r")
        download_subject(sid, data_path)
    print(f"\nDone. Data stored at: {data_path}")


if __name__ == "__main__":
    download_all(n_subjects=20)
