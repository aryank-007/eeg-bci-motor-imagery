"""
EEG preprocessing pipeline for motor imagery classification.
Implements bandpass filtering, epoching, and artifact rejection.
"""

import mne
import numpy as np
from pathlib import Path

# Mu (8-12 Hz) and beta (13-30 Hz) bands are event-related in motor imagery
BANDPASS_LOW = 8.0
BANDPASS_HIGH = 30.0
EPOCH_TMIN = 0.5   # seconds after cue onset (avoid cue artifact)
EPOCH_TMAX = 3.5   # seconds after cue onset
BASELINE = None    # no baseline correction — use band power features


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply standard preprocessing: montage, reference, bandpass filter."""
    raw = raw.copy()

    # Set 10-20 montage for topographic plotting
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    # Common average reference
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)

    # Bandpass filter to mu/beta bands
    raw.filter(
        BANDPASS_LOW,
        BANDPASS_HIGH,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
        verbose=False,
    )

    return raw


def extract_epochs(raw: mne.io.BaseRaw, event_id: dict) -> mne.Epochs:
    """Extract epochs time-locked to motor imagery cue events."""
    events, _ = mne.events_from_annotations(raw, verbose=False)

    # PhysioNet event mapping: T0=rest, T1=task1, T2=task2
    # For LR runs: T1=left fist, T2=right fist
    # For FB runs: T1=both fists, T2=both feet
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        preload=True,
        verbose=False,
    )

    # Drop epochs with peak-to-peak amplitude > 150 µV (gross artifact rejection)
    epochs.drop_bad(reject={"eeg": 150e-6}, verbose=False)

    return epochs


def get_epochs_data(epochs: mne.Epochs):
    """Return (X, y) arrays from epochs. X shape: (n_trials, n_channels, n_times)."""
    X = epochs.get_data()
    y = epochs.events[:, 2]
    # Remap labels to 0-indexed
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.array([label_map[v] for v in y])
    return X.astype(np.float32), y.astype(np.int64)


def load_subject_epochs(
    subject_id: int,
    data_path: Path,
    task: str = "LR",
):
    """Full pipeline: load → preprocess → epoch for one subject."""
    from data.download import load_subject_raw, IMAGERY_RUNS_LR, IMAGERY_RUNS_FB

    raw = load_subject_raw(subject_id, data_path, task=task)
    raw = preprocess_raw(raw)

    # Both tasks share the same annotation codes T1/T2 in PhysioNet
    event_id = {"T1": 1, "T2": 2}
    epochs = extract_epochs(raw, event_id)
    return epochs
