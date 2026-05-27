"""
Visualization utilities for EEG BCI results.
Produces topographic maps, confusion matrices, PSD plots, and training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import mne
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
}


def plot_confusion_matrix(cm, class_names, model_name="Model", save=True):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    if save:
        path = FIG_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


def plot_training_curves(history, model_name="EEGNet", save=True):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#2196F3")
    axes[0].plot(history["val_loss"], label="Val", color="#FF5722")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train", color="#2196F3")
    axes[1].plot(history["val_acc"], label="Val", color="#FF5722")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Chance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()

    fig.suptitle(f"{model_name} — Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = FIG_DIR / f"training_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


def plot_csp_patterns(csp, info, n_components=6, save=True):
    """
    Plot CSP spatial patterns as scalp topomaps.
    csp: fitted CommonSpatialPatterns object with filters_ attribute.
    info: MNE Info object with channel positions.
    """
    plt.rcParams.update(STYLE)
    n = min(n_components, csp.filters_.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        pattern = csp.filters_[i]
        mne.viz.plot_topomap(
            pattern,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-np.abs(pattern).max(), np.abs(pattern).max()),
        )
        ax.set_title(f"CSP {i+1}", fontsize=10)

    fig.suptitle("CSP Spatial Patterns", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = FIG_DIR / "csp_patterns.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


def plot_erds(epochs_class0, epochs_class1, channel="C3", sfreq=160.0, save=True):
    """
    Event-Related (De)Synchronization plot comparing two classes for one channel.
    Shows mu/beta power over time — the core neural signature of motor imagery.
    """
    from scipy.signal import welch

    plt.rcParams.update(STYLE)
    ch_names = epochs_class0.ch_names
    ch_idx = ch_names.index(channel) if channel in ch_names else 0

    def band_power_over_time(epochs, ch_idx, sfreq, window=0.5):
        """Sliding window band power (8-30 Hz) across the trial."""
        data = epochs.get_data()[:, ch_idx, :]
        n_times = data.shape[1]
        win = int(window * sfreq)
        step = win // 4
        times = []
        powers = []
        for start in range(0, n_times - win, step):
            segment = data[:, start:start+win]
            freqs, psd = welch(segment, fs=sfreq, nperseg=win, axis=1)
            mask = (freqs >= 8) & (freqs <= 30)
            power = psd[:, mask].mean(axis=1).mean()
            times.append((start + win / 2) / sfreq + epochs.tmin)
            powers.append(power)
        return np.array(times), np.array(powers)

    times0, pwr0 = band_power_over_time(epochs_class0, ch_idx, sfreq)
    times1, pwr1 = band_power_over_time(epochs_class1, ch_idx, sfreq)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times0, pwr0, label=epochs_class0.event_id and "Class 0" or "Class 0", color="#2196F3", linewidth=2)
    ax.plot(times1, pwr1, label="Class 1", color="#FF5722", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mu/Beta Power (µV²/Hz)")
    ax.set_title(f"ERD/ERS — Channel {channel}")
    ax.legend()
    plt.tight_layout()
    if save:
        path = FIG_DIR / f"erds_{channel}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


def plot_model_comparison(summary_df, save=True):
    """Bar chart comparing accuracy across models."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(
        summary_df["model"],
        summary_df["mean_acc"],
        yerr=summary_df["std_acc"],
        color=colors[:len(summary_df)],
        capsize=5,
        width=0.5,
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Chance (50%)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — Motor Imagery Classification\n(Left vs Right Fist, PhysioNet Dataset)")
    ax.legend()
    for bar, acc in zip(bars, summary_df["mean_acc"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    if save:
        path = FIG_DIR / "model_comparison.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


def plot_psd(epochs, title="Power Spectral Density", save=True):
    """Log-scale PSD for all channels."""
    plt.rcParams.update(STYLE)
    fig = epochs.compute_psd(fmin=1, fmax=45).plot(show=False)
    fig.suptitle(title)
    if save:
        path = FIG_DIR / "psd.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig
