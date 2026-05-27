# EEG-Based Motor Imagery Classification for Brain-Computer Interfaces

A complete, research-grade pipeline for decoding motor imagery from EEG signals using classical signal processing and deep learning. Benchmarked on the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 64 channels).

---

## Overview

Motor imagery BCI systems allow users to control external devices — prosthetics, wheelchairs, communication aids — by imagining limb movements without physically performing them. The brain generates characteristic oscillatory patterns (Event-Related Desynchronization in the mu/beta bands, 8–30 Hz) that can be decoded from scalp EEG.

This project implements and compares four approaches:

| Model | Type | Description |
|---|---|---|
| **CSP + LDA** | Classical | Common Spatial Patterns + Linear Discriminant Analysis |
| **LogPower + LDA** | Classical | Log band-power features + LDA (baseline) |
| **EEGNet** | Deep Learning | Compact CNN with depthwise separable convolutions (Lawhern et al., 2018) |
| **ShallowConvNet** | Deep Learning | Shallow CNN with square/log nonlinearity (Schirrmeister et al., 2017) |

---

## Architecture

```
Raw EEG (64ch, 160Hz)
        │
        ▼
┌───────────────────┐
│   Preprocessing   │  Bandpass 8–30Hz (Butterworth IIR)
│                   │  Common Average Reference
│                   │  Epoch extraction (0.5s – 3.5s post-cue)
│                   │  Artifact rejection (>150µV)
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Classical   Neural
Pipeline    Pipeline
    │         │
    ▼         ▼
CSP+LDA   EEGNet / ShallowConvNet
    │         │
    └────┬────┘
         │
    5-Fold Stratified Cross-Validation
         │
    Accuracy · Kappa · Confusion Matrix
```

### EEGNet Architecture

```
Input (64ch × 480 samples)
    → Temporal Conv (F1=8 filters, kernel=0.5s)
    → Depthwise Spatial Conv (D=2, learns electrode patterns)
    → BatchNorm → ELU → AvgPool → Dropout
    → Separable Conv (F2=16 filters)
    → BatchNorm → ELU → AvgPool → Dropout
    → Dense → Softmax
```

Total parameters: ~2,600 (versus millions in standard CNNs).

---

## Results

Evaluated on left fist vs. right fist imagery (Task LR), 5-fold cross-validation, averaged across 10 subjects.

| Model | Accuracy | Cohen's κ |
|---|---|---|
| Chance | 0.500 | 0.000 |
| LogPower + LDA | ~0.62 | ~0.24 |
| CSP + LDA | ~0.72 | ~0.44 |
| ShallowConvNet | ~0.74 | ~0.48 |
| **EEGNet** | **~0.76** | **~0.52** |

> Results vary across subjects due to individual differences in EEG signal quality and motor imagery ability (BCI illiteracy affects ~20% of users).

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset** (Goldberger et al., 2000; Schalk et al., 2004)
- 109 subjects
- 64 EEG channels (10-20 system)
- 160 Hz sampling rate
- Tasks: imagined left fist, right fist, both fists, both feet

Downloaded automatically via MNE-Python on first run. No manual setup required.

---

## Setup

```bash
git clone https://github.com/yourusername/eeg-bci-motor-imagery
cd eeg-bci-motor-imagery
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Usage

### Quick demo (3 subjects, CSP+LDA, ~5 minutes)
```bash
python scripts/quick_demo.py
```

### Full pipeline
```bash
# Left fist vs. right fist, 10 subjects, CPU
python scripts/run_pipeline.py --n_subjects 10 --task LR --device cpu

# Both fists vs. both feet, 20 subjects, GPU
python scripts/run_pipeline.py --n_subjects 20 --task FB --device cuda
```

### Options
```
--n_subjects    Number of subjects to process (default: 10)
--task          LR = left/right fist | FB = both fists/both feet (default: LR)
--device        cpu or cuda (default: cpu)
--n_epochs_nn   Max training epochs for neural models (default: 150)
```

Outputs are saved to `results/` (CSV) and `figures/` (PNG).

---

## Project Structure

```
eeg-bci-motor-imagery/
├── data/
│   └── download.py          # PhysioNet data loader (auto-downloads via MNE)
├── src/
│   ├── preprocessing.py     # Filtering, epoching, artifact rejection
│   ├── features.py          # CSP, log-band-power feature extraction
│   ├── train.py             # Training loop, cross-validation (neural models)
│   ├── evaluate.py          # Accuracy, kappa, confusion matrix
│   ├── visualize.py         # Topomaps, training curves, model comparison
│   └── models/
│       ├── eegnet.py        # EEGNet (Lawhern et al., 2018)
│       └── shallow_cnn.py   # ShallowConvNet (Schirrmeister et al., 2017)
├── scripts/
│   ├── run_pipeline.py      # Full multi-model pipeline
│   └── quick_demo.py        # Minimal demo
├── results/                 # CSV outputs
├── figures/                 # Generated plots
└── requirements.txt
```

---

## Key Concepts

**Common Spatial Patterns (CSP):** Finds linear combinations of EEG electrodes that maximize the ratio of variance between two classes. Directly captures the spatial distribution of motor-related oscillations.

**Event-Related Desynchronization (ERD):** During motor imagery, mu (8–12 Hz) and beta (13–30 Hz) power decreases contralaterally to the imagined movement — e.g., imagining right hand movement suppresses power over left motor cortex (C3 electrode).

**EEGNet:** Uses depthwise separable convolutions to learn temporal filters (frequency-selective) and spatial filters (electrode-selective) with very few parameters, making it suitable for small EEG datasets.

---

## References

1. Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*, 15(5), 056013.

2. Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420.

3. Schalk, G., et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.

4. Ang, K. K., et al. (2008). Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface. *IEEE IJCNN*, 2390–2397.

---

## License

MIT
