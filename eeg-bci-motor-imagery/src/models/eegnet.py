"""
EEGNet: Compact CNN for EEG-based BCIs.
Based on: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural
Network for EEG-based Brain-Computer Interfaces." J. Neural Eng., 15(5).

Architecture uses depthwise + separable convolutions to learn
temporal filters → spatial filters → separable temporal features
with far fewer parameters than standard CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 64,
        n_times: int = 480,        # samples per epoch at 160 Hz × 3s
        sfreq: float = 160.0,
        F1: int = 8,               # number of temporal filters
        D: int = 2,                # depth multiplier (spatial filters per temporal)
        F2: int = 16,              # number of separable filters
        dropout_rate: float = 0.5,
        kernel_length: int = 64,   # temporal kernel ≈ 0.5s at 160 Hz (half sfreq)
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Block 1: Temporal convolution + Depthwise spatial convolution
        self.block1 = nn.Sequential(
            # Temporal filter: learns frequency-specific patterns
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise convolution over channels: learns spatial (electrode) patterns
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable convolution — learns combined spatio-temporal features
        self.block2 = nn.Sequential(
            # Depthwise part
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            # Pointwise part
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.block2(self.block1(dummy))
            self._flat_size = out.numel()

        self.classifier = nn.Linear(self._flat_size, n_classes)

    def forward(self, x):
        # x: (batch, n_channels, n_times) → add channel dim
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)
