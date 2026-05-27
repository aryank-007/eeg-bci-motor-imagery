"""
ShallowConvNet: Shallow CNN designed for decoding band-power features from EEG.
Based on: Schirrmeister et al. (2017) "Deep learning with convolutional neural
networks for EEG decoding and visualization." Human Brain Mapping, 38(11).

Uses a single temporal + spatial convolution followed by a square/log
nonlinearity that approximates band-power computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SquareActivation(nn.Module):
    def forward(self, x):
        return x ** 2


class LogActivation(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 64,
        n_times: int = 480,
        n_temporal_filters: int = 40,
        temporal_kernel_size: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        # Temporal convolution — learns frequency filters
        self.temporal_conv = nn.Conv2d(
            1, n_temporal_filters, (1, temporal_kernel_size), bias=False
        )
        # Spatial convolution — learns electrode weighting
        self.spatial_conv = nn.Conv2d(
            n_temporal_filters, n_temporal_filters, (n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_temporal_filters, momentum=0.1, affine=True)

        self.square = SquareActivation()
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.log = LogActivation()
        self.dropout = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self._forward_features(dummy)
            self._flat_size = out.numel()

        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = self.dropout(x)
        x = x.flatten(1)
        return self.classifier(x)
