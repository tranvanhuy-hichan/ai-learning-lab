"""Architectures used in the study."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm


class SimpleCNN(nn.Module):
    """Four Conv-BN-ReLU blocks (the from-scratch CNN of the classification project)."""

    def __init__(self, n_classes: int = 12):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )  # fmt: skip

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build(arch: str, n_classes: int = 12, pretrained: bool = True) -> nn.Module:
    w = "IMAGENET1K_V1" if pretrained else None
    if arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        return m
    if arch == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        return m
    if arch == "simple_cnn":
        return SimpleCNN(n_classes)
    raise ValueError(arch)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
