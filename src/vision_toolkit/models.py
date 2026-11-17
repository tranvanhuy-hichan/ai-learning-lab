"""Model definitions. Layouts match the training code in `training/` so saved weights load as-is."""

import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------------------- classification
def build_waste_classifier(n_classes: int = 12) -> nn.Module:
    """EfficientNet-B0 with a new classification head (the backbone was fine-tuned from ImageNet)."""
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


# --------------------------------------------------------------- segmentation
class DoubleConv(nn.Sequential):
    def __init__(self, i: int, o: int):
        super().__init__(
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Classic U-Net: 4 down-sampling stages, a bottleneck, 4 up-sampling stages with skip connections."""

    def __init__(self, n_classes: int = 3, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([DoubleConv(3, c[0])] + [DoubleConv(c[k], c[k + 1]) for k in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(c[k] * 2, c[k]) for k in reversed(range(4))])
        self.out = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for k, blk in enumerate(self.enc):
            x = blk(x)
            if k < 4:
                skips.append(x)
                x = self.pool(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([s, up(x)], dim=1))
        return self.out(x)


class ResNetUNet(nn.Module):
    """U-Net whose encoder is a ResNet18 (ImageNet-pretrained at training time); hand-written decoder."""

    def __init__(self, n_classes: int = 3, pretrained: bool = False):
        super().__init__()
        r = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.l0 = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.up4, self.d4 = nn.ConvTranspose2d(512, 256, 2, stride=2), DoubleConv(512, 256)
        self.up3, self.d3 = nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128)
        self.up2, self.d2 = nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64)
        self.up1, self.d1 = nn.ConvTranspose2d(64, 64, 2, stride=2), DoubleConv(128, 64)
        self.up0, self.d0 = nn.ConvTranspose2d(64, 32, 2, stride=2), DoubleConv(32, 32)
        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.l0(x)
        x1 = self.l1(self.pool(x0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        y = self.d4(torch.cat([x3, self.up4(x4)], 1))
        y = self.d3(torch.cat([x2, self.up3(y)], 1))
        y = self.d2(torch.cat([x1, self.up2(y)], 1))
        y = self.d1(torch.cat([x0, self.up1(y)], 1))
        return self.out(self.d0(self.up0(y)))
