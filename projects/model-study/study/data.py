"""Waste dataset as cached uint8 tensors, plus batched GPU augmentation.

Decoding JPEGs every epoch dominated the earlier training time. Here every image is decoded and resized once
(then cached on disk), and augmentation runs on the GPU for the whole batch, so a run takes minutes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"  # not committed
VISION = ROOT.parent / "vision-toolkit" / "training"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def waste_dir() -> Path:
    return Path(os.environ.get("WASTE_DATA", VISION / "classification" / "data" / "garbage"))


def pet_dir() -> Path:
    return Path(os.environ.get("PET_DATA", VISION / "segmentation" / "data" / "oxford-iiit-pet"))


def _load(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img).transpose(2, 0, 1)


def _stack(paths: list[Path], size: int) -> torch.Tensor:
    with ThreadPoolExecutor(8) as ex:
        arrays = list(ex.map(lambda p: _load(p, size), paths))
    return torch.from_numpy(np.stack(arrays))


def scan() -> tuple[list[str], list[Path], np.ndarray]:
    root = waste_dir()
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted((root / c).iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(f)
                labels.append(i)
    return classes, paths, np.array(labels)


def split_indices(labels: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    """Stratified 70/15/15, identical to the split used by the classification notebooks."""
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=0.3, stratify=labels, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=seed)
    return {"train": tr, "val": va, "test": te}


def load_waste(size: int) -> dict:
    CACHE.mkdir(exist_ok=True)
    classes, paths, labels = scan()
    f = CACHE / f"waste_{size}.pt"
    if f.exists():
        x = torch.load(f)
    else:
        x = _stack(paths, size)
        torch.save(x, f)
    idx = split_indices(labels)
    y = torch.from_numpy(labels)
    return {"classes": classes, **{k: (x[v], y[v]) for k, v in idx.items()}}


def load_pets(size: int, split: str) -> torch.Tensor:
    """Cats and dogs (Oxford-IIIT Pet): images that are clearly *not* waste, used as out-of-distribution data."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"pets_{split}_{size}.pt"
    if f.exists():
        return torch.load(f)
    root = pet_dir()
    names = [ln.split()[0] for ln in (root / "annotations" / f"{split}.txt").read_text().splitlines() if ln.strip()]
    x = _stack([root / "images" / f"{n}.jpg" for n in names], size)
    torch.save(x, f)
    return x


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,S,S) -> float, ImageNet statistics."""
    mean, std = MEAN.to(x.device), STD.to(x.device)
    return (x.float() / 255 - mean) / std


def gpu_augment(x: torch.Tensor, scale=(0.7, 1.0), jitter: float = 0.2) -> torch.Tensor:
    """Random zoom-crop + horizontal flip + brightness/contrast/saturation jitter, batched on the GPU."""
    b = x.shape[0]
    dev = x.device
    x = x.float() / 255
    side = torch.empty(b, device=dev).uniform_(scale[0], scale[1]).sqrt()  # crop side so that area ~ U(scale)
    tx = (1 - side) * (torch.rand(b, device=dev) * 2 - 1)
    ty = (1 - side) * (torch.rand(b, device=dev) * 2 - 1)
    flip = (torch.rand(b, device=dev) < 0.5).float() * 2 - 1
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = side * flip, tx
    theta[:, 1, 1], theta[:, 1, 2] = side, ty
    x = F.grid_sample(
        x, F.affine_grid(theta, list(x.shape), align_corners=False), padding_mode="reflection", align_corners=False
    )

    def r():
        return 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * jitter

    x = x * r()  # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * r() + m  # contrast
    g = x.mean(1, keepdim=True)
    x = (x - g) * r() + g  # saturation
    x = x.clamp(0, 1)
    return (x - MEAN.to(dev)) / STD.to(dev)
