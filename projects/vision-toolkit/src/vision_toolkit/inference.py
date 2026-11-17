"""Inference helpers: waste classification and pet segmentation.

Pre-processing mirrors the training notebooks exactly (same resize + ImageNet normalisation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .models import ResNetUNet, UNet, build_waste_classifier

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ------------------------------------------------------------------ waste classification
WASTE_WEIGHTS = MODELS_DIR / "efficientnet_b0_waste.pt"
WASTE_LABELS_FILE = MODELS_DIR / "waste_labels.json"
WASTE_IMG_SIZE = 128

# General guidance only: real rules differ by city / waste authority.
RECYCLING_TIPS = {
    "battery": "Hazardous waste. Never put it in household bins; take it to a battery collection point.",
    "biological": "Organic waste. Suitable for compost or a food-waste bin.",
    "brown-glass": "Glass. Rinse and place in the glass recycling container (brown/mixed).",
    "cardboard": "Recyclable. Flatten the box and keep it dry before recycling.",
    "clothes": "Textiles. Donate if still wearable, otherwise use a textile collection bin.",
    "green-glass": "Glass. Rinse and place in the glass recycling container (green/mixed).",
    "metal": "Recyclable. Rinse cans and place them with metal / mixed recycling.",
    "paper": "Recyclable. Keep it clean and dry; greasy or soiled paper is usually not accepted.",
    "plastic": "Often recyclable. Check the resin code, rinse, and follow local rules.",
    "shoes": "Reuse first: donate if wearable, otherwise look for a textile / shoe collection point.",
    "trash": "General waste that cannot be recycled; put it in the residual-waste bin.",
    "white-glass": "Glass. Rinse and place in the glass recycling container (clear).",
}

_waste_tf = transforms.Compose(
    [transforms.Resize((WASTE_IMG_SIZE, WASTE_IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)
_cache: dict = {}


def waste_labels() -> list[str]:
    return json.loads(WASTE_LABELS_FILE.read_text())


def _waste_model() -> torch.nn.Module:
    if "waste" not in _cache:
        m = build_waste_classifier(len(waste_labels()))
        m.load_state_dict(torch.load(WASTE_WEIGHTS, map_location="cpu", weights_only=True))
        _cache["waste"] = m.to(DEVICE).eval()
    return _cache["waste"]


@torch.no_grad()
def predict_waste(image: Image.Image) -> dict[str, float]:
    """Return {label: probability} for all 12 waste classes."""
    x = _waste_tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    p = _waste_model()(x).softmax(1)[0].cpu().tolist()
    return dict(zip(waste_labels(), p, strict=True))


# ------------------------------------------------------------------ pet segmentation
SEG_CLASSES = ["pet", "background", "border"]
SEG_PALETTE = np.array([[230, 60, 60], [60, 60, 200], [250, 220, 60]], dtype=np.uint8)
SEG_MODELS = {
    "U-Net (from scratch, 128px)": dict(file="unet_pet.pt", size=128, build=lambda: UNet(3, 32)),
    "ResNet-U-Net (ResNet18 encoder, 256px)": dict(file="resnet_unet_pet.pt", size=256, build=lambda: ResNetUNet(3)),
}


def available_segmenters() -> list[str]:
    """Only the models whose weight file exists in models/."""
    return [name for name, s in SEG_MODELS.items() if (MODELS_DIR / s["file"]).exists()]


def _seg_model(name: str) -> torch.nn.Module:
    if name not in _cache:
        spec = SEG_MODELS[name]
        m = spec["build"]()
        m.load_state_dict(torch.load(MODELS_DIR / spec["file"], map_location="cpu", weights_only=True))
        _cache[name] = m.to(DEVICE).eval()
    return _cache[name]


def colorize(mask: np.ndarray) -> np.ndarray:
    return SEG_PALETTE[mask]


@torch.no_grad()
def segment_pet(image: Image.Image, model_name: str, alpha: float = 0.45):
    """Return (overlay, mask_image, stats) at the original image resolution.

    stats = {class: share of pixels in %}.
    """
    spec = SEG_MODELS[model_name]
    image = image.convert("RGB")
    w, h = image.size
    x = TF.pil_to_tensor(TF.resize(image, [spec["size"], spec["size"]], antialias=True)).float() / 255
    x = TF.normalize(x, MEAN, STD).unsqueeze(0).to(DEVICE)
    logits = _seg_model(model_name)(x)
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)  # back to original size
    mask = logits.argmax(1)[0].cpu().numpy()
    color = colorize(mask)
    overlay = Image.fromarray((np.asarray(image) * (1 - alpha) + color * alpha).astype(np.uint8))
    share = np.bincount(mask.ravel(), minlength=len(SEG_CLASSES)) / mask.size * 100
    return overlay, Image.fromarray(color), dict(zip(SEG_CLASSES, share.round(1).tolist(), strict=True))
