"""Grad-CAM: which part of the image pushed the classifier towards a class."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# EfficientNet-B0 block whose activations are used. features[5] has stride 16 (an 8x8 map for a 128 px input);
# the study (projects/model-study) compared layers with a deletion test before choosing it.
CAM_LAYER = 5


def gradcam(
    model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor | None = None, layer: int = CAM_LAYER
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cam, logits). cam has shape (B, H, W), scaled to [0, 1] per image, at the input resolution."""
    store: dict[str, torch.Tensor] = {}
    handle = model.features[layer].register_forward_hook(lambda _m, _i, out: store.__setitem__("a", out))
    model.eval()
    try:
        with torch.enable_grad():
            logits = model(x.requires_grad_(False))
            idx = logits.argmax(1) if target is None else target
            score = logits.gather(1, idx[:, None]).sum()
            (grads,) = torch.autograd.grad(score, store["a"])
    finally:
        handle.remove()
    acts = store["a"].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    lo = cam.amin(dim=(1, 2), keepdim=True)
    hi = cam.amax(dim=(1, 2), keepdim=True)
    return (cam - lo) / (hi - lo + 1e-8), logits.detach()


def heat(cam: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1] -> (H, W, 3) uint8, black through red to yellow."""
    c = np.clip(cam, 0, 1)
    r = np.clip(c * 2, 0, 1)
    g = np.clip(c * 2 - 1, 0, 1)
    b = np.clip(c * 4 - 3, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend the heat map over the original image (cam is resized to the image size)."""
    w, h = image.size
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    c = np.asarray(cam_img).astype(np.float32) / 255
    a = (alpha * c)[..., None]  # opaque where the evidence is strong, transparent where it is weak
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    return Image.fromarray((base * (1 - a) + heat(c).astype(np.float32) * a).astype(np.uint8))
