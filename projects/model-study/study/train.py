"""One training run: fine-tune or train from scratch, with optional label smoothing, CutMix and distillation.

Every run saves the logits of the validation set, the test set and the out-of-distribution (pet) images, so
calibration, ensembling and out-of-distribution analysis never need to retrain.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metrics
from .data import gpu_augment, normalize
from .models import build, n_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    name: str
    arch: str = "efficientnet_b0"
    pretrained: bool = True
    size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    batch: int = 64
    seed: int = 0
    label_smoothing: float = 0.0
    cutmix: float = 0.0  # probability of applying CutMix to a batch
    teacher: str | None = None  # path to an efficientnet_b0 state dict -> knowledge distillation
    kd_temperature: float = 4.0
    kd_alpha: float = 0.7


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def logits_of(model: nn.Module, x: torch.Tensor, bs: int = 256) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            xb = normalize(x[i : i + bs].to(DEVICE)).contiguous(memory_format=torch.channels_last)
            out.append(model(xb).float().cpu())
    return torch.cat(out)


def _cutmix(x: torch.Tensor, y: torch.Tensor):
    lam = np.random.beta(1.0, 1.0)
    perm = torch.randperm(x.shape[0], device=x.device)
    h, w = x.shape[2:]
    cut = np.sqrt(1 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2, x1, x2 = max(cy - ch // 2, 0), min(cy + ch // 2, h), max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, y, y[perm], lam


def run(
    cfg: Config, data: dict, outdir: Path, pets: dict[str, torch.Tensor] | None = None, verbose: bool = False
) -> dict:
    """Train, pick the best epoch on the validation set, save logits and weights, return a summary."""
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    n_classes = len(data["classes"])

    model = build(cfg.arch, n_classes, cfg.pretrained).to(DEVICE).to(memory_format=torch.channels_last)
    teacher = None
    if cfg.teacher:
        teacher = build("efficientnet_b0", n_classes, pretrained=False).to(DEVICE)
        teacher.load_state_dict(torch.load(cfg.teacher, map_location=DEVICE, weights_only=True))
        teacher.eval()

    counts = torch.bincount(ytr, minlength=n_classes).float()
    weight = (len(ytr) / (n_classes * counts)).to(DEVICE)  # inverse-frequency class weights (as in the notebooks)
    steps = len(ytr) // cfg.batch
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=steps)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    def ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), target, weight=weight, label_smoothing=cfg.label_smoothing)

    best, best_state, hist, t0 = -1.0, None, [], time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        run_loss = 0.0
        for i in range(steps):
            idx = perm[i * cfg.batch : (i + 1) * cfg.batch]
            xb = gpu_augment(xtr[idx].to(DEVICE, non_blocking=True)).contiguous(memory_format=torch.channels_last)
            yb = ytr[idx].to(DEVICE, non_blocking=True)
            mixed = cfg.cutmix > 0 and np.random.rand() < cfg.cutmix
            if mixed:
                xb, ya, yb2, lam = _cutmix(xb, yb)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(xb)
                loss = lam * ce(out, ya) + (1 - lam) * ce(out, yb2) if mixed else ce(out, yb)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(xb).float()
                    T = cfg.kd_temperature
                    kd = (
                        F.kl_div(F.log_softmax(out.float() / T, 1), F.softmax(tl / T, 1), reduction="batchmean") * T * T
                    )
                    loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
        val_acc = metrics.accuracy(logits_of(model, xva), yva)
        hist.append({"epoch": ep + 1, "train_loss": run_loss / steps, "val_acc": val_acc})
        if verbose:
            print(
                f"  [{cfg.name} s{cfg.seed}] ep {ep + 1}/{cfg.epochs} loss {run_loss / steps:.3f} val {val_acc:.4f}",
                flush=True,
            )
        if val_acc > best:
            best, best_state = val_acc, copy.deepcopy(model.state_dict())
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    tag = f"{cfg.name}_s{cfg.seed}"
    (outdir / "weights").mkdir(exist_ok=True)
    torch.save(best_state, outdir / "weights" / f"{tag}.pt")
    arrays = {"val": logits_of(model, xva), "test": logits_of(model, xte), "y_val": yva, "y_test": yte}
    for k, v in (pets or {}).items():
        arrays[k] = logits_of(model, v)
    np.savez_compressed(outdir / f"{tag}.npz", **{k: v.numpy() for k, v in arrays.items()})

    return {
        **asdict(cfg),
        "params": n_params(model),
        "train_time_s": train_time,
        "best_val_acc": best,
        "test_acc": metrics.accuracy(arrays["test"], yte),
        "test_macro_f1": metrics.macro_f1(arrays["test"], yte),
        "history": hist,
    }
