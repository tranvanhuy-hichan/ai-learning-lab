"""Code dung chung cho bai phan doan anh (Oxford-IIIT Pet) bang U-Net."""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models
from torchvision.transforms import functional as TF

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["pet", "background", "border"]  # trimap goc: 1=pet, 2=background, 3=border -> tru 1
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------- du lieu
def _load_split(split, img_size):
    """Doc 1 lan, resize, luu cache uint8 de cac epoch sau doc rat nhanh."""
    cache = DATA_DIR / f"cache_{split}_{img_size}.pt"
    if cache.exists():
        return torch.load(cache)
    ds = datasets.OxfordIIITPet(str(DATA_DIR), split=split, target_types="segmentation", download=True)
    imgs, masks = [], []
    for img, m in ds:
        img = TF.resize(img.convert("RGB"), [img_size, img_size], antialias=True)
        m = TF.resize(m, [img_size, img_size], interpolation=TF.InterpolationMode.NEAREST)
        imgs.append(TF.pil_to_tensor(img))
        masks.append(torch.from_numpy(np.array(m)).to(torch.uint8) - 1)  # 0,1,2 (uint8 cho nhe cache)
    data = (torch.stack(imgs), torch.stack(masks))
    torch.save(data, cache)
    return data


class SegDS(Dataset):
    def __init__(self, imgs, masks, augment=False):
        self.imgs, self.masks, self.augment = imgs, masks, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x, m = self.imgs[i], self.masks[i]
        if self.augment:
            if torch.rand(1) < 0.5:  # lat ngang ca anh lan mask
                x, m = x.flip(-1), m.flip(-1)
            x = TF.adjust_brightness(x, 0.8 + 0.4 * torch.rand(1).item())
            x = TF.adjust_contrast(x, 0.8 + 0.4 * torch.rand(1).item())
        x = (x.float() / 255 - MEAN) / STD
        return x, m.long()


def get_loaders(img_size=128, batch_size=32, workers=2, seed=42):
    """trainval (3680) chia 90/10 -> train/val; test chinh thuc (3669)."""
    imgs, masks = _load_split("trainval", img_size)
    n_val = int(0.1 * len(imgs))
    g = torch.Generator().manual_seed(seed)
    tr_idx, va_idx = random_split(range(len(imgs)), [len(imgs) - n_val, n_val], generator=g)
    tr_idx, va_idx = list(tr_idx), list(va_idx)
    timgs, tmasks = _load_split("test", img_size)
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    return {
        "train": DataLoader(SegDS(imgs[tr_idx], masks[tr_idx], augment=True), shuffle=True, drop_last=True, **kw),
        "val": DataLoader(SegDS(imgs[va_idx], masks[va_idx]), **kw),
        "test": DataLoader(SegDS(timgs, tmasks), **kw),
    }


def denorm(x):
    return (x.cpu() * STD + MEAN).permute(1, 2, 0).clamp(0, 1)


PALETTE = np.array([[230, 60, 60], [60, 60, 200], [250, 220, 60]], dtype=np.uint8)  # pet, background, border


def colorize(m):
    return PALETTE[np.asarray(m)]


def show_samples(loader, n=6):
    x, m = next(iter(loader))
    fig, ax = plt.subplots(2, n, figsize=(n * 2.2, 4.6))
    for i in range(n):
        ax[0][i].imshow(denorm(x[i])); ax[1][i].imshow(colorize(m[i]))
        ax[0][i].axis("off"); ax[1][i].axis("off")
    ax[0][0].set_title("Anh", loc="left", fontsize=9); ax[1][0].set_title("Mask (do=pet, xanh=nen, vang=vien)", loc="left", fontsize=9)
    plt.tight_layout(); plt.show()


def class_pixel_ratio(loader):
    cnt = torch.zeros(3)
    for _, m in loader:
        cnt += torch.bincount(m.flatten(), minlength=3).float()
    return dict(zip(CLASSES, (cnt / cnt.sum()).tolist()))


# ---------------------------------------------------------------- mo hinh U-Net
class DoubleConv(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net: encoder (4 lan MaxPool) -> bottleneck -> decoder (4 lan ConvTranspose + skip connection)."""

    def __init__(self, n_classes=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]  # 32, 64, 128, 256, 512
        self.enc = nn.ModuleList([DoubleConv(3, c[0])] + [DoubleConv(c[k], c[k + 1]) for k in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(c[k] * 2, c[k]) for k in reversed(range(4))])
        self.out = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for k, blk in enumerate(self.enc):
            x = blk(x)
            if k < 4:
                skips.append(x)
                x = self.pool(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([s, x], dim=1))  # skip connection: noi dac trung encoder vao decoder
        return self.out(x)


class ResNetUNet(nn.Module):
    """U-Net voi encoder = ResNet18 pretrained (ImageNet); decoder + skip connection tu viet."""

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        r = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.l0 = nn.Sequential(r.conv1, r.bn1, r.relu)  # 64 kenh, H/2
        self.pool = r.maxpool                              # H/4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64,128,256,512
        self.up4, self.d4 = nn.ConvTranspose2d(512, 256, 2, stride=2), DoubleConv(512, 256)  # ghep l3
        self.up3, self.d3 = nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128)  # ghep l2
        self.up2, self.d2 = nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64)    # ghep l1
        self.up1, self.d1 = nn.ConvTranspose2d(64, 64, 2, stride=2), DoubleConv(128, 64)     # ghep l0
        self.up0, self.d0 = nn.ConvTranspose2d(64, 32, 2, stride=2), DoubleConv(32, 32)      # ve kich thuoc anh goc
        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
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


# ---------------------------------------------------------------- loss & metric
def dice_loss(logits, target, eps=1e-6):
    p = logits.softmax(1)
    t = F.one_hot(target, p.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * t).sum((0, 2, 3))
    union = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


class SegLoss(nn.Module):
    """CrossEntropy (co the co trong so lop) + Dice."""

    def __init__(self, class_weights=None):
        super().__init__()
        w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("w", w)
        self.ce = nn.CrossEntropyLoss(weight=self.w)

    def _apply(self, fn, *a, **k):  # giu weight cua CE dong bo khi chuyen thiet bi
        out = super()._apply(fn, *a, **k)
        self.ce.weight = self.w
        return out

    def forward(self, logits, target):
        return self.ce(logits, target) + dice_loss(logits, target)


def confusion(pred, target, n=3):
    k = target.flatten() * n + pred.flatten()
    return torch.bincount(k, minlength=n * n).view(n, n)


def metrics_from_confusion(cm):
    cm = cm.double()
    tp = cm.diag()
    iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    dice = 2 * tp / (cm.sum(0) + cm.sum(1))
    return {"pixel_acc": (tp.sum() / cm.sum()).item(), "mIoU": iou.mean().item(), "mDice": dice.mean().item(),
            "IoU": dict(zip(CLASSES, iou.tolist())), "Dice": dict(zip(CLASSES, dice.tolist()))}


# ---------------------------------------------------------------- huan luyen
def _epoch(model, loader, criterion, opt=None, scaler=None, sched=None):
    train = opt is not None
    model.train(train)
    loss_sum, n = 0.0, 0
    cm = torch.zeros(3, 3, dtype=torch.long)
    with torch.set_grad_enabled(train):
        for x, m in loader:
            x, m = x.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(x)
                loss = criterion(out.float(), m)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            cm += confusion(out.argmax(1), m).cpu()
    return loss_sum / n, metrics_from_confusion(cm)["mIoU"]


def train_model(name, model, loaders, epochs=20, lr=2e-3, class_weights=None):
    model = model.to(DEVICE)
    criterion = SegLoss(class_weights).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loaders["train"]))
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    hist = {k: [] for k in ("train_loss", "train_miou", "val_loss", "val_miou")}
    best, t0 = 0.0, time.time()
    ckpt = OUT_DIR / f"{name}.pt"
    for ep in range(1, epochs + 1):
        tl, tm = _epoch(model, loaders["train"], criterion, opt, scaler, sched)
        vl, vm = _epoch(model, loaders["val"], criterion)
        for k, v in zip(hist, (tl, tm, vl, vm)):
            hist[k].append(v)
        if vm > best:
            best = vm
            torch.save(model.state_dict(), ckpt)
        print(f"ep {ep:02d}/{epochs} | train loss {tl:.3f} mIoU {tm:.3f} | val loss {vl:.3f} mIoU {vm:.3f}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model, hist, time.time() - t0


def plot_history(hist):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss (CE + Dice)")
    ax[1].plot(hist["train_miou"], label="train"); ax[1].plot(hist["val_miou"], label="val"); ax[1].set_title("mIoU")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------- danh gia
@torch.no_grad()
def evaluate(name, model, loader, hist=None, train_time=None):
    model.eval()
    cm = torch.zeros(3, 3, dtype=torch.long)
    for x, m in loader:
        cm += confusion(model(x.to(DEVICE)).argmax(1).cpu(), m)
    res = metrics_from_confusion(cm)
    res.update({"model": name, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
                "train_time_s": train_time, "history": hist})
    json.dump(res, open(OUT_DIR / f"{name}_results.json", "w"), indent=1)
    print(f"Pixel acc {res['pixel_acc']:.4f} | mIoU {res['mIoU']:.4f} | mDice {res['mDice']:.4f}")
    for c in CLASSES:
        print(f"  {c:11s} IoU {res['IoU'][c]:.4f}  Dice {res['Dice'][c]:.4f}")
    return res


@torch.no_grad()
def show_predictions(model, loader, n=6):
    model.eval()
    x, m = next(iter(loader))
    p = model(x[:n].to(DEVICE)).argmax(1).cpu()
    fig, ax = plt.subplots(4, n, figsize=(n * 2.2, 9))
    for i in range(n):
        img = denorm(x[i])
        ax[0][i].imshow(img); ax[1][i].imshow(colorize(m[i])); ax[2][i].imshow(colorize(p[i]))
        ax[3][i].imshow(img); ax[3][i].imshow(colorize(p[i]), alpha=0.45)
        for r in range(4):
            ax[r][i].axis("off")
    for r, t in enumerate(["Anh", "Ground truth", "U-Net du doan", "Chong lop"]):
        ax[r][0].set_title(t, loc="left", fontsize=9)
    plt.tight_layout(); plt.show()
