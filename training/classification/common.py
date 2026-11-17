"""Code dung chung cho 3 notebook: du lieu, huan luyen, danh gia."""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "garbage"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- du lieu
def scan_dataset(data_dir=DATA_DIR):
    classes = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted((Path(data_dir) / c).iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(f)
                labels.append(i)
    return classes, np.array(paths), np.array(labels)


class ImageDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), int(self.labels[i])


def get_loaders(img_size=128, batch_size=64, workers=4, seed=42):
    """Chia 70/15/15 theo lop (stratified) -> train/val/test."""
    classes, paths, labels = scan_dataset()
    p_tr, p_rest, y_tr, y_rest = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(p_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    loaders = {
        "train": DataLoader(ImageDS(p_tr, y_tr, train_tf), shuffle=True, **kw),
        "val": DataLoader(ImageDS(p_va, y_va, eval_tf), **kw),
        "test": DataLoader(ImageDS(p_te, y_te, eval_tf), **kw),
    }
    return classes, loaders, y_tr


def class_weights(y_train, n_classes):
    """Trong so nghich dao tan suat de bu lop mat can bang (clothes >> battery)."""
    cnt = np.bincount(y_train, minlength=n_classes)
    w = len(y_train) / (n_classes * cnt)
    return torch.tensor(w, dtype=torch.float32)


def show_samples(classes, per_class=4):
    _, paths, labels = scan_dataset()
    fig, ax = plt.subplots(len(classes), per_class, figsize=(per_class * 2, len(classes) * 1.6))
    for r, c in enumerate(classes):
        idx = np.where(labels == r)[0][:per_class]
        for k, a in enumerate(ax[r]):
            a.imshow(Image.open(paths[idx[k]]).convert("RGB")); a.axis("off")
        ax[r][0].set_title(c, fontsize=8, loc="left")
    plt.tight_layout(); plt.show()


def plot_class_distribution(classes):
    _, _, labels = scan_dataset()
    cnt = np.bincount(labels)
    plt.figure(figsize=(9, 3.5)); plt.bar(classes, cnt)
    plt.xticks(rotation=45, ha="right"); plt.title("So anh moi lop"); plt.tight_layout(); plt.show()
    return dict(zip(classes, cnt.tolist()))


# ---------------------------------------------------------------- huan luyen
def _epoch(model, loader, criterion, opt=None, scaler=None, sched=None):
    train = opt is not None
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return loss_sum / n, correct / n


def train_model(name, model, loaders, y_train, n_classes, epochs=15, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loaders["train"]))
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best, t0 = 0.0, time.time()
    ckpt = OUT_DIR / f"{name}.pt"
    for ep in range(1, epochs + 1):
        tl, ta = _epoch(model, loaders["train"], criterion, opt, scaler, sched)
        vl, va = _epoch(model, loaders["val"], criterion)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        if va > best:
            best = va
            torch.save(model.state_dict(), ckpt)
        print(f"ep {ep:02d}/{epochs} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model, hist, time.time() - t0


def plot_history(hist):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss")
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------- danh gia
@torch.no_grad()
def evaluate(name, model, loader, classes, hist=None, train_time=None):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(DEVICE)).argmax(1).cpu()); ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    print(classification_report(y, p, target_names=classes, digits=4))
    cm = confusion_matrix(y, p)
    cmn = cm / cm.sum(1, keepdims=True)
    plt.figure(figsize=(8, 7)); plt.imshow(cmn, cmap="Blues"); plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=90); plt.yticks(range(len(classes)), classes)
    plt.title(f"Confusion matrix (chuan hoa) - {name}"); plt.xlabel("Du doan"); plt.ylabel("Thuc te")
    plt.tight_layout(); plt.show()
    res = {"model": name, "test_acc": float((y == p).mean()),
           "macro_f1": float(classification_report(y, p, output_dict=True)["macro avg"]["f1-score"]),
           "params_M": sum(q.numel() for q in model.parameters()) / 1e6,
           "train_time_s": train_time, "history": hist}
    json.dump(res, open(OUT_DIR / f"{name}_results.json", "w"), indent=1)
    print(f"Test accuracy: {res['test_acc']:.4f} | macro F1: {res['macro_f1']:.4f}")
    return res


def show_predictions(model, loader, classes, n=12):
    model.eval()
    x, y = next(iter(loader))
    with torch.no_grad():
        p = model(x[:n].to(DEVICE)).argmax(1).cpu()
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    fig, ax = plt.subplots(2, n // 2, figsize=(n * 1.6, 5))
    for i, a in enumerate(ax.flat):
        a.imshow((x[i] * std + mean).permute(1, 2, 0).clamp(0, 1))
        a.set_title(f"{classes[p[i]]}\n(that: {classes[y[i]]})", fontsize=8,
                    color="green" if p[i] == y[i] else "red")
        a.axis("off")
    plt.tight_layout(); plt.show()
