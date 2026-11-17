"""Code dung chung cho RNN, LSTM, Transformer.

Hai bai toan:
  * Chuoi thoi gian: du bao nhiet do ngay mai tu 30 ngay truoc (Melbourne daily min temperature).
  * Van ban: phan loai cam xuc review phim IMDB (25k train / 25k test).
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["negative", "positive"]


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def n_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# =====================================================================
# 1. CHUOI THOI GIAN
# =====================================================================
def load_temperature(window=30, batch_size=64):
    """Chia theo thoi gian 70/15/15 (khong xao tron) de tranh ro ri du lieu tuong lai."""
    df = pd.read_csv(DATA_DIR / "temp.csv", parse_dates=["Date"])
    x = df["Temp"].values.astype(np.float32)
    n = len(x)
    i1, i2 = int(0.7 * n), int(0.85 * n)
    mean, std = x[:i1].mean(), x[:i1].std()  # chuan hoa chi bang thong ke cua tap train
    z = (x - mean) / std

    def make(lo, hi):
        idx = np.arange(max(lo, window), hi)  # moi mau: du doan x[t] tu x[t-window:t]
        X = torch.tensor(np.stack([z[t - window : t] for t in idx])[..., None])
        return TensorDataset(X, torch.tensor(z[idx])), idx

    parts = {"train": make(0, i1), "val": make(i1, i2), "test": make(i2, n)}
    loaders = {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train")) for k, (ds, _) in parts.items()}
    info = {
        "mean": float(mean),
        "std": float(std),
        "series": x,
        "dates": df["Date"].values,
        "idx": {k: idx for k, (_, idx) in parts.items()},
        "window": window,
    }
    return loaders, info


def plot_temperature(info):
    d, x = info["dates"], info["series"]
    i1, i2 = info["idx"]["val"][0], info["idx"]["test"][0]
    plt.figure(figsize=(12, 3.5))
    plt.plot(d[:i1], x[:i1], lw=0.6, label="train")
    plt.plot(d[i1:i2], x[i1:i2], lw=0.6, label="val")
    plt.plot(d[i2:], x[i2:], lw=0.6, label="test")
    plt.ylabel("Nhiet do thap nhat (°C)")
    plt.title("Melbourne - nhiet do thap nhat theo ngay")
    plt.legend()
    plt.tight_layout()
    plt.show()


class SeqRegressor(nn.Module):
    """Chuoi 30 ngay -> RNN/LSTM/GRU -> lay trang thai o buoc cuoi -> 1 so (nhiet do ngay mai)."""

    def __init__(self, kind="rnn", hidden=64, layers=2, dropout=0.1):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(1, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1]).squeeze(-1)


def train_regressor(name, model, loaders, epochs=40, lr=3e-3, verbose=True):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    crit = nn.MSELoss()
    hist = {"train_loss": [], "val_loss": []}
    best, t0, ckpt = 1e9, time.time(), OUT_DIR / f"{name}.pt"
    for ep in range(1, epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(y)
            n += len(y)
        model.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vl += crit(model(x), y).item() * len(y)
                m += len(y)
        tl, vl = tl / n, vl / m
        sched.step(vl)
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), ckpt)
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"ep {ep:02d}/{epochs} | train mse {tl:.4f} | val mse {vl:.4f}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model, hist, time.time() - t0


@torch.no_grad()
def evaluate_regressor(name, model, loaders, info, hist=None, train_time=None, save=True):
    model.eval()
    p = torch.cat([model(x.to(DEVICE)).cpu() for x, _ in loaders["test"]]).numpy()
    pred = p * info["std"] + info["mean"]
    true = info["series"][info["idx"]["test"]]
    res = {
        "model": name,
        "MAE": float(np.abs(pred - true).mean()),
        "RMSE": float(np.sqrt(((pred - true) ** 2).mean())),
        "params_M": n_params(model),
        "train_time_s": train_time,
        "history": hist,
    }
    if save:
        json.dump(res, open(OUT_DIR / f"{name}_results.json", "w"), indent=1)
    print(f"[{name}] test MAE {res['MAE']:.3f} °C | RMSE {res['RMSE']:.3f} °C")
    return res, pred, true


def baselines(info):
    """Hai mo hinh 'khong hoc': hom nay = hom qua, va trung binh 7 ngay gan nhat."""
    x, idx = info["series"], info["idx"]["test"]
    true = x[idx]
    out = {}
    for name, pred in {
        "persistence (hom qua)": x[idx - 1],
        "trung binh 7 ngay": np.stack([x[t - 7 : t].mean() for t in idx]),
    }.items():
        out[name] = {"MAE": float(np.abs(pred - true).mean()), "RMSE": float(np.sqrt(((pred - true) ** 2).mean()))}
    return out


def plot_forecast(info, preds, last=120):
    d = info["dates"][info["idx"]["test"]][-last:]
    plt.figure(figsize=(12, 3.8))
    plt.plot(d, info["series"][info["idx"]["test"]][-last:], "k", lw=1.5, label="thuc te")
    for name, p in preds.items():
        plt.plot(d, p[-last:], lw=1, label=name)
    plt.ylabel("°C")
    plt.title(f"Du bao {last} ngay cuoi cua tap test")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================================
# 2. VAN BAN (IMDB)
# =====================================================================
def load_imdb(seed=42):
    """train 25k -> 90/10 train/val (co phan tang); test 25k chinh thuc."""
    tr = pd.read_parquet(DATA_DIR / "imdb_train.parquet")
    te = pd.read_parquet(DATA_DIR / "imdb_test.parquet")
    i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=tr.label, random_state=seed)
    return tr.iloc[i_tr].reset_index(drop=True), tr.iloc[i_va].reset_index(drop=True), te


_TOKEN = re.compile(r"[a-z0-9']+|[!?]")


def tokenize(text):
    return _TOKEN.findall(text.lower().replace("<br />", " "))


class TextData:
    """Token hoa 1 lan, sau do ma hoa thanh tensor voi max_len tuy y (dung cho RNN/LSTM)."""

    def __init__(self, vocab_size=20000):
        self.tr, self.va, self.te = load_imdb()
        self.tok = {
            k: [tokenize(t) for t in df.text]
            for k, df in zip(("train", "val", "test"), (self.tr, self.va, self.te), strict=True)
        }
        cnt = Counter(w for toks in self.tok["train"] for w in toks)
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in cnt.most_common(vocab_size - 2)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.y = {
            k: torch.tensor(df.label.values)
            for k, df in zip(("train", "val", "test"), (self.tr, self.va, self.te), strict=True)
        }

    def encode_tokens(self, toks, max_len):
        ids = [self.stoi.get(w, 1) for w in toks[:max_len]] or [1]
        return ids

    def loaders(self, max_len=400, batch_size=64):
        out = {}
        for k, seqs in self.tok.items():
            enc = [self.encode_tokens(t, max_len) for t in seqs]
            lens = torch.tensor([len(e) for e in enc])
            ids = torch.zeros(len(enc), max_len, dtype=torch.long)
            for i, e in enumerate(enc):
                ids[i, : len(e)] = torch.tensor(e)
            out[k] = DataLoader(TensorDataset(ids, lens, self.y[k]), batch_size=batch_size, shuffle=(k == "train"))
        return out

    def length_stats(self):
        L = np.array([len(t) for t in self.tok["train"]])
        return {
            "mean": L.mean(),
            "median": float(np.median(L)),
            "p90": float(np.percentile(L, 90)),
            "max": int(L.max()),
        }


class RNNTextClassifier(nn.Module):
    """Embedding -> RNN/LSTM (pack_padded_sequence de bo qua phan padding) -> trang thai cuoi -> Linear."""

    def __init__(self, kind, vocab_size, emb=128, hidden=128, layers=1, bidir=False, dropout=0.3):
        super().__init__()
        self.kind, self.bidir = kind, bidir
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = cls(emb, hidden, layers, batch_first=True, bidirectional=bidir, dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), 2)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.kind == "lstm":
            h = h[0]  # (h_n, c_n) -> h_n
        h = torch.cat([h[-2], h[-1]], 1) if self.bidir else h[-1]
        return self.fc(self.drop(h))


class DistilBertClassifier(nn.Module):
    """Transformer pretrained (DistilBERT) + dau phan loai 2 lop. Fine-tune toan bo mang."""

    def __init__(self, name="distilbert/distilbert-base-uncased"):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    def forward(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask).logits


def bert_loaders(max_len=256, batch_size=16, name="distilbert/distilbert-base-uncased"):
    """Token hoa bang tokenizer cua DistilBERT (WordPiece); cache lai de lan sau chay nhanh."""
    from transformers import AutoTokenizer

    cache = DATA_DIR / f"bert_{max_len}.pt"
    if cache.exists():
        data = torch.load(cache)
    else:
        tok = AutoTokenizer.from_pretrained(name)
        tr, va, te = load_imdb()
        data = {}
        for k, df in zip(("train", "val", "test"), (tr, va, te), strict=True):
            enc = tok(
                [t.replace("<br />", " ") for t in df.text],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            data[k] = (enc["input_ids"], enc["attention_mask"], torch.tensor(df.label.values))
        torch.save(data, cache)
    return {
        k: DataLoader(TensorDataset(*v), batch_size=batch_size * (2 if k != "train" else 1), shuffle=(k == "train"))
        for k, v in data.items()
    }


def train_classifier(name, model, loaders, epochs=5, lr=2e-3, clip=1.0, verbose=True, amp=True):
    """Vong lap huan luyen chung: batch = (x, aux, y); aux la do dai (RNN/LSTM) hoac attention mask (Transformer)."""
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loaders["train"]), pct_start=0.1, anneal_strategy="linear"
    )
    use_amp = amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    crit = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best, t0, ckpt = 0.0, time.time(), OUT_DIR / f"{name}.pt"

    def run(loader, train):
        model.train(train)
        ls, ok, n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, aux, y in loader:
                x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
                with torch.autocast(DEVICE.type, enabled=use_amp):
                    out = model(x, aux)
                    loss = crit(out.float(), y)
                if train:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                ls += loss.item() * len(y)
                ok += (out.argmax(1) == y).sum().item()
                n += len(y)
        return ls / n, ok / n

    for ep in range(1, epochs + 1):
        tl, ta = run(loaders["train"], True)
        vl, va = run(loaders["val"], False)
        for k, v in zip(hist, (tl, ta, vl, va), strict=True):
            hist[k].append(v)
        if va > best:
            best = va
            torch.save(model.state_dict(), ckpt)
        if verbose:
            print(f"ep {ep:02d}/{epochs} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model, hist, time.time() - t0


@torch.no_grad()
def evaluate_classifier(name, model, loader, hist=None, train_time=None, save=True, plot=True, report=True):
    model.eval()
    ys, ps = [], []
    for x, aux, y in loader:
        ps.append(model(x.to(DEVICE), aux.to(DEVICE)).argmax(1).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    if report:
        print(classification_report(y, p, target_names=LABELS, digits=4))
    if plot:
        cm = confusion_matrix(y, p)
        plt.figure(figsize=(3.8, 3.4))
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black"
                )
        plt.xticks([0, 1], LABELS)
        plt.yticks([0, 1], LABELS)
        plt.xlabel("Du doan")
        plt.ylabel("Thuc te")
        plt.title(f"Confusion matrix - {name}")
        plt.tight_layout()
        plt.show()
    res = {
        "model": name,
        "test_acc": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "params_M": n_params(model),
        "train_time_s": train_time,
        "history": hist,
    }
    if save:
        json.dump(res, open(OUT_DIR / f"{name}_results.json", "w"), indent=1)
    print(f"[{name}] test accuracy {res['test_acc']:.4f} | macro F1 {res['macro_f1']:.4f}")
    return res


def length_stress_test(data, lengths=(50, 100, 200, 400), kinds=("rnn", "lstm"), epochs=6, lr=2e-3, seed=0):
    """Cung kien truc, cung du lieu, chi doi do dai toi da cua review -> xem RNN thuan va LSTM xuong cap the nao."""
    res = {k: {} for k in kinds}
    for L in lengths:
        loaders = data.loaders(L, 64)
        for kind in kinds:
            set_seed(seed)
            m = RNNTextClassifier(kind, len(data.itos))
            m, h, t = train_classifier(f"_stress_{kind}_{L}", m, loaders, epochs, lr, verbose=False)
            r = evaluate_classifier(
                f"_stress_{kind}_{L}", m, loaders["test"], h, t, save=False, plot=False, report=False
            )
            res[kind][L] = r["test_acc"]
    for f in OUT_DIR.glob("_stress_*.pt"):
        f.unlink()
    json.dump(
        {k: {str(L): v for L, v in d.items()} for k, d in res.items()},
        open(OUT_DIR / "length_stress.json", "w"),
        indent=1,
    )
    return res


def plot_stress(res):
    plt.figure(figsize=(6, 3.8))
    for k, d in res.items():
        plt.plot(list(d), list(d.values()), "o-", label=k.upper())
    plt.axhline(0.5, ls="--", c="gray", lw=0.8)
    plt.text(list(next(iter(res.values())))[0], 0.51, "doan ngau nhien", fontsize=8, color="gray")
    plt.xlabel("Do dai toi da cua review (so tu)")
    plt.ylabel("Test accuracy")
    plt.title("RNN thuan vs LSTM khi chuoi dai ra")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history(hist, keys=(("train_loss", "val_loss"), ("train_acc", "val_acc"))):
    keys = [k for k in keys if k[0] in hist]
    fig, ax = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 3.6), squeeze=False)
    for a, (tr, va) in zip(ax[0], keys, strict=True):
        a.plot(hist[tr], label="train")
        a.plot(hist[va], label="val")
        a.set_title(tr.split("_")[1])
        a.set_xlabel("epoch")
        a.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict_texts_rnn(model, data, texts, max_len=400):
    model.eval()
    for t in texts:
        ids = torch.tensor([data.encode_tokens(tokenize(t), max_len)])
        p = model(ids.to(DEVICE), torch.tensor([ids.shape[1]])).softmax(1)[0].cpu()
        print(f"{LABELS[p.argmax()]:8s} ({p.max():.2f}) | {t[:90]}")


@torch.no_grad()
def predict_texts_bert(model, texts, max_len=256, name="distilbert/distilbert-base-uncased"):
    from transformers import AutoTokenizer

    model.eval()
    tok = AutoTokenizer.from_pretrained(name)
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
        p = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)).softmax(1)[0].cpu()
        print(f"{LABELS[p.argmax()]:8s} ({p.max():.2f}) | {t[:90]}")
