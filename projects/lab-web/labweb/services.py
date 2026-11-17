"""Model loading and prediction for every tool in the web app. Models load lazily and are cached."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from vision_toolkit import inference as vision

from . import text as T
from .seq_models import RNNTextClassifier, SeqRegressor

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEVICE = vision.DEVICE
HF_NAME = "distilbert/distilbert-base-uncased"
_lock = threading.Lock()  # one inference at a time keeps memory predictable on small machines

# Vietnamese versions of the recycling tips (English ones live in vision_toolkit).
TIPS_VI = {
    "battery": "Rác thải nguy hại. Không bỏ vào thùng rác sinh hoạt; hãy mang đến điểm thu gom pin.",
    "biological": "Rác hữu cơ. Phù hợp để ủ phân hoặc bỏ vào thùng rác thực phẩm.",
    "brown-glass": "Thủy tinh. Rửa sạch và bỏ vào thùng tái chế thủy tinh (nâu hoặc hỗn hợp).",
    "cardboard": "Tái chế được. Gấp phẳng hộp và giữ khô trước khi tái chế.",
    "clothes": "Hàng dệt may. Quyên góp nếu còn mặc được, nếu không hãy dùng thùng thu gom quần áo.",
    "green-glass": "Thủy tinh. Rửa sạch và bỏ vào thùng tái chế thủy tinh (xanh hoặc hỗn hợp).",
    "metal": "Tái chế được. Rửa sạch lon và bỏ cùng rác kim loại hoặc rác tái chế hỗn hợp.",
    "paper": "Tái chế được. Giữ sạch và khô; giấy dính dầu mỡ thường không được nhận.",
    "plastic": "Thường tái chế được. Xem mã nhựa, rửa sạch và theo quy định địa phương.",
    "shoes": "Ưu tiên tái sử dụng: quyên góp nếu còn dùng được, nếu không hãy tìm điểm thu gom giày dép.",
    "trash": "Rác thải thông thường không tái chế được; bỏ vào thùng rác còn lại.",
    "white-glass": "Thủy tinh. Rửa sạch và bỏ vào thùng tái chế thủy tinh (trong suốt).",
}


# ------------------------------------------------------------------ vision
def classify_waste(image) -> dict:
    with _lock:
        probs = vision.predict_waste(image)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[0][0]
    return {
        "label": top,
        "top": [{"label": k, "prob": round(v, 4)} for k, v in ranked[:5]],
        "tip": {"en": vision.RECYCLING_TIPS[top], "vi": TIPS_VI[top]},
    }


def segment(image, model: str | None) -> dict:
    available = vision.available_segmenters()
    if not available:
        raise LookupError("no segmentation weights available")
    name = model if model in available else available[0]
    with _lock:
        _, mask, stats = vision.segment_pet(image, name)
    return {"model": name, "mask": mask, "stats": stats}


# ------------------------------------------------------------------ sentiment
@lru_cache(maxsize=1)
def _lstm():
    vocab = json.loads((MODELS / "vocab.json").read_text())
    m = RNNTextClassifier("lstm", len(vocab), emb=128, hidden=128, layers=1, bidir=True, dropout=0.4)
    m.load_state_dict(torch.load(MODELS / "lstm_imdb.pt", map_location="cpu", weights_only=True))
    return m.to(DEVICE).eval(), {w: i for i, w in enumerate(vocab)}


def distilbert_weights() -> Path | None:
    """Optional 268 MB checkpoint: models/distilbert_imdb.pt or $LABWEB_DISTILBERT_WEIGHTS."""
    for p in (os.environ.get("LABWEB_DISTILBERT_WEIGHTS"), MODELS / "distilbert_imdb.pt"):
        if p and Path(p).exists() and importlib.util.find_spec("transformers"):
            return Path(p)
    return None


@lru_cache(maxsize=1)
def _distilbert():
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    m = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(HF_NAME, num_labels=2))
    state = torch.load(distilbert_weights(), map_location="cpu", weights_only=True)
    m.load_state_dict({k.removeprefix("m."): v for k, v in state.items()})
    return m.to(DEVICE).eval(), AutoTokenizer.from_pretrained(HF_NAME)


def sentiment_models() -> list[str]:
    out = []
    if (MODELS / "lstm_imdb.pt").exists() and (MODELS / "vocab.json").exists():
        out.append("lstm")
    if distilbert_weights():
        out.append("distilbert")
    return out


@torch.no_grad()
def sentiment(text: str, model: str) -> dict:
    with _lock:
        if model == "distilbert":
            m, tok = _distilbert()
            enc = tok(text, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            p = m(**enc).logits.softmax(1)[0].cpu()
            used, truncated = int(enc["input_ids"].shape[1]), bool(len(tok.tokenize(text)) + 2 > 256)
        else:
            m, stoi = _lstm()
            tokens = T.tokenize(text)
            ids = torch.tensor([T.encode(tokens, stoi, 400)])
            p = m(ids.to(DEVICE), torch.tensor([ids.shape[1]])).softmax(1)[0].cpu()
            used, truncated = int(ids.shape[1]), len(tokens) > 400
    return {
        "model": model,
        "label": "positive" if p[1] >= p[0] else "negative",
        "probs": {"negative": round(float(p[0]), 4), "positive": round(float(p[1]), 4)},
        "tokens_used": used,
        "truncated": truncated,
    }


# ------------------------------------------------------------------ forecast
@lru_cache(maxsize=1)
def _forecasters():
    out = {}
    for kind in ("rnn", "lstm"):
        m = SeqRegressor(kind, hidden=64, layers=2)
        m.load_state_dict(torch.load(MODELS / f"{kind}_temp.pt", map_location="cpu", weights_only=True))
        out[kind] = m.eval()
    return out


def forecast_available() -> bool:
    return all((MODELS / f).exists() for f in ("rnn_temp.pt", "lstm_temp.pt", "temp_stats.json"))


@torch.no_grad()
def forecast(values: list[float]) -> dict:
    stats = json.loads((MODELS / "temp_stats.json").read_text())
    x = torch.tensor(((np.asarray(values, dtype=np.float32) - stats["mean"]) / stats["std"])[None, :, None])
    with _lock:
        preds = {k: float(m(x).item()) * stats["std"] + stats["mean"] for k, m in _forecasters().items()}
    preds["persistence"] = float(values[-1])
    preds["mean7"] = float(np.mean(values[-7:]))
    return {k: round(v, 2) for k, v in preds.items()}


def forecast_samples() -> list[dict]:
    return json.loads((MODELS / "forecast_samples.json").read_text())


def metrics() -> dict:
    p = MODELS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}
