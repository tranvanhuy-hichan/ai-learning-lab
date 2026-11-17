"""Compression helpers: pruning, ONNX export, int8 quantisation, CPU benchmarks and file sizes."""

from __future__ import annotations

import gzip
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from . import metrics
from .data import gpu_augment, normalize
from .models import build
from .train import DEVICE, logits_of


def file_mb(path: Path) -> float:
    return Path(path).stat().st_size / 1e6


def gzip_mb(path: Path) -> float:
    return len(gzip.compress(Path(path).read_bytes(), compresslevel=6)) / 1e6


# ------------------------------------------------------------------ pruning
def prunable(model: nn.Module):
    return [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d | nn.Linear)]


def sparsity(model: nn.Module) -> float:
    z = t = 0
    for m, _ in prunable(model):
        z += int((m.weight == 0).sum())
        t += m.weight.numel()
    return z / t


def prune_finetune(
    weights: Path, amount: float, data: dict, epochs: int = 2, lr: float = 1e-4, seed: int = 0
) -> nn.Module:
    """Global L1 unstructured pruning of all conv/linear weights, then a short fine-tune with the mask kept."""
    torch.manual_seed(seed)
    n_classes = len(data["classes"])
    model = build("efficientnet_b0", n_classes, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    prune.global_unstructured(prunable(model), pruning_method=prune.L1Unstructured, amount=amount)
    xtr, ytr = data["train"]
    counts = torch.bincount(ytr, minlength=n_classes).float()
    weight = (len(ytr) / (n_classes * counts)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = len(ytr) // 64
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(steps):
            idx = perm[i * 64 : (i + 1) * 64]
            xb, yb = gpu_augment(xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = F.cross_entropy(model(xb).float(), yb, weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for m, name in prunable(model):
        prune.remove(m, name)  # bake the mask into the weights
    return model.eval()


# ------------------------------------------------------------------ ONNX + int8
def export_onnx(model: nn.Module, size: int, path: Path) -> None:
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, dummy, str(path), **kwargs)


def quantize_int8(fp32: Path, int8: Path, calib_uint8: torch.Tensor) -> None:
    """Static post-training quantisation (QDQ, per-channel weights) calibrated on a few training images."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    batches = [normalize(calib_uint8[i : i + 1]).numpy() for i in range(len(calib_uint8))]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(batches)

        def get_next(self):
            b = next(self.it, None)
            return None if b is None else {"x": b}

    quantize_static(
        str(pre),
        str(int8),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    pre.unlink(missing_ok=True)


def ort_session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logits(path: Path, x_uint8: torch.Tensor, bs: int = 64) -> torch.Tensor:
    sess = ort_session(path)
    out = [sess.run(None, {"x": normalize(x_uint8[i : i + bs]).numpy()})[0] for i in range(0, len(x_uint8), bs)]
    return torch.from_numpy(np.concatenate(out))


# ------------------------------------------------------------------ benchmarks
def _time(fn, n: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def bench_torch(model: nn.Module, size: int, threads: int, n: int = 100) -> dict:
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.randn(1, 3, size, size)

    def fn():
        with torch.inference_mode():
            model(x)

    return _time(fn, n, 20)


def bench_ort(path: Path, size: int, threads: int, n: int = 100) -> dict:
    sess = ort_session(path, threads)
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    return _time(lambda: sess.run(None, {"x": x}), n, 20)


def accuracy_of(model: nn.Module, data: dict) -> float:
    return metrics.accuracy(logits_of(model.to(DEVICE), data["test"][0]), data["test"][1])
