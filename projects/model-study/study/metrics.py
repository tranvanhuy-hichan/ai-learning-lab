"""Accuracy, calibration, selective prediction and out-of-distribution metrics (all from saved logits)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def softmax(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits / T, dim=1)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def macro_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(f1_score(y.numpy(), logits.argmax(1).numpy(), average="macro"))


def nll(logits: torch.Tensor, y: torch.Tensor, T: float = 1.0) -> float:
    return torch.nn.functional.cross_entropy(logits / T, y).item()


def reliability(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> dict:
    """Expected calibration error with equal-width confidence bins, plus the per-bin numbers for a diagram."""
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean()).item()
            ece += m.float().mean().item() * gap
            bins.append(
                {
                    "lo": lo.item(),
                    "hi": hi.item(),
                    "n": int(m.sum()),
                    "conf": conf[m].mean().item(),
                    "acc": correct[m].mean().item(),
                }
            )
    return {"ece": ece, "bins": bins}


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single scalar T minimising NLL on held-out validation logits (temperature scaling)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def max_softmax(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return softmax(logits, T).max(1).values


def energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Negative free energy: higher means more in-distribution (Liu et al. 2020)."""
    return T * torch.logsumexp(logits / T, dim=1)


def ood_scores(id_score: torch.Tensor, ood_score: torch.Tensor) -> dict:
    """Both inputs: higher = more in-distribution. AUROC and false-positive rate at 95% true-positive rate."""
    a, b = id_score.numpy(), ood_score.numpy()
    auroc = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
    thr = float(np.percentile(a, 5))  # keeps 95% of in-distribution samples
    return {"auroc": auroc, "fpr95": float((b >= thr).mean()), "threshold": thr}


def risk_coverage(conf: torch.Tensor, correct: torch.Tensor, coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.5)) -> list[dict]:
    """Accuracy on the most confident fraction of the data (selective prediction)."""
    order = conf.argsort(descending=True)
    c = correct[order].float()
    out = []
    for cov in coverages:
        k = max(1, int(round(cov * len(c))))
        out.append({"coverage": cov, "accuracy": c[:k].mean().item()})
    return out


def mean_std(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
        "values": [float(v) for v in a],
    }
