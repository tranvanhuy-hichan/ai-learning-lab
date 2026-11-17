import numpy as np
import torch
from study import data, metrics
from study.compress import prunable, sparsity
from study.models import build


def test_ece_is_near_zero_when_confidence_matches_accuracy():
    n = 1000
    logits = torch.zeros(n, 2)
    logits[:, 0] = 1.3863  # two classes: softmax -> p(class 0) = e^1.3863 / (e^1.3863 + 1) = 0.8
    y = torch.zeros(n, dtype=torch.long)
    y[800:] = 1  # 80% right
    assert metrics.reliability(metrics.softmax(logits), y)["ece"] < 0.01


def test_temperature_scaling_fixes_overconfidence():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 4, (2000,), generator=g)
    logits = torch.randn(2000, 4, generator=g) * 0.8
    logits[torch.arange(2000), y] += 1.2  # informative but noisy
    over = logits * 5  # make it overconfident
    T = metrics.fit_temperature(over, y)
    assert T > 2 and metrics.nll(over, y, T) < metrics.nll(over, y)


def test_ood_scores_separate_clearly_different_populations():
    out = metrics.ood_scores(torch.randn(500) + 5, torch.randn(500))
    assert out["auroc"] > 0.99 and out["fpr95"] < 0.05


def test_risk_coverage_accuracy_does_not_drop_when_keeping_confident_samples():
    g = torch.Generator().manual_seed(1)
    conf = torch.rand(1000, generator=g)
    correct = (torch.rand(1000, generator=g) < conf).float()  # confidence is informative
    rc = metrics.risk_coverage(conf, correct)
    assert rc[-1]["accuracy"] > rc[0]["accuracy"]


def test_split_is_stratified_disjoint_and_complete():
    labels = np.repeat(np.arange(4), 100)
    s = data.split_indices(labels)
    assert sum(len(v) for v in s.values()) == 400
    assert not (set(s["train"]) & set(s["val"])) and not (set(s["train"]) & set(s["test"]))
    assert all(np.bincount(labels[s["train"]]) == 70)


def test_gpu_augment_keeps_shape_and_is_finite():
    x = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)
    y = data.gpu_augment(x)
    assert y.shape == (4, 3, 32, 32) and torch.isfinite(y).all()


def test_global_pruning_reaches_the_requested_sparsity():
    import torch.nn.utils.prune as prune

    m = build("simple_cnn", 12, pretrained=False)
    prune.global_unstructured(prunable(m), pruning_method=prune.L1Unstructured, amount=0.6)
    for mod, name in prunable(m):
        prune.remove(mod, name)
    assert abs(sparsity(m) - 0.6) < 0.01
