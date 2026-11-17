"""Direction 3: can the classifier tell when it should not answer?

  1. calibration: expected calibration error, before and after temperature scaling
  2. selective prediction: accuracy on the most confident fraction of the data
  3. out-of-distribution detection: cat/dog photos should be rejected as "not waste"

All numbers come from the logits saved by the training runs. Thresholds are chosen on validation data only.
Also fits the calibration of the *deployed* model (vision-toolkit/models/efficientnet_b0_waste.pt).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from study import data, metrics, plots  # noqa: E402
from study.models import build  # noqa: E402
from study.train import DEVICE, logits_of  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DEPLOYED = HERE.parent / "vision-toolkit" / "models"


def load_run(tag: str) -> dict[str, torch.Tensor]:
    z = np.load(RESULTS / "runs" / f"{tag}.npz")
    return {k: torch.from_numpy(z[k]) for k in z.files}


def analyse(z: dict, name: str = "") -> dict:
    """Everything reliability-related for one set of logits."""
    y, yv = z["y_test"], z["y_val"]
    T = metrics.fit_temperature(z["val"], yv)
    raw = metrics.reliability(metrics.softmax(z["test"]), y)
    cal = metrics.reliability(metrics.softmax(z["test"], T), y)
    scores = {
        "msp": lambda lg: metrics.max_softmax(lg),
        "max_logit": lambda lg: lg.max(1).values,
        "energy": lambda lg: metrics.energy(lg),
    }
    ood = {k: metrics.ood_scores(f(z["test"]), f(z["pets_test"])) for k, f in scores.items()}
    ood_val = {k: metrics.ood_scores(f(z["val"]), f(z["pets_val"]))["auroc"] for k, f in scores.items()}
    return {
        "accuracy": metrics.accuracy(z["test"], y),
        "temperature": T,
        "ece_raw": raw["ece"], "ece_calibrated": cal["ece"],
        "nll_raw": metrics.nll(z["test"], y), "nll_calibrated": metrics.nll(z["test"], y, T),
        "bins_raw": raw["bins"], "bins_calibrated": cal["bins"],
        "ood_test": ood, "ood_val_auroc": ood_val,
    }  # fmt: skip


def ensemble(zs: list[dict]) -> dict:
    """Average the logits of several seeds."""
    return {k: (torch.stack([z[k] for z in zs]).mean(0) if k not in ("y_val", "y_test") else zs[0][k]) for k in zs[0]}


def deployed_logits() -> dict:
    d = data.load_waste(128)
    model = build("efficientnet_b0", len(d["classes"]), pretrained=False)
    model.load_state_dict(torch.load(DEPLOYED / "efficientnet_b0_waste.pt", map_location="cpu", weights_only=True))
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    z = {
        "val": logits_of(model, d["val"][0]),
        "test": logits_of(model, d["test"][0]),
        "y_val": d["val"][1],
        "y_test": d["test"][1],
    }
    z["pets_val"] = logits_of(model, data.load_pets(128, "trainval"))
    z["pets_test"] = logits_of(model, data.load_pets(128, "test"))
    return z


def figures(z: dict, dep: dict) -> None:
    T = dep["temperature"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    for a, key, title in zip(
        ax, ("bins_raw", "bins_calibrated"), ("Before temperature scaling", f"After (T = {T:.2f})"), strict=True
    ):
        b = dep[key]
        a.plot([0, 1], [0, 1], color=plots.GRAY, lw=1, ls="--")
        a.bar([x["conf"] for x in b], [x["acc"] for x in b], width=0.05, color=plots.BLUE, alpha=0.85)
        a.set_xlim(0.2, 1.0)
        a.set_ylim(0.2, 1.0)
        a.set_xlabel("confidence")
        a.set_title(f"{title}\nECE {dep['ece_raw' if key == 'bins_raw' else 'ece_calibrated']:.3f}")
    ax[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "reliability.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, (name, f) in zip(ax, (("Max softmax", metrics.max_softmax), ("Energy", metrics.energy)), strict=True):
        a.hist(f(z["test"]).numpy(), bins=50, alpha=0.8, color=plots.BLUE, label="waste (test)", density=True)
        a.hist(f(z["pets_test"]).numpy(), bins=50, alpha=0.7, color=plots.ORANGE, label="cats and dogs", density=True)
        a.set_title(f"{name} score (right = looks like waste)")
        a.set_yticks([])
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "ood.png")
    plt.close(fig)

    conf = metrics.max_softmax(z["test"], T)
    correct = (z["test"].argmax(1) == z["y_test"]).float()
    cov = np.linspace(0.3, 1.0, 36)
    order = conf.argsort(descending=True)
    acc = [correct[order][: max(1, int(c * len(correct)))].mean().item() for c in cov]
    fig, a = plt.subplots(figsize=(5.2, 3.6))
    a.plot(cov * 100, np.array(acc) * 100, color=plots.BLUE, lw=2)
    a.set_xlabel("share of photos answered (%)")
    a.set_ylabel("accuracy on those (%)")
    a.set_title("Answering only when confident")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "risk_coverage.png")
    plt.close(fig)


def main() -> None:
    plots.setup()
    out: dict = {}

    # ---- the deployed model: temperature and decision thresholds used by the web app
    z = deployed_logits()
    dep = analyse(z)
    T = dep["temperature"]
    methods = {"msp": metrics.max_softmax, "max_logit": lambda lg: lg.max(1).values, "energy": metrics.energy}
    best = max(dep["ood_val_auroc"], key=dep["ood_val_auroc"].get)  # chosen on validation data only
    f = methods[best]
    thr = float(np.percentile(f(z["val"]).numpy(), 5))  # keeps 95% of real waste in validation
    cv = metrics.max_softmax(z["val"], T)
    okv = (z["val"].argmax(1) == z["y_val"]).float()
    tau = 0.0  # smallest calibrated confidence whose accepted predictions reach 98% accuracy on validation
    for cand in np.linspace(0.3, 0.99, 70):
        sel = cv >= cand
        if sel.sum() > 0.5 * len(cv) and okv[sel].mean() >= 0.98:
            tau = float(cand)
            break
    ct = metrics.max_softmax(z["test"], T)
    okt = (z["test"].argmax(1) == z["y_test"]).float()
    calib = {
        "model": "efficientnet_b0_waste.pt",
        "temperature": T,
        "ood": {
            "method": best, "threshold": thr, "kept_waste_val": 0.95,
            "waste_rejected_test": float((f(z["test"]) < thr).float().mean()),
            "pets_rejected_test": float((f(z["pets_test"]) < thr).float().mean()),
            "auroc_test": dep["ood_test"][best]["auroc"],
        },
        "uncertain": {"threshold": tau, "accuracy_when_accepted_test": float(okt[ct >= tau].mean()), "coverage_test": float((ct >= tau).float().mean())},
        "ece_raw": dep["ece_raw"], "ece_calibrated": dep["ece_calibrated"], "nll_raw": dep["nll_raw"], "nll_calibrated": dep["nll_calibrated"],
        "accuracy": dep["accuracy"],
        "risk_coverage": metrics.risk_coverage(ct, okt),
        "ood_all_methods_test": dep["ood_test"],
    }  # fmt: skip
    (DEPLOYED / "waste_calibration.json").write_text(json.dumps(calib, indent=1))
    out["deployed"] = calib
    figures(z, dep)
    print(json.dumps({k: v for k, v in calib.items() if k not in ("ood_all_methods_test",)}, indent=1), flush=True)

    # ---- every training variant and the seed ensemble (needs the ablation runs)
    ab_file = RESULTS / "ablation.json"
    if ab_file.exists():
        by_variant: dict[str, list[dict]] = {}
        for key in json.loads(ab_file.read_text()):
            name, seed = key.split("/")
            by_variant.setdefault(name, []).append(load_run(f"{name}_{seed}"))
        out["variants"], out["ensemble"] = {}, {}
        for name, zs in by_variant.items():
            per_seed = [analyse(v) for v in zs]
            agg = {
                m: metrics.mean_std([r[m] for r in per_seed])
                for m in ("accuracy", "ece_raw", "ece_calibrated", "nll_raw", "nll_calibrated", "temperature")
            }
            for meth in ("msp", "max_logit", "energy"):
                agg[f"ood_auroc_{meth}"] = metrics.mean_std([r["ood_test"][meth]["auroc"] for r in per_seed])
                agg[f"ood_fpr95_{meth}"] = metrics.mean_std([r["ood_test"][meth]["fpr95"] for r in per_seed])
            out["variants"][name] = agg
            if len(zs) > 1:
                ens = analyse(ensemble(zs))
                out["ensemble"][name] = {k: v for k, v in ens.items() if not k.startswith("bins")}
    (RESULTS / "reliability.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
