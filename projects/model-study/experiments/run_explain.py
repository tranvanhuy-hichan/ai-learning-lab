"""Direction 4: does the model look at the right thing?

1. Which Grad-CAM layer is most faithful? Delete the pixels the map marks as most important and see how fast the
   predicted-class probability falls, compared with deleting the same amount guided by another image's map.
2. Does the classifier lean on the background? Keep only the border, or only the centre, of each photo.
3. Figures: one example per class, and a gallery of mistakes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision-toolkit" / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from study import data, plots  # noqa: E402
from study.models import build  # noqa: E402
from study.train import DEVICE, logits_of  # noqa: E402

from vision_toolkit.explain import gradcam, heat  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DEPLOYED = HERE.parent / "vision-toolkit" / "models" / "efficientnet_b0_waste.pt"
LAYERS = {3: "stride 8 (16x16)", 5: "stride 16 (8x8)", 8: "stride 32 (4x4)"}
FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
N = 400


def cams_for(model, xn: torch.Tensor, layer: int) -> torch.Tensor:
    out = []
    for i in range(0, len(xn), 50):
        cam, _ = gradcam(model, xn[i : i + 50].to(DEVICE), layer=layer)
        out.append(cam.cpu())
    return torch.cat(out)


@torch.no_grad()
def prob_of_class(model, xn: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    out = []
    for i in range(0, len(xn), 100):
        out.append(model(xn[i : i + 100].to(DEVICE)).float().softmax(1).cpu())
    p = torch.cat(out)
    return p.gather(1, cls[:, None])[:, 0]


def delete_top(xn: torch.Tensor, cam: torch.Tensor, frac: float) -> torch.Tensor:
    """Replace the `frac` most important pixels (by cam) with the dataset mean (0 after normalisation)."""
    if frac == 0:
        return xn
    k = int(frac * cam.shape[1] * cam.shape[2])
    thr = cam.flatten(1).kthvalue(cam.shape[1] * cam.shape[2] - k + 1, dim=1).values[:, None, None]
    mask = (cam >= thr)[:, None]
    return xn.masked_fill(mask, 0.0)


def main() -> None:
    plots.setup()
    d = data.load_waste(128)
    classes = d["classes"]
    xte, yte = d["test"]
    model = build("efficientnet_b0", len(classes), pretrained=False)
    model.load_state_dict(torch.load(DEPLOYED, map_location="cpu", weights_only=True))
    model = model.to(DEVICE).eval().to(memory_format=torch.channels_last)

    lg = logits_of(model, xte)
    pred = lg.argmax(1)
    correct = (pred == yte).nonzero()[:, 0]
    gen = torch.Generator().manual_seed(0)
    pick = correct[torch.randperm(len(correct), generator=gen)[:N]]
    xn = data.normalize(xte[pick])
    cls = yte[pick]
    base_p = prob_of_class(model, xn, cls)
    out: dict = {"n_images": int(len(pick)), "layers": {}}

    # ---- 1. faithfulness of each layer
    curves = {}
    for layer, name in LAYERS.items():
        cam = cams_for(model, xn, layer)
        own, other = [], []
        for f in FRACTIONS:
            own.append(float(prob_of_class(model, delete_top(xn, cam, f), cls).mean()))
            other.append(
                float(prob_of_class(model, delete_top(xn, cam.roll(1, 0), f), cls).mean())
            )  # another image's map
        auc = lambda c: float(np.trapezoid(c, FRACTIONS))  # noqa: E731
        out["layers"][str(layer)] = {
            "name": name,
            "own_map": own,
            "other_image_map": other,
            "deletion_auc_own": auc(own),
            "deletion_auc_other": auc(other),
            "faithfulness_gain": auc(other) - auc(own),
        }
        curves[layer] = (own, other)
        print(layer, name, "AUC own", round(auc(own), 4), "other", round(auc(other), 4), flush=True)
    best = max(out["layers"], key=lambda k: out["layers"][k]["faithfulness_gain"])
    out["best_layer"] = int(best)
    out["base_prob"] = float(base_p.mean())

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for a, (layer, (own, other)) in zip(ax, curves.items(), strict=True):
        a.plot(FRACTIONS, own, color=plots.BLUE, lw=2, label="delete what its own map marks")
        a.plot(FRACTIONS, other, color=plots.GRAY, lw=2, ls="--", label="delete what another image's map marks")
        a.set_title(f"features[{layer}]: {LAYERS[layer]}")
        a.set_xlabel("fraction of pixels deleted")
    ax[0].set_ylabel("probability of the true class")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "gradcam_deletion.png")
    plt.close(fig)

    # ---- 2. does it use the background? (accuracy on the 400 *correctly classified* images, so base = 100%)
    s = xn.shape[-1]
    side = int(round(s * np.sqrt(0.5)))
    lo = (s - side) // 2

    def keep(region: str, seed: int = 0) -> torch.Tensor:
        z = xn.clone()
        m = torch.zeros(s, s, dtype=torch.bool)
        if region == "centre_only":
            m[lo : lo + side, lo : lo + side] = True
        elif region == "border_only":
            m[:] = True
            m[lo : lo + side, lo : lo + side] = False
        else:  # control: a random square of the same area
            r = np.random.RandomState(seed)
            y0, x0 = r.randint(0, s - side + 1, 2)
            m[y0 : y0 + side, x0 : x0 + side] = True
        return z.masked_fill(~m[None, None], 0.0)

    ctx = {}
    for region in ("centre_only", "border_only"):
        ctx[region] = float((logits_of_norm(model, keep(region)).argmax(1) == cls).float().mean())
    ctx["random_square_mean"] = float(
        np.mean([(logits_of_norm(model, keep("random", sd)).argmax(1) == cls).float().mean().item() for sd in range(8)])
    )
    ctx["chance"] = 1 / len(classes)
    out["context"] = ctx
    print("context reliance", ctx, flush=True)

    # ---- 3. figures with the chosen layer
    layer = int(best)
    one = {}
    for i in correct.tolist():
        c = int(yte[i])
        one.setdefault(c, i)
        if len(one) == len(classes):
            break
    ids = [one[c] for c in sorted(one)]
    cam = cams_for(model, data.normalize(xte[ids]), layer)
    fig, ax = plt.subplots(4, 6, figsize=(10.5, 7.4))
    for k, i in enumerate(ids):
        r, c = (k // 6) * 2, k % 6
        img = xte[i].permute(1, 2, 0).numpy()
        ax[r][c].imshow(img)
        ax[r][c].set_title(classes[int(yte[i])], fontsize=9)
        ax[r + 1][c].imshow(img)
        ax[r + 1][c].imshow(heat(cam[k].numpy()), alpha=0.55)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "gradcam_examples.png")
    plt.close(fig)

    wrong = (pred != yte).nonzero()[:, 0][:12]
    cam = cams_for(model, data.normalize(xte[wrong]), layer)
    fig, ax = plt.subplots(4, 6, figsize=(10.5, 7.6))
    for k, i in enumerate(wrong.tolist()):
        r, c = (k // 6) * 2, k % 6
        img = xte[i].permute(1, 2, 0).numpy()
        ax[r][c].imshow(img)
        ax[r][c].set_title(f"true: {classes[int(yte[i])]}\npred: {classes[int(pred[i])]}", fontsize=8)
        ax[r + 1][c].imshow(img)
        ax[r + 1][c].imshow(heat(cam[k].numpy()), alpha=0.55)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "gradcam_errors.png")
    plt.close(fig)
    (RESULTS / "explain.json").write_text(json.dumps(out, indent=1))
    print("best layer", best)


@torch.no_grad()
def logits_of_norm(model, xn: torch.Tensor) -> torch.Tensor:
    return torch.cat([model(xn[i : i + 100].to(DEVICE)).float().cpu() for i in range(0, len(xn), 100)])


if __name__ == "__main__":
    main()
