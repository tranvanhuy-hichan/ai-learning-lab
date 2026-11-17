"""Why was the ResNet18-encoder U-Net better than the from-scratch U-Net? A chain of single changes.

  A  U-Net from scratch                         (128 px, 30 epochs, lr 2e-3)
  B  A, but with a pretrained ResNet18 encoder  <- only the encoder changes
  C  B, with lr 1e-3 and 25 epochs              <- only the training recipe changes
  D  C, with border-class loss weight 3         <- only the loss weight changes
  E  D, at 256 px                               <- only the resolution changes

The steps are cumulative, so the effect of each change is measured on top of the previous ones (order matters).
All scored on the official 3,669-image test set at the resolution the model was trained at.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SEG = HERE.parent / "vision-toolkit" / "training" / "segmentation"
sys.path.insert(0, str(SEG))

import torch  # noqa: E402

import common as seg  # noqa: E402  (training helpers of the vision-toolkit)

RESULTS = HERE / "results"
CONFIGS = {
    "A_scratch": dict(model="unet", size=128, epochs=30, lr=2e-3, weights=None, batch=32),
    "B_pretrained_encoder": dict(model="resnet", size=128, epochs=30, lr=2e-3, weights=None, batch=32),
    "C_recipe": dict(model="resnet", size=128, epochs=25, lr=1e-3, weights=None, batch=32),
    "D_border_weight": dict(model="resnet", size=128, epochs=25, lr=1e-3, weights=[1.0, 1.0, 3.0], batch=32),
    "E_256px": dict(model="resnet", size=256, epochs=25, lr=1e-3, weights=[1.0, 1.0, 3.0], batch=16),
}
JOBS = [("E_256px", 0)] + [(n, s) for s in (0, 1) for n in CONFIGS if n != "E_256px"]  # longest first, then both seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    shard, nshards = (int(v) for v in args.shard.split("/"))
    pet = os.environ.get("PET_DATA")
    if pet:
        seg.DATA_DIR = Path(pet).parent  # torchvision root that contains oxford-iiit-pet/
    seg.OUT_DIR = RESULTS / "seg_runs"
    seg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("seg_ablation.json" if nshards == 1 else f"seg_ablation.shard{shard}.json")
    done = json.loads(out.read_text()) if out.exists() else {}
    for n, (name, seed) in enumerate(JOBS):
        key = f"{name}/s{seed}"
        if n % nshards != shard or key in done:
            continue
        cfg = CONFIGS[name]
        print("==", key, flush=True)
        torch.manual_seed(seed)
        loaders = seg.get_loaders(cfg["size"], cfg["batch"])
        model = seg.UNet(3, 32) if cfg["model"] == "unet" else seg.ResNetUNet(3, pretrained=True)
        model, hist, t = seg.train_model(f"{name}_s{seed}", model, loaders, cfg["epochs"], cfg["lr"], cfg["weights"])
        res = seg.evaluate(f"{name}_s{seed}", model, loaders["test"], hist, t)
        done[key] = {k: res[k] for k in ("pixel_acc", "mIoU", "mDice", "IoU", "Dice", "params_M", "train_time_s")} | cfg
        out.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
