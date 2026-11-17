"""Generate assets/segmentation_demo.png from the example images (run from the repo root)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from vision_toolkit import inference as inf  # noqa: E402

paths = sorted((ROOT / "examples").glob("pet_*.jpg"))
models = inf.available_segmenters()
fig, ax = plt.subplots(len(paths), 1 + len(models), figsize=(3.2 * (1 + len(models)), 2.6 * len(paths)), squeeze=False)
for r, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    ax[r][0].imshow(img)
    ax[r][0].set_title("Input" if r == 0 else "", fontsize=10)
    for c, name in enumerate(models, 1):
        overlay, _, _ = inf.segment_pet(img, name)
        ax[r][c].imshow(overlay)
        ax[r][c].set_title(name.split(" (")[0] if r == 0 else "", fontsize=10)
    for a in ax[r]:
        a.axis("off")
plt.tight_layout()
(ROOT / "assets").mkdir(exist_ok=True)
plt.savefig(ROOT / "assets" / "segmentation_demo.png", dpi=130)
print("saved assets/segmentation_demo.png with models:", models)
