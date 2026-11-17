"""Run all the heavy training on Kaggle (2 x T4): one process per GPU, then merge.

    python experiments/kaggle_pipeline.py [--skip-seg]

Stages: data download -> ablation -> students -> post-training compression -> segmentation ablation -> zip.
CPU latency, calibration/OOD analysis and Grad-CAM are run afterwards on a local machine (see README).
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
PY = sys.executable
WORK = Path(os.environ.get("WORK", "/kaggle/working"))
WASTE_URL = "https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes/resolve/main/garbage_classification.zip"


def gpus() -> int:
    import torch

    return max(1, torch.cuda.device_count())


def run_parallel(cmds: list[list[str]], tag: str) -> None:
    procs = []
    for i, cmd in enumerate(cmds):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
        log = open(RESULTS / f"{tag}.shard{i}.log", "w")  # noqa: SIM115
        procs.append((subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    for p, log in procs:
        code = p.wait()
        log.close()
        if code:
            raise SystemExit(f"{tag}: a shard failed (exit {code}); see results/{tag}.shard*.log")


def merge(pattern: str, target: str) -> None:
    merged: dict = {}
    for f in sorted(RESULTS.glob(pattern)):
        merged.update(json.loads(f.read_text()))
    (RESULTS / target).write_text(json.dumps(merged, indent=1))
    print(f"merged {pattern} -> {target}: {len(merged)} entries", flush=True)


def download_data() -> None:
    waste = WORK / "waste"
    if not (waste / "garbage_classification").exists():
        waste.mkdir(parents=True, exist_ok=True)
        print("downloading the waste dataset ...", flush=True)
        zp = waste / "g.zip"
        urllib.request.urlretrieve(WASTE_URL, zp)
        zipfile.ZipFile(zp).extractall(waste)
        zp.unlink()
    os.environ["WASTE_DATA"] = str(waste / "garbage_classification")
    from torchvision import datasets

    print("downloading Oxford-IIIT Pet ...", flush=True)
    for split in ("trainval", "test"):
        datasets.OxfordIIITPet(str(WORK / "pet"), split=split, target_types="segmentation", download=True)
    os.environ["PET_DATA"] = str(WORK / "pet" / "oxford-iiit-pet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seg", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    download_data()
    n = gpus()
    print(f"{n} GPU(s)", flush=True)
    ex = HERE / "experiments"

    run_parallel([[PY, str(ex / "run_ablation.py"), "--shard", f"{i}/{n}"] for i in range(n)], "ablation")
    merge("ablation.shard*.json", "ablation.json")
    run_parallel(
        [[PY, str(ex / "run_compression.py"), "students", "--shard", f"{i}/{n}"] for i in range(n)], "students"
    )
    subprocess.run([PY, str(ex / "run_compression.py"), "post"], cwd=HERE, check=True)
    if not args.skip_seg:
        run_parallel([[PY, str(ex / "run_seg_ablation.py"), "--shard", f"{i}/{n}"] for i in range(n)], "seg")
        merge("seg_ablation.shard*.json", "seg_ablation.json")

    out = WORK / "model-study-results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in RESULTS.rglob("*"):
            if f.is_file() and "seg_runs" not in f.parts:
                z.write(f, Path("results") / f.relative_to(RESULTS))
    print(f"done: {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
