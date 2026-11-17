"""Direction 1: make the model smaller and cheaper, and measure what it costs in accuracy.

  * knowledge distillation: a small student learns from the EfficientNet-B0 teacher's soft predictions
  * global magnitude pruning (+ short fine-tune)
  * ONNX export and static int8 quantisation

Latency is measured separately (run_latency.py) on an otherwise idle machine.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from study import compress, data, metrics, train  # noqa: E402
from study.models import build, n_params  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2]
SIZE = 128


def pick_teacher() -> tuple[str, Path]:
    """Best 128 px variant by mean *validation* accuracy (the test set plays no part in this choice)."""
    ab = json.loads((RESULTS / "ablation.json").read_text())
    by_variant: dict[str, list[float]] = {}
    for r in ab.values():
        if r["size"] == SIZE:
            by_variant.setdefault(r["name"], []).append(r["best_val_acc"])
    best = max(by_variant, key=lambda k: np.mean(by_variant[k]))
    return best, RESULTS / "runs" / "weights" / f"{best}_s0.pt"


def students_stage(shard: int, nshards: int) -> None:
    """Train the small students (with and without distillation). Shardable across GPUs."""
    out_file = RESULTS / ("compression.students.json" if nshards == 1 else f"compression.students.shard{shard}.json")
    done = json.loads(out_file.read_text()) if out_file.exists() else {}
    teacher_name, teacher = pick_teacher()
    print("teacher:", teacher_name, flush=True)
    d = data.load_waste(SIZE)
    pets = {"pets_val": data.load_pets(SIZE, "trainval"), "pets_test": data.load_pets(SIZE, "test")}
    students = {
        "cnn_scratch": dict(arch="simple_cnn", pretrained=False, epochs=20, lr=2e-3),
        "cnn_kd": dict(arch="simple_cnn", pretrained=False, epochs=20, lr=2e-3, teacher=str(teacher)),
        "mnv3_ce": dict(arch="mobilenet_v3_small", pretrained=True, epochs=10, lr=1e-3),
        "mnv3_kd": dict(arch="mobilenet_v3_small", pretrained=True, epochs=10, lr=1e-3, teacher=str(teacher)),
    }
    jobs = [(seed, name) for seed in SEEDS for name in students]
    for n, (seed, name) in enumerate(jobs):
        key = f"{name}/s{seed}"
        if n % nshards != shard or key in done:
            continue
        print("==", key, flush=True)
        done[key] = train.run(
            train.Config(name=name, seed=seed, size=SIZE, **students[name]), d, RESULTS / "runs", pets
        )
        print(f"   test acc {done[key]['test_acc']:.4f}", flush=True)
        out_file.write_text(json.dumps(done, indent=1))


def post_stage() -> None:
    """Compress the teacher after training: ONNX, static int8, global pruning (+ short fine-tune)."""
    out_file = RESULTS / "compression.json"
    out = json.loads(out_file.read_text()) if out_file.exists() else {}
    out.setdefault("post", {})
    out["students"] = {}
    for f in sorted(RESULTS.glob("compression.students*.json")):
        out["students"].update(json.loads(f.read_text()))
    teacher_name, teacher = pick_teacher()
    out["teacher"] = {"variant": teacher_name, "weights": str(teacher.relative_to(RESULTS.parent))}
    d = data.load_waste(SIZE)
    work = RESULTS / "compressed"
    work.mkdir(exist_ok=True)
    xte, yte = d["test"]
    model = build("efficientnet_b0", len(d["classes"]), pretrained=False)
    model.load_state_dict(torch.load(teacher, map_location="cpu", weights_only=True))
    fp32, int8 = work / "teacher_fp32.onnx", work / "teacher_int8.onnx"
    if "onnx_fp32" not in out["post"]:
        compress.export_onnx(model, SIZE, fp32)
        gen = torch.Generator().manual_seed(0)
        calib = d["train"][0][torch.randperm(len(d["train"][0]), generator=gen)[:200]]
        compress.quantize_int8(fp32, int8, calib)
        for key, path in (("onnx_fp32", fp32), ("onnx_int8", int8)):
            lg = compress.ort_logits(path, xte)
            out["post"][key] = {
                "test_acc": metrics.accuracy(lg, yte),
                "macro_f1": metrics.macro_f1(lg, yte),
                "size_mb": compress.file_mb(path),
                "path": str(path.relative_to(RESULTS.parent)),
            }
            print(key, out["post"][key], flush=True)
        out_file.write_text(json.dumps(out, indent=1))
    for amount in (0.3, 0.5, 0.7, 0.9):
        key = f"prune_{int(amount * 100)}"
        if key in out["post"]:
            continue
        pm = compress.prune_finetune(teacher, amount, d)
        path = work / f"{key}.pt"
        torch.save(pm.state_dict(), path)
        out["post"][key] = {
            "amount": amount, "sparsity": compress.sparsity(pm), "test_acc": compress.accuracy_of(pm, d),
            "size_mb": compress.file_mb(path), "gzip_mb": compress.gzip_mb(path), "path": str(path.relative_to(RESULTS.parent)),
        }  # fmt: skip
        print(key, out["post"][key], flush=True)
        out_file.write_text(json.dumps(out, indent=1))
    out["teacher"]["params"] = n_params(model)
    out["teacher"]["size_mb"] = compress.file_mb(teacher)
    out["teacher"]["gzip_mb"] = compress.gzip_mb(teacher)
    out_file.write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["students", "post"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    if args.stage == "students":
        i, n = args.shard.split("/")
        students_stage(int(i), int(n))
    else:
        post_stage()


if __name__ == "__main__":
    main()
