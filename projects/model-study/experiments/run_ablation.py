"""Direction 2: does each change really help? One change at a time, several seeds, same test set.

python experiments/run_ablation.py                  # everything, one process
python experiments/run_ablation.py --shard 0/2      # first half of the jobs (use one GPU per shard)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import data, train  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
VARIANTS = {
    "baseline": {},
    "label_smoothing": {"label_smoothing": 0.1},
    "cutmix": {"cutmix": 0.5},
    "epochs20": {"epochs": 20},
    "size192": {"size": 192},
}
SEEDS = [0, 1, 2]


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    return int(i), int(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n: run jobs where index %% n == i")
    args = ap.parse_args()
    shard, nshards = parse_shard(args.shard)
    out = RESULTS / ("ablation.json" if nshards == 1 else f"ablation.shard{shard}.json")
    done = json.loads(out.read_text()) if out.exists() else {}
    jobs = [
        (seed, name) for seed in SEEDS for name in VARIANTS
    ]  # seed-major: an early stop keeps every variant's early seeds
    cache: dict[int, tuple] = {}
    for n, (seed, name) in enumerate(jobs):
        key = f"{name}/s{seed}"
        if n % nshards != shard or key in done:
            continue
        cfg = train.Config(name=name, seed=seed, **VARIANTS[name])
        if cfg.size not in cache:
            cache[cfg.size] = (
                data.load_waste(cfg.size),
                {"pets_val": data.load_pets(cfg.size, "trainval"), "pets_test": data.load_pets(cfg.size, "test")},
            )
        d, pets = cache[cfg.size]
        print(f"== {key}", flush=True)
        done[key] = train.run(cfg, d, RESULTS / "runs", pets)
        print(
            f"   test acc {done[key]['test_acc']:.4f} | val {done[key]['best_val_acc']:.4f} | {done[key]['train_time_s'] / 60:.1f} min",
            flush=True,
        )
        out.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
