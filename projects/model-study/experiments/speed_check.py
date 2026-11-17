"""Time two epochs of the baseline to size the experiment budget."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402

from study import data, train  # noqa: E402

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t = time.time()
    d = data.load_waste(size)
    print(f"load/cache {size}px: {time.time() - t:.0f}s", {k: len(v[1]) for k, v in d.items() if k != "classes"})
    cfg = train.Config(name="speed", size=size, epochs=2)
    r = train.run(cfg, d, Path("/tmp/study_speed"), verbose=True)
    print(f"{r['train_time_s'] / 2:.1f}s/epoch | test acc after 2 epochs {r['test_acc']:.4f}")
