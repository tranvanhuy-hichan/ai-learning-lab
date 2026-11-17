"""Build the small files the web app needs from the sequence-modeling project.

    python scripts/export_assets.py --weights-dir ../sequence-modeling/outputs

Reads:   sequence-modeling/data (IMDB train + temperature CSV) and trained *.pt in --weights-dir
Writes:  models/vocab.json, temp_stats.json, forecast_samples.json, metrics.json and copies the small weights.
Run `python ../sequence-modeling/scripts/download_data.py` first if data/ is missing.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SEQ = HERE.parent / "sequence-modeling"
sys.path.insert(0, str(SEQ))

import common as C  # noqa: E402  (the sequence-modeling helpers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", type=Path, default=SEQ / "outputs")
    args = ap.parse_args()
    out = HERE / "models"
    out.mkdir(exist_ok=True)

    for name in ("lstm_imdb.pt", "rnn_temp.pt", "lstm_temp.pt"):
        shutil.copy(args.weights_dir / name, out / name)

    data = C.TextData(vocab_size=20000)
    (out / "vocab.json").write_text(json.dumps(data.itos))

    _, info = C.load_temperature(window=30)
    (out / "temp_stats.json").write_text(json.dumps({"mean": info["mean"], "std": info["std"], "window": 30}))

    idx = info["idx"]["test"]
    picks = [idx[int(i)] for i in [len(idx) * k // 6 + 10 for k in range(6)]]
    samples = []
    for t in picks:
        samples.append(
            {
                "id": str(info["dates"][t])[:10],
                "start": str(info["dates"][t - 30])[:10],
                "end": str(info["dates"][t - 1])[:10],
                "values": [round(float(v), 1) for v in info["series"][t - 30 : t]],
                "actual": round(float(info["series"][t]), 1),
            }
        )
    (out / "forecast_samples.json").write_text(json.dumps(samples, indent=1))

    seq = {
        n: json.loads((args.weights_dir / f"{n}_results.json").read_text())
        for n in ("rnn_temp", "lstm_temp", "lstm_imdb", "distilbert_imdb")
    }
    metrics = {
        # vision numbers: projects/vision-toolkit/models/MODEL_CARD.md (single runs, held-out test sets)
        "waste": {"accuracy": 0.9613, "macro_f1": 0.953, "params_m": 4.02},
        "unet": {"miou": 0.767, "border_iou": 0.548, "params_m": 7.76},
        "resnet_unet": {"miou": 0.809, "border_iou": 0.617, "params_m": 14.34},
        "lstm": {
            "accuracy": round(seq["lstm_imdb"]["test_acc"], 4),
            "params_m": round(seq["lstm_imdb"]["params_M"], 2),
        },
        "distilbert": {
            "accuracy": round(seq["distilbert_imdb"]["test_acc"], 4),
            "params_m": round(seq["distilbert_imdb"]["params_M"], 1),
        },
        "rnn_temp": {"mae": round(seq["rnn_temp"]["MAE"], 3), "params_m": round(seq["rnn_temp"]["params_M"], 4)},
        "lstm_temp": {"mae": round(seq["lstm_temp"]["MAE"], 3), "params_m": round(seq["lstm_temp"]["params_M"], 4)},
        "persistence_mae": round(C.baselines(info)["persistence (hom qua)"]["MAE"], 3),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("exported:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
