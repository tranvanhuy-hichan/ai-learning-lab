"""Download the two small public datasets used by the notebooks into ./data.

    python scripts/download_data.py            # into ../data (next to common.py)
    python scripts/download_data.py --dest /tmp/seq_data

Datasets
  * Melbourne daily minimum temperatures (1981-1990), 3,650 rows, from the jbrownlee/Datasets GitHub repo.
  * IMDB movie reviews (25k train / 25k test), from the Hugging Face dataset `stanfordnlp/imdb`.
Check each dataset's own license before reusing it elsewhere.
"""

import argparse
import urllib.request
from pathlib import Path

FILES = {
    "temp.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    "imdb_train.parquet": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet",
    "imdb_test.parquet": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = args.dest / name
        if target.exists():
            print(f"skip {name} (already there)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, target)
        print(f"  saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
