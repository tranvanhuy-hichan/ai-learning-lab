"""Open question from the roadmap: how good is a plain TF-IDF + linear model on IMDB?

DistilBERT scored 91.6% and the BiLSTM 87.7%; without this baseline neither number says much.
Uses the same train/validation split as the sequence-modeling notebooks (seed 42, 90/10).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parents[1]
DATA = HERE.parent / "sequence-modeling" / "data"


def main() -> None:
    tr = pd.read_parquet(DATA / "imdb_train.parquet")
    te = pd.read_parquet(DATA / "imdb_test.parquet")
    i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=tr.label, random_state=42)
    clean = lambda s: s.str.replace("<br />", " ", regex=False)  # noqa: E731
    xtr, ytr = clean(tr.text.iloc[i_tr]), tr.label.iloc[i_tr].values
    xva, yva = clean(tr.text.iloc[i_va]), tr.label.iloc[i_va].values
    xte, yte = clean(te.text), te.label.values

    out = {}
    for ngrams in ((1, 1), (1, 2)):
        t0 = time.time()
        vec = TfidfVectorizer(ngram_range=ngrams, min_df=2, sublinear_tf=True, max_features=400_000)
        a, b, c = vec.fit_transform(xtr), vec.transform(xva), vec.transform(xte)  # vocabulary fitted on train only
        for name, make, grid in (
            ("logreg", lambda C: LogisticRegression(C=C, max_iter=2000), [1, 4, 16, 64, 256]),
            ("linear_svm", lambda C: LinearSVC(C=C), [0.05, 0.2, 0.8, 3.2, 12.8]),
        ):
            scored = []
            for C in grid:  # choose C on the validation split, never on the test set
                m = make(C).fit(a, ytr)
                scored.append((float((m.predict(b) == yva).mean()), C, m))
            val, C, m = max(scored, key=lambda s: s[0])
            key = f"{name}_{ngrams[0]}-{ngrams[1]}gram"
            out[key] = {
                "val_acc": val,
                "test_acc": float((m.predict(c) == yte).mean()),
                "C": C,
                "features": a.shape[1],
                "seconds": time.time() - t0,
            }
            print(key, out[key], flush=True)
    (HERE / "results" / "tfidf_imdb.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
