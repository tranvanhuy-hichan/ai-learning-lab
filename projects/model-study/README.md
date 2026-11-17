# Model study

> Part of [AI Learning Lab](../../README.md). **Training output downloaded:** [results and limitations](results/REPORT.md). Waste ablation and students have 3 seeds each; segmentation uses the earlier recipe (2 seeds for A–D, 1 for E), with a corrected rerun still pending.

One model, four questions, each measured:

| Question | What is done | Script |
|---|---|---|
| Can it be **better**? | Label smoothing, CutMix, more epochs and larger images, one change at a time, 3 seeds each; plus a chain of single changes for the segmentation model | `experiments/run_ablation.py`, `run_seg_ablation.py` |
| Can it be **smaller and cheaper**? | Knowledge distillation, global pruning, ONNX export and int8 quantisation; accuracy against size and CPU latency | `run_compression.py` (CPU latency pending) |
| Can it tell when it **should not answer**? | Calibration (ECE, temperature scaling), accuracy on its most confident predictions, rejecting cat and dog photos as "not waste" | `run_reliability.py` |
| Can I see **what it looks at**? | Grad-CAM, a deletion test of how faithful the maps are, and a check for reliance on the background | `run_explain.py` |

Also: [`run_tfidf.py`](experiments/run_tfidf.py) answers an open question from the roadmap with a plain TF-IDF + linear model on IMDB (see [results](results/tfidf_imdb.json)).

## Run

Heavy training is meant for a GPU. On Kaggle (2 x T4, Internet on) open [`kaggle_study.ipynb`](kaggle_study.ipynb) and *Save & Run All*; it clones this repository, runs every stage on both GPUs and zips `results/`. Locally:

```bash
pip install -r requirements.txt
export WASTE_DATA=/path/to/garbage_classification PET_DATA=/path/to/oxford-iiit-pet
python experiments/run_ablation.py
```

Images are decoded once into a cache, and augmentation runs on the GPU, so one 10-epoch run takes about 2 to 4 minutes on a laptop RTX 3050.

## Tests

```bash
pip install -r requirements-dev.txt
pytest        # synthetic data: calibration, OOD scores, risk-coverage, splits, augmentation, pruning
```

## IMDB baseline (completed locally)

Vocabulary is fitted on training data only; C is selected on validation. The official IMDB test set
and 90/10 training/validation split (seed 42) match the sequence-modeling notebooks.

| Model | Test accuracy |
|---|---:|
| BiLSTM | 87.660% |
| TF-IDF (unigrams + bigrams) + Logistic Regression | 90.552% |
| TF-IDF (unigrams + bigrams) + Linear SVM | 90.356% |
| DistilBERT | 91.592% |

Logistic Regression exceeds BiLSTM by 2.892 percentage points and trails DistilBERT by 1.040 points.
These are single runs. Sources: `results/tfidf_imdb.json` and
`../sequence-modeling/outputs/{lstm,distilbert}_imdb_results.json`.

## Segmentation ablation

The script holds epochs (30), batch size (16), split and seed pairs (0, 1) fixed:
scratch U-Net → pretrained ResNet18 encoder → learning rate 2e-3 to 1e-3 → border weight 3 → 128 to 256 px.
Effects depend on earlier steps; resolution also changes evaluation resolution.
The earlier configuration changed epochs with learning rate and batch size with resolution,
so those results cannot isolate these effects. The completed Kaggle job used its earlier code snapshot;
these corrections apply to the next run. Resume skips only matching configurations.

Rebuild the report and web tables after importing results:

```bash
python experiments/summarize_results.py
```
