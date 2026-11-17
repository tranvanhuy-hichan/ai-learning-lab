# Model study

> Part of [AI Learning Lab](../../README.md). **Work in progress:** the code and the TF-IDF result are in; the training results are being produced on Kaggle and will be added here.

One model, four questions, each measured:

| Question | What is done | Script |
|---|---|---|
| Can it be **better**? | Label smoothing, CutMix, more epochs and larger images, one change at a time, 3 seeds each; plus a chain of single changes for the segmentation model | `experiments/run_ablation.py`, `run_seg_ablation.py` |
| Can it be **smaller and cheaper**? | Knowledge distillation, global pruning, ONNX export and int8 quantisation; accuracy against size and CPU latency | `run_compression.py`, `run_latency.py` |
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
