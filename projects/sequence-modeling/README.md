# Sequence Modeling: RNN, LSTM and Transformer

[English](README.md) | [Tiếng Việt](README_vi.md)

> Part of [AI Learning Lab](../../README.md), a personal learning repository. Each model gets a task that shows what it is good at, and the notebooks say plainly where an experiment did **not** show what I expected.

| Notebook | Model | Task | Test result |
|---|---|---|---|
| [01_rnn_timeseries.ipynb](01_rnn_timeseries.ipynb) | RNN | Next-day minimum temperature in Melbourne from the last 30 days | MAE 1.73 °C (LSTM 1.71, "same as today" 2.04) |
| [02_lstm_text.ipynb](02_lstm_text.ipynb) | LSTM | IMDB review sentiment, plus RNN vs LSTM at several review lengths | BiLSTM 87.7% |
| [03_transformer_text.ipynb](03_transformer_text.ipynb) | Transformer | Same IMDB task with a fine-tuned DistilBERT | **91.6%** |
| [04_compare.ipynb](04_compare.ipynb) | All | Side-by-side tables and the RNN vs LSTM length plot | |

The notebooks are saved with their outputs, so you can read them without running anything. The notes behind them: [RNN and LSTM](../../learning/notes/05-rnn-lstm.md), [Transformer](../../learning/notes/06-transformer-attention.md).

## Run it

```bash
pip install -r requirements.txt
python scripts/download_data.py      # ~41 MB: Melbourne temperatures + IMDB (needs internet)
jupyter lab                          # run notebooks 01 to 04 in order
```

DistilBERT fine-tuning (notebook 03) takes about 10 minutes on a 4 GB GPU. The small result files are in `outputs/`; trained weights (`*.pt`) are not committed.

## Honest notes

- Every result is a single run with one seed.
- RNN vs LSTM on the temperature task differ by 0.02 °C, too small to call.
- The RNN vs LSTM gap on IMDB did **not** grow with review length, so that experiment does not prove the LSTM remembers far-away words.
- DistilBERT's gain mixes architecture and pretraining; a TF-IDF + SVM baseline is still missing.
- The IMDB models only understand English.

## Data

- Melbourne daily minimum temperatures, from the `jbrownlee/Datasets` GitHub repository.
- IMDB reviews, from the Hugging Face dataset `stanfordnlp/imdb`.

Check each dataset's license before reusing it; the data itself is not included here.
