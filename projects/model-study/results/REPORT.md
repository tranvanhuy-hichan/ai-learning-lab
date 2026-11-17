# Downloaded Kaggle results

Source: `hichandut/model-study`; downloaded after CLI reported COMPLETE on 2026-11-17 (Asia/Bangkok).
Raw files: [ablation](ablation.json), [compression](compression.json), [segmentation](seg_ablation.json).
Regenerate: `python experiments/summarize_results.py` from `projects/model-study`.

All spreads below are sample standard deviations across seeds, in percentage points; they are not confidence intervals.
Test data was not used to select the teacher: it is `epochs20`, chosen by mean validation accuracy among 128 px variants.

## Waste classification

| Variant | Test accuracy (%) | Seeds |
|---|---:|---:|
| baseline | 95.991 ± 0.237 | 3 |
| label_smoothing | 96.392 ± 0.114 | 3 |
| cutmix | 96.234 ± 0.286 | 3 |
| epochs20 | 96.306 ± 0.000 | 3 |
| size192 | 97.022 ± 0.151 | 3 |

192 px has the highest observed mean test accuracy; the gap from baseline is 1.031 percentage points.
These variants change one baseline setting each; they do not test combinations of changes.

## Small students

| Student | Test accuracy (%) | Seeds |
|---|---:|---:|
| cnn_scratch | 82.489 ± 0.674 | 3 |
| mnv3_ce | 94.058 ± 0.194 | 3 |
| cnn_kd | 87.829 ± 0.203 | 3 |
| mnv3_kd | 94.459 ± 0.129 | 3 |

## Post-training compression (one teacher / one run per method)

| Method | Test accuracy (%) | File MB | gzip MB |
|---|---:|---:|---:|
| onnx_fp32 | 96.306 | 16.082 | — |
| onnx_int8 | 64.175 | 4.902 | — |
| prune_30 | 96.091 | 16.390 | 11.920 |
| prune_50 | 96.306 | 16.390 | 9.352 |
| prune_70 | 95.232 | 16.390 | 6.573 |
| prune_90 | 46.134 | 16.390 | 3.384 |

An em dash means gzip size was not measured. ONNX INT8 loses substantial accuracy here; it should not replace the current model.
Pruning leaves dense tensor shapes and ordinary checkpoint sizes unchanged. Smaller gzip files do not establish faster inference.
CPU latency has not been measured for these downloaded models.

## Segmentation: historical recipe

| Step | mIoU (%) | Border IoU (%) | Seeds | Epochs | Batch |
|---|---:|---:|---:|---:|---:|
| A_scratch | 76.827 ± 0.163 | 55.220 ± 0.225 | 2 | 30 | 32 |
| B_pretrained_encoder | 77.832 ± 0.095 | 55.911 ± 0.200 | 2 | 30 | 32 |
| C_recipe | 78.541 ± 0.021 | 56.873 ± 0.017 | 2 | 25 | 32 |
| D_border_weight | 78.240 ± 0.075 | 57.471 ± 0.139 | 2 | 25 | 32 |
| E_256px | 80.632 (single run) | 61.337 (single run) | 1 | 25 | 16 |

A→B replaces the scratch encoder with pretrained ResNet18; model architecture and parameter count also change.
B→C changes both learning rate (0.002→0.001) and epochs (30→25), so this is a recipe comparison.
C→D changes border loss weight to 3: inspect both mIoU and border IoU rather than assuming improvement.
D→E changes resolution (128→256) and batch size (32→16); E has only seed 0. Evaluation resolution changes too.
The downloaded measurements must not be described as the corrected 30-epoch / batch-16 ablation now in the script.
A corrected rerun is still pending; no downloaded segmentation checkpoint has replaced the web model.

## IMDB (local results, one run)

| Model | Test accuracy (%) |
|---|---:|
| BiLSTM | 87.660 |
| TF-IDF + Logistic Regression | 90.552 |
| TF-IDF + Linear SVM | 90.356 |
| DistilBERT | 91.592 |
