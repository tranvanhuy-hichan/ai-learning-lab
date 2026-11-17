# AI Vision Toolkit

[English](README.md) | [Tiếng Việt](README_vi.md)

[![CI](https://github.com/tranvanhuy-hichan/ai-vision-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/tranvanhuy-hichan/ai-vision-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

A small computer-vision web app that combines two deep-learning models trained from scratch / fine-tuned in PyTorch:

| Tab | Task | Model | Data |
|---|---|---|---|
| **Waste classifier** | Which of 12 waste types is in the photo, plus a recycling tip | EfficientNet-B0 (ImageNet-pretrained, fine-tuned) | 15,515 images, 12 classes |
| **Pet segmentation** | Label every pixel of a cat/dog photo as pet / background / border | U-Net (written from scratch); optional ResNet18-encoder U-Net | Oxford-IIIT Pet |

![Segmentation demo](assets/segmentation_demo.png)

*Left: input. Middle: U-Net from scratch. Right: ResNet18-encoder U-Net (red = pet, blue = background, yellow = border). In the first photo the from-scratch U-Net misses most of the cat's dark body while the ResNet version finds it; in the second photo the ResNet version adds a spurious border region on the right.*

## Quick start

```bash
git clone <this-repo> && cd ai-vision-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # opens http://127.0.0.1:7860
```

Trained weights for all three models are included in `models/` (about 105 MB in total; the largest file is 57 MB). A GPU is not required.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
make test    # pytest
make lint    # ruff check + format check
```

See [CONTRIBUTING.md](CONTRIBUTING.md), the [model card](models/MODEL_CARD.md) and the [changelog](CHANGELOG.md).

## Results (held-out test sets)

| Model | Metric | Result |
|---|---|---|
| EfficientNet-B0, waste (12 classes, 128 px, 10 epochs) | Accuracy / macro F1 | **96.1%** / 0.953 |
| U-Net from scratch, pets (128 px, 30 epochs) | mIoU (pixel accuracy) | **0.767** (91.3%) |
| ResNet18-encoder U-Net, pets (256 px, 25 epochs) | mIoU (pixel accuracy) | **0.809** (93.1%) |

Notes on how to read these numbers:

- **Waste:** the app reproduces the training-notebook accuracy exactly on the same 2,328-image test split (96.13%). The same notebooks also train a plain CNN and a ResNet18 for comparison (`training/classification/04_compare.ipynb`).
- **Segmentation:** mIoU for the from-scratch U-Net is measured on the 3,669-image official test set resized to 128 px. Scored against the original-resolution masks on 400 random test images, the from-scratch U-Net gives 0.733 and the ResNet18-encoder U-Net gives 0.772 (same images, same scoring), because thin borders are hard to match at full resolution. The gap between the two models is similar in both measurements.
- **The `border` class is the weakest** (IoU 0.548 for the from-scratch U-Net, 0.617 for the ResNet18-encoder version) since it is only a few pixels wide.
- The ResNet18-encoder model differs from the from-scratch U-Net in several ways at once (pretrained encoder, 256 px input, border-class weight, epochs), so its gain cannot be attributed to a single change.
- Everything is a single training run with one seed.

### Segmentation model choice

The app lists every segmentation model whose weights are present in `models/`; pick one in the dropdown. The ResNet18-encoder weights (`resnet_unet_pet.pt`, 57 MB) were trained with `training/segmentation/02_resnet_unet_colab_kaggle.ipynb` (about 11 minutes on a Kaggle T4). GitHub warns about files over 50 MB; if you prefer, host that file as a release asset instead of committing it.

## Limitations

- **Waste classifier:** the training photos mostly show a single item on a plain background. Accuracy on cluttered, real-world photos will be lower than 96% and has not been measured. It always picks one of the 12 classes, even for objects outside them.
- **Segmentation:** trained only on cats and dogs from Oxford-IIIT Pet; it is not meant for other animals or objects.
- Recycling tips are general guidance; rules differ by city.

## Project structure

```
app.py                      Gradio app (two tabs)
src/vision_toolkit/
    models.py               EfficientNet head, U-Net, ResNet-U-Net
    inference.py            pre-processing, prediction, overlay
models/                     trained weights, class labels, MODEL_CARD.md
examples/                   sample pet photos
tests/                      pytest unit tests
scripts/make_demo_figure.py regenerates assets/segmentation_demo.png
training/
    classification/         notebooks: CNN vs ResNet18 vs EfficientNet-B0
    segmentation/           notebooks: U-Net and ResNet18-encoder U-Net
```

The training notebooks expect the datasets in a `data/` folder next to each `common.py` (not included, see below).

## Data and credits

- **Oxford-IIIT Pet** (Parkhi et al., 2012), licensed CC BY-SA 4.0. The four example photos in `examples/` come from this dataset.
- **Garbage Classification, 12 classes**: obtained from the Hugging Face dataset `UdaraChamidu/Garbage-Classification-with-12-classes`. Its images are not redistributed here; check the original license before reusing them.
- Pretrained backbones (EfficientNet-B0, ResNet18) come from torchvision (ImageNet weights).

## License

Code: MIT, see [LICENSE](LICENSE). Datasets keep their own licenses.
