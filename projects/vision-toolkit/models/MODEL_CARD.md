# Model card

Three models are shipped in this folder. All numbers come from single training runs (one seed) on held-out test sets.

## `efficientnet_b0_waste.pt` — waste classifier

| | |
|---|---|
| Task | Single-label image classification, 12 classes (`waste_labels.json`) |
| Architecture | torchvision EfficientNet-B0, new 12-way head, 4.02 M parameters |
| Initialisation | ImageNet weights, all layers fine-tuned |
| Input | RGB, resized to 128x128, ImageNet mean/std normalisation |
| Training data | 15,515 images, stratified 70/15/15 split (seed 42); class-weighted cross-entropy |
| Training | AdamW, OneCycleLR (max lr 1e-3), batch 64, 10 epochs, mixed precision |
| Test result | **accuracy 96.13%, macro F1 0.953** (2,328 images) |

Weakest classes: plastic (F1 0.911), metal (0.918), white-glass (0.926).

## `unet_pet.pt` — U-Net from scratch

| | |
|---|---|
| Task | Semantic segmentation: pet / background / border (Oxford-IIIT trimap, labels shifted to 0..2) |
| Architecture | U-Net, base width 32, 7.76 M parameters, no pretrained weights |
| Input | RGB, 128x128, ImageNet mean/std |
| Training | 3,312 train / 368 val images; cross-entropy + Dice, AdamW, OneCycleLR (max lr 2e-3), batch 32, 30 epochs |
| Test result | **mIoU 0.767, mDice 0.859, pixel accuracy 91.3%** (3,669 official test images at 128 px) |
| Per class IoU | pet 0.839, background 0.914, border 0.548 |

Scored at original resolution on 400 random test images the same model gives mIoU 0.733.

## `resnet_unet_pet.pt` — ResNet18-encoder U-Net

| | |
|---|---|
| Architecture | ResNet18 encoder (ImageNet-pretrained at training time) + hand-written decoder with skip connections, 14.34 M parameters |
| Input | RGB, 256x256, ImageNet mean/std |
| Training | Same data split; cross-entropy (class weights pet 1, background 1, border 3) + Dice, AdamW, OneCycleLR (max lr 1e-3), batch 16, 25 epochs (Kaggle T4, about 11 minutes) |
| Test result | **mIoU 0.809, mDice 0.887, pixel accuracy 93.1%** (3,669 official test images at 256 px) |
| Per class IoU | pet 0.873, background 0.937, border 0.617 |

At original resolution on the same 400 images as above it gives mIoU 0.772. This model differs from the from-scratch U-Net in several ways at once (pretrained encoder, input size, class weights, epochs), so the improvement cannot be credited to one change.

## Intended use and limits

- Educational and portfolio demonstration. Not validated for production or safety-critical decisions.
- The waste classifier was trained on photos that mostly show one item on a plain background; accuracy on cluttered real-world photos has not been measured and is expected to be lower. It always returns one of its 12 classes.
- The segmentation models were trained only on cats and dogs.
- Datasets: Oxford-IIIT Pet (CC BY-SA 4.0); a 12-class garbage dataset from the Hugging Face hub (check the original license before reuse).
