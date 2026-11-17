# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-17

### Added
- Gradio web app with two tabs: waste classification and pet segmentation.
- EfficientNet-B0 waste classifier (12 classes) with recycling tips; 96.1% test accuracy.
- From-scratch U-Net (mIoU 0.767) and ResNet18-encoder U-Net (mIoU 0.809) for pet segmentation, selectable in the app.
- Inference pipeline that reproduces the training-time pre-processing.
- Unit tests, GitHub Actions CI (lint, format check, tests on Python 3.10 to 3.12) and Dependabot.
- Training notebooks for reproducing every model, a model card, and bilingual (English / Vietnamese) documentation.
