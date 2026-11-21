# Traffic Sign Recognition on GTSRB: Classical vs. Deep Learning Under Limited Compute

CPS 843 Computer Vision, Final Project (Fall 2026)

**Authors:** Riya Dave and Jannis Saini


## Overview

This project studies traffic sign recognition, a core perception task for driver-assistance systems and self-driving cars. Given a photo of a road sign, the model predicts which of 43 sign classes it belongs to (stop, yield, speed limit 50, and so on).

**Research question:** How well do a classical method, a small CNN trained from scratch, and pretrained networks recognize traffic signs, and how much does pretraining help when compute and training data are limited?

All experiments run on a standard laptop CPU (no GPU), which mirrors the constraints of low-power embedded hardware in vehicles.

## Dataset

[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/): 43 classes, loaded through `torchvision.datasets.GTSRB`.

| Split | Images | Use |
|---|---|---|
| Train | ~21,300 | Model fitting (80% of the official training set) |
| Validation | ~5,300 | Model selection (20%, fixed random seed 42) |
| Test | 12,630 | Final reported results (official test set) |

Notes:
- Training images come in bursts of near-identical photos of the same physical sign, so validation accuracy is optimistic. **Test accuracy is the main reported metric.**
- Horizontal flip is deliberately not used for augmentation, because mirroring changes the meaning of signs (turn left becomes turn right).

## Methods compared

| Model | Description |
|---|---|
| HOG + linear SVM | Classical baseline using hand-designed gradient features, no neural network |
| Baseline CNN | Small 3-block CNN (~99k parameters) trained from scratch |
| ResNet-18 (frozen) | ImageNet-pretrained, only the final layer is trained |
| ResNet-18 (full) | ImageNet-pretrained, all layers fine-tuned |
| MobileNetV3-Small | Lightweight pretrained network designed for low-power devices |

## Experiments

- Classical vs. from-scratch CNN vs. pretrained models
- Last-layer-only vs. full fine-tuning
- Data augmentation on vs. off
- Training data size: 10%, 50%, 100%
- Cost comparison: parameters, training time, inference time per image

Metrics: accuracy, macro F1, per-class results, confusion matrices, and misclassified examples.

## Repository structure

```
.
├── data.py             # GTSRB loading, fixed train/val split, augmentation switches
├── models.py           # Baseline CNN, ResNet-18, MobileNetV3-Small, EfficientNet-B0
├── engine.py           # Training loop and evaluation functions
├── results_utils.py    # Saves results (JSON) and figures (PNG) to results/
├── run_hog_svm.py      # Classical baseline: HOG + SVM
├── run_resnet.py       # Train/fine-tune any CNN model with command-line options
├── run_baseline.py     # Quick baseline CNN run
├── test.py             # Environment check
├── requirements.txt    # Python dependencies
└── results/            # Saved results of every run (JSON, curves, confusion matrices)
```

The dataset (`data/`), virtual environment (`venv/`), and model weights (`weights/`, `*.pth`) are not committed.

## Setup

Tested on Windows with Python 3.12 (very new Python versions may lack prebuilt PyTorch/OpenCV packages).

```bash
git clone https://github.com/RD797/CPS843-Final-Project.git
cd CPS843-Final-Project
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python test.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Running the experiments

The first run downloads GTSRB into `data/` automatically.

```bash
# Classical baseline
python run_hog_svm.py

# Baseline CNN trained from scratch
python run_resnet.py --model baseline --img_size 32 --epochs 10

# Pretrained models (64x64 images for CPU)
python run_resnet.py --model resnet18 --freeze --img_size 64 --epochs 3
python run_resnet.py --model resnet18 --img_size 64 --epochs 3
python run_resnet.py --model mobilenet --img_size 64 --epochs 3

# Experiments
python run_resnet.py --model resnet18 --freeze --no_augment --img_size 64
python run_resnet.py --model resnet18 --freeze --fraction 0.1 --img_size 64
python run_resnet.py --model resnet18 --freeze --fraction 0.5 --img_size 64
```

Each run saves `results/<run_name>.json` plus training-curve and confusion-matrix images. Run names encode the settings, for example `resnet18_frozen_64px_aug_frac1.0`.

## Results

*To be filled in as experiments finish. Test set, 12,630 images.*

| Model | Image size | Test accuracy | Macro F1 | Trainable params | Inference (ms/img) |
|---|---|---|---|---|---|
| HOG + SVM | 64 | | | | |
| Baseline CNN | 32 | | | | |
| ResNet-18 (frozen) | 64 | | | | |
| ResNet-18 (full) | 64 | | | | |
| MobileNetV3-Small | 64 | | | | |

## Team contributions

| Member | Responsibilities |
|---|---|
| [Name] | [e.g., data pipeline, HOG + SVM, ResNet-18 experiments, report sections] |
| [Name] | [e.g., MobileNet, data-size experiments, evaluation figures, report sections] |

## References and acknowledgments

- J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition," *Neural Networks*, 2012 (GTSRB dataset).
- K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition," CVPR, 2016 (ResNet).
- A. Howard et al., "Searching for MobileNetV3," ICCV, 2019.
- N. Dalal, B. Triggs, "Histograms of oriented gradients for human detection," CVPR, 2005 (HOG).
- Pretrained weights and GTSRB loader from [PyTorch / torchvision](https://pytorch.org/vision/stable/index.html); classical features and SVM from scikit-image and scikit-learn.
