# Traffic Sign Recognition on GTSRB: Classical vs. Deep Learning Under Limited Compute

CPS 843 Computer Vision, Final Project (Fall 2026)

**Authors:** Riya Dave and Jannis Saini
**Group #:** 28

## Overview

This project studies traffic sign recognition, a core perception task for driver-assistance systems and self-driving cars. Given a photo of a road sign, the model predicts which of 43 sign classes it belongs to (stop, yield, speed limit 50, and so on).

**Research question:** How well do a classical method, a small CNN trained from scratch, and pretrained networks recognize traffic signs, and how much does pretraining help when compute and training data are limited?

All experiments run on standard laptop CPUs (no GPU), which mirrors the constraints of low-power embedded hardware in vehicles. To keep training times practical, images are reduced to 32x32 or 64x64 pixels.

## Dataset

[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/): 43 classes, loaded through `torchvision.datasets.GTSRB`.

| Split | Images | Use |
|---|---|---|
| Train | 21,312 | Model fitting (80% of the official training set) |
| Validation | 5,328 | Model selection (20%, fixed random seed 42) |
| Test | 12,630 | Final reported results (official test set) |

Notes:
- Training images come in bursts of near-identical photos of the same physical sign, so validation accuracy is optimistic. **Test accuracy is the main reported metric.**
- Horizontal flip is deliberately not used for augmentation, because mirroring changes the meaning of signs (turn left becomes turn right).

## Methods compared

| Model | Description |
|---|---|
| HOG + linear SVM | Classical baseline using hand-designed gradient features, no neural network |
| Baseline CNN | Small 3-block CNN (99k parameters) trained from scratch |
| ResNet-18 (frozen) | ImageNet-pretrained, only the final layer is trained |
| ResNet-18 (full) | ImageNet-pretrained, all layers fine-tuned |
| MobileNetV3-Small (full) | Lightweight pretrained network designed for low-power devices, all layers fine-tuned |

## Results

Test set, 12,630 images. Every accuracy below comes from a single run, so small differences should be read as trends, not proof. Full per-run data (including training curves and confusion matrices) is in `results/`; `python summarize_results.py` prints all runs in one table and writes `results/summary.csv`.

### Main comparison (100% of training data, with augmentation unless noted)

| Model | Image size | Epochs | Trainable params | Test accuracy | Macro F1 | Inference (ms/img) |
|---|---|---|---|---|---|---|
| HOG + SVM | 64 | n/a | n/a | 90.63% | 0.894 | 1.45 (HOG features; SVM adds about 0.02) |
| Baseline CNN (scratch) | 32 | 10 | 99,243 | 75.43% | 0.696 | 1.00 |
| ResNet-18, frozen | 64 | 5 | 22,059 | 43.75% | 0.336 | 10.45 |
| **ResNet-18, full fine-tuning** | 64 | 3 | 11,198,571 | **96.76%** | **0.954** | 10.45 |
| MobileNetV3-Small, full fine-tuning | 64 | 3 | 1,561,931 | 79.28% | 0.725 | 9.63 |

Inference times are medians from the controlled benchmark below, not from the training runs.

### Augmentation (ResNet-18, 64 px, 100% data)

| Setup | With augmentation | Without augmentation |
|---|---|---|
| Frozen, 5 epochs | 43.75% | 49.74% |
| Full fine-tuning, 3 epochs | 96.76% | 94.66% |

### Training set size (ResNet-18 frozen, 64 px, 5 epochs, with augmentation)

| Training data | Test accuracy | Macro F1 |
|---|---|---|
| 10% | 37.03% | 0.253 |
| 50% | 40.82% | 0.306 |
| 100% | 43.75% | 0.336 |

### Inference benchmark

`benchmark_inference.py` times every model in a single session on one laptop: 4 CPU threads, one image per forward pass, 30 warm-up runs and 300 timed runs. Weights do not affect timing, so randomly initialized models are used. Saved in `results/inference_benchmark.json`.

| Model | Image size | Median (ms) | Mean ± std (ms) | Parameters |
|---|---|---|---|---|
| Scratch CNN | 32 | 1.00 | 1.04 ± 0.20 | 99,243 |
| ResNet-18 | 64 | 10.45 | 11.33 ± 5.48 | 11,198,571 |
| MobileNetV3-Small | 64 | 9.63 | 9.49 ± 1.54 | 1,561,931 |
| HOG feature extraction | 64 | 1.45 | n/a | n/a |

ResNet-18's timing varies a lot between runs (large standard deviation), so medians are used. MobileNetV3-Small has about 7x fewer parameters than ResNet-18 but is only slightly faster here.

### Per-class analysis

`analyze.py` re-evaluates a trained model on the test set and saves per-class precision/recall/F1, the most common confusions, and a grid of misclassified images (`results/analysis/`). Run on the three models trained by Riya:

**ResNet-18, full fine-tuning (96.76%)**
- Weakest classes (F1): Ice/snow 0.754, Double curve 0.822, End of no passing 0.838, Bicycles crossing 0.845.
- Most common confusions: No vehicles → Speed limit 60 (30 images), Double curve → Right-of-way at intersection (22), Ice/snow → Bicycles crossing (18), Speed limit 100 → 120 (18), General caution → Road work (17).
- Errors mostly involve signs of similar shape and colour that differ in small inner details. Bicycles crossing has recall 1.000 but F1 0.845, so other classes are being predicted as it.

**ResNet-18, frozen (43.75%)**
- Errors are dominated by speed-limit signs: Speed limit 30 → 50 (255 images), 120 → 80 (127), 70 → 50 (118).
- Speed limit 20 and Road narrows (right) have F1 of 0.000, so these classes are never predicted correctly.

**Baseline CNN (75.43%)**
- Many different speed limits (50, 100, 60, 120, 70, 30) are predicted as Speed limit 80, which has recall 0.954 but F1 0.504. This is consistent with a model that has not converged after 10 epochs.

The explanations above are hypotheses. The misclassified-image grids in `results/analysis/` are the evidence for checking them (for example, whether failures are small, blurry or dark images).

### Observations so far

- Fine-tuned ResNet-18 is the best model (96.76%), ahead of the classical HOG + SVM baseline (90.63%) and far ahead of the CNN trained from scratch (75.43%).
- Pretraining only pays off when the weights are fine-tuned: the same ResNet-18 reaches 43.75% frozen and 96.76% fine-tuned. At 64 px the frozen ImageNet features appear to be a poor match for the task.
- Augmentation hurt the frozen model (about 6 points lower) but helped the fine-tuned model (about 2 points higher). The no-augmentation fine-tuned run reached 99.5% training accuracy but only 94.66% test accuracy, which indicates more overfitting.
- For the frozen model, more training data helps only slightly (37% to 44% from 10% to 100%).
- The baseline CNN had not converged after 10 epochs (training loss still falling), so its score is a lower bound.
- HOG + SVM trained in about 2 minutes on a laptop CPU and runs at about 1.5 ms per image, which makes it a strong cost-efficiency reference.

### Caveats

- Runs were split between two laptops. Training times are only comparable between runs made on the same machine, so they are not reported in the tables. Inference times come from the single-machine benchmark.
- Per-class analysis needs the trained weights, which are not committed to GitHub, so each model is analyzed on the machine that trained it.
- In the "frozen" models the weights are frozen, but batch-normalization layers still update their running statistics because the model is in training mode.
- Reduced image sizes (32/64 px) lower the accuracy of all models relative to full-resolution benchmarks.

## Repository structure

```
.
├── data.py                 # GTSRB loading, fixed train/val split, augmentation switches
├── models.py               # Baseline CNN, ResNet-18, MobileNetV3-Small, EfficientNet-B0
├── engine.py               # Training loop and evaluation functions
├── results_utils.py        # Saves results (JSON) and figures (PNG) to results/
├── gtsrb_classes.py        # Names of the 43 GTSRB classes
├── run_hog_svm.py          # Classical baseline: HOG + SVM
├── run_resnet.py           # Train/fine-tune any CNN model with command-line options
├── run_baseline.py         # Quick baseline CNN run
├── summarize_results.py    # Prints one table of all runs, saves results/summary.csv
├── compare_models.py       # Comparison charts from saved results
├── analyze.py              # Per-class metrics, top confusions, misclassified examples
├── benchmark_inference.py  # Controlled CPU inference-time benchmark
├── test.py                 # Environment check
├── requirements.txt        # Python dependencies
└── results/                # Saved results of every run
    ├── *.json, *_curves.png, *_confusion.png   # one set per training run
    ├── analysis/           # comparison charts and per-class analysis
    ├── inference_benchmark.json
    └── summary.csv
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

## Reproducing the experiments

The first run downloads GTSRB into `data/` automatically. Run names encode the settings, for example `resnet18_frozen_64px_aug_frac1.0`.

```bash
# Classical baseline
python run_hog_svm.py

# Baseline CNN trained from scratch
python run_resnet.py --model baseline --img_size 32 --epochs 10

# Pretrained models (64x64 images for CPU)
python run_resnet.py --model resnet18 --freeze --img_size 64 --epochs 5
python run_resnet.py --model resnet18 --img_size 64 --epochs 3
python run_resnet.py --model mobilenet --img_size 64 --epochs 3

# Augmentation experiments
python run_resnet.py --model resnet18 --freeze --no_augment --img_size 64 --epochs 5
python run_resnet.py --model resnet18 --no_augment --img_size 64 --epochs 3

# Training-set-size experiments
python run_resnet.py --model resnet18 --freeze --fraction 0.1 --img_size 64 --epochs 5
python run_resnet.py --model resnet18 --freeze --fraction 0.5 --img_size 64 --epochs 5
```

### Analysis

```bash
python summarize_results.py          # table of all runs -> results/summary.csv
python compare_models.py             # comparison charts -> results/analysis/
python analyze.py resnet18_full_64px_aug_frac1.0   # per-class analysis (needs weights/<run>.pth)
python benchmark_inference.py        # inference timing -> results/inference_benchmark.json
```

## Team contributions

| Member | Responsibilities |
|---|---|
| Riya Dave | Data pipeline, HOG + SVM, baseline CNN, ResNet-18 frozen and full fine-tuning runs, no-augmentation fine-tuning run, per-class analysis and inference benchmark |
| Jannis Saini | MobileNetV3-Small, frozen ResNet-18 no-augmentation and 10%/50% training-size runs, evaluation figures |

## References and acknowledgments

- J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition," *Neural Networks*, 2012 (GTSRB dataset).
- K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition," CVPR, 2016 (ResNet).
- A. Howard et al., "Searching for MobileNetV3," ICCV, 2019.
- N. Dalal, B. Triggs, "Histograms of oriented gradients for human detection," CVPR, 2005 (HOG).
- Pretrained weights and GTSRB loader from [PyTorch / torchvision](https://pytorch.org/vision/stable/index.html); classical features and SVM from scikit-image and scikit-learn.
