"""Per-class analysis of one trained model (needs its weights in weights/).

Usage:  python analyze.py resnet18_full_64px_aug_frac1.0
Outputs in results/analysis/: per-class CSV, per-class F1 chart, top confusions CSV,
and a grid of misclassified test images.
"""
import csv
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from data import get_dataloaders, MEAN, STD
from models import BaselineCNN, get_resnet18, get_mobilenet_v3_small, get_efficientnet_b0
from gtsrb_classes import CLASS_NAMES

OUT = os.path.join("results", "analysis")


def build_model(model_name):
    return {"baseline": BaselineCNN,
            "resnet18": lambda: get_resnet18(pretrained=False),
            "mobilenet": lambda: get_mobilenet_v3_small(pretrained=False),
            "efficientnet": lambda: get_efficientnet_b0(pretrained=False)}[model_name]()


if __name__ == "__main__":
    name = sys.argv[1]
    m = re.match(r"(\w+?)_(frozen|full)_(\d+)px_(aug|noaug)_frac([\d.]+)$", name)
    if not m:
        sys.exit("Run name must look like resnet18_full_64px_aug_frac1.0")
    model_name, _, size, _, _ = m.groups()
    weights = os.path.join("weights", f"{name}.pth")
    if not os.path.exists(weights):
        sys.exit(f"Missing {weights}. Weights are not on GitHub; run this on the laptop that trained the model.")
    os.makedirs(OUT, exist_ok=True)

    model = build_model(model_name)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    _, _, test_loader, _ = get_dataloaders(img_size=int(size), num_workers=0)

    preds, labels, err_imgs, err_idx = [], [], [], []
    seen = 0
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x).argmax(1)
            for j in range(len(y)):
                if p[j] != y[j] and len(err_imgs) < 400:
                    err_imgs.append(x[j].clone())
                    err_idx.append(len(preds) + j)
            preds += p.tolist()
            labels += y.tolist()
    preds, labels = np.array(preds), np.array(labels)
    print(f"Test accuracy: {(preds == labels).mean():.4f}")

    # --- per-class metrics ---
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=range(43), zero_division=0)
    with open(os.path.join(OUT, f"{name}_per_class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "precision", "recall", "f1", "support"])
        for c in range(43):
            w.writerow([c, CLASS_NAMES[c], round(prec[c], 4), round(rec[c], 4),
                        round(f1[c], 4), int(sup[c])])

    order = np.argsort(f1)
    plt.figure(figsize=(7, 9))
    plt.barh([f"{c}: {CLASS_NAMES[c]}" for c in order], f1[order], color="#3b6fb6")
    plt.xlabel("F1 score")
    plt.title(f"Per-class F1: {name}")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{name}_per_class_f1.png"), dpi=200)
    plt.close()
    print("\nWorst 8 classes (F1):")
    for c in order[:8]:
        print(f"  {c:2d} {CLASS_NAMES[c]:32s} F1 {f1[c]:.3f}  recall {rec[c]:.3f}")

    # --- most common confusions ---
    cm = confusion_matrix(labels, preds, labels=range(43))
    np.fill_diagonal(cm, 0)
    flat = np.dstack(np.unravel_index(np.argsort(-cm, axis=None), cm.shape))[0][:15]
    print("\nTop confusions (true -> predicted, count):")
    with open(os.path.join(OUT, f"{name}_top_confusions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class", "predicted_class", "count"])
        for t, p in flat:
            if cm[t, p] == 0:
                break
            w.writerow([CLASS_NAMES[t], CLASS_NAMES[p], int(cm[t, p])])
            print(f"  {CLASS_NAMES[t]}  ->  {CLASS_NAMES[p]}: {cm[t, p]}")

    # --- misclassified examples ---
    if err_imgs:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(err_imgs), size=min(16, len(err_imgs)), replace=False)
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        fig, axes = plt.subplots(4, 4, figsize=(9, 10))
        for ax in axes.flat:
            ax.axis("off")
        for ax, k in zip(axes.flat, pick):
            img = (err_imgs[k] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            i = err_idx[k]
            ax.imshow(img)
            ax.set_title(f"true: {CLASS_NAMES[labels[i]]}\npred: {CLASS_NAMES[preds[i]]}", fontsize=7)
        plt.suptitle(f"Misclassified test images: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"{name}_misclassified.png"), dpi=200)
        plt.close()
    print("\nSaved analysis files to", OUT)
