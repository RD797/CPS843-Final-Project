"""Helpers that save every run's results to results/ so they can be committed to GitHub as proof."""
import json
import os
import matplotlib
matplotlib.use("Agg")  # no pop-up windows; just save PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

RESULTS_DIR = "results"


def save_json(name, data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved", path)


def plot_confusion(name, labels, preds):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Confusion matrix: {name}")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_confusion.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved", path)


def plot_curves(name, history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epochs = range(1, len(history["val_acc"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(epochs, history["train_loss"], marker="o")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Train loss"); ax[0].set_title("Loss")
    ax[1].plot(epochs, history["train_acc"], marker="o", label="train")
    ax[1].plot(epochs, history["val_acc"], marker="o", label="validation")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].set_title("Accuracy")
    ax[1].legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_curves.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved", path)
