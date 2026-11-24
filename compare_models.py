"""Comparison charts built from the saved results/*.json (no weights needed).
Saves PNGs to results/analysis/. Run after pulling everyone's results."""
import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("results", "analysis")
os.makedirs(OUT, exist_ok=True)

runs = []
for p in sorted(glob.glob("results/*.json")):
    r = json.load(open(p))
    r["file"] = os.path.basename(p)[:-5]
    runs.append(r)


def label(r):
    if r["model"].startswith("HOG"):
        return "HOG + SVM"
    names = {"baseline": "Scratch CNN", "resnet18": "ResNet-18", "mobilenet": "MobileNetV3-S",
             "efficientnet": "EfficientNet-B0"}
    if r["model"] == "baseline":
        return f"Scratch CNN ({r['img_size']}px)"
    mode = "frozen" if r["frozen_backbone"] else "fine-tuned"
    aug = "aug" if r["augmentation"] else "no aug"
    return f"{names[r['model']]} {mode}, {aug}"


# 1. Main comparison: all runs on 100% of the data
full = [r for r in runs if r["train_fraction"] == 1.0]
full.sort(key=lambda r: r["test_acc"])
plt.figure(figsize=(8, 4.5))
bars = plt.barh([label(r) for r in full], [100 * r["test_acc"] for r in full], color="#3b6fb6")
for b, r in zip(bars, full):
    plt.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2,
             f"{100 * r['test_acc']:.1f}%", va="center", fontsize=8)
plt.xlim(0, 105)
plt.xlabel("Test accuracy (%)")
plt.title("Test accuracy by model (100% of training data)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "compare_accuracy.png"), dpi=200)
plt.close()

# 2. Augmentation effect: frozen vs fine-tuned ResNet-18
def find(model, frozen, aug, frac=1.0):
    for r in runs:
        if (r["model"] == model and r.get("frozen_backbone") == frozen
                and r.get("augmentation") == aug and r["train_fraction"] == frac):
            return r
rows = [("Frozen", find("resnet18", True, True), find("resnet18", True, False)),
        ("Fine-tuned", find("resnet18", False, True), find("resnet18", False, False))]
if all(a and b for _, a, b in rows):
    plt.figure(figsize=(5.5, 4))
    for i, (g, a, b) in enumerate(rows):
        plt.bar(i - 0.2, 100 * a["test_acc"], 0.4, color="#3b6fb6", label="with augmentation" if i == 0 else "")
        plt.bar(i + 0.2, 100 * b["test_acc"], 0.4, color="#c0504d", label="no augmentation" if i == 0 else "")
        plt.text(i - 0.2, 100 * a["test_acc"] + 1, f"{100 * a['test_acc']:.1f}", ha="center", fontsize=8)
        plt.text(i + 0.2, 100 * b["test_acc"] + 1, f"{100 * b['test_acc']:.1f}", ha="center", fontsize=8)
    plt.xticks(range(2), [g for g, _, _ in rows])
    plt.ylabel("Test accuracy (%)")
    plt.ylim(0, 105)
    plt.title("Effect of augmentation (ResNet-18, 64 px)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "compare_augmentation.png"), dpi=200)
    plt.close()

# 3. Training set size (frozen ResNet-18, with augmentation)
pts = [(r["train_fraction"], r["test_acc"]) for r in runs
       if r["model"] == "resnet18" and r.get("frozen_backbone") and r.get("augmentation")
       and r.get("epochs") == 5]
if len(pts) > 1:
    pts.sort()
    plt.figure(figsize=(5, 3.8))
    plt.plot([100 * f for f, _ in pts], [100 * a for _, a in pts], marker="o")
    plt.xlabel("Training data used (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Data size (ResNet-18 frozen, 64 px)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "compare_data_size.png"), dpi=200)
    plt.close()

# 4. Validation accuracy per epoch for each trained model (100% data)
plt.figure(figsize=(7, 4.2))
for r in runs:
    if "history" in r and r["train_fraction"] == 1.0:
        h = r["history"]["val_acc"]
        if len(h) > 1:
            plt.plot(range(1, len(h) + 1), [100 * v for v in h], marker="o", label=label(r))
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy (%)")
plt.title("Validation accuracy during training")
plt.legend(fontsize=6)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "compare_val_curves.png"), dpi=200)
plt.close()
print("Saved comparison charts to", OUT)
