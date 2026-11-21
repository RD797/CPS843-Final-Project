"""Classical baseline: HOG features + linear SVM (no neural network)."""
import argparse
import time
import numpy as np
import torch
from torchvision import datasets, transforms
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from results_utils import save_json, plot_confusion


def extract(dataset, indices=None):
    feats, labels = [], []
    idx = indices if indices is not None else range(len(dataset))
    for i in idx:
        img, y = dataset[i]
        gray = np.array(img.convert("L"))
        feats.append(hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
        labels.append(y)
    return np.array(feats), np.array(labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of training data to use")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    tf = transforms.Resize((args.img_size, args.img_size))
    train_full = datasets.GTSRB("data", split="train", download=True, transform=tf)
    test_set = datasets.GTSRB("data", split="test", download=True, transform=tf)

    # Same fixed-seed split as data.py, so numbers are comparable with the CNNs
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    n_val = int(len(perm) * 0.2)
    train_idx = perm[n_val:]
    train_idx = train_idx[: int(len(train_idx) * args.fraction)]
    val_idx = perm[:n_val]

    t0 = time.time()
    print("Extracting HOG features (takes a few minutes)...")
    X_tr, y_tr = extract(train_full, train_idx)
    X_val, y_val = extract(train_full, val_idx)
    X_te, y_te = extract(test_set)
    t_feat = time.time() - t0
    print("Feature shape:", X_tr.shape)

    t0 = time.time()
    clf = LinearSVC(C=args.C, dual=False, max_iter=2000).fit(X_tr, y_tr)
    t_train = time.time() - t0

    val_acc = accuracy_score(y_val, clf.predict(X_val))
    t0 = time.time()
    preds = clf.predict(X_te)
    t_inf = (time.time() - t0) / len(y_te) * 1000
    test_acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")
    print(f"Val acc {val_acc:.4f} | TEST acc {test_acc:.4f} | macro F1 {f1:.4f}")

    name = f"hog_svm_{args.img_size}_frac{args.fraction}"
    save_json(name, {
        "model": "HOG + LinearSVC", "img_size": args.img_size, "train_fraction": args.fraction,
        "C": args.C, "feature_dim": int(X_tr.shape[1]), "val_acc": val_acc,
        "test_acc": test_acc, "test_macro_f1": f1, "feature_time_s": t_feat,
        "train_time_s": t_train, "inference_ms_per_image_(classifier_only)": t_inf})
    plot_confusion(name, y_te, preds)
