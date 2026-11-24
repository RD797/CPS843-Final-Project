"""Prints one table of every run in results/ and saves it to results/summary.csv."""
import csv
import glob
import json
import os

rows = []
for path in sorted(glob.glob("results/*.json")):
    with open(path) as f:
        r = json.load(f)
    rows.append({
        "run": os.path.basename(path)[:-5],
        "epochs": r.get("epochs", "-"),
        "lr": r.get("lr", "-"),
        "aug": r.get("augmentation", "-"),
        "fraction": r.get("train_fraction", "-"),
        "test_acc": round(r["test_acc"], 4),
        "macro_f1": round(r["test_macro_f1"], 4),
        "trainable_params": r.get("params_trainable", "-"),
        "train_time_s": round(r.get("total_train_time_s", r.get("train_time_s", 0))),
        "ms_per_img": round(r.get("inference_ms_per_image",
                            r.get("inference_ms_per_image_(classifier_only)", 0)), 2),
    })

cols = list(rows[0].keys())
print(" | ".join(cols))
for r in rows:
    print(" | ".join(str(r[c]) for c in cols))

with open("results/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print("\nSaved results/summary.csv")
