"""Controlled CPU inference-time benchmark: every model timed in one session, one image at a time.
Close other apps and plug in the laptop first. Weights are not needed for timing."""
import json
import os
import time
import numpy as np
import torch
from skimage.feature import hog
from models import BaselineCNN, get_resnet18, get_mobilenet_v3_small, count_parameters

WARMUP, RUNS = 30, 300
torch.set_num_threads(max(1, os.cpu_count() // 2))

cases = [("Scratch CNN", BaselineCNN(), 32),
         ("ResNet-18", get_resnet18(pretrained=False), 64),
         ("MobileNetV3-Small", get_mobilenet_v3_small(pretrained=False), 64)]

results = {}
print(f"CPU threads: {torch.get_num_threads()}\n")
for name, model, size in cases:
    model.eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    total, _ = count_parameters(model)
    results[name] = {"img_size": size, "params": total,
                     "median_ms": float(np.median(times)), "mean_ms": float(np.mean(times)),
                     "std_ms": float(np.std(times))}
    print(f"{name:20s} {size}px  median {np.median(times):6.2f} ms  "
          f"mean {np.mean(times):6.2f} +/- {np.std(times):5.2f}  params {total:,}")

img = (np.random.rand(64, 64) * 255).astype(np.uint8)
t = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    t.append((time.perf_counter() - t0) * 1000)
results["HOG features (SVM step excluded)"] = {"img_size": 64, "median_ms": float(np.median(t)),
                                               "mean_ms": float(np.mean(t)), "std_ms": float(np.std(t))}
print(f"{'HOG features':20s} 64px  median {np.median(t):6.2f} ms  (the linear SVM adds ~0.02 ms)")

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "inference_benchmark.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/inference_benchmark.json")
