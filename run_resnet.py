"""Train/fine-tune a pretrained model (or the baseline) and save results to results/.

Examples (CPU-friendly):
  python run_resnet.py --model resnet18 --freeze --img_size 64 --epochs 3
  python run_resnet.py --model resnet18 --img_size 64 --epochs 3
  python run_resnet.py --model mobilenet --img_size 64 --epochs 3
  python run_resnet.py --model resnet18 --freeze --no_augment --img_size 64   (augmentation experiment)
  python run_resnet.py --model resnet18 --freeze --fraction 0.1 --img_size 64 (data-size experiment)
"""
import argparse
import time
import torch
from sklearn.metrics import f1_score
from data import get_dataloaders
from models import (BaselineCNN, get_resnet18, get_mobilenet_v3_small,
                    get_efficientnet_b0, count_parameters)
from engine import train_model, evaluate, get_device
from results_utils import save_json, plot_confusion, plot_curves

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18",
                    choices=["baseline", "resnet18", "mobilenet", "efficientnet"])
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--freeze", action="store_true", help="train only the final layer")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    args = ap.parse_args()

    lr = args.lr or (1e-3 if (args.freeze or args.model == "baseline") else 3e-4)
    name = (f"{args.model}_{'frozen' if args.freeze else 'full'}_{args.img_size}px"
            f"_{'noaug' if args.no_augment else 'aug'}_frac{args.fraction}")
    print("Run name:", name, "| device:", get_device())

    train_loader, val_loader, test_loader, nc = get_dataloaders(
        img_size=args.img_size, batch_size=args.batch_size,
        augment=not args.no_augment, train_fraction=args.fraction)

    if args.model == "baseline":
        model = BaselineCNN(nc)
    elif args.model == "resnet18":
        model = get_resnet18(nc, freeze_backbone=args.freeze)
    elif args.model == "mobilenet":
        model = get_mobilenet_v3_small(nc, freeze_backbone=args.freeze)
    else:
        model = get_efficientnet_b0(nc, freeze_backbone=args.freeze)
    total, trainable = count_parameters(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    import os
    os.makedirs("weights", exist_ok=True)  # *.pth is git-ignored
    t0 = time.time()
    history = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=lr,
                          save_path=f"weights/{name}.pth", optimizer_name=args.optimizer)
    train_time = time.time() - t0

    t0 = time.time()
    test_acc, preds, labels = evaluate(model, test_loader)
    inf_ms = (time.time() - t0) / len(labels) * 1000
    f1 = f1_score(labels, preds, average="macro")
    print(f"TEST accuracy {test_acc:.4f} | macro F1 {f1:.4f} | {inf_ms:.2f} ms/image")

    save_json(name, {
        "model": args.model, "frozen_backbone": args.freeze, "img_size": args.img_size,
        "epochs": args.epochs, "lr": lr, "optimizer": args.optimizer,
        "augmentation": not args.no_augment, "train_fraction": args.fraction,
        "params_total": total, "params_trainable": trainable,
        "test_acc": test_acc, "test_macro_f1": f1,
        "best_val_acc": max(history["val_acc"]), "total_train_time_s": train_time,
        "inference_ms_per_image": inf_ms, "device": str(get_device()), "history": history})
    plot_curves(name, history)
    plot_confusion(name, labels, preds)
