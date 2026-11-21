"""Training and evaluation functions, reused for every model and experiment."""
import time
import torch
import torch.nn as nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, device=None):
    """Returns (accuracy, predictions list, true labels list)."""
    device = device or get_device()
    model.to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return acc, preds, labels


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3,
                save_path="model.pth", optimizer_name="adam", device=None):
    """Trains, keeps the best-validation weights, returns a history dict for plotting."""
    device = device or get_device()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.Adam(params, lr=lr) if optimizer_name == "adam"
           else torch.optim.SGD(params, lr=lr, momentum=0.9))
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    best = 0.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        val_acc, _, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0
        history["train_loss"].append(total_loss / n)
        history["train_acc"].append(correct / n)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(dt)
        print(f"Epoch {ep}/{epochs} | train loss {total_loss/n:.4f} | "
              f"train acc {correct/n:.4f} | val acc {val_acc:.4f} | {dt:.0f}s")
        if val_acc >= best:
            best = val_acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return history
