from data import get_dataloaders
from models import BaselineCNN, count_parameters
from engine import train_model, evaluate

if __name__ == "__main__":  # this guard is required on Windows
    train_loader, val_loader, test_loader, _ = get_dataloaders(img_size=32)
    model = BaselineCNN()
    print("Parameters (total, trainable):", count_parameters(model))
    train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, save_path="baseline.pth")
    acc, _, _ = evaluate(model, test_loader)
    print(f"Baseline TEST accuracy: {acc:.4f}")
