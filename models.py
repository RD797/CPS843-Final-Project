"""Model definitions: a small baseline CNN plus pretrained ResNet-18 / EfficientNet-B0."""
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Small CNN trained from scratch (~99k parameters). Works at any image size."""
    def __init__(self, num_classes=43):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128),
                                      nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def get_resnet18(num_classes=43, freeze_backbone=False, pretrained=True):
    """freeze_backbone=True trains only the final layer (last-layer vs full fine-tuning experiment)."""
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # new layer is trainable
    return m


def get_efficientnet_b0(num_classes=43, freeze_backbone=False, pretrained=True):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_mobilenet_v3_small(num_classes=43, freeze_backbone=False, pretrained=True):
    """Lightweight pretrained model built for low-power hardware."""
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m
