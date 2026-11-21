"""Shared data pipeline for GTSRB (traffic sign recognition).
Both teammates import get_dataloaders() so splits are identical."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet stats (needed for pretrained models)


def get_dataloaders(img_size=224, batch_size=64, augment=True, train_fraction=1.0,
                    val_split=0.2, seed=42, num_workers=0, root="data"):
    """Returns train_loader, val_loader, test_loader, num_classes.

    augment=False        -> 'with vs. without augmentation' experiment
    train_fraction=0.1   -> 'less training data' experiment (0.1, 0.5, 1.0)
    No horizontal flip on purpose: flipping 'turn left' makes 'turn right'.
    """
    plain = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = plain

    # Two copies of the train set so train and val get different transforms
    full_train = datasets.GTSRB(root, split="train", download=True, transform=train_tf)
    full_val = datasets.GTSRB(root, split="train", download=True, transform=plain)
    test_set = datasets.GTSRB(root, split="test", download=True, transform=plain)

    # Fixed-seed split: identical for both teammates
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(len(perm) * val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_idx = train_idx[: int(len(train_idx) * train_fraction)]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(Subset(full_val, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, 43
