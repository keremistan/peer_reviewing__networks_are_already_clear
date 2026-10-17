import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional


class SyntheticImageNetDataset(Dataset):
    """
    Synthetic ImageNet dataset generating standard (3, 224, 224) tensors
    and random labels in [0, 999] for benchmarking throughput and debugging.
    """
    def __init__(self, length: int = 1280, num_classes: int = 1000, img_size: int = 224):
        self.length = length
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Deterministic-like or pseudo-random tensor
        img = torch.randn(3, self.img_size, self.img_size)
        label = idx % self.num_classes
        return img, label


def build_transforms(img_size: int = 224, rand_augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build modern data augmentation transforms for ImageNet-1k.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if rand_augment:
        train_transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    train_transform_list.extend([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform_list = [
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ]

    return transforms.Compose(train_transform_list), transforms.Compose(val_transform_list)


def build_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    synthetic: bool = False,
    synthetic_length: int = 1280,
    img_size: int = 224,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders.
    If synthetic is True or data_dir does not exist, falls back to SyntheticImageNetDataset.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train_transform, val_transform = build_transforms(img_size=img_size)

    if synthetic or not data_dir or not os.path.exists(data_dir):
        if not synthetic:
            print(f"[Dataset] ImageNet data directory '{data_dir}' not found. Using Synthetic ImageNet dataset.")
        train_dataset = SyntheticImageNetDataset(length=synthetic_length, img_size=img_size)
        val_dataset = SyntheticImageNetDataset(length=max(synthetic_length // 5, 256), img_size=img_size)
    else:
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Missing train directory at {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Missing val directory at {val_dir}")

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        print(f"[Dataset] Loaded ImageNet-1k: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
