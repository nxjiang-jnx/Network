from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def build_imagenet_loaders(
    data_root: str,
    batch_size: int,
    workers: int,
    val_batch_size: int | None = None,
    train_crop_size: int = 224,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
    train_sampler: Optional[DistributedSampler] = None,
    val_sampler: Optional[DistributedSampler] = None,
    build_val: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet folders not found. Need {train_dir} and {val_dir}."
        )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(train_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(val_resize_size),
            transforms.CenterCrop(val_crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)

    val_batch_size = val_batch_size or batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )
    if not build_val:
        return train_loader, None

    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )
    return train_loader, val_loader


def build_imagenet_val_loader(
    data_root: str,
    val_batch_size: int,
    workers: int,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
    prefetch_factor: int = 2,
) -> DataLoader:
    root = Path(data_root)
    val_dir = root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(val_resize_size),
            transforms.CenterCrop(val_crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    return DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )


def device_from_flag(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)
