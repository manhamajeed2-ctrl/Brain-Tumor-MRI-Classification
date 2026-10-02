"""Dataset and transforms for Brain Tumor MRI Classification (minimal dependencies)."""

import os
import random
from pathlib import Path
from typing import Tuple, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# Class names in consistent order
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_tensor_and_normalize(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> normalized CHW float tensor."""
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()  # CHW
    return tensor


def get_transforms(img_size: int = 224, is_train: bool = True) -> Callable:
    """Return a callable transform (PIL -> Tensor)."""

    def train_transform(img: Image.Image) -> torch.Tensor:
        img = img.resize((img_size + 32, img_size + 32), Image.BILINEAR)
        w, h = img.size
        th, tw = img_size, img_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img = img.crop((j, i, j + tw, i + th))
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        if random.random() < 0.2:
            img = ImageOps.flip(img)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        return _to_tensor_and_normalize(img)

    def eval_transform(img: Image.Image) -> torch.Tensor:
        img = img.resize((img_size, img_size), Image.BILINEAR)
        return _to_tensor_and_normalize(img)

    return train_transform if is_train else eval_transform


class BrainTumorDataset(Dataset):
    """Simple folder-based dataset: root/class_name/*.jpg"""

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for cls_name in CLASS_NAMES:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((img_path, CLASS_TO_IDX[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}. Check folder structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def create_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / valid / test dataloaders."""
    train_ds = BrainTumorDataset(
        os.path.join(data_dir, "train"),
        transform=get_transforms(img_size, is_train=True),
    )
    valid_ds = BrainTumorDataset(
        os.path.join(data_dir, "valid"),
        transform=get_transforms(img_size, is_train=False),
    )
    test_ds = BrainTumorDataset(
        os.path.join(data_dir, "test"),
        transform=get_transforms(img_size, is_train=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, valid_loader, test_loader
