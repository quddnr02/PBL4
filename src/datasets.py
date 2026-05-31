from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from augmentation import AugmentationConfig, train_augmentation, val_preprocessing


class VOCSegmentationDataset(Dataset):
    def __init__(
        self,
        voc_root: str,
        split_txt: str,
        image_size: int = 512,
        train: bool = False,
        augmentation: dict | None = None,
    ):
        self.voc_root = Path(voc_root)
        self.split_txt = Path(split_txt)
        self.image_size = image_size
        self.train = train
        self.augmentation = AugmentationConfig.from_dict(augmentation)
        with self.split_txt.open("r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.image_dir = self.voc_root / "JPEGImages"
        self.mask_dir = self.voc_root / "SegmentationClass"
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"VOC root must contain JPEGImages and SegmentationClass: {self.voc_root}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[idx]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{image_id}.png")
        if self.train:
            image, mask = train_augmentation(image, mask, self.image_size, self.augmentation)
        else:
            image, mask = val_preprocessing(image, mask, self.image_size, self.augmentation)
        return image, mask.long()

    def mask_paths(self) -> list[Path]:
        return [self.mask_dir / f"{image_id}.png" for image_id in self.ids]
