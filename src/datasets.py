from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from utils import IMAGENET_MEAN, IMAGENET_STD, VOC_IGNORE_INDEX

AUG_SCALE_RANGE = (0.75, 1.50)


class VOCSegmentationDataset(Dataset):
    def __init__(self, voc_root: str, split_txt: str, image_size: int = 512, train: bool = False):
        self.voc_root = Path(voc_root)
        self.split_txt = Path(split_txt)
        self.image_size = image_size
        self.train = train
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
        image, mask = self._transform(image, mask)
        return image, mask.long()

    def mask_paths(self) -> list[Path]:
        return [self.mask_dir / f"{image_id}.png" for image_id in self.ids]

    def _transform(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            image, mask = self._train_transform(image, mask)
        else:
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image_tensor, mask_tensor

    def _train_transform(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        image, mask = self._random_scale(image, mask)
        image, mask = self._pad_if_needed(image, mask)
        top, left = self._random_crop_params(image)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def _random_scale(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        scale = torch.empty(1).uniform_(*AUG_SCALE_RANGE).item()
        width, height = image.size
        short_side = max(1, int(self.image_size * scale))
        if width <= height:
            new_width = short_side
            new_height = int(height * short_side / width)
        else:
            new_height = short_side
            new_width = int(width * short_side / height)
        image = TF.resize(image, [new_height, new_width], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_height, new_width], interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def _pad_if_needed(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        pad_w = max(self.image_size - width, 0)
        pad_h = max(self.image_size - height, 0)
        if pad_w > 0 or pad_h > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=VOC_IGNORE_INDEX)
        return image, mask

    def _random_crop_params(self, image: Image.Image) -> tuple[int, int]:
        width, height = image.size
        top = torch.randint(0, height - self.image_size + 1, (1,)).item()
        left = torch.randint(0, width - self.image_size + 1, (1,)).item()
        return top, left
