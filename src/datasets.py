from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from utils import IMAGENET_MEAN, IMAGENET_STD, VOC_IGNORE_INDEX, load_config

AUG_SCALE_RANGE = (0.75, 1.50)


class VOCSegmentationDataset(Dataset):
    def __init__(
        self,
        voc_root: str,
        split_txt: str,
        image_size: int = 512,
        train: bool = False,
        aug_config: str | None = None,
    ):
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
        self.aug_config = load_config(aug_config) if aug_config and self.train else None
        self.dataset_expansion = self._build_dataset_expansion()

    @property
    def base_length(self) -> int:
        return len(self.ids)

    @property
    def expansion_enabled(self) -> bool:
        return self.train and self.dataset_expansion["enabled"]

    def __len__(self) -> int:
        if not self.expansion_enabled:
            return self.base_length
        repeat_count = self.dataset_expansion["original_repeat"] + self.dataset_expansion["augmented_repeat"]
        return self.base_length * repeat_count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id, is_augmented = self._index_to_variant(idx)
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{image_id}.png")
        image, mask = self._transform(image, mask, is_augmented=is_augmented)
        return image, mask.long()

    def mask_paths(self) -> list[Path]:
        return [self.mask_dir / f"{image_id}.png" for image_id in self.ids]

    def _build_dataset_expansion(self) -> dict[str, int | bool]:
        expansion_cfg = (self.aug_config or {}).get("dataset_expansion", {})
        return {
            "enabled": bool(expansion_cfg.get("enabled", False)),
            "original_repeat": max(1, int(expansion_cfg.get("original_repeat", 1))),
            "augmented_repeat": max(0, int(expansion_cfg.get("augmented_repeat", 0))),
        }

    def _index_to_variant(self, idx: int) -> tuple[str, bool]:
        if not self.expansion_enabled:
            return self.ids[idx], False
        base_len = self.base_length
        original_total = base_len * self.dataset_expansion["original_repeat"]
        if idx < original_total:
            return self.ids[idx % base_len], False
        augmented_idx = idx - original_total
        return self.ids[augmented_idx % base_len], True

    def _extra_augmentation(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        aug_cfg = (self.aug_config or {}).get("extra_augmentation", {})
        image, mask = self._apply_random_rotation(image, mask, aug_cfg.get("rotation", {}))
        image = self._apply_color_jitter(image, aug_cfg.get("color_jitter", {}))
        image = self._apply_gaussian_blur(image, aug_cfg.get("gaussian_blur", {}))
        image = self._apply_grayscale(image, aug_cfg.get("grayscale", {}))
        return image, mask

    def _should_apply(self, cfg: dict) -> bool:
        return bool(cfg.get("enabled", False)) and torch.rand(1).item() < float(cfg.get("prob", 0.0))

    def _apply_color_jitter(self, image: Image.Image, cfg: dict) -> Image.Image:
        if not self._should_apply(cfg):
            return image
        jitter = ColorJitter(
            brightness=float(cfg.get("brightness", 0.0)),
            contrast=float(cfg.get("contrast", 0.0)),
            saturation=float(cfg.get("saturation", 0.0)),
            hue=float(cfg.get("hue", 0.0)),
        )
        return jitter(image)

    def _apply_gaussian_blur(self, image: Image.Image, cfg: dict) -> Image.Image:
        if not self._should_apply(cfg):
            return image
        kernel_size = int(cfg.get("kernel_size", 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma_min = float(cfg.get("sigma_min", 0.1))
        sigma_max = float(cfg.get("sigma_max", 2.0))
        sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def _apply_grayscale(self, image: Image.Image, cfg: dict) -> Image.Image:
        if not self._should_apply(cfg):
            return image
        return TF.rgb_to_grayscale(image, num_output_channels=3)

    def _apply_random_rotation(
        self, image: Image.Image, mask: Image.Image, cfg: dict
    ) -> tuple[Image.Image, Image.Image]:
        if not self._should_apply(cfg):
            return image, mask
        degrees = float(cfg.get("degrees", 0.0))
        angle = torch.empty(1).uniform_(-degrees, degrees).item()
        image_fill = cfg.get("image_fill", [0, 0, 0])
        mask_fill = int(cfg.get("mask_fill", VOC_IGNORE_INDEX))
        image = TF.rotate(
            image,
            angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=image_fill,
        )
        mask = TF.rotate(
            mask,
            angle,
            interpolation=TF.InterpolationMode.NEAREST,
            fill=mask_fill,
        )
        return image, mask

    def _transform(
        self, image: Image.Image, mask: Image.Image, is_augmented: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            image, mask = self._train_transform(image, mask)
            if is_augmented:
                image, mask = self._extra_augmentation(image, mask)
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
