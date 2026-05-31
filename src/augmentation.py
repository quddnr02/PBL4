from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from utils import IMAGENET_MEAN, IMAGENET_STD, VOC_IGNORE_INDEX


@dataclass(frozen=True)
class AugmentationConfig:
    scale_range: tuple[float, float] = (0.75, 1.50)
    hflip_prob: float = 0.5
    random_crop: bool = True
    random_scale: bool = True
    normalize_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalize_std: tuple[float, float, float] = IMAGENET_STD

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "AugmentationConfig":
        cfg = cfg or {}
        scale_range = cfg.get("scale_range", cls.scale_range)
        return cls(
            scale_range=(float(scale_range[0]), float(scale_range[1])),
            hflip_prob=float(cfg.get("hflip_prob", cls.hflip_prob)),
            random_crop=bool(cfg.get("random_crop", cls.random_crop)),
            random_scale=bool(cfg.get("random_scale", cls.random_scale)),
            normalize_mean=tuple(float(v) for v in cfg.get("normalize_mean", cls.normalize_mean)),
            normalize_std=tuple(float(v) for v in cfg.get("normalize_std", cls.normalize_std)),
        )


def train_augmentation(
    image: Image.Image,
    mask: Image.Image,
    image_size: int,
    cfg: AugmentationConfig | dict | None = None,
    ignore_index: int = VOC_IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    aug = cfg if isinstance(cfg, AugmentationConfig) else AugmentationConfig.from_dict(cfg)
    if aug.random_scale:
        image, mask = _random_scale(image, mask, image_size, aug.scale_range)
    image, mask = _pad_if_needed(image, mask, image_size, ignore_index)
    if aug.random_crop:
        top, left = _random_crop_params(image, image_size)
        image = TF.crop(image, top, left, image_size, image_size)
        mask = TF.crop(mask, top, left, image_size, image_size)
    else:
        image = TF.resize(image, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [image_size, image_size], interpolation=TF.InterpolationMode.NEAREST)
    if torch.rand(1).item() < aug.hflip_prob:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    return _to_tensors(image, mask, aug)


def val_preprocessing(
    image: Image.Image,
    mask: Image.Image,
    image_size: int,
    cfg: AugmentationConfig | dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    aug = cfg if isinstance(cfg, AugmentationConfig) else AugmentationConfig.from_dict(cfg)
    image = TF.resize(image, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, [image_size, image_size], interpolation=TF.InterpolationMode.NEAREST)
    return _to_tensors(image, mask, aug)


def _to_tensors(image: Image.Image, mask: Image.Image, aug: AugmentationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = TF.normalize(TF.to_tensor(image), aug.normalize_mean, aug.normalize_std)
    mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
    return image_tensor, mask_tensor


def _random_scale(
    image: Image.Image,
    mask: Image.Image,
    image_size: int,
    scale_range: tuple[float, float],
) -> tuple[Image.Image, Image.Image]:
    scale = torch.empty(1).uniform_(*scale_range).item()
    width, height = image.size
    short_side = max(1, int(image_size * scale))
    if width <= height:
        new_width = short_side
        new_height = int(height * short_side / width)
    else:
        new_height = short_side
        new_width = int(width * short_side / height)
    image = TF.resize(image, [new_height, new_width], interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, [new_height, new_width], interpolation=TF.InterpolationMode.NEAREST)
    return image, mask


def _pad_if_needed(
    image: Image.Image,
    mask: Image.Image,
    image_size: int,
    ignore_index: int,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    pad_w = max(image_size - width, 0)
    pad_h = max(image_size - height, 0)
    if pad_w > 0 or pad_h > 0:
        image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
        mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=ignore_index)
    return image, mask


def _random_crop_params(image: Image.Image, image_size: int) -> tuple[int, int]:
    width, height = image.size
    top = torch.randint(0, height - image_size + 1, (1,)).item()
    left = torch.randint(0, width - image_size + 1, (1,)).item()
    return top, left
