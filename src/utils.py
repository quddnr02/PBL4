import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


VOC_NUM_CLASSES = 21
VOC_IGNORE_INDEX = 255
VOC_CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def voc_label_mappings(num_classes: int = VOC_NUM_CLASSES) -> tuple[dict[int, str], dict[str, int]]:
    if num_classes == len(VOC_CLASS_NAMES):
        id2label = {idx: name for idx, name in enumerate(VOC_CLASS_NAMES)}
    else:
        id2label = {idx: f"class_{idx}" for idx in range(num_classes)}
    label2id = {name: idx for idx, name in id2label.items()}
    return id2label, label2id


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int | None, deterministic: bool = True) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1_000_000


def checkpoint_size_mb(path: str | os.PathLike) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 2)


def estimate_state_dict_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


class SegmentationOutputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["logits"]


def compute_gflops(model: torch.nn.Module, image_size: int, device: torch.device) -> float:
    try:
        from thop import profile
    except ImportError:
        print("[WARN] thop is not installed; GFLOPs will be recorded as 0.0")
        return 0.0

    training = model.training
    wrapper = SegmentationOutputWrapper(model).to(device)
    wrapper.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        with torch.no_grad():
            flops, _ = profile(wrapper, inputs=(dummy,), verbose=False)
        return float(flops) / 1_000_000_000
    except Exception as exc:
        print(f"[WARN] GFLOPs calculation failed: {exc}")
        return 0.0
    finally:
        model.train(training)
