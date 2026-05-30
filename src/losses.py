from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils import VOC_IGNORE_INDEX, VOC_NUM_CLASSES


def compute_class_weights(
    mask_paths: list[Path],
    num_classes: int = VOC_NUM_CLASSES,
    ignore_index: int = VOC_IGNORE_INDEX,
    method: str = "median_frequency",
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path), dtype=np.int64)
        valid = (mask != ignore_index) & (mask >= 0) & (mask < num_classes)
        if valid.any():
            counts += np.bincount(mask[valid], minlength=num_classes)
    frequencies = counts / max(counts.sum(), 1.0)
    present = frequencies > 0
    weights = np.ones(num_classes, dtype=np.float64)
    if method == "inverse_frequency":
        weights[present] = 1.0 / np.maximum(frequencies[present], 1e-12)
        weights[present] /= weights[present].mean()
    else:
        median = np.median(frequencies[present]) if present.any() else 1.0
        weights[present] = median / np.maximum(frequencies[present], 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def make_boundary_targets(masks: torch.Tensor, ignore_index: int = VOC_IGNORE_INDEX, dilation: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    valid = masks != ignore_index
    safe = masks.clone()
    safe[~valid] = 0
    boundary = torch.zeros_like(safe, dtype=torch.bool)
    boundary[:, 1:, :] |= (safe[:, 1:, :] != safe[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    boundary[:, :-1, :] |= (safe[:, :-1, :] != safe[:, 1:, :]) & valid[:, :-1, :] & valid[:, 1:, :]
    boundary[:, :, 1:] |= (safe[:, :, 1:] != safe[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    boundary[:, :, :-1] |= (safe[:, :, :-1] != safe[:, :, 1:]) & valid[:, :, :-1] & valid[:, :, 1:]
    boundary_f = boundary.float().unsqueeze(1)
    if dilation > 1:
        boundary_f = F.max_pool2d(boundary_f, kernel_size=dilation, stride=1, padding=dilation // 2)
    return boundary_f, valid.float().unsqueeze(1)


class MultiHeadSegmentationLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = VOC_IGNORE_INDEX,
        boundary_weight: float = 0.2,
        context_weight: float = 0.4,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight
        self.context_weight = context_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs: dict[str, torch.Tensor], masks: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        final_loss = self.ce(outputs["logits"], masks)
        total = final_loss
        parts = {"seg_loss": float(final_loss.detach().cpu())}
        if "context_logits" in outputs:
            context_loss = self.ce(outputs["context_logits"], masks)
            total = total + self.context_weight * context_loss
            parts["context_loss"] = float(context_loss.detach().cpu())
        if "boundary_logits" in outputs:
            boundary_targets, valid = make_boundary_targets(masks, ignore_index=self.ignore_index)
            boundary_targets = boundary_targets.to(outputs["boundary_logits"].device)
            valid = valid.to(outputs["boundary_logits"].device)
            bce = self.bce(outputs["boundary_logits"], boundary_targets)
            boundary_loss = (bce * valid).sum() / valid.sum().clamp_min(1.0)
            total = total + self.boundary_weight * boundary_loss
            parts["boundary_loss"] = float(boundary_loss.detach().cpu())
        parts["total_loss"] = float(total.detach().cpu())
        return total, parts
