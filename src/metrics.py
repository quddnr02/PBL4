from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion_matrix.fill(0)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        labels = targets.detach().cpu().numpy().astype(np.int64)
        mask = (labels != self.ignore_index) & (labels >= 0) & (labels < self.num_classes)
        encoded = labels[mask] * self.num_classes + preds[mask]
        counts = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        cm = self.confusion_matrix.astype(np.float64)
        diag = np.diag(cm)
        total = cm.sum()
        pa = float(diag.sum() / total) if total > 0 else 0.0
        denom = cm.sum(axis=1) + cm.sum(axis=0) - diag
        iou = np.divide(diag, denom, out=np.full_like(diag, np.nan), where=denom > 0)
        miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
        return {"PA": pa, "mIoU": miou}

    def save_confusion_matrix(self, path: str) -> None:
        np.savetxt(path, self.confusion_matrix, fmt="%d", delimiter=",")
