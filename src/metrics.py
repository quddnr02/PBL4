from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from utils import VOC_CLASS_NAMES


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255, class_names: tuple[str, ...] = VOC_CLASS_NAMES):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names[:num_classes]
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

    def save_confusion_matrix(self, path: str | Path) -> None:
        self._write_matrix(Path(path), self.confusion_matrix, float_format=None)

    def save_normalized_confusion_matrix(self, path: str | Path) -> None:
        cm = self.confusion_matrix.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        self._write_matrix(Path(path), normalized, float_format=".6f")

    def save_confusion_outputs(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        self.save_confusion_matrix(output_dir / "confusion_matrix_full.csv")
        self.save_normalized_confusion_matrix(output_dir / "normalized_confusion_matrix.csv")

    def _write_matrix(self, path: Path, matrix: np.ndarray, float_format: str | None) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Ground Truth \\ Prediction", *self.class_names])
            for class_name, row in zip(self.class_names, matrix):
                if float_format is None:
                    values = [int(v) for v in row]
                else:
                    values = [format(float(v), float_format) for v in row]
                writer.writerow([class_name, *values])
