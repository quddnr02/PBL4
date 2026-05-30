import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)
from torchvision.transforms import functional as TF

from metrics import SegMetrics


VOC_NUM_CLASSES = 21
VOC_IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AUG_SCALE_RANGE = (0.75, 1.50)


EXPERIMENTS = {
    "fcn_resnet50": {
        "label": "FCN + ResNet-50",
        "builder": fcn_resnet50,
        "backbone_weights": ResNet50_Weights.IMAGENET1K_V1,
    },
    "deeplabv3_resnet50": {
        "label": "DeepLabV3 + ResNet-50",
        "builder": deeplabv3_resnet50,
        "backbone_weights": ResNet50_Weights.IMAGENET1K_V1,
    },
    "deeplabv3_mobilenetv3": {
        "label": "DeepLabV3 + MobileNetV3",
        "builder": deeplabv3_mobilenet_v3_large,
        "backbone_weights": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    },
    "lraspp_mobilenetv3": {
        "label": "LR-ASPP + MobileNetV3",
        "builder": lraspp_mobilenet_v3_large,
        "backbone_weights": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    },
}


class VOCSegmentationDataset(Dataset):
    def __init__(self, voc_root, split_txt, image_size=384, train=False):
        self.voc_root = Path(voc_root)
        self.image_size = image_size
        self.train = train

        with open(split_txt, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.image_dir = self.voc_root / "JPEGImages"
        self.mask_dir = self.voc_root / "SegmentationClass"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                "VOC root must contain JPEGImages and SegmentationClass folders: "
                f"{self.voc_root}"
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{image_id}.png")

        image, mask = self._transform(image, mask)
        return image, mask.long()

    def _transform(self, image, mask):
        if self.train:
            image, mask = self._train_transform(image, mask)
        else:
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

    def _train_transform(self, image, mask):
        image, mask = self._random_scale(image, mask)
        image, mask = self._pad_if_needed(image, mask)

        top, left = self._random_crop_params(image)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask

    def _random_scale(self, image, mask):
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

    def _pad_if_needed(self, image, mask):
        width, height = image.size
        pad_right = max(0, self.image_size - width)
        pad_bottom = max(0, self.image_size - height)

        if pad_right > 0 or pad_bottom > 0:
            padding = [0, 0, pad_right, pad_bottom]
            image = TF.pad(image, padding, fill=0)
            mask = TF.pad(mask, padding, fill=VOC_IGNORE_INDEX)

        return image, mask

    def _random_crop_params(self, image):
        width, height = image.size
        max_top = height - self.image_size
        max_left = width - self.image_size
        top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
        left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
        return top, left



def build_model(experiment_name, num_classes, pretrained_backbone=True):
    cfg = EXPERIMENTS[experiment_name]
    weights_backbone = cfg["backbone_weights"] if pretrained_backbone else None
    return cfg["builder"](
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    start_time = time.perf_counter()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)

    elapsed = time.perf_counter() - start_time
    return total_loss / len(loader.dataset), elapsed


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    metrics = SegMetrics(num_classes=num_classes)
    total_loss = 0.0
    start_time = time.perf_counter()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)

        total_loss += loss.item() * images.size(0)
        metrics.update(outputs.cpu(), masks.cpu())

    pa, miou = metrics.get_result()
    elapsed = time.perf_counter() - start_time
    return total_loss / len(loader.dataset), pa, miou, elapsed


def peak_memory_mb(device):
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def run_experiment(args, experiment_name):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VOCSegmentationDataset(args.voc_root, args.train_txt, args.image_size, train=True)
    val_dataset = VOCSegmentationDataset(args.voc_root, args.val_txt, args.image_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(experiment_name, args.num_classes, pretrained_backbone=not args.no_pretrained_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=VOC_IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_miou = -1.0
    rows = []

    print(f"\n[{experiment_name}] {EXPERIMENTS[experiment_name]['label']}")
    print(f"Device: {device} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_loss, train_time = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, pa, miou, val_time = evaluate(model, val_loader, criterion, device, args.num_classes)
        scheduler.step()

        mem_mb = peak_memory_mb(device)
        row = {
            "experiment": experiment_name,
            "label": EXPERIMENTS[experiment_name]["label"],
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "pixel_accuracy": pa,
            "miou": miou,
            "train_time_sec": train_time,
            "val_time_sec": val_time,
            "peak_gpu_memory_mb": mem_mb,
            "lr": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"PA={pa:.4f} mIoU={miou:.4f} "
            f"time={train_time:.1f}s mem={mem_mb:.1f}MB"
        )

        if miou > best_miou:
            best_miou = miou
            if not args.no_save_checkpoint:
                torch.save(
                    {
                        "experiment": experiment_name,
                        "label": EXPERIMENTS[experiment_name]["label"],
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "miou": miou,
                        "pixel_accuracy": pa,
                        "args": vars(args),
                    },
                    output_dir / "best.pt",
                )

    write_csv(output_dir / "history.csv", rows)
    return rows[-1], max(rows, key=lambda x: x["miou"])


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    default_root = r"C:\Users\syumi\Desktop\머러기\pbl_seg\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"
    default_train = r"C:\Users\syumi\Desktop\머러기\pbl_seg\pbl_train.txt"
    default_val = r"C:\Users\syumi\Desktop\머러기\pbl_seg\pbl_val.txt"
    default_output = r"C:\Users\syumi\Desktop\머러기\pbl_seg\runs"

    parser = argparse.ArgumentParser(description="VOC2012 segmentation backbone/head comparison")
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS.keys()], default="all")
    parser.add_argument("--voc-root", default=default_root)
    parser.add_argument("--train-txt", default=default_train)
    parser.add_argument("--val-txt", default=default_val)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--num-classes", type=int, default=VOC_NUM_CLASSES)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true", help="Do not save best.pt checkpoint files")
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]
    all_rows = []
    summary_rows = []

    for experiment_name in experiments:
        last_row, best_row = run_experiment(args, experiment_name)
        all_rows.append(last_row)
        summary_rows.append(best_row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "last_results.csv", all_rows)
    write_csv(output_dir / "best_results.csv", summary_rows)

    print("\nBest validation results")
    for row in summary_rows:
        print(
            f"{row['label']}: mIoU={row['miou']:.4f}, "
            f"PA={row['pixel_accuracy']:.4f}, "
            f"epoch={row['epoch']}, mem={row['peak_gpu_memory_mb']:.1f}MB"
        )


if __name__ == "__main__":
    main()
