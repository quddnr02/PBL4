from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VOCSegmentationDataset
from losses import MultiHeadSegmentationLoss, compute_class_weights
from metrics import SegmentationMetrics
from model import SegFormerMultiHead
from utils import (
    VOC_IGNORE_INDEX,
    VOC_NUM_CLASSES,
    checkpoint_size_mb,
    compute_gflops,
    count_parameters_m,
    ensure_dir,
    estimate_state_dict_size_mb,
    load_config,
    peak_memory_mb,
    set_seed,
)

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "PA",
    "mIoU",
    "Params_M",
    "GFLOPs",
    "peak_gpu_memory_mb",
    "lr",
    "train_time_sec",
    "val_time_sec",
]


def make_loaders(cfg: dict) -> tuple[VOCSegmentationDataset, VOCSegmentationDataset, DataLoader, DataLoader]:
    train_dataset = VOCSegmentationDataset(
        cfg["voc_root"], cfg["train_txt"], cfg["image_size"], train=True, augmentation=cfg.get("augmentation")
    )
    val_dataset = VOCSegmentationDataset(
        cfg["voc_root"], cfg["val_txt"], cfg["image_size"], train=False, augmentation=cfg.get("augmentation")
    )
    common = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_dataset, val_dataset, train_loader, val_loader


def build_loss(cfg: dict, train_dataset: VOCSegmentationDataset, device: torch.device) -> MultiHeadSegmentationLoss:
    class_weights = None
    if bool(cfg.get("use_class_weight", True)):
        class_weights = compute_class_weights(train_dataset.mask_paths(), method=cfg.get("class_weight_method", "median_frequency"))
        print("Calculated class weights:", [round(v, 6) for v in class_weights.tolist()])
        class_weights = class_weights.to(device)
    return MultiHeadSegmentationLoss(
        class_weights=class_weights,
        ignore_index=VOC_IGNORE_INDEX,
        boundary_weight=float(cfg.get("boundary_loss_weight", 0.3)),
        context_weight=float(cfg.get("context_loss_weight", 0.3)),
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch: int) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    start = time.perf_counter()
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss, _ = criterion(outputs, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(loader.dataset), 1), time.perf_counter() - start


@torch.no_grad()
def validate(model, loader, criterion, device, epoch: int) -> tuple[float, dict[str, float], SegmentationMetrics, float]:
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(VOC_NUM_CLASSES, VOC_IGNORE_INDEX)
    start = time.perf_counter()
    progress = tqdm(loader, desc=f"val epoch {epoch}", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        loss, _ = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)
        metrics.update(outputs["logits"], masks)
    return total_loss / max(len(loader.dataset), 1), metrics.compute(), metrics, time.perf_counter() - start


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_checkpoint(path: Path, model, optimizer, epoch: int, row: dict, cfg: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "config": cfg,
        },
        path,
    )


def run_experiment(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42), deterministic=bool(cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    experiment_name = cfg["experiment_name"]
    output_dir = ensure_dir(Path(cfg.get("output_dir", "runs")) / experiment_name)
    print(f"\n===== {experiment_name} =====")
    print(f"Config: {config_path}")
    print(f"Device: {device}")

    train_dataset, _, train_loader, val_loader = make_loaders(cfg)
    model = SegFormerMultiHead(
        num_classes=VOC_NUM_CLASSES,
        use_boundary_head=bool(cfg.get("use_boundary_head", False)),
        use_context_head=bool(cfg.get("use_context_head", False)),
        pretrained_model_name=cfg.get("pretrained_model_name", "nvidia/segformer-b1-finetuned-ade-512-512"),
    ).to(device)
    params_m = count_parameters_m(model)
    gflops = compute_gflops(model, int(cfg["image_size"]), device)
    print(f"Params(M): {params_m:.3f}")
    print(f"GFLOPs: {gflops:.3f}")
    print(f"Model size(MB, estimated state_dict / best.pt function): {estimate_state_dict_size_mb(model):.3f} / {checkpoint_size_mb(output_dir / 'best.pt'):.3f}")

    criterion = build_loss(cfg, train_dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.01)))
    scaler = GradScaler(enabled=device.type == "cuda") if device.type == "cuda" else None

    history = []
    best_row = None
    best_miou = -1.0
    best_cm = None
    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss, train_time = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, metric_values, val_metrics, val_time = validate(model, val_loader, criterion, device, epoch)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "PA": metric_values["PA"],
            "mIoU": metric_values["mIoU"],
            "Params_M": params_m,
            "GFLOPs": gflops,
            "peak_gpu_memory_mb": peak_memory_mb(device),
            "lr": optimizer.param_groups[0]["lr"],
            "train_time_sec": train_time,
            "val_time_sec": val_time,
        }
        history.append(row)
        write_rows(output_dir / "history.csv", history, HISTORY_FIELDS)
        val_metrics.save_confusion_outputs(output_dir)
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"PA={metric_values['PA']:.4f} mIoU={metric_values['mIoU']:.4f}"
        )
        if row["mIoU"] > best_miou:
            best_miou = row["mIoU"]
            best_row = row.copy()
            best_cm = val_metrics.confusion_matrix.copy()
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, row, cfg)
            print(f"Saved best checkpoint: {output_dir / 'best.pt'} ({checkpoint_size_mb(output_dir / 'best.pt'):.3f} MB)")

    last_row = history[-1]
    if best_row is not None:
        write_rows(output_dir / "best_results.csv", [best_row], HISTORY_FIELDS)
        write_rows(output_dir / "summary.csv", [best_row], HISTORY_FIELDS)
        if best_cm is not None:
            best_metrics = SegmentationMetrics(VOC_NUM_CLASSES, VOC_IGNORE_INDEX)
            best_metrics.confusion_matrix = best_cm
            best_metrics.save_confusion_outputs(output_dir)
    write_rows(output_dir / "last_results.csv", [last_row], HISTORY_FIELDS)
    save_checkpoint(output_dir / "last.pt", model, optimizer, epochs, last_row, cfg)
    print(f"Saved last checkpoint: {output_dir / 'last.pt'} ({checkpoint_size_mb(output_dir / 'last.pt'):.3f} MB)")
    print(f"Finished {experiment_name}. Best mIoU={best_miou:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven SegFormer-B1 VOC2012 semantic segmentation trainer")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.config)
