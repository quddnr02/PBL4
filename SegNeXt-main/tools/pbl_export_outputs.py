#!/usr/bin/env python
"""Export PBL-friendly CSV/PNG outputs from an official MMSeg/SegNeXt run.

This script intentionally uses MMSegmentation's config/model/inference APIs. It
is a reporting layer only; it does not reimplement SegNeXt.
"""

import argparse
import csv
import glob
import json
import os
import shutil
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor
from PIL import Image, ImageDraw, ImageFont

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

VOC_PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
    [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
    [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
], dtype=np.uint8)

IGNORE_COLOR = np.array([255, 255, 255], dtype=np.uint8)
ERROR_COLOR = np.array([255, 0, 0], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='Export PBL SegNeXt outputs.')
    parser.add_argument('--config', required=True, help='MMSeg config path.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path or glob, e.g. best_mIoU*.pth.')
    parser.add_argument('--work-dir', required=True, help='Run output directory.')
    parser.add_argument('--voc-root', required=True, help='VOC2012 root containing JPEGImages and SegmentationClass.')
    parser.add_argument('--val-txt', required=True, help='Validation image-id split file. Never pass pbl_train.txt here.')
    parser.add_argument('--save-wrong', type=int, default=12, help='Number of lowest per-image mIoU samples to visualize.')
    parser.add_argument('--device', default='cuda:0', help='Inference device, e.g. cuda:0 or cpu.')
    parser.add_argument('--ignore-index', type=int, default=255)
    return parser.parse_args()


def resolve_one(pattern):
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    if os.path.exists(pattern):
        return pattern
    raise FileNotFoundError(f'No file matched: {pattern}')


def read_split(path):
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            value = line.strip()
            if value:
                ids.append(os.path.splitext(value)[0])
    return ids


def find_log_jsons(work_dir):
    patterns = [
        os.path.join(work_dir, '*.log.json'),
        os.path.join(work_dir, '**', '*.log.json'),
        os.path.join(work_dir, '*.json'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def load_history(work_dir):
    rows = []
    for log_path in find_log_jsons(work_dir):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and ('epoch' in row or 'iter' in row):
                    row['_source_log'] = os.path.relpath(log_path, work_dir)
                    rows.append(row)
    return rows


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        keys = OrderedDict()
        for row in rows:
            for key in row:
                keys[key] = None
        fieldnames = list(keys)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scalar(row, key):
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def pick_metric_rows(rows):
    metric_rows = [r for r in rows if scalar(r, 'mIoU') is not None]
    if not metric_rows:
        metric_rows = [r for r in rows if scalar(r, 'IoU') is not None]
    best = max(metric_rows, key=lambda r: scalar(r, 'mIoU') if scalar(r, 'mIoU') is not None else scalar(r, 'IoU')) if metric_rows else {}
    last = metric_rows[-1] if metric_rows else (rows[-1] if rows else {})
    return best, last


def plot_loss_miou(rows, out_path):
    train_rows = [r for r in rows if scalar(r, 'loss') is not None]
    metric_rows = [r for r in rows if scalar(r, 'mIoU') is not None]
    plt.figure(figsize=(10, 5))
    if train_rows:
        x = [scalar(r, 'iter') or idx for idx, r in enumerate(train_rows)]
        y = [scalar(r, 'loss') for r in train_rows]
        plt.plot(x, y, label='train loss')
    if metric_rows:
        x = [scalar(r, 'epoch') or idx for idx, r in enumerate(metric_rows)]
        y = [scalar(r, 'mIoU') for r in metric_rows]
        plt.plot(x, y, label='val mIoU')
    plt.xlabel('iteration / epoch')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def colorize(mask, ignore_index=255):
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(VOC_PALETTE))
    result[valid] = VOC_PALETTE[mask[valid]]
    result[mask == ignore_index] = IGNORE_COLOR
    return result


def update_confusion(confusion, gt, pred, num_classes=21, ignore_index=255):
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    encoded = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    bincount = np.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bincount.reshape(num_classes, num_classes)


def iou_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
    return iou


def sample_miou(gt, pred, num_classes=21, ignore_index=255):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    update_confusion(confusion, gt, pred, num_classes, ignore_index)
    iou = iou_from_confusion(confusion)
    return float(np.nanmean(iou)) if np.isfinite(iou).any() else float('nan')


def panel_title(image, title):
    canvas = Image.new('RGB', (image.width, image.height + 24), (255, 255, 255))
    canvas.paste(image, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def make_wrong_panel(image_path, gt, pred, out_path, ignore_index=255):
    image = Image.open(image_path).convert('RGB')
    gt_img = Image.fromarray(colorize(gt, ignore_index)).resize(image.size, Image.NEAREST)
    pred_img = Image.fromarray(colorize(pred, ignore_index)).resize(image.size, Image.NEAREST)

    error = np.array(image).copy()
    gt_resized = np.array(Image.fromarray(gt).resize(image.size, Image.NEAREST))
    pred_resized = np.array(Image.fromarray(pred).resize(image.size, Image.NEAREST))
    mismatch = (gt_resized != ignore_index) & (gt_resized != pred_resized)
    error[mismatch] = (0.55 * error[mismatch] + 0.45 * ERROR_COLOR).astype(np.uint8)
    error_img = Image.fromarray(error)

    panels = [
        panel_title(image, 'image'),
        panel_title(gt_img, 'GT'),
        panel_title(pred_img, 'pred'),
        panel_title(error_img, 'error overlay'),
    ]
    total_w = sum(p.width for p in panels)
    total_h = max(p.height for p in panels)
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(out_path)


def save_wrong_grid(image_paths, out_path):
    images = [Image.open(path).convert('RGB') for path in image_paths]
    if not images:
        return
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    cols = 1
    rows = len(images)
    canvas = Image.new('RGB', (cols * width, rows * height), (255, 255, 255))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += height
    canvas.save(out_path)


def export_validation_outputs(args, checkpoint):
    cfg = Config.fromfile(args.config)
    model = init_segmentor(cfg, checkpoint, device=args.device)
    ids = read_split(args.val_txt)
    confusion = np.zeros((len(VOC_CLASSES), len(VOC_CLASSES)), dtype=np.int64)
    sample_rows = []
    predictions = {}

    print('[PBL export] Running validation inference')
    print(f'  val images = {len(ids)}')
    for idx, image_id in enumerate(ids, 1):
        image_path = os.path.join(args.voc_root, 'JPEGImages', f'{image_id}.jpg')
        gt_path = os.path.join(args.voc_root, 'SegmentationClass', f'{image_id}.png')
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(gt_path)
        pred = inference_segmentor(model, image_path)[0].astype(np.uint8)
        gt = np.array(Image.open(gt_path), dtype=np.uint8)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST), dtype=np.uint8)
        update_confusion(confusion, gt, pred, len(VOC_CLASSES), args.ignore_index)
        miou = sample_miou(gt, pred, len(VOC_CLASSES), args.ignore_index)
        sample_rows.append({'rank': '', 'image_id': image_id, 'sample_mIoU': miou, 'image_path': image_path, 'gt_path': gt_path})
        predictions[image_id] = pred
        if idx % 25 == 0 or idx == len(ids):
            print(f'  processed {idx}/{len(ids)}')

    write_csv(os.path.join(args.work_dir, 'confusion_matrix_full.csv'), [
        dict(class_name=VOC_CLASSES[i], **{VOC_CLASSES[j]: int(confusion[i, j]) for j in range(len(VOC_CLASSES))})
        for i in range(len(VOC_CLASSES))
    ])

    ious = iou_from_confusion(confusion)
    per_class_rows = []
    for i, class_name in enumerate(VOC_CLASSES):
        per_class_rows.append({
            'class_id': i,
            'class_name': class_name,
            'iou': ious[i],
            'tp': int(confusion[i, i]),
            'gt_pixels': int(confusion[i, :].sum()),
            'pred_pixels': int(confusion[:, i].sum()),
        })
    per_class_rows.append({'class_id': 'mean', 'class_name': 'mIoU', 'iou': float(np.nanmean(ious))})
    write_csv(os.path.join(args.work_dir, 'per_class_iou.csv'), per_class_rows)

    wrong_dir = os.path.join(args.work_dir, 'wrong_sample_images')
    os.makedirs(wrong_dir, exist_ok=True)
    wrong_rows = sorted(sample_rows, key=lambda r: (np.inf if np.isnan(r['sample_mIoU']) else r['sample_mIoU']))[:args.save_wrong]
    wrong_images = []
    for rank, row in enumerate(wrong_rows, 1):
        row['rank'] = rank
        image_id = row['image_id']
        gt = np.array(Image.open(row['gt_path']), dtype=np.uint8)
        pred = predictions[image_id]
        out_path = os.path.join(wrong_dir, f'{rank:02d}_{image_id}.png')
        make_wrong_panel(row['image_path'], gt, pred, out_path, args.ignore_index)
        row['panel_path'] = out_path
        wrong_images.append(out_path)
    write_csv(os.path.join(args.work_dir, 'wrong_samples.csv'), wrong_rows)
    save_wrong_grid(wrong_images, os.path.join(args.work_dir, 'wrong_samples.png'))


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    checkpoint = resolve_one(args.checkpoint)

    print('[PBL export]')
    print(f'  config     = {args.config}')
    print(f'  checkpoint = {checkpoint}')
    print(f'  work_dir   = {args.work_dir}')
    print(f'  voc_root   = {args.voc_root}')
    print(f'  val_txt    = {args.val_txt}')

    history = load_history(args.work_dir)
    write_csv(os.path.join(args.work_dir, 'history.csv'), history)
    best, last = pick_metric_rows(history)
    write_csv(os.path.join(args.work_dir, 'best_results.csv'), [best])
    write_csv(os.path.join(args.work_dir, 'last_results.csv'), [last])
    plot_loss_miou(history, os.path.join(args.work_dir, 'loss_miou.png'))
    shutil.copy2(checkpoint, os.path.join(args.work_dir, 'best.pt'))

    cfg = Config.fromfile(args.config)
    cfg.dump(os.path.join(args.work_dir, 'config_exported.py'))
    with open(os.path.join(args.work_dir, 'args_export.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    export_validation_outputs(args, checkpoint)
    print('[PBL export] Done.')


if __name__ == '__main__':
    main()
