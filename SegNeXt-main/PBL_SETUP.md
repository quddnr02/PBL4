# PBL VOC2012 SegNeXt-S CE baseline setup

This folder is intended to be used inside the official SegNeXt repository. The
PBL experiment **does not reimplement SegNeXt in a single Python file**. Training
must go through the official MMSegmentation registry/config flow and therefore
uses:

- `mmseg/models/backbones/mscan.py` for the MSCAN-S backbone
- `mmseg/models/decode_heads/ham_head.py` for `LightHamHead` / Ham decoder
- `tools/train.py` and `tools/test.py` for training and evaluation

## Expected project layout

```text
PBL4/
├─ SegNeXt-main/
│  ├─ mmseg/
│  ├─ tools/
│  ├─ configs/
│  ├─ local_configs/
│  ├─ pretrained/
│  │  └─ mscan_s.pth
│  └─ PBL_SETUP.md
├─ VOCdevkit/
│  └─ VOC2012/
│     ├─ JPEGImages/
│     ├─ SegmentationClass/
│     └─ ImageSets/
├─ pbl_train.txt
├─ pbl_val.txt
└─ runs/
```

`pbl_train.txt` and `pbl_val.txt` are image-id lists. The config uses
`pbl_train.txt` only for training and `pbl_val.txt` only for validation/test.

## Create a new virtual environment

PowerShell:

```powershell
cd .\PBL4
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux/WSL2:

```bash
cd PBL4
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Install PyTorch / torchvision

Choose the command that matches your CUDA driver. Example for CUDA 11.3:

```powershell
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

CPU-only fallback:

```powershell
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2
```

## Install MMCV / MMSeg dependencies

The official SegNeXt codebase is based on the MMSegmentation 0.x / MMCV 1.x
style config system. Use a matching `mmcv-full` wheel for your Torch/CUDA pair.
For the CUDA 11.3 + Torch 1.10 example above:

```powershell
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install timm==0.6.12 matplotlib pandas pillow opencv-python
cd .\SegNeXt-main
pip install -v -e .
```

If the Windows `mmcv-full` wheel or local compilation fails, use WSL2 or Colab
with a supported CUDA/Torch/MMCV combination. Do not mix MMSegmentation 1.x/2.x
configs with this official SegNeXt 0.x-style repository unless you also migrate
all APIs and configs.

## Download the MSCAN-S ImageNet checkpoint

The CE baseline must **not** silently fall back to random initialization. The
config raises a clear `FileNotFoundError` if this file is missing:

```text
SegNeXt-main/pretrained/mscan_s.pth
```

Automatic download attempt:

```powershell
cd .\SegNeXt-main
python .\tools\download_mscan.py --variant s
```

If automatic download fails, manually download the official MSCAN-S ImageNet
checkpoint and save it exactly as:

```text
SegNeXt-main/pretrained/mscan_s.pth
```

The downloader currently tries the OpenMMLab mirror:

```text
https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth
```

## Experiment config

Main config:

```text
SegNeXt-main/local_configs/segnext/pbl/segnext_s_512x512_voc_pbl_30epoch.py
```

Key settings:

- model: official `EncoderDecoder` with `MSCAN` backbone and `LightHamHead`
- variant: SegNeXt-S / MSCAN-S
- loss: CrossEntropyLoss baseline only
- classes: 21 VOC classes
- ignore index: 255
- crop size: 512 x 512
- batch size: 4 images/GPU
- optimizer: AdamW, lr `6e-5`, weight decay `0.01`
- decode head lr multiplier: `10`
- scheduler: linear warmup for 3 epochs + polynomial decay
- runner: `EpochBasedRunner`, `max_epochs=30`
- work dir: `..\runs\SegNeXt_S_CE_Official`
- best checkpoint: `evaluation.save_best='mIoU'`

The config prints a readable preflight summary at load time: config path,
`work_dir`, official model name, checkpoint path, split paths/counts, class
count, ignore index, crop size, optimizer/lr/scheduler, and best-mIoU saving.

## Train

PowerShell:

```powershell
cd .\SegNeXt-main
python .\tools\train.py `
  .\local_configs\segnext\pbl\segnext_s_512x512_voc_pbl_30epoch.py `
  --work-dir ..\runs\SegNeXt_S_CE_Official
```

## Test best mIoU checkpoint

PowerShell:

```powershell
cd .\SegNeXt-main
python .\tools\test.py `
  .\local_configs\segnext\pbl\segnext_s_512x512_voc_pbl_30epoch.py `
  ..\runs\SegNeXt_S_CE_Official\best_mIoU*.pth `
  --eval mIoU
```

## Export PBL-style outputs

PowerShell:

```powershell
cd .\SegNeXt-main
python .\tools\pbl_export_outputs.py `
  --config .\local_configs\segnext\pbl\segnext_s_512x512_voc_pbl_30epoch.py `
  --checkpoint ..\runs\SegNeXt_S_CE_Official\best_mIoU*.pth `
  --work-dir ..\runs\SegNeXt_S_CE_Official `
  --voc-root ..\VOCdevkit\VOC2012 `
  --val-txt ..\pbl_val.txt `
  --save-wrong 12
```

Use `--device cpu` if you need to run export without CUDA.

## Expected output structure

After training, testing, and export, `runs/SegNeXt_S_CE_Official/` should contain
MMSeg outputs plus PBL-compatible files:

```text
runs/SegNeXt_S_CE_Official/
├─ *.log
├─ *.log.json
├─ best_mIoU*.pth
├─ latest.pth
├─ history.csv
├─ best_results.csv
├─ last_results.csv
├─ best.pt
├─ confusion_matrix_full.csv
├─ per_class_iou.csv
├─ loss_miou.png
├─ wrong_samples.csv
├─ wrong_samples.png
├─ wrong_sample_images/
│  ├─ 01_<image_id>.png
│  ├─ 02_<image_id>.png
│  └─ ...
├─ config_exported.py
└─ args_export.json
```

`wrong_sample_images/*.png` are 4-panel visualizations in the order
`image / GT / pred / error overlay`. VOC colors are used for ground truth and
predictions. Pixels with `ignore_index=255` are rendered white.

## Self-check checklist

Before launching the full run, verify:

1. `SegNeXt-main/pretrained/mscan_s.pth` exists.
2. `pbl_train.txt` and `pbl_val.txt` exist in the PBL4 root.
3. The config load summary reports `num_classes=21`, `ignore_index=255`, and
   nonzero train/val image counts.
4. The training command writes to `runs/SegNeXt_S_CE_Official`.
5. MMSeg saves `best_mIoU*.pth` after validation.
6. The export script creates all CSV/PNG files listed above.
