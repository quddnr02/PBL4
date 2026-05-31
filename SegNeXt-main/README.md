# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation (NeurIPS 2022)

This directory vendors the official PyTorch SegNeXt project from
<https://github.com/Visual-Attention-Network/SegNeXt> and keeps the PBL-specific
configuration and helper tools in place.

PBL additions retained in this repository include:

- `configs/_base_/datasets/voc12_pbl.py`
- `local_configs/segnext/pbl/segnext_s_512x512_voc_pbl_30epoch.py`
- `tools/download_mscan.py`
- `tools/pbl_export_outputs.py`
- `PBL_SETUP.md`

The upstream project is based on MMSegmentation v0.24.1 and provides the MSCAN
backbone and LightHamHead decode head used by SegNeXt.
