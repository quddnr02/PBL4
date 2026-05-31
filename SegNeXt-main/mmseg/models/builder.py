# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    return SEGMENTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
