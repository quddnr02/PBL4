# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    return F.interpolate(input, size, scale_factor, mode, align_corners)
