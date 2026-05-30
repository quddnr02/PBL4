from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SemanticHead(nn.Sequential):
    def __init__(self, channels: int, num_classes: int):
        super().__init__(ConvBNReLU(channels, channels, 3), nn.Conv2d(channels, num_classes, 1))


class BoundaryHead(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(ConvBNReLU(channels, channels, 3), ConvBNReLU(channels, channels, 3, dilation=2), nn.Conv2d(channels, 1, 1))


class ContextHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        branch_channels = channels // 2
        self.branches = nn.ModuleList([ConvBNReLU(channels, branch_channels, 3, dilation=d) for d in (1, 3, 5)])
        self.fusion = nn.Sequential(ConvBNReLU(branch_channels * 3, channels, 1), nn.Conv2d(channels, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


class SegFormerMultiHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 21,
        use_boundary_head: bool = False,
        use_context_head: bool = False,
        decoder_channels: int = 256,
        pretrained_model_name: str = "nvidia/mit-b1",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_boundary_head = use_boundary_head
        self.use_context_head = use_context_head
        try:
            self.encoder = SegformerModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        except Exception as exc:
            print(f"[WARN] Could not load pretrained '{pretrained_model_name}' ({exc}); using random SegFormer-B1 config.")
            config = SegformerConfig(
                output_hidden_states=True,
                hidden_sizes=[64, 128, 320, 512],
                depths=[2, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                decoder_hidden_size=256,
            )
            self.encoder = SegformerModel(config)
        hidden_sizes = list(self.encoder.config.hidden_sizes)
        self.projections = nn.ModuleList([nn.Conv2d(ch, decoder_channels, 1) for ch in hidden_sizes])
        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_channels * len(hidden_sizes), decoder_channels, 1),
            ConvBNReLU(decoder_channels, decoder_channels, 3),
        )
        self.semantic_head = SemanticHead(decoder_channels, num_classes)
        self.boundary_head = BoundaryHead(decoder_channels) if use_boundary_head else None
        self.context_head = ContextHead(decoder_channels, num_classes) if use_context_head else None
        fusion_channels = num_classes
        if use_context_head:
            fusion_channels += num_classes
        if use_boundary_head:
            fusion_channels += 1
        self.logit_fusion = nn.Conv2d(fusion_channels, num_classes, 1) if fusion_channels > num_classes else nn.Identity()

    def _hidden_states(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        features = []
        for feat in hidden_states[-4:]:
            if feat.ndim == 3:
                b, n, c = feat.shape
                h = w = int(n ** 0.5)
                feat = feat.transpose(1, 2).reshape(b, c, h, w)
            features.append(feat)
        return tuple(features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self._hidden_states(x)
        target_size = features[0].shape[-2:]
        fused_features = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode="bilinear", align_corners=False)
            fused_features.append(projected)
        fused = self.fuse(torch.cat(fused_features, dim=1))
        semantic_logits = self.semantic_head(fused)
        logits_to_fuse = [semantic_logits]
        outputs: dict[str, torch.Tensor] = {"semantic_logits": F.interpolate(semantic_logits, size=input_size, mode="bilinear", align_corners=False)}
        if self.context_head is not None:
            context_logits = self.context_head(fused)
            logits_to_fuse.append(context_logits)
            outputs["context_logits"] = F.interpolate(context_logits, size=input_size, mode="bilinear", align_corners=False)
        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(fused)
            logits_to_fuse.append(boundary_logits)
            outputs["boundary_logits"] = F.interpolate(boundary_logits, size=input_size, mode="bilinear", align_corners=False)
        fused_logits = self.logit_fusion(torch.cat(logits_to_fuse, dim=1) if len(logits_to_fuse) > 1 else semantic_logits)
        outputs["logits"] = F.interpolate(fused_logits, size=input_size, mode="bilinear", align_corners=False)
        return outputs
