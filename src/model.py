from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from utils import VOC_CLASS_NAMES, VOC_IGNORE_INDEX


OFFICIAL_PRETRAINED_MODEL = "nvidia/segformer-b1-finetuned-ade-512-512"


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BoundaryHead(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )


class ContextHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNReLU(channels, channels, 3, dilation=d) for d in (2, 4, 6)])
        self.fusion = nn.Conv2d(channels * 3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


class SegFormerMultiHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 21,
        use_boundary_head: bool = False,
        use_context_head: bool = False,
        pretrained_model_name: str = OFFICIAL_PRETRAINED_MODEL,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_boundary_head = use_boundary_head
        self.use_context_head = use_context_head
        self.segformer = self._load_official_segformer(pretrained_model_name, num_classes)
        decoder_channels = int(self.segformer.config.decoder_hidden_size)
        self.boundary_head = BoundaryHead(decoder_channels) if use_boundary_head else None
        self.context_head = ContextHead(decoder_channels, num_classes) if use_context_head else None

    def _load_official_segformer(self, pretrained_model_name: str, num_classes: int) -> SegformerForSemanticSegmentation:
        id2label = {idx: name for idx, name in enumerate(VOC_CLASS_NAMES)}
        label2id = {name: idx for idx, name in id2label.items()}
        try:
            return SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                semantic_loss_ignore_index=VOC_IGNORE_INDEX,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            print(
                f"[WARN] Could not load pretrained '{pretrained_model_name}' ({exc}); "
                "using a randomly initialized official SegFormer-B1 segmentation config."
            )
            config = SegformerConfig(
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                semantic_loss_ignore_index=VOC_IGNORE_INDEX,
                hidden_sizes=[64, 128, 320, 512],
                depths=[2, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                decoder_hidden_size=256,
            )
            return SegformerForSemanticSegmentation(config)

    def _official_decode(self, encoder_hidden_states: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        decode_head = self.segformer.decode_head
        batch_size = encoder_hidden_states[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, linear_proj in zip(encoder_hidden_states, decode_head.linear_projections):
            if self.segformer.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = linear_proj(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.transpose(1, 2)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        decoder_feature = decode_head.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        decoder_feature = decode_head.batch_norm(decoder_feature)
        decoder_feature = decode_head.activation(decoder_feature)
        decoder_feature = decode_head.dropout(decoder_feature)
        logits = decode_head.classifier(decoder_feature)
        return logits, decoder_feature

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        encoder_outputs = self.segformer.segformer(pixel_values=x, output_hidden_states=True, return_dict=True)
        official_logits, decoder_feature = self._official_decode(encoder_outputs.hidden_states)
        outputs: dict[str, torch.Tensor] = {
            "logits": F.interpolate(official_logits, size=input_size, mode="bilinear", align_corners=False),
            "semantic_logits": F.interpolate(official_logits, size=input_size, mode="bilinear", align_corners=False),
        }
        if self.context_head is not None:
            context_logits = self.context_head(decoder_feature)
            outputs["context_logits"] = F.interpolate(context_logits, size=input_size, mode="bilinear", align_corners=False)
        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(decoder_feature)
            outputs["boundary_logits"] = F.interpolate(boundary_logits, size=input_size, mode="bilinear", align_corners=False)
        return outputs
