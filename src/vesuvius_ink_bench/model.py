"""Pretrained ResNet3D-152 + 3D-decoder ink-detection architecture (scrollprize/ink_canonical_2um).

Matches ``ScrollPrize/villa`` (``ink-detection/optimized_inference/model_resnet3d_3d_decoder.py``)
with ``GroupNorm(min(32, C), C)`` and ``ReLU`` activations throughout the 3D FPN decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck3D(nn.Module):
    """3D ResNet bottleneck block matching ScrollPrize/villa i3d backbone keys."""

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int | Tuple[int, int, int] = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3DBackbone(nn.Module):
    """3D ResNet-152 encoder returning multi-scale feature maps (c1, c2, c3, c4)."""

    def __init__(self, layers: Tuple[int, int, int, int] = (3, 8, 36, 3)) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def _make_layer(
        self, planes: int, blocks: int, stride: int | Tuple[int, int, int] = 1
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        out_planes = planes * Bottleneck3D.expansion
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_planes),
            )
        layers_list: List[nn.Module] = [
            Bottleneck3D(self.in_planes, planes, stride=stride, downsample=downsample)
        ]
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers_list.append(Bottleneck3D(self.in_planes, planes))
        return nn.Sequential(*layers_list)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class DecoderResBlock3D(nn.Module):
    """3D GroupNorm residual decoder block matching ``ResConvBlock3D``."""

    def __init__(self, in_ch: int, out_ch: int, max_groups: int = 32) -> None:
        super().__init__()
        g = min(max_groups, out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, out_ch)
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        h = F.relu(self.gn1(self.conv1(x)), inplace=True)
        h = self.gn2(self.conv2(h))
        return F.relu(h + identity, inplace=True)


class DepthAttentionCollapse(nn.Module):
    """Attention-weighted collapse along the Z depth dimension."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn_conv = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn_conv(x), dim=2)
        return (x * weights).sum(dim=2)


class AuxHead2D(nn.Module):
    """Auxiliary 2D segmentation head present in the Lightning checkpoint."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x).squeeze(2))


class InkDecoder3D(nn.Module):
    """3D FPN-style decoder matching ``Decoder3DUNet`` in ScrollPrize/villa."""

    def __init__(self) -> None:
        super().__init__()
        in_channels = (256, 512, 1024, 2048)
        out_channels = (64, 128, 256, 512)
        self.channel_reduce = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(ic, oc, kernel_size=1, bias=False),
                    nn.GroupNorm(min(32, oc), oc),
                    nn.ReLU(inplace=True),
                )
                for ic, oc in zip(in_channels, out_channels)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderResBlock3D(512 + 256, 256),
                DecoderResBlock3D(256 + 128, 128),
                DecoderResBlock3D(128 + 64, 64),
            ]
        )
        self.depth_collapse = DepthAttentionCollapse(64)
        self.logit = nn.Conv2d(64, 1, kernel_size=1)
        self.aux_head_s2 = AuxHead2D(256)
        self.aux_head_s1 = AuxHead2D(128)

    def forward(
        self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        r1, r2, r3, r4 = [
            reduce(f) for reduce, f in zip(self.channel_reduce, feats)
        ]

        x = F.interpolate(r4, size=r3.shape[2:], mode="trilinear", align_corners=False)
        x = self.decoder_blocks[0](torch.cat([x, r3], dim=1))

        x = F.interpolate(x, size=r2.shape[2:], mode="trilinear", align_corners=False)
        x = self.decoder_blocks[1](torch.cat([x, r2], dim=1))

        x = F.interpolate(x, size=r1.shape[2:], mode="trilinear", align_corners=False)
        x = self.decoder_blocks[2](torch.cat([x, r1], dim=1))

        x2d = self.depth_collapse(x)
        return self.logit(x2d)


class VillaInkResNet3D(nn.Module):
    """Canonical ScrollPrize/villa ResNet3D-152 + 3D decoder ink detector.

    Matches ``scrollprize/ink_canonical_2um`` (``r152_3ddec_v2_l5_epoch13.ckpt``)
    and outputs 2D ink surface logits ``(B, 1, H, W)`` upsampled to input spatial size.
    """

    def __init__(self, with_norm: bool = True) -> None:
        super().__init__()
        self.with_norm = with_norm
        self.normalization = nn.BatchNorm3d(1) if with_norm else nn.Identity()
        self.backbone = ResNet3DBackbone(layers=(3, 8, 36, 3))
        self.decoder = InkDecoder3D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, h, w = x.shape
        x_norm = self.normalization(x)
        feats = self.backbone(x_norm)
        logits_2d = self.decoder(feats)
        if logits_2d.shape[-2:] != (h, w):
            logits_2d = F.interpolate(
                logits_2d, size=(h, w), mode="bilinear", align_corners=False
            )
        return logits_2d

    @classmethod
    def from_villa_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
    ) -> Tuple["VillaInkResNet3D", Dict[str, Any]]:
        """Load pretrained ScrollPrize/villa checkpoint using safe ``weights_only=True``."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found at {ckpt_path}. "
                "Download it from Hugging Face (scrollprize/ink_canonical_2um)."
            )

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        hparams: Dict[str, Any] = dict(ckpt.get("hyper_parameters", {}))
        with_norm = bool(hparams.get("with_norm", True))
        model = cls(with_norm=with_norm)

        raw_state: Dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)
        cleaned = {
            k.removeprefix("model.").removeprefix("module."): v
            for k, v in raw_state.items()
            if not k.removeprefix("model.").removeprefix("module.").startswith("backbone.fc.")
        }
        model.load_state_dict(cleaned, strict=strict)

        meta: Dict[str, Any] = {
            "checkpoint_path": str(ckpt_path),
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "hyper_parameters": hparams,
            "num_state_dict_tensors": len(cleaned),
        }
        return model, meta
