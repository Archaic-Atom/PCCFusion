from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.act = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(channels, channels, 3, dilation=dilation),
            ConvGNAct(channels, channels, 3, dilation=dilation, activation=False),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class InvertedResidual(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, dilation: int = 1):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            ConvGNAct(channels, hidden, 1),
            ConvGNAct(hidden, hidden, 3, dilation=dilation, groups=hidden),
            ConvGNAct(hidden, channels, 1, activation=False),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2):
        super().__init__()
        self.down = ConvGNAct(in_channels, out_channels, 3, stride=2)
        self.blocks = nn.Sequential(*[InvertedResidual(out_channels) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.down(x))


class ModalityStem(nn.Module):
    def __init__(self, channels: int = 32, depth: int = 3):
        super().__init__()
        self.in_conv = ConvGNAct(1, channels, 3)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.in_conv(x))


class SharedPyramidEncoder(nn.Module):
    def __init__(self, channels=(32, 48, 80, 128, 160), depths=(2, 2, 2, 2)):
        super().__init__()
        self.stages = nn.ModuleList([
            DownStage(channels[i], channels[i + 1], depths[i])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        feats = [x]
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class FlowCorrectionBlock(nn.Module):
    def __init__(self, content_channels: int):
        super().__init__()
        # ref, warped moving, absolute difference, 4 phase hints
        in_channels = content_channels * 3 + 4
        self.pre = ConvGNAct(in_channels, content_channels, 3)
        self.body = nn.Sequential(
            InvertedResidual(content_channels, expansion=2, dilation=3),
            InvertedResidual(content_channels, expansion=2, dilation=3),
        )
        self.out = nn.Conv2d(content_channels, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ref, moving_warped, phase_hints):
        x = torch.cat([ref, moving_warped, torch.abs(ref - moving_warped), phase_hints], dim=1)
        x = self.body(self.pre(x))
        out = self.out(x)
        return out[:, :2], out[:, 2:3]


class DeepFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, residual_type: str = 'irb'):
        super().__init__()
        self.pre = ConvGNAct(in_channels, out_channels, 3)
        if residual_type == 'irb':
            blocks = [InvertedResidual(out_channels, expansion=2, dilation=3) for _ in range(depth)]
        else:
            blocks = [ResidualBlock(out_channels) for _ in range(depth)]
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return self.body(self.pre(x))


def resize_flow(flow: torch.Tensor, size) -> torch.Tensor:
    """Resize pixel-unit flow while preserving displacement in the target coordinate system."""
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = int(size[0]), int(size[1])
    if (old_h, old_w) == (new_h, new_w):
        return flow
    resized = F.interpolate(flow, size=(new_h, new_w), mode='bilinear', align_corners=True)
    resized[:, 0] *= new_w / max(old_w, 1)
    resized[:, 1] *= new_h / max(old_h, 1)
    return resized


def warp_tensor(src: torch.Tensor, flow: torch.Tensor, padding_mode: str = 'reflection') -> torch.Tensor:
    """
    Warp using ME-PMA's flow convention: output(y,x) samples src(y+dy,x+dx).
    flow channel 0 is dx and channel 1 is dy.
    """
    b, _, h, w = src.shape
    if flow.shape[-2:] != (h, w):
        flow = resize_flow(flow, (h, w))

    yy, xx = torch.meshgrid(
        torch.arange(h, device=src.device, dtype=src.dtype),
        torch.arange(w, device=src.device, dtype=src.dtype),
        indexing='ij',
    )
    base = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    loc = base + flow
    if w > 1:
        loc[:, 0] = 2.0 * loc[:, 0] / (w - 1) - 1.0
    else:
        loc[:, 0] = 0
    if h > 1:
        loc[:, 1] = 2.0 * loc[:, 1] / (h - 1) - 1.0
    else:
        loc[:, 1] = 0
    grid = loc.permute(0, 2, 3, 1)
    return F.grid_sample(
        src,
        grid.clamp(-1, 1),
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True,
    )


class QualityWeightHead(nn.Module):
    """Very small local quality estimator; outputs one IR/VIS weight per pixel."""
    def __init__(self, channels: int):
        super().__init__()
        self.ir_energy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.vis_energy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 1),
        )

    def forward(self, ir, vis, confidence, residual):
        ir_score = self.ir_energy(ir)
        vis_score = self.vis_energy(vis)
        logits = self.mix(torch.cat([ir_score, vis_score, confidence, residual], dim=1))
        # Low confidence should produce a sharper modality choice.
        temperature = 0.25 + 0.75 * confidence
        return torch.softmax(logits / temperature.clamp_min(0.1), dim=1)
