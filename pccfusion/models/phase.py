from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, warp_tensor


def _gabor_quadrature_bank(
    kernel_size: int = 21,
    num_scales: int = 3,
    num_orientations: int = 6,
    gamma: float = 0.65,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create fixed zero-mean oriented quadrature filters used by phase congruency."""
    radius = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing='ij',
    )
    even_filters = []
    odd_filters = []
    wavelengths = torch.logspace(math.log10(3.5), math.log10(10.0), num_scales)
    for wavelength in wavelengths:
        sigma = 0.56 * float(wavelength)
        for orient in range(num_orientations):
            theta = orient * math.pi / num_orientations
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            envelope = torch.exp(-(x_theta.square() + gamma ** 2 * y_theta.square()) / (2 * sigma ** 2))
            phase = 2 * math.pi * x_theta / float(wavelength)
            even = envelope * torch.cos(phase)
            odd = envelope * torch.sin(phase)
            even = even - even.mean()
            odd = odd - odd.mean()
            even = even / even.abs().sum().clamp_min(1e-6)
            odd = odd / odd.abs().sum().clamp_min(1e-6)
            even_filters.append(even)
            odd_filters.append(odd)
    even_bank = torch.stack(even_filters, dim=0).unsqueeze(1)
    odd_bank = torch.stack(odd_filters, dim=0).unsqueeze(1)
    return even_bank, odd_bank


class PhaseCongruencyExtractor(nn.Module):
    """
    Multi-scale, multi-orientation phase-congruency structure carrier.

    The quadrature bank is fixed. A very small shared calibrator suppresses sensor-specific
    noise without learning cross-modal correspondences.
    """

    def __init__(
        self,
        kernel_size: int = 21,
        num_scales: int = 3,
        num_orientations: int = 6,
        noise_threshold: float = 0.02,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.noise_threshold = noise_threshold
        even, odd = _gabor_quadrature_bank(kernel_size, num_scales, num_orientations)
        self.register_buffer('even_bank', even, persistent=True)
        self.register_buffer('odd_bank', odd, persistent=True)
        self.padding = kernel_size // 2
        channels = num_orientations * 2
        self.calibrator = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(6 if channels % 6 == 0 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.GroupNorm(6 if channels % 6 == 0 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # The filters are evaluated in float32 for numerical stability under AMP.
        dtype = image.dtype
        image32 = image.float()
        even = F.conv2d(image32, self.even_bank.float(), padding=self.padding)
        odd = F.conv2d(image32, self.odd_bank.float(), padding=self.padding)
        b, _, h, w = even.shape
        even = even.view(b, self.num_scales, self.num_orientations, h, w)
        odd = odd.view(b, self.num_scales, self.num_orientations, h, w)

        amplitude = torch.sqrt(even.square() + odd.square() + 1e-8)
        sum_even = even.sum(dim=1)
        sum_odd = odd.sum(dim=1)
        local_energy = torch.sqrt(sum_even.square() + sum_odd.square() + 1e-8)
        sum_amplitude = amplitude.sum(dim=1)
        threshold = self.noise_threshold * sum_amplitude.detach().mean(dim=(-2, -1), keepdim=True)
        pc = F.relu(local_energy - threshold) / (sum_amplitude + 1e-6)
        orientation_energy = sum_amplitude / (sum_amplitude.sum(dim=1, keepdim=True) + 1e-6)
        carrier = torch.cat([pc, orientation_energy], dim=1)
        carrier = carrier + 0.1 * self.calibrator(carrier)
        return carrier.to(dtype=dtype)


class LocalPhaseCorrelation(nn.Module):
    """Differentiable local phase correlation with soft-argmax displacement."""

    def __init__(
        self,
        window_size: int,
        stride: int,
        padding: int,
        temperature: float = 24.0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.temperature = float(temperature)
        hann = torch.hann_window(self.window_size, periodic=False)
        self.register_buffer('window', torch.outer(hann, hann), persistent=False)

        coords = torch.arange(self.window_size, dtype=torch.float32) - self.window_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('offset_x', xx.reshape(-1), persistent=False)
        self.register_buffer('offset_y', yy.reshape(-1), persistent=False)

        freq = torch.fft.fftfreq(self.window_size)
        fy, fx = torch.meshgrid(freq, freq, indexing='ij')
        self.register_buffer('freq_x', fx, persistent=False)
        self.register_buffer('freq_y', fy, persistent=False)

    def _patches(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.window_size,
            stride=self.stride,
            padding=self.padding,
        )
        num_patches = patches.shape[-1]
        patches = patches.view(b, c, self.window_size, self.window_size, num_patches)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()
        return patches

    def forward(self, reference: torch.Tensor, moving: torch.Tensor) -> Dict[str, torch.Tensor]:
        if reference.shape != moving.shape:
            raise ValueError(f"Phase correlation inputs must match, got {reference.shape} and {moving.shape}")
        b, c, h, w = reference.shape
        with torch.autocast(device_type=reference.device.type, enabled=False):
            ref_patch = self._patches(reference.float()) * self.window
            mov_patch = self._patches(moving.float()) * self.window
            ref_patch = ref_patch - ref_patch.mean(dim=(-2, -1), keepdim=True)
            mov_patch = mov_patch - mov_patch.mean(dim=(-2, -1), keepdim=True)

            ref_fft = torch.fft.fft2(ref_patch, dim=(-2, -1))
            mov_fft = torch.fft.fft2(mov_patch, dim=(-2, -1))
            cross = ref_fft * torch.conj(mov_fft)
            cross = cross / cross.abs().clamp_min(1e-6)

            # Average orientation channels after phase-only normalization.
            cross_mean = cross.mean(dim=2)
            corr = torch.fft.ifft2(cross_mean, dim=(-2, -1)).real
            corr = torch.fft.fftshift(corr, dim=(-2, -1))
            flat = corr.flatten(-2)
            probabilities = torch.softmax(self.temperature * flat, dim=-1)

            peak_x = (probabilities * self.offset_x).sum(dim=-1)
            peak_y = (probabilities * self.offset_y).sum(dim=-1)
            # ref * conj(moving) produces a peak opposite to the sampling flow convention.
            flow_x = -peak_x
            flow_y = -peak_y
            patch_flow = torch.stack([flow_x, flow_y], dim=2)

            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / math.log(self.window_size * self.window_size)

            model_phase = 2 * math.pi * (
                self.freq_x[None, None] * flow_x[..., None, None]
                + self.freq_y[None, None] * flow_y[..., None, None]
            )
            phase_error = torch.angle(cross_mean) - model_phase
            residual = (1.0 - torch.cos(phase_error)).mean(dim=(-2, -1)) * 0.5
            confidence = ((1.0 - entropy) * torch.exp(-residual / 0.25)).clamp(0.0, 1.0)

            out_h = (h + 2 * self.padding - self.window_size) // self.stride + 1
            out_w = (w + 2 * self.padding - self.window_size) // self.stride + 1
            flow_grid = patch_flow.permute(0, 2, 1).reshape(b, 2, out_h, out_w)
            confidence_grid = confidence.reshape(b, 1, out_h, out_w)
            residual_grid = residual.reshape(b, 1, out_h, out_w)
            peak_grid = flat.max(dim=-1).values.reshape(b, 1, out_h, out_w)

        return {
            'flow_grid': flow_grid.to(reference.dtype),
            'confidence_grid': confidence_grid.to(reference.dtype),
            'residual_grid': residual_grid.to(reference.dtype),
            'peak_grid': peak_grid.to(reference.dtype),
        }


class MultiScalePhaseGeometry(nn.Module):
    """Coarse-to-fine phase geometry at 1/16, 1/8 and 1/4 resolution."""

    def __init__(self, phase_channels: int = 12):
        super().__init__()
        self.extractor = PhaseCongruencyExtractor()
        self.global_pc = LocalPhaseCorrelation(window_size=16, stride=16, padding=0)
        self.local_pc32 = LocalPhaseCorrelation(window_size=16, stride=8, padding=8)
        self.local_pc64 = LocalPhaseCorrelation(window_size=16, stride=8, padding=8)

    @staticmethod
    def _resize_image(image: torch.Tensor, size):
        return F.interpolate(image, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def _dense(result: Dict[str, torch.Tensor], size):
        dense = {}
        for key, value in result.items():
            dense[key.replace('_grid', '')] = F.interpolate(value, size=size, mode='bilinear', align_corners=True)
        return dense

    def forward(self, ir: torch.Tensor, vis: torch.Tensor):
        sizes = {
            2: (max(ir.shape[-2] // 4, 16), max(ir.shape[-1] // 4, 16)),
            3: (max(ir.shape[-2] // 8, 16), max(ir.shape[-1] // 8, 16)),
            4: (max(ir.shape[-2] // 16, 16), max(ir.shape[-1] // 16, 16)),
        }
        # The main model pads dimensions to multiples of 16, so these map to encoder sizes.
        pc_ir = {}
        pc_vis = {}
        for level in (2, 3, 4):
            pc_ir[level] = self.extractor(self._resize_image(ir, sizes[level]))
            pc_vis[level] = self.extractor(self._resize_image(vis, sizes[level]))

        result4 = self._dense(self.global_pc(pc_vis[4], pc_ir[4]), sizes[4])
        flow4 = result4['flow']

        prior3 = F.interpolate(flow4, size=sizes[3], mode='bilinear', align_corners=True) * 2.0
        ir_pc3_warped = warp_tensor(pc_ir[3], prior3)
        result3 = self._dense(self.local_pc32(pc_vis[3], ir_pc3_warped), sizes[3])
        flow3 = prior3 + result3['flow']

        prior2 = F.interpolate(flow3, size=sizes[2], mode='bilinear', align_corners=True) * 2.0
        ir_pc2_warped = warp_tensor(pc_ir[2], prior2)
        result2 = self._dense(self.local_pc64(pc_vis[2], ir_pc2_warped), sizes[2])
        flow2 = prior2 + result2['flow']

        outputs = {
            4: {**result4, 'flow': flow4, 'pc_ir': pc_ir[4], 'pc_vis': pc_vis[4]},
            3: {**result3, 'flow': flow3, 'pc_ir': pc_ir[3], 'pc_vis': pc_vis[3]},
            2: {**result2, 'flow': flow2, 'pc_ir': pc_ir[2], 'pc_vis': pc_vis[2]},
        }
        return outputs
