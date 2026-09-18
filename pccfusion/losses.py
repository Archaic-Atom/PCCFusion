from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        gy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('gx', gx[None, None])
        self.register_buffer('gy', gy[None, None])

    def forward(self, x):
        gx = F.conv2d(x, self.gx.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.gy.to(dtype=x.dtype), padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    window = _gaussian_window(window_size, 1.5, x.device, x.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding)
    mu_y = F.conv2d(y, window, padding=padding)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x.square(), window, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(y.square(), window, padding=padding) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding) - mu_xy
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )
    return score.mean()


class PC2Loss(nn.Module):
    def __init__(
        self,
        w_intensity: float = 1.0,
        w_ssim: float = 5.0,
        w_gradient: float = 20.0,
        w_registration: float = 4.0,
        w_phase: float = 0.5,
        w_confidence: float = 0.5,
        w_smoothness: float = 0.2,
    ):
        super().__init__()
        self.w_intensity = w_intensity
        self.w_ssim = w_ssim
        self.w_gradient = w_gradient
        self.w_registration = w_registration
        self.w_phase = w_phase
        self.w_confidence = w_confidence
        self.w_smoothness = w_smoothness
        self.gradient = SobelGradient()

    def _flow_smoothness(self, flow, edge_reference):
        flow_dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        flow_dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        edge_dx = torch.abs(edge_reference[:, :, :, 1:] - edge_reference[:, :, :, :-1])
        edge_dy = torch.abs(edge_reference[:, :, 1:, :] - edge_reference[:, :, :-1, :])
        return (flow_dx * torch.exp(-10 * edge_dx)).mean() + (
            flow_dy * torch.exp(-10 * edge_dy)
        ).mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        ir_aligned: torch.Tensor,
        vis_aligned: torch.Tensor,
    ):
        # All loss terms are intentionally evaluated in FP32. The model forward
        # can still use AMP, while SSIM, gradients, phase residuals and confidence
        # calibration remain numerically stable. Explicit .float() keeps gradients.
        fused = outputs['fused'].float()
        warped_ir = outputs['warped_ir'].float()
        flow = outputs['flow'].float()
        confidence_logits = outputs['confidence_logits'].float()
        phase_residual = outputs['phase_residual'].float()
        overlap = outputs['structure_overlap'].detach().float()
        ir_aligned = ir_aligned.float()
        vis_aligned = vis_aligned.float()

        intensity_target = torch.maximum(ir_aligned, vis_aligned)
        loss_intensity = F.l1_loss(fused, intensity_target)

        grad_ir = self.gradient(ir_aligned)
        grad_vis = self.gradient(vis_aligned)
        grad_fused = self.gradient(fused)
        grad_target = torch.maximum(grad_ir, grad_vis)
        loss_gradient = F.l1_loss(grad_fused, grad_target)

        loss_ssim = 1.0 - 0.5 * (ssim(fused, ir_aligned) + ssim(fused, vis_aligned))

        reg_l1 = F.l1_loss(warped_ir, ir_aligned)
        reg_ssim = 1.0 - ssim(warped_ir, ir_aligned)
        reg_grad = F.l1_loss(self.gradient(warped_ir), grad_ir)
        loss_registration = reg_l1 + reg_ssim + reg_grad

        # Phase residual should be low where phase-congruent structures overlap.
        loss_phase = (phase_residual * (0.25 + 0.75 * overlap)).mean()

        local_error = F.avg_pool2d(
            torch.abs(warped_ir.detach() - ir_aligned), kernel_size=9, stride=1, padding=4
        )
        confidence_target = torch.exp(-local_error / 0.08).clamp(0, 1)
        # BCEWithLogits is AMP-safe; plain BCE on a sigmoid probability is
        # explicitly rejected by PyTorch autocast.
        loss_confidence = F.binary_cross_entropy_with_logits(
            confidence_logits, confidence_target
        )

        loss_smoothness = self._flow_smoothness(flow, vis_aligned)

        total = (
            self.w_intensity * loss_intensity
            + self.w_ssim * loss_ssim
            + self.w_gradient * loss_gradient
            + self.w_registration * loss_registration
            + self.w_phase * loss_phase
            + self.w_confidence * loss_confidence
            + self.w_smoothness * loss_smoothness
        )
        return {
            'loss': total,
            'intensity': loss_intensity,
            'ssim': loss_ssim,
            'gradient': loss_gradient,
            'registration': loss_registration,
            'phase': loss_phase,
            'confidence': loss_confidence,
            'smoothness': loss_smoothness,
        }
