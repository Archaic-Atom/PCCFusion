from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    DeepFusionBlock,
    FlowCorrectionBlock,
    ModalityStem,
    QualityWeightHead,
    ResidualBlock,
    SharedPyramidEncoder,
    resize_flow,
    warp_tensor,
)
from .phase import MultiScalePhaseGeometry


class PC2Fusion(nn.Module):
    """
    Phase Congruency and Phase Correlation guided unaligned IR-VIS fusion.

    The analytical phase branch proposes geometry and confidence. Learned flow heads are
    confidence-gated residual correctors rather than standalone cross-modal flow networks.
    """

    def __init__(
        self,
        channels=(32, 48, 80, 128, 160),
        stem_depth: int = 3,
        encoder_depths=(2, 2, 2, 2),
    ):
        super().__init__()
        self.channels = tuple(channels)
        self.ir_stem = ModalityStem(channels[0], stem_depth)
        self.vis_stem = ModalityStem(channels[0], stem_depth)
        self.shared_encoder = SharedPyramidEncoder(channels, encoder_depths)
        self.phase_geometry = MultiScalePhaseGeometry()

        self.flow_correct4 = FlowCorrectionBlock(channels[4])
        self.flow_correct3 = FlowCorrectionBlock(channels[3])
        self.flow_correct2 = FlowCorrectionBlock(channels[2])

        self.quality_heads = nn.ModuleDict({
            str(level): QualityWeightHead(channels[level]) for level in range(5)
        })

        self.deep_fusion = DeepFusionBlock(channels[4] * 4 + 4, channels[4], depth=4, residual_type='irb')
        self.decode3 = DeepFusionBlock(channels[4] + channels[3] * 4 + 4, channels[3], depth=2, residual_type='irb')
        self.decode2 = DeepFusionBlock(channels[3] + channels[2] * 4 + 4, channels[2], depth=2, residual_type='irb')
        self.decode1 = DeepFusionBlock(channels[2] + channels[1] * 4 + 4, channels[1], depth=3, residual_type='res')
        self.decode0 = DeepFusionBlock(channels[1] + channels[0] * 4 + 4, 64, depth=3, residual_type='res')
        self.reconstruction = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.out = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16):
        h, w = x.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        # Reflection requires padding less than input size. Replicate is safe for tiny images.
        mode = 'reflect' if h > pad_h and w > pad_w else 'replicate'
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (pad_h, pad_w)

    @staticmethod
    def _crop(x: torch.Tensor, pad):
        pad_h, pad_w = pad
        if pad_h > 0:
            x = x[..., :-pad_h, :]
        if pad_w > 0:
            x = x[..., :, :-pad_w]
        return x

    @staticmethod
    def _logit(x: torch.Tensor):
        x = x.clamp(1e-4, 1 - 1e-4)
        return torch.log(x) - torch.log1p(-x)

    @staticmethod
    def _structure_overlap(pc_ir: torch.Tensor, pc_vis: torch.Tensor):
        # First six channels are directional phase congruency maps.
        ir = pc_ir[:, :6].mean(dim=1, keepdim=True)
        vis = pc_vis[:, :6].mean(dim=1, keepdim=True)
        return (2 * ir * vis / (ir.square() + vis.square() + 1e-6)).clamp(0, 1)

    def _phase_hints(self, phase_level: Dict[str, torch.Tensor], size, flow_override=None):
        flow = resize_flow(phase_level['flow'], size) if flow_override is None else flow_override
        confidence = F.interpolate(phase_level['confidence'], size=size, mode='bilinear', align_corners=True)
        residual = F.interpolate(phase_level['residual'], size=size, mode='bilinear', align_corners=True).clamp(0, 1)
        overlap = self._structure_overlap(phase_level['pc_ir'], phase_level['pc_vis'])
        overlap = F.interpolate(overlap, size=size, mode='bilinear', align_corners=True)
        flow_mag = torch.linalg.vector_norm(flow, dim=1, keepdim=True)
        flow_mag = torch.tanh(flow_mag / 8.0)
        hints = torch.cat([confidence, residual, flow_mag, overlap], dim=1)
        return flow, confidence, residual, overlap, hints

    def _correct_flow(self, level, phase_level, ir_feat, vis_feat, base_flow):
        _, confidence, residual, overlap, hints = self._phase_hints(
            phase_level, ir_feat.shape[-2:], flow_override=base_flow
        )
        ir_warp = warp_tensor(ir_feat, base_flow)
        correction_block = getattr(self, f'flow_correct{level}')
        delta_flow, delta_confidence = correction_block(vis_feat, ir_warp, hints)
        final_flow = base_flow + (1.0 - confidence) * delta_flow
        # Keep the pre-sigmoid confidence for an AMP-safe BCEWithLogits loss.
        final_confidence_logits = self._logit(confidence) + delta_confidence
        final_confidence = torch.sigmoid(final_confidence_logits)
        final_ir_warp = warp_tensor(ir_feat, final_flow)
        final_hints = torch.cat([
            final_confidence,
            residual,
            torch.tanh(torch.linalg.vector_norm(final_flow, dim=1, keepdim=True) / 8.0),
            overlap,
        ], dim=1)
        return {
            'flow': final_flow,
            'confidence': final_confidence,
            'confidence_logits': final_confidence_logits,
            'residual': residual,
            'overlap': overlap,
            'hints': final_hints,
            'ir_warped': final_ir_warp,
        }

    def _guided_pair(self, level, ir_warped, vis, confidence, residual):
        weights = self.quality_heads[str(level)](ir_warped, vis, confidence, residual)
        selected = weights[:, 0:1] * ir_warped + weights[:, 1:2] * vis
        common = 0.5 * (ir_warped + vis)
        guided = confidence * common + (1.0 - confidence) * selected
        return guided, weights

    def forward(self, ir_moved: torch.Tensor, vis: torch.Tensor) -> Dict[str, torch.Tensor]:
        if ir_moved.ndim != 4 or vis.ndim != 4 or ir_moved.shape[1] != 1 or vis.shape[1] != 1:
            raise ValueError('PC2Fusion expects Bx1xHxW infrared and visible luminance tensors.')
        if ir_moved.shape != vis.shape:
            raise ValueError(f'Input shapes must match, got {ir_moved.shape} and {vis.shape}.')

        ir, pad = self._pad_to_multiple(ir_moved)
        vis_pad, pad_vis = self._pad_to_multiple(vis)
        if pad != pad_vis:
            raise RuntimeError('Paired padding mismatch.')

        ir_feats = self.shared_encoder(self.ir_stem(ir))
        vis_feats = self.shared_encoder(self.vis_stem(vis_pad))
        phase = self.phase_geometry(ir, vis_pad)

        # Coarse 1/16 geometry.
        phase_flow4 = resize_flow(phase[4]['flow'], ir_feats[4].shape[-2:])
        geo4 = self._correct_flow(4, phase[4], ir_feats[4], vis_feats[4], phase_flow4)

        # Preserve the analytical fine-scale residual while replacing its coarse prior
        # with the corrected flow from the previous level.
        p4_at_p3 = resize_flow(phase[4]['flow'], phase[3]['flow'].shape[-2:])
        phase_delta3 = phase[3]['flow'] - p4_at_p3
        base3 = resize_flow(geo4['flow'], ir_feats[3].shape[-2:]) + resize_flow(
            phase_delta3, ir_feats[3].shape[-2:]
        )
        geo3 = self._correct_flow(3, phase[3], ir_feats[3], vis_feats[3], base3)

        p3_at_p2 = resize_flow(phase[3]['flow'], phase[2]['flow'].shape[-2:])
        phase_delta2 = phase[2]['flow'] - p3_at_p2
        base2 = resize_flow(geo3['flow'], ir_feats[2].shape[-2:]) + resize_flow(
            phase_delta2, ir_feats[2].shape[-2:]
        )
        geo2 = self._correct_flow(2, phase[2], ir_feats[2], vis_feats[2], base2)

        geometries = {2: geo2, 3: geo3, 4: geo4}
        # High-resolution flows are propagated from the finest analytical/corrected scale.
        for level in (1, 0):
            size = ir_feats[level].shape[-2:]
            flow = resize_flow(geo2['flow'], size)
            confidence_logits = F.interpolate(
                geo2['confidence_logits'], size=size, mode='bilinear', align_corners=True
            )
            confidence = torch.sigmoid(confidence_logits)
            residual = F.interpolate(geo2['residual'], size=size, mode='bilinear', align_corners=True)
            overlap = F.interpolate(geo2['overlap'], size=size, mode='bilinear', align_corners=True)
            ir_warped = warp_tensor(ir_feats[level], flow)
            hints = torch.cat([
                confidence,
                residual,
                torch.tanh(torch.linalg.vector_norm(flow, dim=1, keepdim=True) / 8.0),
                overlap,
            ], dim=1)
            geometries[level] = {
                'flow': flow,
                'confidence': confidence,
                'confidence_logits': confidence_logits,
                'residual': residual,
                'overlap': overlap,
                'hints': hints,
                'ir_warped': ir_warped,
            }

        guided = {}
        modality_weights = {}
        for level in range(5):
            guided[level], modality_weights[level] = self._guided_pair(
                level,
                geometries[level]['ir_warped'],
                vis_feats[level],
                geometries[level]['confidence'],
                geometries[level]['residual'],
            )

        def fusion_inputs(level):
            ir_w = geometries[level]['ir_warped']
            vis_f = vis_feats[level]
            return [ir_w, vis_f, torch.abs(ir_w - vis_f), guided[level], geometries[level]['hints']]

        x4 = self.deep_fusion(torch.cat(fusion_inputs(4), dim=1))
        x3 = F.interpolate(x4, size=vis_feats[3].shape[-2:], mode='bilinear', align_corners=False)
        x3 = self.decode3(torch.cat([x3, *fusion_inputs(3)], dim=1))
        x2 = F.interpolate(x3, size=vis_feats[2].shape[-2:], mode='bilinear', align_corners=False)
        x2 = self.decode2(torch.cat([x2, *fusion_inputs(2)], dim=1))
        x1 = F.interpolate(x2, size=vis_feats[1].shape[-2:], mode='bilinear', align_corners=False)
        x1 = self.decode1(torch.cat([x1, *fusion_inputs(1)], dim=1))
        x0 = F.interpolate(x1, size=vis_feats[0].shape[-2:], mode='bilinear', align_corners=False)
        x0 = self.decode0(torch.cat([x0, *fusion_inputs(0)], dim=1))
        fused = self.out(self.reconstruction(x0))

        full_flow = geometries[0]['flow']
        warped_ir_image = warp_tensor(ir, full_flow)

        output = {
            'fused': self._crop(fused, pad),
            'warped_ir': self._crop(warped_ir_image, pad),
            'flow': self._crop(full_flow, pad),
            'confidence': self._crop(geometries[0]['confidence'], pad),
            'confidence_logits': self._crop(geometries[0]['confidence_logits'], pad),
            'phase_residual': self._crop(geometries[0]['residual'], pad),
            'structure_overlap': self._crop(geometries[0]['overlap'], pad),
            'modality_weights': {
                level: modality_weights[level] for level in modality_weights
            },
            'multi_scale': geometries,
        }
        return output


if __name__ == '__main__':
    model = PC2Fusion()
    ir = torch.rand(1, 1, 256, 256)
    vis = torch.rand(1, 1, 256, 256)
    out = model(ir, vis)
    print({k: v.shape for k, v in out.items() if torch.is_tensor(v)})
