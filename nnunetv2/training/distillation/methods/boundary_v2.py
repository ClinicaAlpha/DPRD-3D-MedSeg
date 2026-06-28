"""
Boundary distillation variant that derives an emphasis mask purely from teacher features.

Workflow per stage:
1. Extract spatial gradients from the teacher feature map (Sobel magnitude).
2. Normalize the gradient magnitude to obtain a reweighting mask.
3. Apply the mask to both teacher and student high-frequency maps.
4. Compute an L2 loss on the masked responses.

This stays agnostic to annotation targets and logits; the only supervision signal
comes from teacher features, matching the lightweight approach requested.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


class BoundaryStageModuleV2(nn.Module):
    """
    Stage-level module that builds a teacher-derived emphasis mask and applies it
    to student/teacher high-frequency features before measuring their discrepancy.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        sobel_scale: float = 1.0,
        softmax_temperature: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.teacher_channels = teacher_channels
        self.sobel_scale = float(sobel_scale)
        self.eps = float(eps)
        self.softmax_temperature = float(softmax_temperature)

        self.align: Optional[nn.Conv3d] = None
        if student_channels != teacher_channels:
            self._reset_align(student_channels)

        self._init_sobel_kernels()

    def _reset_align(self, in_channels: int, device: Optional[torch.device] = None) -> None:
        conv = nn.Conv3d(in_channels, self.teacher_channels, kernel_size=1, bias=False)
        if device is not None:
            conv = conv.to(device)
        self.align = conv

    def _init_sobel_kernels(self) -> None:
        """Define 3D Sobel kernels once."""
        sobel_z = torch.tensor(
            [
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            ],
            dtype=torch.float32,
        ) / 16.0

        sobel_y = torch.tensor(
            [
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            ],
            dtype=torch.float32,
        ) / 16.0

        sobel_x = torch.tensor(
            [
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            ],
            dtype=torch.float32,
        ) / 16.0

        self.register_buffer("sobel_z", sobel_z.view(1, 1, 3, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3, 3), persistent=False)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3, 3), persistent=False)

    def extract_high_frequency(self, features: torch.Tensor) -> torch.Tensor:
        """Apply Sobel filters channel-wise and return gradient magnitude."""
        b, c, d, h, w = features.shape
        reshaped = features.contiguous().view(b * c, 1, d, h, w)
        hf_z = F.conv3d(reshaped, self.sobel_z, padding=1)
        hf_y = F.conv3d(reshaped, self.sobel_y, padding=1)
        hf_x = F.conv3d(reshaped, self.sobel_x, padding=1)
        magnitude = torch.sqrt(hf_x**2 + hf_y**2 + hf_z**2 + self.eps)
        return magnitude.view(b, c, d, h, w) * self.sobel_scale

    def _build_reweight_mask(self, teacher_hf: torch.Tensor) -> torch.Tensor:
        """
        Build a per-channel softmax mask over spatial positions.

        Each channel's weights sum to 1, stabilising the loss scale without touching
        ground-truth signals. Temperature controls how peaky the focus becomes.
        """
        b, c, d, h, w = teacher_hf.shape
        logits = teacher_hf.view(b, c, -1)

        tau = max(self.softmax_temperature, self.eps)
        logits = logits / tau
        logits = logits - logits.max(dim=2, keepdim=True).values
        weights = torch.softmax(logits, dim=2)
        mask = weights.view(b, c, d, h, w)

        # Rescale so that each channel's mean stays close to 1, avoiding resolution-dependent shrinkage.
        voxels = float(d * h * w)
        if voxels > 0:
            mask = mask * voxels
        return mask

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.align is not None:
            if self.align.weight.device != student_feat.device:
                self.align = self.align.to(student_feat.device)
            if student_feat.shape[1] != self.align.in_channels:
                self._reset_align(student_feat.shape[1], student_feat.device)
            student_feat = self.align(student_feat)
        elif student_feat.shape[1] != self.teacher_channels:
            self._reset_align(student_feat.shape[1], student_feat.device)
            student_feat = self.align(student_feat)

        teacher_hf = self.extract_high_frequency(teacher_feat.detach())
        student_hf = self.extract_high_frequency(student_feat)

        mask_t = self._build_reweight_mask(teacher_hf)
        mask_s = self._build_reweight_mask(student_hf)
        mask = mask_t * mask_s
        teacher_masked = teacher_hf * mask
        student_masked = student_hf * mask

        loss = F.mse_loss(student_masked, teacher_masked, reduction="mean")
        # loss = loss / mask.shape[0]  # average over batch
        return loss, mask


class BoundaryDistillationV2(DistillationMethod, nn.Module):
    """
    Pure feature-based boundary distillation using teacher-derived emphasis masks.
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        required = ["student_channels", "teacher_channels"]
        for key in required:
            if key not in config:
                raise ValueError(f"BoundaryDistillationV2 requires '{key}' in config")

        student_channels = config["student_channels"]
        teacher_channels = config["teacher_channels"]
        student_channels = list(student_channels) if isinstance(student_channels, (list, tuple)) else [student_channels]
        teacher_channels = list(teacher_channels) if isinstance(teacher_channels, (list, tuple)) else [teacher_channels]

        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        self.layer_indices: List[int] = self._parse_indices(
            config.get("layer_indices"),
            len(student_channels),
        )
        sobel_scale = float(config.get("sobel_scale", 1.0))
        softmax_temperature = float(config.get("softmax_temperature", 0.5))
        eps = float(config.get("eps", 1e-6))

        self.stage_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            module = BoundaryStageModuleV2(
                student_channels=student_channels[idx],
                teacher_channels=teacher_channels[idx],
                sobel_scale=sobel_scale,
                softmax_temperature=softmax_temperature,
                eps=eps,
            )
            self.stage_modules[str(idx)] = module

    @staticmethod
    def _parse_indices(
        raw_indices: Optional[Union[int, str, List[int], Tuple[int, ...]]],
        num_stages: int,
    ) -> List[int]:
        if raw_indices is None:
            indices: List[int] = [num_stages - 1]
        elif isinstance(raw_indices, int):
            indices = [raw_indices]
        elif isinstance(raw_indices, str):
            lowered = raw_indices.lower()
            if lowered in ("all", "*"):
                indices = list(range(num_stages))
            else:
                indices = [int(raw_indices)]
        else:
            indices = [int(i) for i in raw_indices]

        processed: List[int] = []
        for idx in indices:
            if idx < 0:
                idx = num_stages + idx
            if idx < 0 or idx >= num_stages:
                raise ValueError(f"layer index {idx} out of bounds for {num_stages} stages")
            processed.append(idx)
        return sorted(set(processed))

    # ---------- Stage-wise helpers ----------

    def get_stage_indices(self) -> List[int]:
        return list(self.layer_indices)

    def compute_stage_loss(
        self,
        stage_idx: int,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        student_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del target, student_output  # unused on purpose
        if str(stage_idx) not in self.stage_modules:
            raise ValueError(f"Stage {stage_idx} not configured for BoundaryDistillationV2")

        module = self.stage_modules[str(stage_idx)]
        loss, mask = module(student_feat, teacher_feat)

        return loss, {
            "boundary": loss.detach(),
            "mask_mean": mask.mean().detach(),
        }

    # ---------- DistillationMethod compatibility ----------

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, float] = {}

        for idx in self.layer_indices:
            key = f"stage{idx}"
            student_feat = student_features.get(key)
            teacher_feat = teacher_features.get(key)
            if student_feat is None or teacher_feat is None:
                raise ValueError(
                    f"BoundaryDistillationV2 requires '{key}' features. "
                    f"Student keys: {list(student_features.keys())}, "
                    f"Teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                idx,
                student_feat,
                teacher_feat,
                target,
                student_output=None,
            )
            total_loss = stage_loss if total_loss is None else total_loss + stage_loss
            for name, value in stage_loss_dict.items():
                value_item = value.item() if isinstance(value, torch.Tensor) else float(value)
                loss_dict[f"stage{idx}_{name}"] = value_item

        if total_loss is None:
            total_loss = torch.zeros((), device=next(iter(student_features.values())).device)
        return total_loss, loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{idx}": f"encoder.stages[{idx}]" for idx in self.layer_indices}

    def to(self, device: torch.device) -> "BoundaryDistillationV2":
        self.stage_modules = self.stage_modules.to(device)
        return self


__all__ = ["BoundaryStageModuleV2", "BoundaryDistillationV2"]
