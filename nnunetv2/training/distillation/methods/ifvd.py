"""
Inter-class feature variance distillation (IFVD) wrapper.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.kd_losses import IFVD_Loss

from .base import DistillationMethod


class IFVDDistillation(DistillationMethod, nn.Module):
    """
    IFVD distillation with per-class feature variance matching.

    Config parameters:
        student_channels (Sequence[int])
        teacher_channels (Sequence[int])
        num_classes (int)
        layer_indices: stages to supervise
        stage_weights: optional per-stage weights
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        student_channels = config.get("student_channels")
        teacher_channels = config.get("teacher_channels")
        num_classes = config.get("num_classes")

        if student_channels is None or teacher_channels is None:
            raise ValueError("IFVDDistillation requires student_channels and teacher_channels")
        if num_classes is None:
            raise ValueError("IFVDDistillation requires 'num_classes' in config")

        student_channels = list(student_channels)
        teacher_channels = list(teacher_channels)
        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        self.num_classes = int(num_classes)
        self.layer_indices = self._parse_indices(config.get("layer_indices"), len(student_channels))
        self.stage_weights = self._resolve_stage_weights(config.get("stage_weights"))

        self.loss_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            self.loss_modules[str(idx)] = IFVD_Loss(
                student_channels=student_channels[idx],
                teacher_channels=teacher_channels[idx],
                num_classes=self.num_classes,
            )

    @staticmethod
    def _parse_indices(
        raw_indices: Optional[Union[int, str, List[int], Tuple[int, ...]]], num_stages: int
    ) -> List[int]:
        if raw_indices is None:
            indices = [num_stages - 1]
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

    def _resolve_stage_weights(self, stage_weights: Optional[List[float]]) -> Dict[int, float]:
        if stage_weights is not None:
            if len(stage_weights) != len(self.layer_indices):
                raise ValueError("stage_weights length must match number of selected stages")
            return {idx: float(w) for idx, w in zip(self.layer_indices, stage_weights)}

        num = len(self.layer_indices)
        return {idx: 1.0 / num for idx in self.layer_indices} if num > 0 else {}

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
                    f"IFVDDistillation requires '{key}' features; "
                    f"student keys: {list(student_features.keys())}, "
                    f"teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                idx, student_feat, teacher_feat, target
            )
            if total_loss is None:
                total_loss = stage_loss
            else:
                total_loss = total_loss + stage_loss

            for name, value in stage_loss_dict.items():
                value_item = value.item() if isinstance(value, torch.Tensor) else float(value)
                loss_dict[f"stage{idx}_{name}"] = value_item

        if total_loss is None:
            total_loss = torch.zeros((), device=target.device)
        return total_loss, loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{idx}": f"encoder.stages[{idx}]" for idx in self.layer_indices}

    def to(self, device: torch.device) -> "IFVDDistillation":
        self.loss_modules = self.loss_modules.to(device)
        return self

    def get_stage_indices(self) -> List[int]:
        return list(self.layer_indices)

    def compute_stage_loss(
        self,
        stage_idx: int,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if str(stage_idx) not in self.loss_modules:
            raise ValueError(f"Stage {stage_idx} not configured for IFVDDistillation")

        loss_module = self.loss_modules[str(stage_idx)]
        _, _, D, H, W = student_feat.shape
        if target.shape[2:] != (D, H, W):
            target_resized = F.interpolate(target.float(), size=(D, H, W), mode="nearest")
        else:
            target_resized = target.float()

        stage_loss = loss_module(student_feat, teacher_feat, target_resized)
        weight = float(self.stage_weights.get(stage_idx, 1.0))
        weighted_loss = stage_loss * weight
        return weighted_loss, {"ifvd": stage_loss}


__all__ = ["IFVDDistillation"]
