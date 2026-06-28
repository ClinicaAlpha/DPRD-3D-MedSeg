"""
Cross-Image Relational Knowledge Distillation (CIRKD) wrapper.

Paper:
  Yang et al., "Cross-image Relational Knowledge Distillation for
  Semantic Segmentation", CVPR 2022.
  https://arxiv.org/abs/2204.06986
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from nnunetv2.utilities.kd_losses import CIRKD_Loss

from .base import DistillationMethod


class CIRKDDistillation(DistillationMethod, nn.Module):
    """
    Stage-wise CIRKD distillation for 3D encoder features.

    Config parameters:
      - student_channels (Sequence[int])
      - teacher_channels (Sequence[int])
      - layer_indices: stages to supervise (default: last)
      - stage_weights: optional per-stage weights
      - temperature (float): softmax temperature in CIRKD relation matching
      - target_size (tuple/list[int]): pooled spatial size for relation maps
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        student_channels = config.get("student_channels")
        teacher_channels = config.get("teacher_channels")
        if student_channels is None or teacher_channels is None:
            raise ValueError("CIRKDDistillation requires student_channels and teacher_channels")

        student_channels = list(student_channels)
        teacher_channels = list(teacher_channels)
        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        self.layer_indices = self._parse_indices(config.get("layer_indices"), len(student_channels))
        self.stage_weights = self._resolve_stage_weights(config.get("stage_weights"))

        temperature = float(config.get("temperature", 0.1))
        target_size = config.get("target_size", (8, 8, 8))
        target_size = self._parse_target_size(target_size)

        self.loss_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            self.loss_modules[str(idx)] = CIRKD_Loss(
                s_channels=int(student_channels[idx]),
                t_channels=int(teacher_channels[idx]),
                temperature=temperature,
                target_size=target_size,
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

    @staticmethod
    def _parse_target_size(v: Union[int, List[int], Tuple[int, ...]]) -> Tuple[int, int, int]:
        if isinstance(v, int):
            return (int(v), int(v), int(v))
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return (int(v[0]), int(v[1]), int(v[2]))
        raise ValueError("target_size must be int or sequence of length 3")

    def compute_stage_loss(
        self,
        stage_idx: int,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        key = str(stage_idx)
        if key not in self.loss_modules:
            raise ValueError(f"Stage {stage_idx} not configured for CIRKDDistillation")

        module = self.loss_modules[key]
        stage_loss = module(student_feat, teacher_feat.detach())
        weight = float(self.stage_weights.get(stage_idx, 1.0))
        return stage_loss * weight, {"cirkd": stage_loss}

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
                    f"CIRKD requires '{key}' features; "
                    f"student keys: {list(student_features.keys())}, "
                    f"teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(idx, student_feat, teacher_feat, target)
            total_loss = stage_loss if total_loss is None else total_loss + stage_loss

            for name, value in stage_loss_dict.items():
                value_item = value.item() if isinstance(value, torch.Tensor) else float(value)
                loss_dict[f"stage{idx}_{name}"] = value_item

        if total_loss is None:
            total_loss = torch.zeros((), device=target.device)
        return total_loss, loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{idx}": f"encoder.stages[{idx}]" for idx in self.layer_indices}

    def to(self, device: torch.device) -> "CIRKDDistillation":
        self.loss_modules = self.loss_modules.to(device)
        return self

    def get_stage_indices(self) -> List[int]:
        return list(self.layer_indices)


__all__ = ["CIRKDDistillation"]
