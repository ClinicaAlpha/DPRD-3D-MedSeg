"""
Simple feature-matching distillation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


class FeatureDistillation(DistillationMethod, nn.Module):
    """
    Baseline L2 feature matching between student and teacher activations.
    """

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        student_channels = config.get("student_channels")
        teacher_channels = config.get("teacher_channels")
        self.layer = config.get("layer", "encoder")
        self.stage_index = int(config.get("stage_index", -1))

        if student_channels is None or teacher_channels is None:
            raise ValueError("FeatureDistillation requires student_channels and teacher_channels")

        if student_channels != teacher_channels:
            self.align = nn.Conv3d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        feature_key = f"{self.layer}_output"

        student_feat = student_features.get(feature_key)
        teacher_feat = teacher_features.get(feature_key)

        if student_feat is None or teacher_feat is None:
            raise ValueError(f"FeatureDistillation requires '{feature_key}' in features")

        if isinstance(student_feat, (list, tuple)) or isinstance(teacher_feat, (list, tuple)):
            if not (isinstance(student_feat, (list, tuple)) and isinstance(teacher_feat, (list, tuple))):
                raise ValueError("Student/teacher feature types do not match for FeatureDistillation")
            if not student_feat or not teacher_feat:
                raise ValueError("Empty feature list received for FeatureDistillation")
            if len(student_feat) != len(teacher_feat):
                raise ValueError(
                    f"Student/teacher feature list length mismatch: {len(student_feat)} vs {len(teacher_feat)}"
                )
            idx = self.stage_index
            if idx < 0:
                idx = len(student_feat) + idx
            if idx < 0 or idx >= len(student_feat):
                raise ValueError(f"stage_index {self.stage_index} out of bounds for {len(student_feat)} stages")
            student_feat = student_feat[idx]
            teacher_feat = teacher_feat[idx]

        if self.align is not None:
            student_feat = self.align(student_feat)

        loss = F.mse_loss(student_feat, teacher_feat)
        return loss, {"mse_loss": loss.item()}

    def get_required_features(self) -> Dict[str, str]:
        return {f"{self.layer}_output": self.layer}

    def to(self, device: torch.device) -> "FeatureDistillation":
        if self.align is not None:
            self.align = self.align.to(device)
        return self


__all__ = ["FeatureDistillation"]
