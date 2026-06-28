"""
Utility helpers for the ReCo distillation strategy.

This module provides a drop-in FeatureLoss implementation that matches the API
expected by `nnunetv2.training.distillation.methods.reco.ReCoDistillation`. It
aggregates several interpretable components (foreground/background feature
alignment, relation matching and a lightweight mask consistency term) while
remaining robust to different target encodings and feature tensor shapes.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FeatureLoss"]


def _normalise_target(target: torch.Tensor) -> torch.Tensor:
    """
    Convert targets to integer labels with shape (N, 1, D, H, W).

    Accepts:
        * integer labels with optional singleton channel dimension
        * one-hot masks (N, K, D, H, W)
        * floating masks with values in [0, 1]
    """
    if target.dim() != 5:
        raise ValueError(f"Expected target tensor in NxCxDxHxW layout, got {target.shape}")

    if target.dtype.is_floating_point:
        if target.shape[1] == 1:
            labels = target.round().long()
        else:
            labels = torch.argmax(target, dim=1, keepdim=True)
    else:
        labels = target

    if labels.shape[1] != 1:
        labels = torch.argmax(labels, dim=1, keepdim=True)

    return labels.long()


def _masked_mse(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error restricted to voxels selected by mask (broadcast to channels).
    """
    eps = 1e-6
    active = mask.sum()
    if active <= eps:
        return torch.zeros((), dtype=student.dtype, device=student.device)
    diff = (student - teacher) ** 2 * mask
    return diff.sum() / active.clamp_min(eps)


def _relation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """
    Cosine-similarity matrix alignment between student and teacher features.
    """
    n, c, d, h, w = student.shape
    student_flat = student.view(n, c, -1)
    teacher_flat = teacher.view(n, c, -1)

    student_norm = F.normalize(student_flat, dim=2)
    teacher_norm = F.normalize(teacher_flat, dim=2)

    student_rel = torch.bmm(student_norm, student_norm.transpose(1, 2))
    teacher_rel = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2))
    return F.mse_loss(student_rel, teacher_rel)


class FeatureLoss(nn.Module):
    """
    Composite feature loss used by ReCo distillation.

    Parameters
    ----------
    student_channels, teacher_channels : int
        Channel counts for student/teacher features. When they differ, a 1x1x1
        projection adapts the student channels.
    temp, tau : float
        Scaling factors for the masked losses (temp) and relation term (tau).
    coef_fg, coef_bg, coef_mask, coef_rel : float
        Weights for the respective loss components. Setting any of them to zero
        disables that component.
    chunk_d : int, optional
        Placeholder for future memory-optimised depth chunking (currently unused).
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        temp: float = 1.0,
        tau: float = 1.0,
        coef_fg: float = 1.0,
        coef_bg: float = 1.0,
        coef_mask: float = 1.0,
        coef_rel: float = 1.0,
        chunk_d: int | None = None,
    ):
        super().__init__()
        self.temp = float(temp)
        self.tau = float(tau)
        self.coef_fg = float(coef_fg)
        self.coef_bg = float(coef_bg)
        self.coef_mask = float(coef_mask)
        self.coef_rel = float(coef_rel)
        self.chunk_d = chunk_d

        self.proj: nn.Module | None = None
        if student_channels != teacher_channels:
            self.proj = nn.Conv3d(student_channels, teacher_channels, kernel_size=1, bias=False)

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if student.dim() != 5 or teacher.dim() != 5:
            raise ValueError("FeatureLoss expects 5D tensors (N, C, D, H, W)")

        if self.proj is not None:
            student = self.proj(student)

        if student.shape != teacher.shape:
            raise ValueError("Student and teacher features must match after projection")

        labels = _normalise_target(target)
        fg_mask = (labels > 0).float()
        bg_mask = 1.0 - fg_mask

        fg_mask = fg_mask.expand_as(student)
        bg_mask = bg_mask.expand_as(student)

        stats: Dict[str, torch.Tensor] = {}
        losses = []

        if self.coef_fg > 0:
            fg_loss = _masked_mse(student, teacher, fg_mask) * self.temp
            losses.append(self.coef_fg * fg_loss)
            stats["fg_loss"] = fg_loss.detach()

        if self.coef_bg > 0:
            bg_loss = _masked_mse(student, teacher, bg_mask) * self.temp
            losses.append(self.coef_bg * bg_loss)
            stats["bg_loss"] = bg_loss.detach()

        if self.coef_rel > 0:
            rel_loss = _relation_loss(student, teacher) * self.tau
            losses.append(self.coef_rel * rel_loss)
            stats["rel_loss"] = rel_loss.detach()

        if self.coef_mask > 0:
            eps = 1e-6
            student_norm = torch.linalg.norm(student, dim=1, keepdim=True)
            teacher_norm = torch.linalg.norm(teacher, dim=1, keepdim=True)

            fg_student = (student_norm * fg_mask[:, :1]).sum() / fg_mask[:, :1].sum().clamp_min(eps)
            fg_teacher = (teacher_norm * fg_mask[:, :1]).sum() / fg_mask[:, :1].sum().clamp_min(eps)
            mask_loss = (fg_student - fg_teacher).abs()

            losses.append(self.coef_mask * mask_loss)
            stats["mask_loss"] = mask_loss.detach()

        if losses:
            total = torch.stack(losses).sum()
        else:
            total = torch.zeros((), dtype=student.dtype, device=student.device)

        stats["total"] = total.detach()
        return total, stats
