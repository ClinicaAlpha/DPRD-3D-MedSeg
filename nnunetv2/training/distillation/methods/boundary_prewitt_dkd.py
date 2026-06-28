"""
Hybrid distillation: logit-based Prewitt boundary loss + DKD-style logits loss.

This file is self-contained and does not depend on boundary_prewitt.py.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


TensorOrSequence = Union[torch.Tensor, Tuple[torch.Tensor, ...], list]


class BoundaryLogitsPrewitt(DistillationMethod, nn.Module):
    """
    Boundary distillation on model outputs using 3D Prewitt filters.

    Parameters
    ----------
    use_softmax:
        If True, apply activation before edge extraction:
        - multi-label targets: sigmoid
        - multi-class targets: softmax
        If False, use raw logits.
    prewitt_temperature:
        Temperature used only for the boundary branch when `use_softmax=True`.
    """

    def __init__(self, use_softmax: bool = True, prewitt_temperature: float = 1.0, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        self.use_softmax = bool(use_softmax)
        self.prewitt_temperature = float(prewitt_temperature)
        self.eps = 1e-8

        self._init_prewitt_kernels()

    def _init_prewitt_kernels(self) -> None:
        diff = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        smooth = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        k_z = torch.einsum("i,j,k->ijk", diff, smooth, smooth)
        k_y = torch.einsum("i,j,k->ijk", smooth, diff, smooth)
        k_x = torch.einsum("i,j,k->ijk", smooth, smooth, diff)

        self.register_buffer("prewitt_z", k_z.view(1, 1, 3, 3, 3))
        self.register_buffer("prewitt_y", k_y.view(1, 1, 3, 3, 3))
        self.register_buffer("prewitt_x", k_x.view(1, 1, 3, 3, 3))

    @staticmethod
    def _unwrap_output(output: Optional[TensorOrSequence]) -> Optional[torch.Tensor]:
        if output is None:
            return None
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                return None
            return output[0]
        return output

    def _get_aligned_logits(
        self,
        student_output: Optional[TensorOrSequence],
        teacher_output: Optional[TensorOrSequence],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        student_logits = self._unwrap_output(student_output)
        teacher_logits = self._unwrap_output(teacher_output)

        if student_logits is None or teacher_logits is None:
            return None, None

        if student_logits.dim() != 5 or teacher_logits.dim() != 5:
            raise ValueError(
                "BoundaryLogitsPrewitt expects 3D logits with shape BxCxDxHxW. "
                f"Got student={tuple(student_logits.shape)}, teacher={tuple(teacher_logits.shape)}"
            )

        if student_logits.shape[1] != teacher_logits.shape[1]:
            raise ValueError(
                "Student and teacher logits must have same channel count. "
                f"Got student C={student_logits.shape[1]}, teacher C={teacher_logits.shape[1]}"
            )

        if student_logits.shape[2:] != teacher_logits.shape[2:]:
            student_logits = F.interpolate(
                student_logits,
                size=teacher_logits.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return student_logits, teacher_logits

    @staticmethod
    def _is_multilabel_target(target: torch.Tensor) -> bool:
        return target.dim() == 5 and target.shape[1] > 1

    def _prepare_boundary_input(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_softmax:
            return logits

        tau = max(self.prewitt_temperature, 1e-6)
        if self._is_multilabel_target(target):
            return torch.sigmoid(logits / tau)
        return F.softmax(logits / tau, dim=1)

    def extract_edges(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        b, c, d, h, w = x.shape
        x_flat = x.view(b * c, 1, d, h, w)

        k_z = self.prewitt_z.to(x.device)
        k_y = self.prewitt_y.to(x.device)
        k_x = self.prewitt_x.to(x.device)

        edge_z = F.conv3d(x_flat, k_z, padding=1)
        edge_y = F.conv3d(x_flat, k_y, padding=1)
        edge_x = F.conv3d(x_flat, k_x, padding=1)

        magnitude = torch.sqrt(edge_z * edge_z + edge_y * edge_y + edge_x * edge_x + self.eps)
        return magnitude.view(b, c, d, h, w)

    def compute_prewitt_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        s_input = self._prepare_boundary_input(student_logits, target)
        with torch.no_grad():
            t_input = self._prepare_boundary_input(teacher_logits, target)
            t_edges = self.extract_edges(t_input)

        s_edges = self.extract_edges(s_input)

        s_weighted = s_input * s_edges
        t_weighted = t_input * t_edges
        return F.mse_loss(s_weighted, t_weighted, reduction="mean")

    def forward(
        self,
        student_features,
        teacher_features,
        target: torch.Tensor,
        student_output: Optional[TensorOrSequence] = None,
        teacher_output: Optional[TensorOrSequence] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del student_features, teacher_features, kwargs

        student_logits, teacher_logits = self._get_aligned_logits(student_output, teacher_output)
        if student_logits is None or teacher_logits is None:
            return torch.tensor(0.0, device=target.device), {}

        loss = self.compute_prewitt_loss(student_logits, teacher_logits, target)
        return loss, {"prewitt_loss": float(loss.detach())}

    def get_required_features(self) -> Dict[str, str]:
        return {}

    def to(self, device: torch.device) -> "BoundaryLogitsPrewitt":
        nn.Module.to(self, device)
        return self


class BoundaryPrewittDKD(BoundaryLogitsPrewitt):
    """
    Hybrid distillation:
    1) Prewitt boundary alignment on output maps
    2) DKD-style logits distillation

    Parameters
    ----------
    dkd_alpha:
        Weight for target-class component (TCKD / positive branch).
    dkd_beta:
        Weight for non-target component (NCKD / negative branch).
    dkd_temperature:
        Temperature for DKD branch.
    dkd_weight:
        Overall weight for DKD branch in total loss.
    prewitt_weight:
        Overall weight for Prewitt boundary branch in total loss.
    dkd_edge_source:
        Source for edge mask used in DKD target branch: "teacher" | "student" | "both".
    dkd_target_include_background:
        Only for multi-class targets. If False, target-region mask is foreground-only (label > 0).
        If True, include all valid labels as target region.
    use_target_ignore_channel:
        If True and target has C+1 channels in multi-label mode,
        treat the last channel as ignore mask.
    ignore_index:
        Ignore index for multi-class labels.
    """

    def __init__(
        self,
        dkd_alpha: float = 1.0,
        dkd_beta: float = 2.0,
        dkd_temperature: float = 2.0,
        dkd_weight: float = 1.0,
        prewitt_weight: float = 1.0,
        dkd_edge_source: str = "teacher",
        dkd_target_include_background: bool = False,
        use_target_ignore_channel: bool = False,
        ignore_index: Optional[int] = None,
        **config,
    ):
        super().__init__(**config)

        self.dkd_alpha = float(dkd_alpha)
        self.dkd_beta = float(dkd_beta)
        self.dkd_temperature = float(dkd_temperature)
        self.dkd_weight = float(dkd_weight)
        self.prewitt_weight = float(prewitt_weight)
        self.dkd_edge_source = str(dkd_edge_source).lower()
        if self.dkd_edge_source not in ("teacher", "student", "both"):
            raise ValueError("dkd_edge_source must be one of: 'teacher', 'student', 'both'")
        self.dkd_target_include_background = bool(dkd_target_include_background)
        self.use_target_ignore_channel = bool(use_target_ignore_channel)
        self.ignore_index = ignore_index

    @staticmethod
    def _resize_target_to_logits(target: torch.Tensor, spatial_size: Tuple[int, int, int]) -> torch.Tensor:
        if target.dim() == 5 and target.shape[2:] != spatial_size:
            return F.interpolate(target.float(), size=spatial_size, mode="nearest")
        return target

    def _extract_ignore_mask_multilabel(
        self,
        target: torch.Tensor,
        num_classes: int,
    ) -> Optional[torch.Tensor]:
        if not self.use_target_ignore_channel:
            return None
        if target.dim() != 5:
            return None
        if target.shape[1] == num_classes + 1:
            return (target[:, -1:] > 0.5).float()
        return None

    def _prepare_multiclass_labels(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        if target.dim() == 5:
            if target.shape[1] == 1:
                labels = target[:, 0]
            else:
                labels = torch.argmax(target, dim=1)
        elif target.dim() == 4:
            labels = target
        else:
            raise ValueError(f"Unsupported target shape for multiclass DKD: {tuple(target.shape)}")

        labels = labels.long()
        labels = torch.clamp(labels, min=-1, max=max(num_classes - 1, 0))
        return labels

    def _build_dkd_edge_mask(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Build a spatial edge mask (B, 1, D, H, W), normalized to mean ~= 1 per sample.
        with torch.no_grad():
            s_input = self._prepare_boundary_input(logits_student.detach(), target)
            t_input = self._prepare_boundary_input(logits_teacher.detach(), target)
            s_edge = self.extract_edges(s_input).mean(dim=1, keepdim=True)
            t_edge = self.extract_edges(t_input).mean(dim=1, keepdim=True)

            if self.dkd_edge_source == "teacher":
                edge = t_edge
            elif self.dkd_edge_source == "student":
                edge = s_edge
            else:
                edge = 0.5 * (s_edge + t_edge)

            denom = edge.mean(dim=(2, 3, 4), keepdim=True)
            edge = edge / (denom + self.eps)
            return edge

    def _prepare_multilabel_target_map(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        if target.dim() != 5:
            raise ValueError(f"Expected 5D multi-label target, got {tuple(target.shape)}")
        if target.shape[1] == num_classes + 1 and self.use_target_ignore_channel:
            target_map = target[:, :-1]
        elif target.shape[1] >= num_classes:
            target_map = target[:, :num_classes]
        else:
            raise ValueError(
                f"Multi-label target channels ({target.shape[1]}) do not match num classes ({num_classes})"
            )
        return target_map.float().clamp(0.0, 1.0)

    def dkd_loss(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_student = logits_student.float()
        logits_teacher = logits_teacher.float().detach()

        b, c, d, h, w = logits_student.shape
        target_resized = self._resize_target_to_logits(target, (d, h, w))
        edge_mask = self._build_dkd_edge_mask(logits_student, logits_teacher, target_resized)

        tau = max(self.dkd_temperature, 1e-6)
        scale = tau * tau

        # -------- Multi-label / region targets --------
        if self._is_multilabel_target(target_resized):
            teacher_prob = torch.sigmoid(logits_teacher / tau)
            target_map = self._prepare_multilabel_target_map(target_resized, c)
            non_target_map = 1.0 - target_map

            student_scaled = logits_student / tau
            log_student_pos = F.logsigmoid(student_scaled)
            log_student_neg = F.logsigmoid(-student_scaled)

            loss_pos = -(teacher_prob * log_student_pos)
            loss_neg = -((1.0 - teacher_prob) * log_student_neg)
            target_weight = target_map * edge_mask
            non_target_weight = non_target_map

            ignore_mask = self._extract_ignore_mask_multilabel(target_resized, c)
            if ignore_mask is not None:
                valid_mask = 1.0 - ignore_mask.to(target_weight.dtype)
                target_weight = target_weight * valid_mask
                non_target_weight = non_target_weight * valid_mask

            target_denom = torch.clamp(target_weight.sum(), min=1.0)
            nontarget_denom = torch.clamp(non_target_weight.sum(), min=1.0)
            target_loss = (loss_pos * target_weight).sum() / target_denom
            nontarget_loss = (loss_neg * non_target_weight).sum() / nontarget_denom

            loss_tckd_component = self.dkd_alpha * target_loss * scale
            loss_nckd_component = self.dkd_beta * nontarget_loss * scale
            return loss_tckd_component + loss_nckd_component, loss_tckd_component, loss_nckd_component

        # -------- Multi-class targets --------
        # labels: [B, D, H, W], integer class ids prepared from one-hot/logit-style or index-style targets.
        labels = self._prepare_multiclass_labels(target_resized, c)
        # valid_mask: [B, D, H, W], keep only voxels whose label is inside [0, C-1].
        valid_mask = (labels >= 0) & (labels < c)
        if self.ignore_index is not None:
            # Exclude ignore-index voxels from DKD supervision.
            valid_mask = valid_mask & (labels != int(self.ignore_index))

        if valid_mask.sum() == 0:
            # No valid supervision signal in this batch -> return 0 to avoid invalid reductions.
            zero = logits_student.new_tensor(0.0)
            return zero, zero, zero

        # Flatten spatial voxels: [B, C, D, H, W] -> [B*D*H*W, C] so DKD is computed per voxel row.
        s_flat = logits_student.permute(0, 2, 3, 4, 1).reshape(-1, c)
        t_flat = logits_teacher.permute(0, 2, 3, 4, 1).reshape(-1, c)
        # Flatten labels/mask to align with flattened logits.
        labels_flat = labels.reshape(-1)
        valid_flat = valid_mask.reshape(-1)

        # Keep only valid voxels: resulting shape [M, C] / [M], where M = number of valid voxels.
        s_flat = s_flat[valid_flat]
        t_flat = t_flat[valid_flat]
        labels_flat = labels_flat[valid_flat]

        # One-hot GT mask per valid voxel, shape [M, C].
        gt_mask = F.one_hot(labels_flat, num_classes=c).to(s_flat.dtype)

        # Temperature-scaled predictive distributions for student/teacher, shape [M, C].
        pred_s = F.softmax(s_flat / tau, dim=1)
        pred_t = F.softmax(t_flat / tau, dim=1)

        # Clamp for numerical stability in subsequent log/division operations.
        pred_s = torch.clamp(pred_s, min=1e-7, max=1.0 - 1e-7)
        pred_t = torch.clamp(pred_t, min=1e-7, max=1.0 - 1e-7)

        # Target-class probabilities only, shape [M, 1].
        pt_tgt = (pred_t * gt_mask).sum(dim=1, keepdim=True)
        ps_tgt = (pred_s * gt_mask).sum(dim=1, keepdim=True)
        # TCKD: binary KL on (target vs non-target) per voxel.
        # cat(...) gives [M, 2], kl_div(..., reduction="none") -> [M, 2], then sum -> [M].
        tckd_per_voxel = F.kl_div(
            torch.log(torch.cat([ps_tgt, 1.0 - ps_tgt], dim=1)),
            torch.cat([pt_tgt, 1.0 - pt_tgt], dim=1),
            reduction="none",
        ).sum(dim=1)

        if self.dkd_target_include_background:
            # Use all valid labels (including background) as TCKD region.
            target_region = valid_mask.float().unsqueeze(1)
        else:
            # Use foreground-only voxels (label > 0) as TCKD region.
            target_region = (labels > 0).float().unsqueeze(1)
        # Boundary-aware weighting for target branch; shapes broadcast to [B, 1, D, H, W].
        target_region = target_region * edge_mask
        # Flatten + valid filtering -> [M], aligned with tckd_per_voxel.
        target_region_flat = target_region.reshape(-1)[valid_flat]
        # Safe denominator prevents division by zero when weighted region is empty.
        target_region_denom = torch.clamp(target_region_flat.sum(), min=1.0)
        # Weighted mean TCKD over selected region.
        loss_tckd = (tckd_per_voxel * target_region_flat).sum() / target_region_denom

        # Non-target class mask and conditional distributions over non-target classes.
        other_mask = 1.0 - gt_mask
        pt_other = pred_t * other_mask
        ps_other = pred_s * other_mask

        # Normalize by (1 - p_target) to get proper non-target distributions, shape [M, C].
        pt_other = pt_other / (1.0 - pt_tgt + 1e-8)
        ps_other = ps_other / (1.0 - ps_tgt + 1e-8)

        # NCKD: KL on non-target classes; per-entry output [M, C].
        loss_nckd_full = F.kl_div(torch.log(ps_other + 1e-8), pt_other, reduction="none")
        # Non-target branch does not use edge mask by design.
        # Average over non-target classes per voxel, then mean over voxels -> scalar.
        nontarget_count = torch.clamp(other_mask.sum(dim=1), min=1.0)
        loss_nckd = ((loss_nckd_full * other_mask).sum(dim=1) / nontarget_count).mean()

        # Final DKD objective split into target/non-target components (both already scaled).
        loss_tckd_component = self.dkd_alpha * loss_tckd * scale
        loss_nckd_component = self.dkd_beta * loss_nckd * scale
        return loss_tckd_component + loss_nckd_component, loss_tckd_component, loss_nckd_component

    def forward(
        self,
        student_features,
        teacher_features,
        target: torch.Tensor,
        student_output: Optional[TensorOrSequence] = None,
        teacher_output: Optional[TensorOrSequence] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del student_features, teacher_features, kwargs

        student_logits, teacher_logits = self._get_aligned_logits(student_output, teacher_output)
        if student_logits is None or teacher_logits is None:
            return torch.tensor(0.0, device=target.device), {}

        # Important for debugging and stability:
        # If a branch weight is 0, skip computing that branch entirely.
        # This avoids "0 * NaN = NaN" contaminating total loss.
        eps_w = 1e-12
        zero = student_logits.new_tensor(0.0)

        if abs(self.prewitt_weight) > eps_w:
            loss_prewitt = self.compute_prewitt_loss(student_logits, teacher_logits, target)
            if not torch.isfinite(loss_prewitt):
                raise RuntimeError("Non-finite prewitt loss detected.")
        else:
            loss_prewitt = zero

        if abs(self.dkd_weight) > eps_w:
            loss_dkd, loss_tckd_component, loss_nckd_component = self.dkd_loss(student_logits, teacher_logits, target)
            if not torch.isfinite(loss_dkd):
                raise RuntimeError("Non-finite DKD loss detected.")
        else:
            loss_dkd = zero
            loss_tckd_component = zero
            loss_nckd_component = zero

        total_loss = self.prewitt_weight * loss_prewitt + self.dkd_weight * loss_dkd
        if not torch.isfinite(total_loss):
            raise RuntimeError("Non-finite total distillation loss detected.")

        # Report effective components (already scaled by method-internal weights).
        # Trainer-level kd_weight is still applied outside this method.
        loss_prewitt_effective = self.prewitt_weight * loss_prewitt
        loss_dkd_effective = self.dkd_weight * loss_dkd
        loss_tckd_effective = self.dkd_weight * loss_tckd_component
        loss_nckd_effective = self.dkd_weight * loss_nckd_component

        return total_loss, {
            "loss_prewitt_effective": float(loss_prewitt_effective.detach()),
            "loss_dkd_effective": float(loss_dkd_effective.detach()),
            "loss_tckd_effective": float(loss_tckd_effective.detach()),
            "loss_nckd_effective": float(loss_nckd_effective.detach()),
        }


__all__ = ["BoundaryLogitsPrewitt", "BoundaryPrewittDKD"]
