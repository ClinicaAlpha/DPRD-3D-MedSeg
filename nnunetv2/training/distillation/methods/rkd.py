"""
Relational Knowledge Distillation (RKD) for nnU-Net style 3D features.

Paper:
  Park et al., "Relational Knowledge Distillation", CVPR 2019.
  https://arxiv.org/abs/1904.05068

Core idea:
  - Convert each sample feature map to one embedding vector.
  - Distill pairwise sample relations in a batch:
    1) distance-wise relation
    2) angle-wise relation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


class _SingleStageRKDLoss(nn.Module):
    """
    Single-stage RKD over batch-sample embeddings.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        proj_dim: int,
        distance_weight: float = 25.0,
        angle_weight: float = 50.0,
        smooth_l1_beta: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        if proj_dim <= 0:
            raise ValueError("proj_dim must be > 0")

        self.distance_weight = float(distance_weight)
        self.angle_weight = float(angle_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.eps = float(eps)

        if teacher_channels == proj_dim:
            self.proj_t: nn.Module = nn.Identity()
        else:
            self.proj_t = nn.Conv3d(teacher_channels, proj_dim, kernel_size=1, bias=False)

        if student_channels == proj_dim:
            self.proj_s: nn.Module = nn.Identity()
        else:
            self.proj_s = nn.Conv3d(student_channels, proj_dim, kernel_size=1, bias=False)

    def _pool(self, feat: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # feat: (B, C, D, H, W), returns (B, C)
        if mask is None:
            return feat.mean(dim=(2, 3, 4))

        weighted = (feat * mask).sum(dim=(2, 3, 4))
        denom = mask.sum(dim=(2, 3, 4)).clamp_min(self.eps)
        return weighted / denom

    def _pairwise_dist(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) -> pairwise upper-triangle distances (P,)
        b = x.shape[0]
        if b < 2:
            return x.new_zeros((0,))
        d = torch.cdist(x, x, p=2)
        tri = torch.triu_indices(b, b, offset=1, device=x.device)
        return d[tri[0], tri[1]]

    def _distance_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        td = self._pairwise_dist(t)
        sd = self._pairwise_dist(s)
        if td.numel() == 0:
            return s.new_zeros(())

        t_mean = td[td > 0].mean() if (td > 0).any() else td.mean()
        s_mean = sd[sd > 0].mean() if (sd > 0).any() else sd.mean()
        t_norm = td / (t_mean + self.eps)
        s_norm = sd / (s_mean + self.eps)
        return F.smooth_l1_loss(s_norm, t_norm, beta=self.smooth_l1_beta, reduction="mean")

    def _angle_terms(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) -> flattened angle tensor (B*B*B,)
        diff = x.unsqueeze(0) - x.unsqueeze(1)  # (B, B, C), row i-j
        diff = F.normalize(diff, p=2, dim=2, eps=self.eps)
        angle = torch.bmm(diff, diff.transpose(1, 2))  # (B, B, B): cos((i-j),(i-k))
        return angle.reshape(-1)

    def _angle_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if s.shape[0] < 2:
            return s.new_zeros(())
        ta = self._angle_terms(t)
        sa = self._angle_terms(s)
        return F.smooth_l1_loss(sa, ta, beta=self.smooth_l1_beta, reduction="mean")

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if student_feat.dim() != 5 or teacher_feat.dim() != 5:
            raise ValueError("RKD expects 3D features: (B, C, D, H, W)")

        ft = self.proj_t(teacher_feat.detach())
        fs = self.proj_s(student_feat)
        if ft.shape != fs.shape:
            raise ValueError(
                "Projected teacher/student features must match. "
                f"Got teacher={tuple(ft.shape)}, student={tuple(fs.shape)}"
            )

        if mask is not None:
            if mask.dim() != 5:
                raise ValueError(f"mask must be (B, 1, D, H, W), got {tuple(mask.shape)}")
            if mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
            if mask.shape[2:] != ft.shape[2:]:
                mask = F.interpolate(mask.float(), size=ft.shape[2:], mode="nearest")
            mask = mask.to(dtype=ft.dtype, device=ft.device)

        t_vec = self._pool(ft, mask)
        s_vec = self._pool(fs, mask)

        l_dist = self._distance_loss(s_vec, t_vec)
        l_angle = self._angle_loss(s_vec, t_vec)
        loss = self.distance_weight * l_dist + self.angle_weight * l_angle
        return loss, {
            "rkd": loss.detach(),
            "rkd_distance": l_dist.detach(),
            "rkd_angle": l_angle.detach(),
        }


class RKDDistillation(DistillationMethod, nn.Module):
    """
    Stage-wise RKD wrapper aligned with nnUNet-KD method interface.
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        student_channels = config.get("student_channels")
        teacher_channels = config.get("teacher_channels")
        if student_channels is None or teacher_channels is None:
            raise ValueError("RKDDistillation requires student_channels and teacher_channels")

        student_channels = self._as_channel_list(student_channels)
        teacher_channels = self._as_channel_list(teacher_channels)
        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        self.layer_indices = self._parse_indices(config.get("layer_indices"), len(student_channels))
        self.stage_weights = self._resolve_stage_weights(config.get("stage_weights"))

        self.use_mask = bool(config.get("use_mask", False))
        self.mask_threshold = float(config.get("mask_threshold", 0.0))
        self.assume_background_channel = bool(config.get("assume_background_channel", True))

        stage_proj_dim = self._resolve_proj_dim(
            raw_proj_dim=config.get("proj_dim"),
            teacher_channels=teacher_channels,
            num_stages=len(student_channels),
        )
        distance_weight = float(config.get("distance_weight", 25.0))
        angle_weight = float(config.get("angle_weight", 50.0))
        smooth_l1_beta = float(config.get("smooth_l1_beta", 1.0))
        eps = float(config.get("eps", 1e-12))

        self.loss_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            self.loss_modules[str(idx)] = _SingleStageRKDLoss(
                student_channels=student_channels[idx],
                teacher_channels=teacher_channels[idx],
                proj_dim=stage_proj_dim[idx],
                distance_weight=distance_weight,
                angle_weight=angle_weight,
                smooth_l1_beta=smooth_l1_beta,
                eps=eps,
            )

    @staticmethod
    def _as_channel_list(value: Union[int, Sequence[int]]) -> List[int]:
        if isinstance(value, int):
            return [int(value)]
        return [int(v) for v in value]

    @staticmethod
    def _parse_indices(
        raw_indices: Optional[Union[int, str, List[int], Tuple[int, ...]]],
        num_stages: int,
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
    def _resolve_proj_dim(
        raw_proj_dim: Optional[Union[int, Sequence[int]]],
        teacher_channels: Sequence[int],
        num_stages: int,
    ) -> List[int]:
        if raw_proj_dim is None:
            return [int(c) for c in teacher_channels]
        if isinstance(raw_proj_dim, int):
            return [int(raw_proj_dim)] * num_stages
        dims = [int(v) for v in raw_proj_dim]
        if len(dims) != num_stages:
            raise ValueError("proj_dim list length must equal number of stages")
        return dims

    def _build_mask(self, target: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        if target.dim() == 4:
            target = target.unsqueeze(1)
        if target.dim() != 5:
            raise ValueError(f"Expected target dims=4 or 5 for mask creation, got {target.dim()}")

        if target.shape[1] == 1:
            mask = (target > self.mask_threshold).float()
        else:
            source = target[:, 1:, ...] if self.assume_background_channel and target.shape[1] > 1 else target
            mask = (source.sum(dim=1, keepdim=True) > self.mask_threshold).float()

        if mask.shape[2:] != spatial_shape:
            mask = F.interpolate(mask, size=spatial_shape, mode="nearest")
        return mask

    def _resolve_stage_target(
        self,
        stage_idx: int,
        target: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if isinstance(target_list, (list, tuple)):
            if stage_idx >= len(target_list):
                raise ValueError(f"target_list length {len(target_list)} < stage index {stage_idx}")
            stage_target = target_list[stage_idx]
        elif isinstance(target, (list, tuple)):
            if stage_idx >= len(target):
                raise ValueError(f"target list length {len(target)} < stage index {stage_idx}")
            stage_target = target[stage_idx]
        else:
            stage_target = target

        if not isinstance(stage_target, torch.Tensor):
            raise ValueError("RKDDistillation requires tensor target for mask creation")
        return stage_target.float()

    def compute_stage_loss(
        self,
        stage_idx: int,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if str(stage_idx) not in self.loss_modules:
            raise ValueError(f"Stage {stage_idx} not configured for RKD")

        mask = None
        if self.use_mask:
            stage_target = self._resolve_stage_target(
                stage_idx=stage_idx,
                target=target,
                target_list=kwargs.get("target_list"),
            )
            mask = self._build_mask(stage_target, student_feat.shape[2:]).to(student_feat.device)

        module = self.loss_modules[str(stage_idx)]
        stage_loss, stage_loss_dict = module(student_feat, teacher_feat, mask=mask)

        weight = float(self.stage_weights.get(stage_idx, 1.0))
        return stage_loss * weight, stage_loss_dict

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
                    f"RKD requires '{key}' features; "
                    f"student keys: {list(student_features.keys())}, "
                    f"teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                idx, student_feat, teacher_feat, target, target_list=kwargs.get("target_list")
            )
            total_loss = stage_loss if total_loss is None else (total_loss + stage_loss)

            for name, value in stage_loss_dict.items():
                val = value.item() if isinstance(value, torch.Tensor) else float(value)
                loss_dict[f"stage{idx}_{name}"] = val

        if total_loss is None:
            total_loss = torch.zeros((), device=target.device)
        return total_loss, loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{idx}": f"encoder.stages[{idx}]" for idx in self.layer_indices}

    def to(self, device: torch.device) -> "RKDDistillation":
        self.loss_modules = self.loss_modules.to(device)
        return self

    def get_stage_indices(self) -> List[int]:
        return list(self.layer_indices)


RKD = RKDDistillation

__all__ = ["RKDDistillation", "RKD"]
