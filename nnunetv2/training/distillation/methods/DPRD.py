"""
Pairwise relation distillation for 3D encoder features.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


class _SingleStageRelationLoss(nn.Module):
    """
    Single-stage relational KD:
      1) project student/teacher features to a shared channel dimension
      2) pool each sample into one feature vector (masked or global average)
      3) align pairwise deltas across all (i, j), i < j
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        proj_dim: int,
        lam_delta: float = 1.0,
        lam_dist: float = 0.0,
        smooth_l1_beta: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if proj_dim <= 0:
            raise ValueError("proj_dim must be > 0")

        self.eps = float(eps)
        self.lam_delta = float(lam_delta)
        self.lam_dist = float(lam_dist)
        self.smooth_l1_beta = float(smooth_l1_beta)

        self.proj_t: nn.Module
        self.proj_s: nn.Module

        if teacher_channels == proj_dim:
            self.proj_t = nn.Identity()
        else:
            self.proj_t = nn.Conv3d(teacher_channels, proj_dim, kernel_size=1, bias=False)

        if student_channels == proj_dim:
            self.proj_s = nn.Identity()
        else:
            self.proj_s = nn.Conv3d(student_channels, proj_dim, kernel_size=1, bias=False)

    def _pool(
        self,
        feat: torch.Tensor,
        mask: Optional[torch.Tensor],
        downsample_factor: int = 4,
        min_valid_mask: float = 1e-4,
    ) -> torch.Tensor:
        # Dynamic stable masked pooling:
        # infer output size from current feature shape using integer downsampling.
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be a positive integer")

        _, _, d, h, w = feat.shape
        min_out = 2
        d0 = max(min_out, int(d) // int(downsample_factor))
        h0 = max(min_out, int(h) // int(downsample_factor))
        w0 = max(min_out, int(w) // int(downsample_factor))

        if mask is None:
            mask = torch.ones(
                (feat.shape[0], 1, feat.shape[2], feat.shape[3], feat.shape[4]),
                dtype=feat.dtype,
                device=feat.device,
            )
        elif mask.shape[2:] != feat.shape[2:]:
            # mask = F.interpolate(mask.float(), size=feat.shape[2:], mode="nearest").to(feat.dtype)
            mask = F.interpolate(mask.float(), size=feat.shape[2:], mode="nearest").to(dtype=feat.dtype, device=feat.device)
        else:
            mask = mask.to(dtype=feat.dtype, device=feat.device)

        f_weighted = feat * mask
        pooled_num = F.adaptive_avg_pool3d(f_weighted, output_size=(d0, h0, w0))
        pooled_den = F.adaptive_avg_pool3d(mask, output_size=(d0, h0, w0))

        thresh = max(self.eps, float(min_valid_mask))
        valid = pooled_den > thresh
        safe_den = torch.where(valid, pooled_den, torch.ones_like(pooled_den))
        pooled = pooled_num / safe_den
        pooled = torch.where(valid, pooled, torch.zeros_like(pooled))
        ## the nan_to_num is a safety measure; ideally there should be no NaNs after the masking, but just in case, we set them to zero to avoid destabilizing the training with large spikes.
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        # print(f"[RelationAllKD _pool] pooled_num shape = {tuple(pooled_num.shape)} | f_weighted shape = {tuple(f_weighted.shape)} | mask shape = {tuple(mask.shape)} | pooled shape = {tuple(pooled.shape)} | pooled.flatten(start_dim=1) shape = {tuple(pooled.flatten(start_dim=1).shape)}")
        return pooled.flatten(start_dim=1)



    @staticmethod
    def _pairwise_deltas(x: torch.Tensor) -> torch.Tensor:
        # x: (B, d) -> (num_pairs, d), for i < j
        b = x.shape[0]
        if b < 2:
            return x.new_zeros((0, x.shape[1]))
        pair_idx = torch.triu_indices(b, b, offset=1, device=x.device)
        diffs = x.unsqueeze(1) - x.unsqueeze(0)
        return diffs[pair_idx[0], pair_idx[1], :]

    @staticmethod
    def _adjacent_deltas(x: torch.Tensor) -> torch.Tensor:
        # x: (B, d) -> (B-1, d)
        b = x.shape[0]
        if b < 2:
            return x.new_zeros((0, x.shape[1]))
        return x[1:] - x[:-1]

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if student_feat.dim() != 5 or teacher_feat.dim() != 5:
            raise ValueError("Relation distillation expects 3D features: (B, C, D, H, W)")

        ft = self.proj_t(teacher_feat.detach())
        fs = self.proj_s(student_feat)

        if ft.shape != fs.shape:
            raise ValueError(
                "Projected teacher/student features must match. "
                f"Got teacher={tuple(ft.shape)}, student={tuple(fs.shape)}"
            )

        if mask is not None:
            if mask.dim() != 5:
                raise ValueError(f"mask must have shape (B, 1, D, H, W), got {tuple(mask.shape)}")
            if mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
            if mask.shape[2:] != ft.shape[2:]:
                mask = F.interpolate(mask.float(), size=ft.shape[2:], mode="nearest")
            mask = mask.to(dtype=ft.dtype, device=ft.device)

        t_vec = self._pool(ft, mask)
        s_vec = self._pool(fs, mask)

        dt = self._pairwise_deltas(t_vec)
        ds = self._pairwise_deltas(s_vec)

        if dt.shape[0] == 0:
            zero = student_feat.new_zeros(())
            return zero, {
                "relation": zero.detach(),
                "relation_delta": zero.detach(),
                "relation_dist": zero.detach(),
            }

        norm_t = torch.norm(dt, p=2, dim=1)
        norm_s = torch.norm(ds, p=2, dim=1)

        mu_t = norm_t.mean()
        mu_s = norm_s.mean()

        dt_norm = dt / (mu_t + self.eps)
        ds_norm = ds / (mu_s + self.eps)

        # Sum over feature dimension, then average over pair dimension.
        l_delta_per_elem = F.smooth_l1_loss(
            ds_norm,
            dt_norm,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        l_delta = l_delta_per_elem.sum(dim=1).mean()

        l_dist = student_feat.new_zeros(())
        if self.lam_dist > 0.0:
            dist_t = norm_t / (mu_t + self.eps)
            dist_s = norm_s / (mu_s + self.eps)
            l_dist = F.smooth_l1_loss(
                dist_s,
                dist_t,
                beta=self.smooth_l1_beta,
                reduction="mean",
            )
        loss = self.lam_delta * l_delta + self.lam_dist * l_dist
        return loss, {
            "relation": loss.detach(),
            "relation_delta": l_delta.detach(),
            "relation_dist": l_dist.detach(),
        }


class DPRD(DistillationMethod, nn.Module):
    """
    Stage-wise pairwise relation distillation.

    Config parameters:
      - student_channels (Sequence[int] | int)
      - teacher_channels (Sequence[int] | int)
      - layer_indices (default: last stage)
      - stage_weights (optional)
      - proj_dim (int | list[int], default: teacher channels per stage)
      - use_mask (bool, default: False): build ROI mask from target and use masked pooling
      - mask_threshold (float, default: 0.0)
      - assume_background_channel (bool, default: True) for multi-channel target masks
      - stage_to_target_idx (optional): explicit mapping from encoder stage idx to target_list idx
      - lam_delta, lam_dist, smooth_l1_beta, eps
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        student_channels = config.get("student_channels")
        teacher_channels = config.get("teacher_channels")
        if student_channels is None or teacher_channels is None:
            raise ValueError("DPRD requires student_channels and teacher_channels")

        student_channels = self._as_channel_list(student_channels)
        teacher_channels = self._as_channel_list(teacher_channels)

        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        self.layer_indices = self._parse_indices(config.get("layer_indices"), len(student_channels))
        self.stage_weights = self._resolve_stage_weights(config.get("stage_weights"))

        self.use_mask = bool(config.get("use_mask", False))
        self.mask_threshold = float(config.get("mask_threshold", 0.0))
        self.assume_background_channel = bool(config.get("assume_background_channel", True))
        self.stage_to_target_idx = self._parse_stage_to_target_idx(config.get("stage_to_target_idx"))

        raw_proj_dim = config.get("proj_dim")
        stage_proj_dim = self._resolve_proj_dim(
            raw_proj_dim=raw_proj_dim,
            teacher_channels=teacher_channels,
            num_stages=len(student_channels),
        )

        lam_delta = float(config.get("lam_delta", 1.0))
        lam_dist = float(config.get("lam_dist", 0.0))
        smooth_l1_beta = float(config.get("smooth_l1_beta", 1.0))
        eps = float(config.get("eps", 1e-6))

        self.loss_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            self.loss_modules[str(idx)] = _SingleStageRelationLoss(
                student_channels=student_channels[idx],
                teacher_channels=teacher_channels[idx],
                proj_dim=stage_proj_dim[idx],
                lam_delta=lam_delta,
                lam_dist=lam_dist,
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
    def _parse_stage_to_target_idx(raw_mapping: Optional[Union[Dict, Sequence[int]]]) -> Dict[int, int]:
        if raw_mapping is None:
            return {}
        if isinstance(raw_mapping, dict):
            return {int(k): int(v) for k, v in raw_mapping.items()}
        if isinstance(raw_mapping, (list, tuple)):
            return {int(stage): int(tgt_idx) for stage, tgt_idx in enumerate(raw_mapping)}
        raise ValueError("stage_to_target_idx must be a dict or a sequence of target indices")

    @staticmethod
    def _resolve_proj_dim(
        raw_proj_dim: Optional[Union[int, Sequence[int]]],
        teacher_channels: Sequence[int],
        num_stages: int,
    ) -> List[int]:
        if raw_proj_dim is None:
            return [int(c) for c in teacher_channels]

        if isinstance(raw_proj_dim, int):
            dim = int(raw_proj_dim)
            return [dim] * num_stages

        dims = [int(v) for v in raw_proj_dim]
        if len(dims) != num_stages:
            raise ValueError("proj_dim list length must equal number of stages")
        return dims

    def _build_mask(
        self,
        target: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
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
            # Priority 1: explicit mapping provided by config
            if stage_idx in self.stage_to_target_idx:
                mapped_idx = int(self.stage_to_target_idx[stage_idx])
                if mapped_idx < 0 or mapped_idx >= len(target_list):
                    raise ValueError(
                        f"stage_to_target_idx[{stage_idx}]={mapped_idx} out of bounds for target_list length {len(target_list)}"
                    )
                stage_target = target_list[mapped_idx]
            else:
                # Strict stage-index semantics: target_list is expected to be stage-aligned.
                if stage_idx < 0 or stage_idx >= len(target_list):
                    raise ValueError(
                        f"target_list length {len(target_list)} < stage index {stage_idx}. "
                        "Expected stage-aligned target_list (or provide stage_to_target_idx)."
                    )
                stage_target = target_list[stage_idx]
        elif isinstance(target, (list, tuple)):
            if stage_idx >= len(target):
                raise ValueError(f"target list length {len(target)} < stage index {stage_idx}")
            stage_target = target[stage_idx]
        else:
            stage_target = target

        if not isinstance(stage_target, torch.Tensor):
            raise ValueError("DPRD requires tensor target for mask creation")
        return stage_target.float()

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
                    f"DPRD requires '{key}' features; "
                    f"student keys: {list(student_features.keys())}, "
                    f"teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                idx,
                student_feat,
                teacher_feat,
                target,
                target_list=kwargs.get("target_list"),
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

    def compute_stage_loss(
        self,
        stage_idx: int,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if str(stage_idx) not in self.loss_modules:
            raise ValueError(f"Stage {stage_idx} not configured for DPRD")

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
        weighted_loss = stage_loss * weight
        return weighted_loss, stage_loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{idx}": f"encoder.stages[{idx}]" for idx in self.layer_indices}

    def to(self, device: torch.device) -> "DPRD":
        self.loss_modules = self.loss_modules.to(device)
        return self

    def get_stage_indices(self) -> List[int]:
        return list(self.layer_indices)


__all__ = ["DPRD"]
