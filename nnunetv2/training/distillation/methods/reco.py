"""
Relation-based Context (ReCo) distillation (ported from Trainer_ReCo_KD).

Inputs per stage:
  - student_feat: encoder.stages[i] output, shape [B, Cs, D, H, W]
  - teacher_feat: encoder.stages[i] output, shape [B, Ct, D, H, W] (detached)
  - gt_seg: target at the matching scale, shape [B, 1, D, H, W] or [B, C, D, H, W]

Outputs:
  - stage_loss: weighted sum of 4 components (see FeatureLoss)
  - metrics: fg_loss, bg_loss, mask_loss, rela_loss (unweighted per-stage values)

Loss components (per stage):
  - fg_loss: area-weighted foreground MSE with teacher attention (mean over voxels & channels)
  - bg_loss: area-weighted background MSE with teacher attention (mean over voxels & channels)
  - mask_loss: L1 on spatial + channel attention maps (student vs teacher)
  - rela_loss: relation/context MSE after attention pooling
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0.0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0.0,
    distribution: str = "normal",
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class FeatureLoss(nn.Module):
    """
    Single-stage ReCo feature distillation.

    Args:
        student_channels, teacher_channels: channel counts for projection.
        temp: attention temperature.
        num_classes: required when gt_seg is a single-channel label map.
        tau: mask exponent for foreground/background weighting.
        chunk_d: depth chunking size.
        coef_fg/bg/mask/rel: per-component weights.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        temp: float = 0.5,
        num_classes: Optional[int] = None,
        tau: float = 1.0,
        chunk_d: int = 16,
        coef_fg: float = 1.0,
        coef_bg: float = 1.0,
        coef_mask: float = 1.0,
        coef_rel: float = 1.0,
    ):
        super().__init__()
        self.temp = float(temp)
        self.num_classes = num_classes
        self.tau = float(tau)
        self.chunk_d = int(chunk_d)

        if student_channels != teacher_channels:
            self.align = nn.Conv3d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        channels = teacher_channels
        self.conv_mask_s = nn.Conv3d(channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv3d(channels, 1, kernel_size=1)

        self.channel_add_conv_s = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=1),
            nn.GroupNorm(1, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, channels, kernel_size=1),
        )
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=1),
            nn.GroupNorm(1, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, channels, kernel_size=1),
        )

        self.coef_fg = float(coef_fg)
        self.coef_bg = float(coef_bg)
        self.coef_mask = float(coef_mask)
        self.coef_rel = float(coef_rel)

        self.reset_parameters()

    def forward(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        gt_seg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.align is not None:
            preds_S = self.align(preds_S)
        if preds_S.shape != preds_T.shape:
            raise ValueError("Teacher & student feature shape mismatch")

        _, channels, depth, height, width = preds_S.shape

        S_t, C_t = self.get_attention(preds_T, self.temp)
        S_s, C_s = self.get_attention(preds_S, self.temp)

        if gt_seg.shape[1] == 1:
            if self.num_classes is None:
                raise ValueError("num_classes must be provided for single-channel label map")
            mask_fg, mask_bg = self.generate_area_weighted_fg_bg_mask(gt_seg, self.num_classes)
        else:
            mask_fg, mask_bg = self.generate_area_weighted_fg_bg_mask(gt_seg, None)

        fg_loss = self.compute_region_loss(preds_S, preds_T, mask_fg, C_t, S_t, tau=self.tau)
        bg_loss = self.compute_region_loss(preds_S, preds_T, mask_bg, C_t, S_t, tau=self.tau)
        mask_loss = self.get_mask_loss(C_s, C_t, S_s, S_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)

        loss = (
            self.coef_fg * fg_loss
            + self.coef_bg * bg_loss
            + self.coef_mask * mask_loss
            + self.coef_rel * rela_loss
        )

        return loss, {
            "fg_loss": (self.coef_fg * fg_loss).detach(),
            "bg_loss": (self.coef_bg * bg_loss).detach(),
            "mask_loss": (self.coef_mask * mask_loss).detach(),
            "rela_loss": (self.coef_rel * rela_loss).detach(),
        }

    def get_attention(self, preds: torch.Tensor, temp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channels, depth, height, width = preds.shape
        values = torch.abs(preds)

        fea_map = values.mean(dim=1, keepdim=True)
        spatial_flat = (fea_map / temp).view(batch, -1)
        S = (depth * height * width) * F.softmax(spatial_flat, dim=1)
        S = S.view(batch, depth, height, width)

        channel_map = values.mean(dim=(2, 3, 4))
        C_att = channels * F.softmax(channel_map / temp, dim=1)
        return S, C_att

    def compute_region_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        mask: torch.Tensor,
        C_t: torch.Tensor,
        S_t: torch.Tensor,
        tau: float = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        _, channels, depth, _, _ = preds_S.shape
        chunk = self.chunk_d

        S_t = torch.sqrt(S_t).unsqueeze(1)
        C_t = torch.sqrt(C_t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        num = preds_S.new_tensor(0.0)
        den = preds_S.new_tensor(0.0)

        for sd in range(0, depth, chunk):
            ed = min(sd + chunk, depth)
            fea_t = preds_T[:, :, sd:ed, :, :] * S_t[:, :, sd:ed, :, :] * C_t
            fea_s = preds_S[:, :, sd:ed, :, :] * S_t[:, :, sd:ed, :, :] * C_t

            diff2 = (fea_s - fea_t) ** 2
            w = (mask[:, :, sd:ed, :, :]).pow(tau)
            num = num + (diff2 * w).sum()
            den = den + w.sum()

        return num / (den.clamp_min(eps) * channels)

    def get_mask_loss(self, C_s: torch.Tensor, C_t: torch.Tensor, S_s: torch.Tensor, S_t: torch.Tensor) -> torch.Tensor:
        loss_channel = F.l1_loss(C_s, C_t, reduction="mean")
        loss_spatial = F.l1_loss(S_s, S_t, reduction="mean")
        return loss_channel + loss_spatial

    def spatial_pool(self, x: torch.Tensor, in_type: int) -> torch.Tensor:
        batch, channels, _, _, _ = x.size()
        input_x = x.view(batch, channels, -1).unsqueeze(1)

        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)

        context_mask = context_mask.view(batch, 1, -1)
        context_mask = F.softmax(context_mask, dim=2).unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)
        return context.view(batch, channels, 1, 1, 1)

    def get_rela_loss(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> torch.Tensor:
        context_s = self.spatial_pool(preds_S, in_type=0)
        context_t = self.spatial_pool(preds_T, in_type=1)
        out_s = preds_S + self.channel_add_conv_s(context_s)
        out_t = preds_T + self.channel_add_conv_t(context_t)
        return F.mse_loss(out_s, out_t, reduction="mean")

    def last_zero_init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Sequential):
            for layer in reversed(module):
                if isinstance(layer, (nn.Conv3d, nn.Conv2d)):
                    constant_init(layer, val=0.0)
                    break
        elif isinstance(module, (nn.Conv3d, nn.Conv2d)):
            constant_init(module, val=0.0)

    def reset_parameters(self) -> None:
        kaiming_init(self.conv_mask_s, mode="fan_in", nonlinearity="relu")
        kaiming_init(self.conv_mask_t, mode="fan_in", nonlinearity="relu")
        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

    def generate_area_weighted_fg_bg_mask(
        self, label_map: torch.Tensor, num_classes: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, channels, _, _, _ = label_map.shape
        if channels == 1:
            if num_classes is None:
                raise ValueError("num_classes required for single-channel labels")
            mask_fg = torch.zeros_like(label_map, dtype=torch.float32)
            for c in range(1, num_classes):
                mc = (label_map == c).float()
                area = mc.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
                mask_fg += mc / area
        else:
            per = []
            for c in range(channels):
                mc = label_map[:, c].unsqueeze(1).float()
                area = mc.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
                per.append(mc / area)
            mask_fg = torch.max(torch.stack(per, dim=1).squeeze(2), dim=1, keepdim=True)[0]

        mask_bg = (mask_fg == 0).float()
        bg_area = mask_bg.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        mask_bg = mask_bg / bg_area
        return mask_fg, mask_bg


class ReCoDistillation(DistillationMethod, nn.Module):
    """
    Stage-wise ReCo distillation (ported from Trainer_ReCo_KD).

    Config parameters:
        student_channels, teacher_channels: sequence[int]
        num_classes: required for single-channel targets
        temp, tau, coef_*: FeatureLoss settings
        chunk_d: depth chunking
        stage_alpha: auto stage-weight decay factor (default 0.4)
        stage_weights/manual_stage_weights: explicit per-stage weights
        layer_indices/stage_indices: optional subset of encoder stages to distill
        drop_last_stage: mirror Trainer_ReCo_KD behavior (default True)
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        if "student_channels" not in config or "teacher_channels" not in config:
            raise ValueError("ReCoDistillation requires student_channels and teacher_channels")

        student_channels = list(config["student_channels"])
        teacher_channels = list(config["teacher_channels"])

        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        drop_last = bool(config.get("drop_last_stage", True))
        if drop_last and len(student_channels) > 1:
            student_channels = student_channels[:-1]
            teacher_channels = teacher_channels[:-1]

        raw_indices = config.get("layer_indices")
        if raw_indices is None:
            raw_indices = config.get("stage_indices")
        self.stage_indices = self._parse_indices(raw_indices, len(student_channels))
        self.reco_modules = nn.ModuleDict()

        for idx in self.stage_indices:
            s_ch = student_channels[idx]
            t_ch = teacher_channels[idx]
            self.reco_modules[str(idx)] = FeatureLoss(
                student_channels=s_ch,
                teacher_channels=t_ch,
                temp=config.get("temp", 0.5),
                num_classes=config.get("num_classes"),
                tau=config.get("tau", 1.0),
                chunk_d=config.get("chunk_d", 16),
                coef_fg=config.get("coef_fg", 1.0),
                coef_bg=config.get("coef_bg", 1.0),
                coef_mask=config.get("coef_mask", 1.0),
                coef_rel=config.get("coef_rel", 1.0),
            )

        manual_weights = config.get("stage_weights")
        if manual_weights is None:
            manual_weights = config.get("manual_stage_weights")

        if manual_weights is not None:
            if len(manual_weights) != len(self.stage_indices):
                raise ValueError("stage_weights length must match number of selected stages")
            weights = [float(w) for w in manual_weights]
        else:
            alpha = float(config.get("stage_alpha", 0.4))
            base = float(config.get("stage_weight_base", 2.0))
            dims = int(config.get("stage_weight_dims", 3))
            weights = self._stage_weights_by_stride(len(self.stage_indices), alpha, base, dims)

        self.stage_weights = {idx: w for idx, w in zip(self.stage_indices, weights)} if weights else {}

    @staticmethod
    def _stage_weights_by_stride(K: int, alpha: float, base: float, dims: int) -> List[float]:
        if K <= 0:
            return []
        r = base ** (-dims * alpha)
        weights = [r ** k for k in range(K)]
        total = sum(weights) + 1e-8
        return [float(w / total) for w in weights]

    @staticmethod
    def _parse_indices(raw_indices: Optional[Union[int, str, List[int], Tuple[int, ...]]], num_stages: int) -> List[int]:
        if raw_indices is None:
            indices = list(range(num_stages))
        elif isinstance(raw_indices, int):
            indices = [raw_indices]
        elif isinstance(raw_indices, str):
            token = raw_indices.strip().lower()
            if token in ("all", "*"):
                indices = list(range(num_stages))
            else:
                raise ValueError("layer_indices string must be 'all'/'*' or an int/list")
        elif isinstance(raw_indices, (list, tuple)):
            indices = [int(i) for i in raw_indices]
        else:
            raise ValueError("layer_indices must be int, list, tuple, or 'all'")

        normalized: List[int] = []
        for idx in indices:
            if idx < 0:
                idx = num_stages + idx
            if idx < 0 or idx >= num_stages:
                raise ValueError(f"layer index {idx} out of bounds for {num_stages} stages")
            normalized.append(idx)
        return normalized

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, float] = {}

        for stage_idx in self.stage_indices:
            student_feat = student_features.get(f"stage{stage_idx}")
            teacher_feat = teacher_features.get(f"stage{stage_idx}")

            if student_feat is None or teacher_feat is None:
                raise ValueError(f"Missing features for stage{stage_idx}")

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                stage_idx, student_feat, teacher_feat, target
            )

            total_loss = stage_loss if total_loss is None else total_loss + stage_loss

            for name, value in stage_loss_dict.items():
                value_item = value.item() if isinstance(value, torch.Tensor) else float(value)
                loss_dict[f"stage{stage_idx}_{name}"] = value_item

        if total_loss is None:
            total_loss = torch.zeros((), device=target.device)
        return total_loss, loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{i}": f"encoder.stages[{i}]" for i in self.stage_indices}

    def to(self, device: torch.device) -> "ReCoDistillation":
        self.reco_modules = self.reco_modules.to(device)
        return self

    def get_stage_indices(self) -> List[int]:
        return list(self.stage_indices)

    def compute_stage_loss(
        self,
        stage_idx: int,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        key = str(stage_idx)
        if key not in self.reco_modules:
            raise ValueError(f"Stage {stage_idx} not configured for ReCoDistillation")
        module = self.reco_modules[key]

        target_list = kwargs.get("target_list")
        if isinstance(target_list, (list, tuple)):
            if stage_idx >= len(target_list):
                raise ValueError(f"Target list length {len(target_list)} < stage index {stage_idx}")
            gt_seg = target_list[stage_idx]
        elif isinstance(target, (list, tuple)):
            if stage_idx >= len(target):
                raise ValueError(f"Target list length {len(target)} < stage index {stage_idx}")
            gt_seg = target[stage_idx]
        else:
            gt_seg = target

        if not isinstance(gt_seg, torch.Tensor):
            raise ValueError("ReCoDistillation requires tensor targets for gt_seg")

        if not isinstance(target_list, (list, tuple)) and not isinstance(target, (list, tuple)):
            spatial_shape = student_feat.shape[2:]
            if gt_seg.shape[2:] != spatial_shape:
                gt_seg = F.interpolate(gt_seg.float(), size=spatial_shape, mode="nearest")
            else:
                gt_seg = gt_seg.float()

        stage_loss, stage_loss_dict = module(student_feat, teacher_feat.detach(), gt_seg=gt_seg)
        weight = float(self.stage_weights.get(stage_idx, 1.0)) if self.stage_weights else 1.0
        return stage_loss * weight, stage_loss_dict


__all__ = ["ReCoDistillation"]
