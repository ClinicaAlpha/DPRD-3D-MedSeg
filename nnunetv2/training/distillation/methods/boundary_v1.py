"""
Improved boundary-aware distillation with warmup and EMA-smoothed targets.

This variant keeps the Sobel-based high-frequency supervision but adds:
    - Multi-class boundary extraction with optional class weighting
    - Per-voxel uncertainty reweighting using the student's prediction
    - EMA statistics over teacher high-frequency responses for stabilisation
    - Large warmup window before the boundary signal becomes dominant
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


class BoundaryStageModuleV1(nn.Module):
    """
    Stage-level boundary distillation primitive.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        num_classes: int,
        boundary_width: int = 3,
        warmup_iters: int = 5000,
        teacher_stat_ema_decay: float = 0.97,
        uncertainty_scale: float = 1.5,
        class_weights: Optional[Iterable[float]] = None,
        chunk_d: Optional[int] = None,
        use_teacher_stats: bool = True,
    ) -> None:
        super().__init__()

        if boundary_width < 1:
            raise ValueError("boundary_width must be >= 1")
        if boundary_width % 2 == 0:
            raise ValueError("boundary_width should be odd for symmetric padding")
        if num_classes is None or num_classes < 2:
            raise ValueError("num_classes must be provided (>=2)")
        if not (0.0 <= teacher_stat_ema_decay < 1.0):
            raise ValueError("teacher_stat_ema_decay must lie in [0, 1)")

        self.boundary_width = boundary_width
        self.pad = boundary_width // 2
        self.num_classes = num_classes
        self.chunk_d = chunk_d
        self.warmup_iters = warmup_iters
        self.teacher_stat_ema_decay = teacher_stat_ema_decay
        self.uncertainty_scale = uncertainty_scale
        self.use_teacher_stats = bool(use_teacher_stats)
        self.eps = 1e-6

        self.teacher_channels = teacher_channels
        self.align: Optional[nn.Conv3d] = None
        if student_channels != teacher_channels:
            self._reset_align(student_channels)

        if class_weights is not None:
            weights = list(class_weights)
            if len(weights) != num_classes - 1:
                raise ValueError(
                    f"class_weights should have length num_classes-1 ({num_classes - 1}), "
                    f"got {len(weights)}"
                )
            self.register_buffer(
                "class_weights", torch.as_tensor(weights, dtype=torch.float32), persistent=False
            )
        else:
            self.class_weights = None

        self._init_sobel_kernels()

        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)
        if self.use_teacher_stats:
            self.register_buffer(
                "teacher_hf_mean", torch.zeros(teacher_channels, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "teacher_hf_var", torch.ones(teacher_channels, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "stats_initialized", torch.zeros((), dtype=torch.bool), persistent=False
            )
        else:
            self.teacher_hf_mean = None
            self.teacher_hf_var = None
            self.stats_initialized = None

    def _init_sobel_kernels(self) -> None:
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

    def _class_weight(self, cls_index: int) -> float:
        if self.class_weights is None:
            return 1.0
        idx = cls_index - 1
        if idx < 0 or idx >= len(self.class_weights):
            return 1.0
        return float(self.class_weights[idx].item())

    def _morphological_gradient(self, mask: torch.Tensor) -> torch.Tensor:
        kernel = self.boundary_width
        padding = self.pad
        dilated = F.max_pool3d(mask, kernel_size=kernel, stride=1, padding=padding)
        eroded = -F.max_pool3d(-mask, kernel_size=kernel, stride=1, padding=padding)
        boundary = (dilated - eroded).clamp_min(0.0)
        return boundary

    def generate_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() != 5:
            raise ValueError("target must be BxCxDxHxW tensor")
        if target.shape[1] > 1:
            hard = torch.argmax(target, dim=1, keepdim=True)
        else:
            hard = target.long()

        boundary = torch.zeros_like(hard, dtype=torch.float32)
        total_weight = 0.0

        for cls in range(1, self.num_classes):
            class_mask = (hard == cls).float()
            if class_mask.sum() <= 0:
                continue
            class_boundary = self._morphological_gradient(class_mask)
            weight = self._class_weight(cls)
            boundary = boundary + class_boundary * weight
            total_weight += weight

        if total_weight == 0.0:
            # Fallback to a coarse foreground boundary if no class is present
            fg = (hard > 0).float()
            if fg.sum() == 0:
                return boundary
            boundary = self._morphological_gradient(fg)

        boundary_sum = boundary.sum()
        if boundary_sum > 0:
            boundary = boundary / (boundary_sum + self.eps)
        return boundary

    def apply_uncertainty_weight(self, boundary: torch.Tensor, student_output: torch.Tensor) -> torch.Tensor:
        logits = student_output
        if logits.shape[2:] != boundary.shape[2:]:
            logits = F.interpolate(logits, size=boundary.shape[2:], mode="trilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        max_prob, _ = probs.max(dim=1, keepdim=True)
        uncertainty = (1.0 - max_prob).detach()
        weighted = boundary * (1.0 + self.uncertainty_scale * uncertainty)
        total = weighted.sum()
        if total > 0:
            weighted = weighted / (total + self.eps)
        return weighted

    def extract_high_frequency(self, features: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = features.shape
        features_flat = features.contiguous().view(B * C, 1, D, H, W)
        hf_z = F.conv3d(features_flat, self.sobel_z, padding=1)
        hf_y = F.conv3d(features_flat, self.sobel_y, padding=1)
        hf_x = F.conv3d(features_flat, self.sobel_x, padding=1)
        high_freq = torch.sqrt(hf_x ** 2 + hf_y ** 2 + hf_z ** 2 + self.eps)
        return high_freq.view(B, C, D, H, W)

    def _reset_align(self, in_channels: int, device: Optional[torch.device] = None) -> None:
        conv = nn.Conv3d(in_channels, self.teacher_channels, kernel_size=1, bias=False)
        if device is not None:
            conv = conv.to(device)
        self.align = conv

    def _update_teacher_stats(self, teacher_hf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_teacher_stats:
            mean = torch.zeros(
                (1, teacher_hf.shape[1], 1, 1, 1), dtype=teacher_hf.dtype, device=teacher_hf.device
            )
            std = torch.ones(
                (1, teacher_hf.shape[1], 1, 1, 1), dtype=teacher_hf.dtype, device=teacher_hf.device
            )
            return mean, std

        hf_flat = teacher_hf.detach().flatten(2)
        mean_batch = hf_flat.mean(dim=2).mean(dim=0)
        var_batch = hf_flat.var(dim=2, unbiased=False).mean(dim=0)
        var_batch = torch.clamp(var_batch, min=self.eps)

        if not self.stats_initialized.item():
            self.teacher_hf_mean.copy_(mean_batch)
            self.teacher_hf_var.copy_(var_batch)
            self.stats_initialized.fill_(True)
        else:
            decay = self.teacher_stat_ema_decay
            self.teacher_hf_mean.mul_(decay).add_(mean_batch * (1.0 - decay))
            self.teacher_hf_var.mul_(decay).add_(var_batch * (1.0 - decay))

        mean = self.teacher_hf_mean.view(1, -1, 1, 1, 1)
        std = torch.sqrt(self.teacher_hf_var.view(1, -1, 1, 1, 1) + self.eps)
        return mean, std

    def _advance_warmup(self, device: torch.device) -> torch.Tensor:
        self.step.add_(1)
        if self.warmup_iters <= 0:
            return torch.ones((), dtype=torch.float32, device=device)
        scale = torch.clamp(self.step.float() / float(self.warmup_iters), max=1.0)
        return scale.to(device)

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        student_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.align is not None:
            if self.align.weight.device != student_feat.device:
                self.align = self.align.to(student_feat.device)
            if student_feat.shape[1] != self.align.in_channels:
                self._reset_align(student_feat.shape[1], student_feat.device)
            student_feat = self.align(student_feat)
        elif student_feat.shape[1] != self.teacher_channels:
            self._reset_align(student_feat.shape[1], student_feat.device)
            student_feat = self.align(student_feat)

        with torch.no_grad():
            boundary_mask = self.generate_boundary_mask(target)
            if student_output is not None and self.uncertainty_scale > 0:
                boundary_mask = self.apply_uncertainty_weight(boundary_mask, student_output)

        if boundary_mask.sum() <= 0:
            zero = student_feat.new_tensor(0.0)
            return zero, {"boundary": zero}

        high_freq_s = self.extract_high_frequency(student_feat)
        high_freq_t = self.extract_high_frequency(teacher_feat.detach())

        mean, std = self._update_teacher_stats(high_freq_t)
        high_freq_s = (high_freq_s - mean) / std
        high_freq_t = (high_freq_t - mean) / std

        diff = (high_freq_s - high_freq_t) ** 2
        boundary_mask = boundary_mask.to(diff.dtype)
        loss_map = diff * boundary_mask
        boundary_loss = loss_map.sum()

        warmup_scale = self._advance_warmup(student_feat.device)
        total_loss = boundary_loss * warmup_scale

        loss_dict: Dict[str, torch.Tensor] = {"boundary": boundary_loss.detach()}
        if self.warmup_iters > 0:
            loss_dict["warmup_scale"] = warmup_scale.detach()
        return total_loss, loss_dict


class BoundaryDistillationV1(DistillationMethod, nn.Module):
    """
    Boundary-aware distillation wrapper that exposes per-stage computation to the trainer.

    Trainer can either call ``forward`` (which loops over stages internally) or, when finer
    control is desired, iterate ``get_stage_indices()`` and call ``compute_stage_loss`` for each.
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        required = ["student_channels", "teacher_channels", "num_classes"]
        for key in required:
            if key not in config:
                raise ValueError(f"BoundaryDistillationV1 requires '{key}' in config")

        student_channels = config["student_channels"]
        teacher_channels = config["teacher_channels"]

        student_channels = (
            list(student_channels)
            if isinstance(student_channels, (list, tuple))
            else [student_channels]
        )
        teacher_channels = (
            list(teacher_channels)
            if isinstance(teacher_channels, (list, tuple))
            else [teacher_channels]
        )

        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        self.num_classes: int = int(config["num_classes"])
        self.layer_indices: List[int] = self._parse_indices(
            config.get("layer_indices"), len(student_channels)
        )

        class_weights = config.get("class_weights")
        boundary_width = int(config.get("boundary_width", 3))
        warmup_iters = int(config.get("warmup_iters", 8000))
        teacher_stat_ema_decay = float(config.get("teacher_stat_ema_decay", 0.97))
        uncertainty_scale = float(config.get("uncertainty_scale", 1.5))
        chunk_d = config.get("chunk_d")
        use_teacher_stats = bool(config.get("use_teacher_stats", True))

        if class_weights is not None and not isinstance(class_weights, (list, tuple)):
            raise ValueError("class_weights should be a list or tuple of floats")

        self.stage_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            module = BoundaryStageModuleV1(
                student_channels=student_channels[idx],
                teacher_channels=teacher_channels[idx],
                num_classes=self.num_classes,
                boundary_width=boundary_width,
                warmup_iters=warmup_iters,
                teacher_stat_ema_decay=teacher_stat_ema_decay,
                uncertainty_scale=uncertainty_scale,
                class_weights=class_weights,
                chunk_d=chunk_d,
                use_teacher_stats=use_teacher_stats,
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
        if str(stage_idx) not in self.stage_modules:
            raise ValueError(f"Stage {stage_idx} not configured for BoundaryDistillationV1")

        module = self.stage_modules[str(stage_idx)]
        target_fp = target.float()
        spatial_shape = student_feat.shape[2:]

        if target_fp.shape[2:] != spatial_shape:
            target_resized = F.interpolate(target_fp, size=spatial_shape, mode="nearest")
        else:
            target_resized = target_fp

        if student_output is not None:
            logits = student_output
            if logits.shape[2:] != spatial_shape:
                logits = F.interpolate(logits, size=spatial_shape, mode="trilinear", align_corners=False)
        else:
            logits = None

        stage_loss, stage_loss_dict = module(
            student_feat,
            teacher_feat,
            target_resized,
            student_output=logits,
        )
        return stage_loss, stage_loss_dict

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
        student_output = kwargs.get("student_output")

        for idx in self.layer_indices:
            key = f"stage{idx}"
            student_feat = student_features.get(key)
            teacher_feat = teacher_features.get(key)
            if student_feat is None or teacher_feat is None:
                raise ValueError(
                    f"BoundaryDistillationV1 requires '{key}' features. "
                    f"Student keys: {list(student_features.keys())}, "
                    f"Teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                idx, student_feat, teacher_feat, target, student_output=student_output
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

    def to(self, device: torch.device) -> "BoundaryDistillationV1":
        self.stage_modules = self.stage_modules.to(device)
        return self


__all__ = ["BoundaryStageModuleV1", "BoundaryDistillationV1"]
