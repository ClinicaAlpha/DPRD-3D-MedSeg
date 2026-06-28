"""
Boundary-aware distillation primitives.

Includes a lightweight boundary-loss module and a high-level method wrapper
that plugs into the generic distiller.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod


# ---------- 初始化工具 ----------
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
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ===========================================================
#   简化版 Boundary Distillation (用于消融实验)
# ===========================================================
class SimpleBoundaryDistillation(nn.Module):
    """
    简化版边界蒸馏，只保留核心功能，便于消融实验

    核心思想:
        1. 从 GT 生成 boundary mask (边界带)
        2. 从教师/学生特征提取高频分量 (Sobel 边缘检测)
        3. 只在边界区域内让学生对齐教师的高频特征
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        boundary_width: int = 3,
        num_classes: Optional[int] = None,
        chunk_d: int = 16,
        use_boundary_loss: bool = True,
        use_attention_loss: bool = False,
        temp: float = 0.5,
    ):
        super().__init__()

        self.boundary_width = boundary_width
        self.num_classes = num_classes
        self.chunk_d = chunk_d
        self.temp = temp

        # Loss 开关 (便于消融实验)
        self.use_boundary_loss = use_boundary_loss
        self.use_attention_loss = use_attention_loss

        # 通道对齐
        if student_channels != teacher_channels:
            self.align = nn.Conv3d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

        C = teacher_channels

        # 边界 attention 模块 (只在需要时使用)
        if self.use_attention_loss:
            self.boundary_attention_s = nn.Sequential(
                nn.Conv3d(C, C // 2, kernel_size=3, padding=1),
                nn.GroupNorm(1, C // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(C // 2, 1, kernel_size=1),
            )
            self.boundary_attention_t = nn.Sequential(
                nn.Conv3d(C, C // 2, kernel_size=3, padding=1),
                nn.GroupNorm(1, C // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(C // 2, 1, kernel_size=1),
            )
        else:
            self.boundary_attention_s = None
            self.boundary_attention_t = None

        # 初始化 3D Sobel 边缘检测卷积核 (固定权重)
        self._init_sobel_kernels()

        if self.use_attention_loss:
            self.reset_parameters()

    def _init_sobel_kernels(self) -> None:
        """初始化 3D Sobel 卷积核 (固定权重, 不参与训练)"""
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

        self.register_buffer("sobel_z", sobel_z.view(1, 1, 3, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3, 3))
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3, 3))

    def extract_high_frequency(self, features: torch.Tensor) -> torch.Tensor:
        """
        从特征图中提取高频/边缘分量 (使用 3D Sobel)
        """
        B, C, D, H, W = features.shape

        # 对每个通道独立应用 Sobel
        features_flat = features.view(B * C, 1, D, H, W)

        hf_z = F.conv3d(features_flat, self.sobel_z, padding=1)
        hf_y = F.conv3d(features_flat, self.sobel_y, padding=1)
        hf_x = F.conv3d(features_flat, self.sobel_x, padding=1)

        # 梯度幅值 (3 个方向的 L2 范数)
        high_freq = torch.sqrt(hf_x ** 2 + hf_y ** 2 + hf_z ** 2 + 1e-8)
        high_freq = high_freq.view(B, C, D, H, W)

        return high_freq

    def generate_boundary_mask(self, label_map: torch.Tensor) -> torch.Tensor:
        """
        从 GT 生成边界 mask

        方法: 形态学膨胀 - 原图 = 边界带
        """
        B, C, D, H, W = label_map.shape
        assert C == 1, "Only single-channel labels supported in simple version"
        assert self.num_classes is not None, "num_classes required"

        foreground = (label_map > 0).float()

        kernel = torch.ones(
            (1, 1, self.boundary_width, self.boundary_width, self.boundary_width),
            device=label_map.device,
        )

        padded = F.pad(foreground, (1, 1, 1, 1, 1, 1), mode="replicate")
        dilated = F.conv3d(padded, kernel, padding=0)
        dilated = torch.clamp(dilated, 0, 1)

        boundary = torch.abs(dilated - foreground)
        if boundary.sum() > 0:
            boundary = boundary / (boundary.sum() + 1e-6)

        return boundary

    def compute_boundary_loss(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, boundary: torch.Tensor
    ) -> torch.Tensor:
        high_freq_s = self.extract_high_frequency(student_feat)
        high_freq_t = self.extract_high_frequency(teacher_feat)

        diff = (high_freq_s - high_freq_t) ** 2
        weighted = diff * boundary
        return weighted.sum()

    def compute_attention_loss(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, boundary: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_attention_loss:
            return student_feat.new_tensor(0.0)

        if self.boundary_attention_s is None or self.boundary_attention_t is None:
            raise RuntimeError("Attention modules not initialized")

        attn_s = self.boundary_attention_s(student_feat.detach())
        attn_t = self.boundary_attention_t(teacher_feat.detach())

        attn_s = attn_s / (attn_s.max() + 1e-6)
        attn_t = attn_t / (attn_t.max() + 1e-6)

        boundary = F.interpolate(boundary, size=attn_s.shape[2:], mode="nearest")

        return F.mse_loss(attn_s * boundary, attn_t * boundary)

    def forward(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.align is not None:
            student_feat = self.align(student_feat)

        with torch.no_grad():
            boundary_mask = self.generate_boundary_mask(target)

        losses = {}
        total = student_feat.new_tensor(0.0)

        if self.use_boundary_loss:
            boundary_loss = self.compute_boundary_loss(
                student_feat, teacher_feat, boundary_mask
            )
            losses["boundary"] = boundary_loss
            total = total + boundary_loss

        if self.use_attention_loss:
            attention_loss = self.compute_attention_loss(
                student_feat, teacher_feat, boundary_mask
            )
            losses["attention"] = attention_loss
            total = total + attention_loss

        return total, losses

    def reset_parameters(self) -> None:
        """Initialize learnable layers."""
        if self.align is not None:
            kaiming_init(self.align, distribution="normal")

        if self.boundary_attention_s is not None:
            for module in self.boundary_attention_s.modules():
                if isinstance(module, nn.Conv3d):
                    kaiming_init(module, distribution="normal")
                elif isinstance(module, nn.GroupNorm):
                    constant_init(module, 1.0, bias=0)

        if self.boundary_attention_t is not None:
            for module in self.boundary_attention_t.modules():
                if isinstance(module, nn.Conv3d):
                    kaiming_init(module, distribution="normal")
                elif isinstance(module, nn.GroupNorm):
                    constant_init(module, 1.0, bias=0)


class BoundaryDistillation(DistillationMethod, nn.Module):
    """
    Boundary-aware high-frequency feature distillation.

    Config parameters mirror the legacy implementation. The wrapper supports
    multiple encoder stages by instantiating a ``SimpleBoundaryDistillation``
    module per stage.
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        required = ["student_channels", "teacher_channels", "num_classes"]
        for key in required:
            if key not in config:
                raise ValueError(f"BoundaryDistillation requires '{key}' in config")

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

        layer_indices = self._parse_indices(config.get("layer_indices"), len(student_channels))
        self.layer_indices: List[int] = layer_indices

        base_kwargs = dict(
            boundary_width=config.get("boundary_width", 3),
            num_classes=config["num_classes"],
            chunk_d=config.get("chunk_d", 16),
            use_boundary_loss=config.get("use_boundary_loss", True),
            use_attention_loss=config.get("use_attention_loss", False),
            temp=config.get("temp", 0.5),
        )

        self.distill_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            module = SimpleBoundaryDistillation(
                student_channels=student_channels[idx],
                teacher_channels=teacher_channels[idx],
                **base_kwargs,
            )
            self.distill_modules[str(idx)] = module

    @staticmethod
    def _parse_indices(
        raw_indices: Optional[Union[int, str, List[int], Tuple[int, ...]]], num_stages: int
    ) -> List[int]:
        if raw_indices is None:
            indices: List[int] = [num_stages - 1]
        elif isinstance(raw_indices, (int,)):
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
        if str(stage_idx) not in self.distill_modules:
            raise ValueError(f"Stage {stage_idx} not configured for BoundaryDistillation")

        module = self.distill_modules[str(stage_idx)]
        target_fp = target.float()
        spatial_shape = student_feat.shape[2:]
        if target_fp.shape[2:] != spatial_shape:
            target_resized = F.interpolate(target_fp, size=spatial_shape, mode="nearest")
        else:
            target_resized = target_fp

        if target_resized.shape[1] != 1:
            target_resized = torch.argmax(target_resized, dim=1, keepdim=True)

        stage_loss, stage_loss_dict = module(student_feat, teacher_feat, target_resized)
        return stage_loss, stage_loss_dict

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.layer_indices:
            raise ValueError("No encoder stages selected for BoundaryDistillation")

        total_loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, float] = {}

        for idx in self.layer_indices:
            key = f"stage{idx}"
            student_feat = student_features.get(key)
            teacher_feat = teacher_features.get(key)

            if student_feat is None or teacher_feat is None:
                raise ValueError(
                    f"BoundaryDistillation requires '{key}' features. "
                    f"Student keys: {list(student_features.keys())}, "
                    f"Teacher keys: {list(teacher_features.keys())}"
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

    def to(self, device: torch.device) -> "BoundaryDistillation":
        self.distill_modules = self.distill_modules.to(device)
        return self


__all__ = ["SimpleBoundaryDistillation", "BoundaryDistillation"]
