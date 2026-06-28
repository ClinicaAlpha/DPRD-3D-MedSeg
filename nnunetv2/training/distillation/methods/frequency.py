"""
Wavelet-based feature distillation (3D), v2_mean.

Same as frequency_v2 but averages the stage losses instead of summing them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .base import DistillationMethod


def _build_filter_bank_3d(wavelet: str, *, mode: str = "dec") -> torch.Tensor:
    wave = pywt.Wavelet(wavelet)
    if mode == "dec":
        lo = torch.tensor(wave.dec_lo, dtype=torch.float32)
        hi = torch.tensor(wave.dec_hi, dtype=torch.float32)
    elif mode == "rec":
        lo = torch.tensor(wave.rec_lo, dtype=torch.float32)
        hi = torch.tensor(wave.rec_hi, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown filter mode '{mode}'. Use 'dec' or 'rec'.")

    kernels = []
    for fz in (lo, hi):
        for fy in (lo, hi):
            for fx in (lo, hi):
                kernel = fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
                kernels.append(kernel)

    filters = torch.stack(kernels, dim=0).unsqueeze(1)  # (8, 1, K, K, K)
    return filters


class SimpleFrequencyDistillationV2Mean(nn.Module):
    """
    Single-stage wavelet MSE distillation for 3D feature maps.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        wavelet: str = "haar",
        levels: int = 1,
        band: Union[str, List[int], Tuple[int, ...]] = "all",
        sample: bool = False,
        sample_dir: Optional[str] = None,
        sample_every: int = 200,
        sample_max: int = 50,
        sample_stages: Union[str, List[int], Tuple[int, ...]] = "all",
        sample_levels: Union[str, List[int], Tuple[int, ...]] = "all",
        sample_channel: Union[str, int] = "mean",
        sample_slice: Union[str, int] = "mid",
    ):
        super().__init__()

        if levels < 1:
            raise ValueError("levels must be >= 1")

        if student_channels != teacher_channels:
            self.align = nn.Conv3d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

        dec_filters = _build_filter_bank_3d(wavelet, mode="dec")
        rec_filters = _build_filter_bank_3d(wavelet, mode="rec")
        self.register_buffer("dec_filters", dec_filters)
        self.register_buffer("rec_filters", rec_filters)
        self.pad = dec_filters.shape[-1] - 1
        self.levels = int(levels)
        self.subband_indices = self._resolve_subbands(band)

        self.sample = bool(sample)
        self.sample_dir = sample_dir
        self.sample_every = max(1, int(sample_every))
        self.sample_max = max(0, int(sample_max))
        self.sample_stages = self._resolve_stages(sample_stages)
        self.sample_levels = self._resolve_levels(sample_levels, self.levels)
        self.sample_channel = sample_channel
        self.sample_slice = sample_slice
        self._sample_step = 0
        self._sample_saved = 0

    @staticmethod
    def _resolve_subbands(band: Union[str, List[int], Tuple[int, ...]]) -> List[int]:
        if isinstance(band, (list, tuple)):
            indices = [int(i) for i in band]
        else:
            lowered = str(band).lower()
            if lowered in ("all", "*"):
                indices = list(range(8))
            elif lowered in ("low", "ll", "lll"):
                indices = [0]
            elif lowered in ("mid", "mf", "middle"):
                indices = [1, 2, 4]
            elif lowered in ("high", "hf"):
                indices = list(range(1, 8))
            else:
                raise ValueError(f"Unknown band selection '{band}'. Use all/low/high or a list.")

        cleaned: List[int] = []
        for idx in indices:
            if idx < 0:
                idx = 8 + idx
            if idx < 0 or idx > 7:
                raise ValueError(f"Subband index {idx} out of range [0..7]")
            cleaned.append(idx)
        return sorted(set(cleaned))

    @staticmethod
    def _resolve_levels(
        levels: Union[str, List[int], Tuple[int, ...]],
        num_levels: int,
    ) -> List[int]:
        if isinstance(levels, (list, tuple)):
            indices = [int(i) for i in levels]
        else:
            lowered = str(levels).lower()
            if lowered in ("all", "*"):
                indices = list(range(num_levels))
            else:
                indices = [int(lowered)]

        cleaned: List[int] = []
        for idx in indices:
            if idx < 0:
                idx = num_levels + idx
            if idx < 0 or idx >= num_levels:
                raise ValueError(f"Level index {idx} out of bounds for {num_levels} levels")
            cleaned.append(idx)
        return sorted(set(cleaned))

    @staticmethod
    def _resolve_stages(stages: Union[str, List[int], Tuple[int, ...]]) -> List[int]:
        if isinstance(stages, (list, tuple)):
            indices = [int(i) for i in stages]
        else:
            lowered = str(stages).lower()
            if lowered in ("all", "*"):
                return []
            indices = [int(lowered)]

        cleaned: List[int] = []
        for idx in indices:
            if idx < 0:
                raise ValueError("Negative stage indices are not supported for sampling.")
            cleaned.append(idx)
        return sorted(set(cleaned))

    def _dwt3_single(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 5:
            raise ValueError("Wavelet distillation expects 3D features: (B, C, D, H, W).")

        b, c, d, h, w = features.shape
        filters = self.dec_filters.to(dtype=features.dtype, device=features.device)
        flat = features.view(b * c, 1, d, h, w)
        coeffs = F.conv3d(flat, filters, stride=2, padding=self.pad)
        coeffs = coeffs.view(b, c, 8, coeffs.shape[-3], coeffs.shape[-2], coeffs.shape[-1])
        return coeffs

    def _idwt3_single(self, coeffs: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
        if coeffs.dim() != 6:
            raise ValueError("Wavelet coefficients must be (B, C, 8, D, H, W).")
        b, c, _, d, h, w = coeffs.shape
        filters = self.rec_filters.to(dtype=coeffs.dtype, device=coeffs.device)
        flat = coeffs.view(b * c, 8, d, h, w)
        recon = F.conv_transpose3d(flat, filters, stride=2, padding=self.pad)
        recon = recon.view(b, c, recon.shape[-3], recon.shape[-2], recon.shape[-1])
        out_d, out_h, out_w = out_shape
        return recon[..., :out_d, :out_h, :out_w]

    def _dwt3_multilevel(
        self, features: torch.Tensor
    ) -> List[Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor]]:
        coeffs_per_level: List[Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor]] = []
        current = features
        for _ in range(self.levels):
            current_shape = (current.shape[-3], current.shape[-2], current.shape[-1])
            coeffs = self._dwt3_single(current)
            coeffs_per_level.append((coeffs, current_shape, current))
            current = coeffs[:, :, 0, ...]
        return coeffs_per_level

    def _maybe_save_samples(
        self,
        teacher_before: torch.Tensor,
        teacher_after: torch.Tensor,
        *,
        level_idx: int,
        stage_idx: Optional[int],
    ) -> None:
        if not self.sample:
            return
        if self._sample_saved >= self.sample_max:
            return
        if self.sample_every > 1 and (self._sample_step % self.sample_every) != 0:
            return
        if level_idx not in self.sample_levels:
            return
        if stage_idx is not None and self.sample_stages:
            if stage_idx not in self.sample_stages:
                return

        sample_dir = self.sample_dir or os.path.join(os.getcwd(), "frequency_v2_samples")
        stage_tag = f"stage{stage_idx}" if stage_idx is not None else "stageX"
        out_dir = os.path.join(sample_dir, stage_tag, f"level{level_idx}")
        os.makedirs(out_dir, exist_ok=True)

        def _to_image(feat: torch.Tensor) -> "torch.Tensor":
            if feat.dim() != 5:
                raise ValueError("Expected feature map shape (B, C, D, H, W)")
            b, c, d, h, w = feat.shape
            b = min(b, 1)
            if isinstance(self.sample_channel, str) and self.sample_channel == "mean":
                chan = feat[:b].mean(dim=1)  # (B, D, H, W)
            else:
                ch_idx = int(self.sample_channel)
                ch_idx = max(0, min(ch_idx, c - 1))
                chan = feat[:b, ch_idx, ...].unsqueeze(1)
                chan = chan.squeeze(1)
            if isinstance(self.sample_slice, str) and self.sample_slice == "mid":
                z = d // 2
            else:
                z = int(self.sample_slice)
                z = max(0, min(z, d - 1))
            img = chan[0, z, ...]  # (H, W)
            return img

        def _save_pair(tag: str, before: torch.Tensor, after: torch.Tensor) -> None:
            try:
                import numpy as np
            except Exception:
                return

            img_before = _to_image(before).detach().cpu()
            img_after = _to_image(after).detach().cpu()
            stacked = torch.stack([img_before, img_after], dim=0)

            def _normalize(x: torch.Tensor) -> torch.Tensor:
                x_min = float(x.min())
                x_max = float(x.max())
                if x_max - x_min < 1e-6:
                    return torch.zeros_like(x)
                return (x - x_min) / (x_max - x_min)

            before_norm = _normalize(stacked[0])
            after_norm = _normalize(stacked[1])
            comp = torch.cat([before_norm, after_norm], dim=-1)
            comp_np = (comp.numpy() * 255.0).clip(0, 255).astype("uint8")

            filename = f"{tag}_step{self._sample_step:06d}.png"
            path = os.path.join(out_dir, filename)

            try:
                import imageio.v2 as imageio
                imageio.imwrite(path, comp_np)
            except Exception:
                try:
                    from PIL import Image
                    Image.fromarray(comp_np).save(path)
                except Exception:
                    np.save(path.replace(".png", ".npy"), comp_np)

        _save_pair("teacher", teacher_before, teacher_after)
        self._sample_saved += 1

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        stage_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del target  # unused
        if self.align is not None:
            student_feat = self.align(student_feat)

        student_levels = self._dwt3_multilevel(student_feat)
        teacher_levels = self._dwt3_multilevel(teacher_feat)

        self._sample_step += 1
        losses: List[torch.Tensor] = []
        for level_idx, ((s_coeffs, s_shape, s_in), (t_coeffs, t_shape, t_in)) in enumerate(
            zip(student_levels, teacher_levels)
        ):
            if s_shape != t_shape:
                raise ValueError("Student/teacher feature shapes must match at each level.")
            if self.subband_indices != list(range(8)):
                mask = torch.zeros_like(s_coeffs)
                mask[:, :, self.subband_indices, ...] = 1
                s_coeffs = s_coeffs * mask
                t_coeffs = t_coeffs * mask
            s_recon = self._idwt3_single(s_coeffs, s_shape)
            t_recon = self._idwt3_single(t_coeffs, t_shape)
            if self.sample:
                with torch.no_grad():
                    self._maybe_save_samples(
                        t_in,
                        t_recon,
                        level_idx=level_idx,
                        stage_idx=stage_idx,
                    )
            losses.append(F.mse_loss(s_recon, t_recon))

        loss = torch.stack(losses).mean()
        return loss, {"wavelet": loss.detach()}


class FrequencyDistillationV2Mean(DistillationMethod, nn.Module):
    """
    Stage-wise wavelet MSE distillation wrapper (mean across stages).
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        required = ["student_channels", "teacher_channels"]
        for key in required:
            if key not in config:
                raise ValueError(f"FrequencyDistillation requires '{key}' in config")

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
            wavelet=config.get("wavelet", "haar"),
            levels=int(config.get("levels", 1)),
            band=config.get("band", "all"),
            sample=bool(config.get("sample", False)),
            sample_dir=config.get("sample_dir"),
            sample_every=int(config.get("sample_every", 200)),
            sample_max=int(config.get("sample_max", 50)),
            sample_stages=config.get("sample_stages", "all"),
            sample_levels=config.get("sample_levels", "all"),
            sample_channel=config.get("sample_channel", "mean"),
            sample_slice=config.get("sample_slice", "mid"),
        )

        self.distill_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            module = SimpleFrequencyDistillationV2Mean(
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
        student_output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del target, student_output  # unused on purpose
        if str(stage_idx) not in self.distill_modules:
            raise ValueError(f"Stage {stage_idx} not configured for FrequencyDistillation")

        module = self.distill_modules[str(stage_idx)]
        stage_loss, stage_loss_dict = module(student_feat, teacher_feat, None, stage_idx=stage_idx)
        return stage_loss, stage_loss_dict

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        student_output = kwargs.get("student_output")
        if not self.layer_indices:
            raise ValueError("No encoder stages selected for FrequencyDistillation")

        total_loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, float] = {}
        stage_count = len(self.layer_indices)
        stage_scale = 1.0 / stage_count if stage_count > 0 else 1.0

        for idx in self.layer_indices:
            key = f"stage{idx}"
            student_feat = student_features.get(key)
            teacher_feat = teacher_features.get(key)

            if student_feat is None or teacher_feat is None:
                raise ValueError(
                    f"FrequencyDistillation requires '{key}' features. "
                    f"Student keys: {list(student_features.keys())}, "
                    f"Teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                idx,
                student_feat,
                teacher_feat,
                target,
                student_output=student_output,
            )
            stage_loss = stage_loss * stage_scale
            if total_loss is None:
                total_loss = stage_loss
            else:
                total_loss = total_loss + stage_loss

            for name, value in stage_loss_dict.items():
                value_item = value.item() if isinstance(value, torch.Tensor) else float(value)
                loss_dict[f"stage{idx}_{name}"] = value_item * stage_scale

        if total_loss is None:
            total_loss = torch.zeros((), device=next(iter(student_features.values())).device)
        return total_loss, loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{idx}": f"encoder.stages[{idx}]" for idx in self.layer_indices}

    def to(self, device: torch.device) -> "FrequencyDistillationV2Mean":
        self.distill_modules = self.distill_modules.to(device)
        return self


__all__ = ["SimpleFrequencyDistillationV2Mean", "FrequencyDistillationV2Mean"]