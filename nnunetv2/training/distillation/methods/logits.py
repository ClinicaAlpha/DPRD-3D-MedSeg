"""
Logits-based knowledge distillation.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F

from .base import DistillationMethod


class LogitsDistillation(DistillationMethod):
    """
    KL divergence on segmentation logits.

    Config parameters:
        temperature: float, softening temperature (default 2.0)
        chunk_size: int, optional chunked computation over batch dimension to save memory
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.temperature: float = float(config.get("temperature", 2.0))
        # chunk_size <= 0 disables chunking
        self.chunk_size: int = int(config.get("chunk_size", 0))

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        student_output: Union[torch.Tensor, Tuple[torch.Tensor, ...], None] = kwargs.get("student_output")
        teacher_output: Union[torch.Tensor, Tuple[torch.Tensor, ...], None] = kwargs.get("teacher_output")

        if student_output is None or teacher_output is None:
            raise ValueError("LogitsDistillation requires both student_output and teacher_output.")

        student_logits = student_output[0] if isinstance(student_output, (list, tuple)) else student_output
        teacher_logits = teacher_output[0] if isinstance(teacher_output, (list, tuple)) else teacher_output

        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Student and teacher logits must have the same shape. "
                f"Got student {tuple(student_logits.shape)}, teacher {tuple(teacher_logits.shape)}"
            )

        loss = self._kl_loss(student_logits, teacher_logits)
        return loss, {"kl": float(loss.detach())}

    def _kl_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature

        def _kl_chunk(s_logits, t_logits):
            # Flatten spatial dims so KL is averaged over voxels, not summed
            s_logits = s_logits.to(torch.float32) / T
            t_logits = t_logits.to(torch.float32).detach() / T
            s_flat = s_logits.flatten(2).transpose(1, 2).reshape(-1, s_logits.shape[1])
            t_flat = t_logits.flatten(2).transpose(1, 2).reshape(-1, t_logits.shape[1])
            log_probs = F.log_softmax(s_flat, dim=1)
            probs = F.softmax(t_flat, dim=1)
            # batchmean divides by total elements (B * voxels)
            return F.kl_div(log_probs, probs, reduction="batchmean", log_target=False) * (T * T)

        if self.chunk_size and self.chunk_size > 0:
            B = student_logits.shape[0]
            losses = []
            for start in range(0, B, self.chunk_size):
                end = min(start + self.chunk_size, B)
                losses.append(_kl_chunk(student_logits[start:end], teacher_logits[start:end]))
            return torch.stack(losses).mean()

        return _kl_chunk(student_logits, teacher_logits)

    def get_required_features(self) -> Dict[str, str]:
        # No hooks needed; we use full-model outputs supplied via kwargs.
        return {}


__all__ = ["LogitsDistillation"]
