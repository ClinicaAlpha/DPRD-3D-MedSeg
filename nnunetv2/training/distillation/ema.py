"""
Utility helpers for maintaining an exponential moving average (EMA) of model parameters.

EMA smoothing can stabilise validation metrics and give a small boost in final
performance, especially when training lightweight students with knowledge
distillation. The implementation is intentionally lightweight and keeps all
logic contained within the distillation package so that the core nnU-Net files
remain untouched.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """
    Maintain an exponential moving average of a model's trainable parameters.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters are tracked. Only parameters with
        ``requires_grad=True`` are included.
    decay : float
        Smoothing factor in [0, 1). Higher values retain longer history.
    device : Optional[torch.device]
        Optional device on which to keep the EMA copy. If omitted the helper
        mirrors the parameter device at creation/update time.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")

        self.decay = decay
        self.device = device

        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}

        self._register(model)

    def _register(self, model: nn.Module):
        """Initialise shadow parameters from the given model."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            tensor = param.detach().clone()
            if self.device is not None:
                tensor = tensor.to(self.device)
            self.shadow_params[name] = tensor

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update the EMA parameters with the current model weights.
        """
        decay = self.decay

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name not in self.shadow_params:
                # Handle parameters introduced after initialisation.
                tensor = param.detach().clone()
                if self.device is not None:
                    tensor = tensor.to(self.device)
                self.shadow_params[name] = tensor
                continue

            shadow = self.shadow_params[name]
            if self.device is None:
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                    self.shadow_params[name] = shadow
                shadow.data.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
            else:
                if shadow.device != self.device:
                    shadow = shadow.to(self.device)
                    self.shadow_params[name] = shadow
                shadow.data.mul_(decay).add_(param.detach().to(self.device), alpha=1.0 - decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """
        Replace model parameters with their EMA counterparts.

        Stores the original weights so they can be restored with ``restore``.
        """
        self.backup_params.clear()

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow_params:
                continue

            self.backup_params[name] = param.detach().clone()
            param.copy_(self.shadow_params[name].to(param.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """
        Restore model parameters that were swapped out via ``apply_shadow``.
        """
        if not self.backup_params:
            return

        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.copy_(self.backup_params[name])

        self.backup_params.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the EMA parameters."""
        return {k: v.clone() for k, v in self.shadow_params.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load EMA parameters from ``state_dict``."""
        self.shadow_params = {k: v.clone() for k, v in state_dict.items()}
        self.backup_params.clear()
