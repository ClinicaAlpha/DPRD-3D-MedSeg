"""
Base abstractions for distillation methods.

Each method implements a unified interface so the distiller can assemble
them dynamically based on the user configuration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class DistillationMethod(ABC):
    """
    Unified interface for knowledge distillation methods.

    Subclasses must implement two things:
    - ``forward``: consume feature dictionaries and return (loss, metrics)
    - ``get_required_features``: describe which network hooks to register
    """

    def __init__(self, **config):
        self.config = config

    @abstractmethod
    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the distillation loss for a batch.
        """

    @abstractmethod
    def get_required_features(self) -> Dict[str, str]:
        """
        Describe which intermediate tensors need to be captured.

        Returns
        -------
        Dict[str, str]
            Mapping from storage key -> dotted module path.
        """

    def to(self, device: torch.device) -> "DistillationMethod":
        """
        Optional device transfer override.

        Base implementation returns ``self`` so subclasses only need to
        override when they hold torch Modules.
        """
        return self

    def __repr__(self) -> str:
        cfg = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({cfg})"

