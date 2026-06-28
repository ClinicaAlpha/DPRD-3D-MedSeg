"""
Method registry for distillation strategies.

This module keeps import-time cost low by lazily importing concrete strategy
classes only when they are actually requested.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Dict, Tuple, Type

import torch

from .base import DistillationMethod


class NoDistillation(DistillationMethod):
    """Baseline placeholder that returns zero loss."""

    def forward(self, student_features, teacher_features, target, **kwargs):
        device = target.device if torch.is_tensor(target) else torch.device("cpu")
        return torch.tensor(0.0, device=device), {}

    def get_required_features(self) -> Dict[str, str]:
        return {}


_METHOD_IMPORTS: Dict[str, Tuple[str, str]] = {
    "boundary": ("nnunetv2.training.distillation.methods.boundary", "BoundaryDistillation"),
    "boundary_v1": ("nnunetv2.training.distillation.methods.boundary_v1", "BoundaryDistillationV1"),
    "boundary_v2": ("nnunetv2.training.distillation.methods.boundary_v2", "BoundaryDistillationV2"),
    "reco": ("nnunetv2.training.distillation.methods.reco", "ReCoDistillation"),
    "feature": ("nnunetv2.training.distillation.methods.feature", "FeatureDistillation"),
    "fitnet": ("nnunetv2.training.distillation.methods.fitnet", "FitNetDistillation"),
    "cwd": ("nnunetv2.training.distillation.methods.cwd", "CWDDistillation"),
    "ifvd": ("nnunetv2.training.distillation.methods.ifvd", "IFVDDistillation"),
    "cirkd": ("nnunetv2.training.distillation.methods.cirkd", "CIRKDDistillation"),
    "skd": ("nnunetv2.training.distillation.methods.skd", "SKDDistillation"),
    "dprd": ("nnunetv2.training.distillation.methods.DPRD", "DPRD"),
    "rkd": ("nnunetv2.training.distillation.methods.rkd", "RKDDistillation"),
    "frequency_v2_mean": ("nnunetv2.training.distillation.methods.frequency", "FrequencyDistillationV2Mean"),
    "logits": ("nnunetv2.training.distillation.methods.logits", "LogitsDistillation"),
    "boundary_prewitt": (
        "nnunetv2.training.distillation.methods.boundary_prewitt_dkd",
        "BoundaryLogitsPrewitt",
    ),
    "boundary_prewitt_dkd": (
        "nnunetv2.training.distillation.methods.boundary_prewitt_dkd",
        "BoundaryPrewittDKD",
    ),
}

METHOD_REGISTRY: Dict[str, str] = {
    **{k: v[1] for k, v in _METHOD_IMPORTS.items()},
    "none": "NoDistillation",
}


@lru_cache(maxsize=None)
def _resolve_method_class(name: str) -> Type[DistillationMethod]:
    if name == "none":
        return NoDistillation
    module_path, class_name = _METHOD_IMPORTS[name]
    module = import_module(module_path)
    return getattr(module, class_name)


def build_method(name: str, **config) -> DistillationMethod:
    key = name.lower()
    if key not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown distillation method '{name}'. Available: {available}")
    method_cls = _resolve_method_class(key)
    return method_cls(**config)


_CLASS_NAME_TO_KEY = {class_name: key for key, class_name in METHOD_REGISTRY.items()}


def __getattr__(name: str):
    if name == "NoDistillation":
        return NoDistillation
    if name == "DistillationMethod":
        return DistillationMethod
    key = _CLASS_NAME_TO_KEY.get(name)
    if key is not None:
        return _resolve_method_class(key)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DistillationMethod",
    "BoundaryDistillation",
    "BoundaryDistillationV1",
    "BoundaryDistillationV2",
    "ReCoDistillation",
    "FeatureDistillation",
    "FitNetDistillation",
    "CWDDistillation",
    "IFVDDistillation",
    "CIRKDDistillation",
    "SKDDistillation",
    "DPRD",
    "RKDDistillation",
    "FrequencyDistillationV2Mean",
    "LogitsDistillation",
    "BoundaryLogitsPrewitt",
    "BoundaryPrewittDKD",
    "NoDistillation",
    "METHOD_REGISTRY",
    "build_method",
]
