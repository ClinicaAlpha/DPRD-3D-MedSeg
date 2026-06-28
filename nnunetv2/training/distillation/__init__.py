"""
nnUNet-KD distillation package.

Keep top-level imports lightweight. Heavy modules (trainer/methods) are loaded
lazily on attribute access.
"""
from __future__ import annotations

from importlib import import_module

from .config import (
    DistillationConfig,
    get_baseline_config,
    get_boundary_config,
    get_boundary_v1_config,
    get_boundary_v2_config,
    get_reco_config,
    get_fitnet_config,
    get_cwd_config,
    get_skd_config,
    get_ifvd_config,
    get_relation_config,
)

_METHOD_EXPORTS = {
    "METHOD_REGISTRY",
    "build_method",
    "BoundaryDistillation",
    "BoundaryDistillationV1",
    "BoundaryDistillationV2",
    "ReCoDistillation",
    "FeatureDistillation",
    "FitNetDistillation",
    "CWDDistillation",
    "IFVDDistillation",
    "SKDDistillation",
    "DPRD",
    "FrequencyDistillationV2Mean",
    "LogitsDistillation",
    "BoundaryLogitsPrewitt",
    "BoundaryPrewittDKD",
}


def __getattr__(name: str):
    if name == "DistillationTrainer":
        from .distiller import DistillationTrainer

        return DistillationTrainer
    if name == "ExponentialMovingAverage":
        from .ema import ExponentialMovingAverage

        return ExponentialMovingAverage
    if name in _METHOD_EXPORTS:
        methods_module = import_module(".methods", __name__)
        return getattr(methods_module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DistillationTrainer",
    "DistillationConfig",
    "BoundaryDistillation",
    "ReCoDistillation",
    "FeatureDistillation",
    "FitNetDistillation",
    "CWDDistillation",
    "IFVDDistillation",
    "SKDDistillation",
    "DPRD",
    "FrequencyDistillationV2Mean",
    "LogitsDistillation",
    "build_method",
    "METHOD_REGISTRY",
    "get_boundary_config",
    "get_boundary_v1_config",
    "get_boundary_v2_config",
    "get_reco_config",
    "get_fitnet_config",
    "get_cwd_config",
    "get_skd_config",
    "get_ifvd_config",
    "get_relation_config",
    "get_baseline_config",
    "ExponentialMovingAverage",
]
