"""
Configuration system for distillation training.

This is adapted from the original nnUNet distillation utilities and extended
to cover multiple KD strategies (boundary, reco, skd, feature, none).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


@dataclass
class DistillationConfig:
    """
    Unified configuration for distillation training.
    """

    # ==================== Teacher Model ====================
    teacher_checkpoint: Optional[str] = None
    teacher_plans: Optional[str] = None
    freeze_teacher: bool = True

    # ==================== Configuration ====================
    configuration: str = "3d_fullres"

    # ==================== Student Model ====================
    reduction_factor: int = 1
    student_plans: Optional[str] = None
    student_architecture: Optional[str] = None
    student_features_per_stage: Optional[List[int]] = None
    # Optional MedNeXt-specific student kwargs (YAML-only in current CLI flow)
    mednext_model_id: Optional[str] = None
    mednext_exp_r: Optional[Union[int, Sequence[int]]] = None
    mednext_block_counts: Optional[List[int]] = None
    mednext_kernel_size: Optional[int] = None
    mednext_enc_kernel_size: Optional[int] = None
    mednext_dec_kernel_size: Optional[int] = None
    mednext_checkpoint_style: Optional[str] = None
    mednext_norm_type: Optional[str] = None
    mednext_grn: Optional[bool] = None
    mednext_do_res: Optional[bool] = None
    mednext_do_res_up_down: Optional[bool] = None

    # ==================== Distillation Strategy ====================
    strategy: str = "boundary"
    strategy_config: Dict[str, Any] = field(default_factory=dict)

    # ==================== Loss Weights ====================
    kd_weight: float = 0.5
    kd_schedule: str = "constant"
    kd_warmup_epochs: int = 0
    kd_warmup_start_epoch: int = 0

    # ==================== Training Configuration ====================
    num_epochs: int = 1000
    batch_size: Optional[int] = None
    initial_lr: float = 1e-2
    weight_decay: float = 3e-5

    # ==================== Validation & Checkpointing ====================
    val_interval: int = 1
    save_interval: int = 100
    early_stopping_patience: int = 40
    eval_with_best: bool = False

    # ==================== Logging ====================
    log_interval: int = 10
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    # ==================== Experiment Naming ====================
    experiment_tag: Optional[str] = None
    experiment_tag_mode: str = "append"

    # ==================== Advanced Options ====================
    mixed_precision: bool = True
    compile_model: Optional[bool] = None
    num_workers: int = 12
    seed: Optional[int] = None
    use_ema: Optional[bool] = None
    ema_decay: float = 0.999

    # ==================== I/O Methods ====================

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DistillationConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        config_dict = asdict(self)
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self

    def __repr__(self) -> str:
        lines = ["DistillationConfig:"]
        for key, value in asdict(self).items():
            if isinstance(value, dict) and value:
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# ==================== Preset Configs ====================

def get_boundary_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="boundary",
        strategy_config={
            "boundary_width": 3,
            "use_boundary_loss": True,
            "use_attention_loss": False,
        },
        kd_weight=0.5,
        kd_schedule="warmup",
        kd_warmup_epochs=50,
    )
    config.update(**kwargs)
    return config


def get_boundary_v1_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="boundary_v1",
        strategy_config={
            "boundary_width": 3,
            "warmup_iters": 12000,
            "teacher_stat_ema_decay": 0.97,
            "uncertainty_scale": 1.5,
            "use_teacher_stats": True,
        },
        kd_weight=0.5,
        kd_schedule="warmup",
        kd_warmup_epochs=150,
    )
    config.update(**kwargs)
    return config


def get_boundary_v2_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="boundary_v2",
        strategy_config={
            "layer_indices": [0, 1, 2, 3],
            "sobel_scale": 1.0,
            "softmax_temperature": 0.5,
        },
        kd_weight=0.5,
        kd_schedule="warmup",
        kd_warmup_epochs=150,
    )
    config.update(**kwargs)
    return config


def get_reco_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="reco",
        strategy_config={
            "temp": 0.5,
            "tau": 1.0,
            "coef_fg": 1.0,
            "coef_bg": 1.0,
            "coef_mask": 1.0,
            "coef_rel": 1.0,
        },
        kd_weight=0.01,
        kd_schedule="warmup",
        kd_warmup_epochs=600,
    )
    config.update(**kwargs)
    return config


def get_fitnet_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="fitnet",
        strategy_config={
            "layer_indices": "all",
            "is_3d": True,
        },
        kd_weight=1.0,
        kd_schedule="warmup",
        kd_warmup_epochs=100,
    )
    config.update(**kwargs)
    return config


def get_cwd_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="cwd",
        strategy_config={
            "layer_indices": "all",
            "norm_type": "channel",
            "divergence": "mse",
            "temperature": 1.0,
        },
        kd_weight=1.0,
        kd_schedule="warmup",
        kd_warmup_epochs=200,
    )
    config.update(**kwargs)
    return config


def get_skd_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="skd",
        strategy_config={
            "patch_size": 2,
            "layer_indices": "all",
        },
        kd_weight=1.0,
        kd_schedule="warmup",
        kd_warmup_epochs=200,
    )
    config.update(**kwargs)
    return config


def get_ifvd_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="ifvd",
        strategy_config={
            "layer_indices": "all",
        },
        kd_weight=1.0,
        kd_schedule="warmup",
        kd_warmup_epochs=200,
    )
    config.update(**kwargs)
    return config


def get_cirkd_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="cirkd",
        strategy_config={
            "layer_indices": "all",
            "temperature": 0.1,
            "target_size": [4, 4, 4],
        },
        kd_weight=0.1,
        kd_schedule="warmup",
        kd_warmup_epochs=100,
    )
    config.update(**kwargs)
    return config


def get_relation_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="DPRD",
        strategy_config={
            "layer_indices": "all",
            "lam_delta": 1.0,
            "lam_dist": 0.0,
            "use_mask": False,
            "proj_dim": 128,
        },
        kd_weight=1.0,
        kd_schedule="warmup",
        kd_warmup_epochs=150,
    )
    config.update(**kwargs)
    return config


def get_rkd_config(teacher_checkpoint: str, **kwargs) -> DistillationConfig:
    config = DistillationConfig(
        teacher_checkpoint=teacher_checkpoint,
        strategy="rkd",
        strategy_config={
            "layer_indices": "all",
            "proj_dim": 128,
            "distance_weight": 25.0,
            "angle_weight": 50.0,
            "use_mask": False,
        },
        kd_weight=1.0,
        kd_schedule="warmup",
        kd_warmup_epochs=150,
    )
    config.update(**kwargs)
    return config


def get_baseline_config(**kwargs) -> DistillationConfig:
    config = DistillationConfig(
        strategy="none",
        kd_weight=0.0,
    )
    config.update(**kwargs)
    return config


__all__ = [
    "DistillationConfig",
    "get_boundary_config",
    "get_boundary_v1_config",
    "get_boundary_v2_config",
    "get_reco_config",
    "get_fitnet_config",
    "get_cwd_config",
    "get_skd_config",
    "get_ifvd_config",
    "get_cirkd_config",
    "get_relation_config",
    "get_rkd_config",
    "get_baseline_config",
]
