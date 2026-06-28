"""
High-level distillation trainer with clean API

This is the main entry point for distillation training.
"""
from typing import Union, Optional, Dict, Any, List, Tuple
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from .config import DistillationConfig
from .methods import build_method, DistillationMethod
from .ema import ExponentialMovingAverage
from .utils import (
    load_teacher_model,
    register_feature_hooks,
    get_kd_weight_scheduler,
    setup_logging
)


class DistillationTrainer(nnUNetTrainer):
    """
    High-level trainer for knowledge distillation in medical image segmentation

    This trainer combines:
    - nnUNet's training infrastructure
    - Flexible distillation strategies
    - Clean configuration system
    - Extensive logging and checkpointing

    Example usage:
        >>> from nnunetv2.training.distillation import DistillationTrainer, DistillationConfig
        >>>
        >>> # Create config
        >>> config = DistillationConfig.from_yaml('config.yaml')
        >>> # Or use preset
        >>> config = get_boundary_config(teacher_checkpoint='/path/to/teacher.pth')
        >>>
        >>> # Create trainer
        >>> trainer = DistillationTrainer(
        ...     plans=plans,
        ...     configuration='3d_fullres',
        ...     fold=0,
        ...     dataset_json=dataset_json,
        ...     distillation_config=config
        ... )
        >>>
        >>> # Initialize and train
        >>> trainer.initialize()
        >>> trainer.run_training()

    The trainer handles:
    - ✅ Teacher model loading
    - ✅ Feature extraction hooks
    - ✅ Distillation loss computation
    - ✅ KD weight scheduling
    - ✅ Logging and checkpointing
    - ✅ Mixed precision training
    """

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            args = (f"{timestamp}:", *args)
            add_timestamp = False
        super().print_to_log_file(*args,
                                  also_print_to_console=also_print_to_console,
                                  add_timestamp=add_timestamp)

    def __init__(self,
                 plans: Union[dict, str],
                 configuration: str,
                 fold: int,
                 dataset_json: Union[dict, str],
                 distillation_config: Optional[DistillationConfig] = None,
                 device: torch.device = torch.device('cuda')):
        """
        Initialize distillation trainer

        Args:
            plans: nnUNet plans (dict or path to plans.json)
            configuration: Configuration name (e.g., '3d_fullres')
            fold: Training fold number
            dataset_json: Dataset JSON (dict or path)
            distillation_config: Distillation configuration
            device: Device to use
        """
        # Store distillation config before super().__init__
        self.distill_config = distillation_config or DistillationConfig()

        # WORKAROUND: Parent's __init__ uses inspect.signature(self.__init__) which sees
        # our child class signature including distillation_config, then tries to save
        # all params via locals()[k]. We need to temporarily override __init__ to hide
        # our extra parameter.
        original_init = self.__init__
        self.__init__ = super(DistillationTrainer, self).__init__

        try:
            # Initialize base trainer
            super().__init__(plans, configuration, fold, dataset_json, device)
        finally:
            # Restore our __init__
            self.__init__ = original_init

        # Add our extra parameter to my_init_kwargs (which parent just created)
        self.my_init_kwargs['distillation_config'] = distillation_config

        # Runtime variables
        self.teacher_model: Optional[nn.Module] = None
        self.distill_strategy: Optional[DistillationMethod] = None
        self.kd_weight_scheduler = None

        # Feature storage
        self.student_features: Dict[str, torch.Tensor] = {}
        self.teacher_features: Dict[str, torch.Tensor] = {}

        # Tracking KD weight history without touching core logger
        self.kd_weight_history: List[float] = []
        self._student_network_descriptor: Optional[Dict[str, Any]] = None
        self._student_plan_written: bool = False
        self._student_plan_path: Optional[Path] = None
        self._distill_config_logged: bool = False
        self.iteration_in_epoch: int = 0
        self._iter_log_interval: int = 50

        # Early stopping
        self.best_epoch = 0
        self.best_val_metric = None
        self.early_stop_triggered = False

        # EMA tracking (optional, enabled automatically for KD unless disabled)
        self.ema_helper: Optional[ExponentialMovingAverage] = None
        self._ema_active_during_validation: bool = False

        # Compile override flag (None -> follow base behavior)
        self._compile_override: Optional[bool] = self.distill_config.compile_model

        # Override base trainer settings from config
        self._apply_config_overrides()
        self._customize_output_paths()

    def _apply_config_overrides(self):
        """Apply distillation config overrides to base trainer settings"""
        cfg = self.distill_config

        if cfg.num_epochs is not None:
            self.num_epochs = cfg.num_epochs

        if cfg.initial_lr is not None:
            self.initial_lr = cfg.initial_lr

        if cfg.weight_decay is not None:
            self.weight_decay = cfg.weight_decay

        if cfg.log_interval is not None:
            # Iter-level KD logging cadence.
            self._iter_log_interval = max(1, int(cfg.log_interval))

        if cfg.batch_size is not None:
            # Ensure dataloaders use the requested batch size instead of plans default.
            batch_size = int(cfg.batch_size)
            old_batch_size = getattr(self.configuration_manager, "batch_size", None) \
                if hasattr(self, "configuration_manager") else None
            self.batch_size = batch_size
            if hasattr(self, "configuration_manager"):
                self.configuration_manager.configuration["batch_size"] = batch_size
            if old_batch_size != batch_size:
                self.print_to_log_file(
                    f"Overriding batch size from plans: {old_batch_size} -> {batch_size}"
                )

        if cfg.early_stopping_patience is not None:
            self.patience = cfg.early_stopping_patience

        if cfg.seed is not None:
            import numpy as np
            import random

            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Update compile override flag in case config was modified after init
        self._compile_override = cfg.compile_model

    def build_network_architecture(self,
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """
        Build student network with optional modifications

        Supports:
        1. Channel reduction via reduction_factor
        2. Manual channel specification via student_features_per_stage
        3. Different architecture via student_architecture

        Args:
            architecture_class_name: Network class name
            arch_init_kwargs: Network initialization kwargs
        """
        cfg = self.distill_config
        modified_kwargs = arch_init_kwargs.copy()
        modified_arch = architecture_class_name

        # Option 1: Use different architecture
        if cfg.student_architecture is not None:
            modified_arch = cfg.student_architecture
            self.print_to_log_file(f"🏗️  Using custom student architecture: {modified_arch}")

        # Option 2: Manual channel specification (highest priority)
        if cfg.student_features_per_stage is not None:
            original_features = arch_init_kwargs.get('features_per_stage', [])
            modified_kwargs['features_per_stage'] = cfg.student_features_per_stage

            self.print_to_log_file(f"📉 Using manual channel specification")
            self.print_to_log_file(f"   Original: {original_features}")
            self.print_to_log_file(f"   Student:  {cfg.student_features_per_stage}")

        # Option 3: Channel reduction via factor
        elif cfg.reduction_factor > 1:
            original_features = arch_init_kwargs.get('features_per_stage', [])

            if original_features:
                reduced_features = [max(f // cfg.reduction_factor, 8) for f in original_features]
                modified_kwargs['features_per_stage'] = reduced_features

                self.print_to_log_file(f"📉 Applying channel reduction (factor={cfg.reduction_factor})")
                self.print_to_log_file(f"   Original: {original_features}")
                self.print_to_log_file(f"   Reduced:  {reduced_features}")

        # Option 4: Optional MedNeXt-specific overrides from YAML config
        mednext_overrides = {
            "mednext_model_id": cfg.mednext_model_id,
            "exp_r": cfg.mednext_exp_r,
            "block_counts": cfg.mednext_block_counts,
            "kernel_size": cfg.mednext_kernel_size,
            "enc_kernel_size": cfg.mednext_enc_kernel_size,
            "dec_kernel_size": cfg.mednext_dec_kernel_size,
            "checkpoint_style": cfg.mednext_checkpoint_style,
            "norm_type": cfg.mednext_norm_type,
            "grn": cfg.mednext_grn,
            "do_res": cfg.mednext_do_res,
            "do_res_up_down": cfg.mednext_do_res_up_down,
        }
        for key, value in mednext_overrides.items():
            if value is not None:
                modified_kwargs[key] = value
                self.print_to_log_file(f"🧩 Applying MedNeXt override: {key}={value}")

        self._student_network_descriptor = {
            'architecture_class_name': modified_arch,
            'arch_kwargs': deepcopy(modified_kwargs),
            'arch_kwargs_req_import': deepcopy(arch_init_kwargs_req_import),
            'num_input_channels': num_input_channels,
            'num_output_channels': num_output_channels,
            'enable_deep_supervision': enable_deep_supervision
        }

        # Build network
        return super().build_network_architecture(
            modified_arch,
            modified_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )

    def _sanitize_tag(self, tag: str) -> str:
        tag = tag.replace(' ', '_')
        return re.sub(r'[^A-Za-z0-9_.-]', '_', tag)

    def _build_experiment_tag(self) -> str:
        cfg = self.distill_config
        parts: List[str] = []
        custom_tag = None
        custom_mode = "append"

        if cfg.experiment_tag:
            custom_tag = self._sanitize_tag(cfg.experiment_tag)
            custom_mode = (cfg.experiment_tag_mode or "append").lower()
            if custom_mode == "override" and custom_tag:
                return custom_tag

        if cfg.strategy:
            parts.append(self._sanitize_tag(cfg.strategy))

        if cfg.reduction_factor and cfg.reduction_factor > 1:
            parts.append(f"R{cfg.reduction_factor}")

        if cfg.student_architecture:
            parts.append(self._sanitize_tag(cfg.student_architecture.split('.')[-1]))

        if cfg.student_features_per_stage:
            parts.append("feat-" + "x".join(str(f) for f in cfg.student_features_per_stage))

        if cfg.kd_schedule and cfg.kd_schedule != 'constant':
            parts.append(self._sanitize_tag(f"kd-{cfg.kd_schedule}"))

        if custom_tag:
            if custom_mode != "append":
                custom_mode = "append"
            parts.append(custom_tag)

        return "__".join(parts)

    def _customize_output_paths(self):
        if self.output_folder_base is None:
            return

        tag = self._build_experiment_tag()
        if not tag:
            return

        current_base = Path(self.output_folder_base)
        dataset_root = current_base.parent
        log_filename = Path(self.log_file).name if hasattr(self, 'log_file') else None

        new_folder_name = f"{self.__class__.__name__}__{tag}__{self.plans_manager.plans_name}__{self.configuration_name}"
        self.output_folder_base = str(dataset_root / new_folder_name)
        self.output_folder = join(self.output_folder_base, f'fold_{self.fold}')
        maybe_mkdir_p(self.output_folder_base)
        maybe_mkdir_p(self.output_folder)

        if log_filename is not None:
            self.log_file = str(Path(self.output_folder) / log_filename)

    def _write_student_plans_once(self):
        if self._student_plan_written:
            return
        descriptor = self._student_network_descriptor
        if descriptor is None:
            return
        if self.output_folder_base is None:
            return

        plans_dict = deepcopy(self.plans_manager.plans)
        configs = plans_dict.setdefault('configurations', {})
        config_entry = configs.setdefault(self.configuration_name, {})
        architecture_entry = config_entry.setdefault('architecture', {})
        architecture_entry['network_class_name'] = descriptor['architecture_class_name']
        architecture_entry['arch_kwargs'] = descriptor['arch_kwargs']
        req_import = descriptor['arch_kwargs_req_import']
        if isinstance(req_import, (tuple, list)):
            architecture_entry['_kw_requires_import'] = list(req_import)
        else:
            architecture_entry['_kw_requires_import'] = req_import

        tag = self._build_experiment_tag()
        if tag:
            plans_dict['plans_name'] = f"{self.plans_manager.plans_name}_{tag}"
        else:
            plans_dict['plans_name'] = f"{self.plans_manager.plans_name}_student"

        save_dir = Path(self.output_folder_base)
        maybe_mkdir_p(str(save_dir))
        save_path = save_dir / 'student_plans.json'
        save_json(plans_dict, str(save_path), sort_keys=False)

        self._student_plan_written = True
        self._student_plan_path = save_path
        self.print_to_log_file(f"Student plans written to {save_path}")

    def _do_i_compile(self):
        """
        Respect distillation config compile flag while keeping nnU-Net defaults.

        If compile_model is False we explicitly disable torch.compile. If compile_model
        is True we defer to the parent implementation (which still performs safety checks).
        When compile_model is None we also fall back to the parent behavior.
        """
        if self._compile_override is False:
            # Mirror nnU-Net log style so users understand why compile is skipped
            if 'nnUNet_compile' in os.environ and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because distillation config set compile_model=False")
            return False
        return super()._do_i_compile()

    def _should_enable_ema(self) -> bool:
        """
        Determine whether EMA should be enabled for the current run.

        Defaults to enabling EMA when any distillation strategy (other than 'none')
        is active. Users can force-enable/disable via the configuration.
        """
        cfg = self.distill_config

        if cfg.use_ema is not None:
            return cfg.use_ema

        # Automatically enable for KD setups (teacher available and non-zero KD weight)
        return (
            cfg.strategy != 'none'
            and cfg.teacher_checkpoint is not None
            and cfg.kd_weight > 0
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.iteration_in_epoch = 0

    def initialize(self):
        """
        Initialize trainer components

        Steps:
        1. Initialize base trainer (network, optimizer, dataloader, etc.)
        2. Load teacher model
        3. Setup distillation strategy
        4. Register feature extraction hooks
        5. Setup KD weight scheduler
        6. Setup EMA tracking (optional)
        7. Setup logging
        """
        if self.was_initialized:
            return

        # Step 1: Base initialization
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("🚀 Initializing Distillation Trainer")
        self.print_to_log_file("=" * 60)

        super().initialize()
        self._write_student_plans_once()

        cfg = self.distill_config

        # Step 2: Load teacher model
        if cfg.strategy != 'none' and cfg.teacher_checkpoint is not None:
            self.print_to_log_file(f"\n👨‍🏫 Loading teacher model...")
            self.teacher_model = self._load_teacher_model()
            self.teacher_model.eval()
            if cfg.freeze_teacher:
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
            self.print_to_log_file(f"   ✓ Teacher loaded from: {cfg.teacher_checkpoint}")
        else:
            self.print_to_log_file(f"\n⚠️  No distillation (baseline training)")

        # Step 3: Setup distillation strategy
        if cfg.strategy != 'none':
            self.print_to_log_file(f"\n🎯 Setting up distillation strategy: {cfg.strategy}")
            self.distill_strategy = self._create_distillation_strategy()
            self.distill_strategy = self.distill_strategy.to(self.device)
            self.print_to_log_file(f"   ✓ Strategy config: {cfg.strategy_config}")
        else:
            self.distill_strategy = build_method('none')

        # Step 4: Register feature hooks
        if cfg.strategy != 'none':
            self.print_to_log_file(f"\n🔗 Registering feature extraction hooks...")
            required_features = self.distill_strategy.get_required_features()
            register_feature_hooks(self.network, self.student_features, required_features, 'student')
            register_feature_hooks(self.teacher_model, self.teacher_features, required_features, 'teacher')
            self.print_to_log_file(f"   ✓ Registered hooks for: {list(required_features.keys())}")

        # Step 5: Setup KD weight scheduler
        self.kd_weight_scheduler = get_kd_weight_scheduler(
            schedule=cfg.kd_schedule,
            max_weight=cfg.kd_weight,
            warmup_epochs=cfg.kd_warmup_epochs,
            total_epochs=cfg.num_epochs,
            warmup_start_epoch=cfg.kd_warmup_start_epoch,
        )
        self.print_to_log_file(f"\n📊 KD weight schedule: {cfg.kd_schedule}")
        self.print_to_log_file(f"   Max weight: {cfg.kd_weight}")
        self.print_to_log_file(f"   Warmup epochs: {cfg.kd_warmup_epochs}")
        if cfg.kd_warmup_start_epoch > 0:
            self.print_to_log_file(f"   Warmup start epoch: {cfg.kd_warmup_start_epoch}")

        # Step 6: Setup EMA (optional)
        if self._should_enable_ema():
            self.ema_helper = ExponentialMovingAverage(
                self.network,
                decay=cfg.ema_decay,
                device=torch.device('cpu')
            )
            self.print_to_log_file(f"\n🧮 EMA tracking enabled (decay={cfg.ema_decay})")
        else:
            self.ema_helper = None

        # Step 7: Setup logging
        if cfg.wandb_project is not None:
            setup_logging(cfg, self)
            self.print_to_log_file(f"\n📝 Weights & Biases logging enabled")
            self.print_to_log_file(f"   Project: {cfg.wandb_project}")
            self.print_to_log_file(f"   Run name: {cfg.wandb_name or 'auto'}")

        # Print summary
        self.print_to_log_file("\n" + "=" * 60)
        self.print_to_log_file("✅ Initialization complete!")
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"\nConfiguration summary:")
        self.print_to_log_file(f"  Strategy: {cfg.strategy}")
        self.print_to_log_file(f"  KD weight: {cfg.kd_weight}")
        self.print_to_log_file(f"  Reduction factor: {cfg.reduction_factor}")
        self.print_to_log_file(f"  Num epochs: {cfg.num_epochs}")
        self.print_to_log_file("=" * 60 + "\n")

    def _load_teacher_model(self) -> nn.Module:
        """Load teacher model from checkpoint"""
        cfg = self.distill_config

        # Determine teacher plans path
        teacher_ckpt_path = Path(cfg.teacher_checkpoint)
        if cfg.teacher_plans is not None:
            teacher_plans_path = cfg.teacher_plans
        else:
            # Assume plans.json is in the same directory or parent directory
            if teacher_ckpt_path.parent.name == 'fold_0':  # nnUNet structure
                teacher_plans_path = teacher_ckpt_path.parent.parent / 'plans.json'
            else:
                teacher_plans_path = teacher_ckpt_path.parent / 'plans.json'

        if not os.path.exists(teacher_plans_path):
            raise FileNotFoundError(f"Teacher plans not found: {teacher_plans_path}")

        # Load teacher model using utility function
        teacher_model, missing_keys, unexpected_keys = load_teacher_model(
            checkpoint_path=str(cfg.teacher_checkpoint),
            plans_path=str(teacher_plans_path),
            configuration=self.configuration_name,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.label_manager.num_segmentation_heads,
            device=self.device
        )

        if missing_keys or unexpected_keys:
            self.print_to_log_file("   ⚠️ Teacher checkpoint loaded with key mismatches")
            if missing_keys:
                preview = ', '.join(missing_keys[:10])
                more = f" (+{len(missing_keys) - 10} more)" if len(missing_keys) > 10 else ""
                self.print_to_log_file(f"      Missing keys: {preview}{more}")
            if unexpected_keys:
                preview = ', '.join(unexpected_keys[:10])
                more = f" (+{len(unexpected_keys) - 10} more)" if len(unexpected_keys) > 10 else ""
                self.print_to_log_file(f"      Unexpected keys: {preview}{more}")
        else:
            self.print_to_log_file("   ✓ Teacher state_dict matches architecture")

        return teacher_model

    def _create_distillation_strategy(self) -> DistillationMethod:
        """Create distillation strategy from config"""
        cfg = self.distill_config

        # Prepare strategy config
        strategy_config = cfg.strategy_config.copy()

        student_channels = self._get_network_channels()
        teacher_channels = self._get_network_channels(self.teacher_model, prefer_teacher=True) if self.teacher_model is not None else None

        def maybe_set(key: str, value):
            if value is not None and key not in strategy_config:
                strategy_config[key] = value

        self.print_to_log_file(f"   Detected student channels: {student_channels}")
        if teacher_channels is not None:
            self.print_to_log_file(f"   Detected teacher channels: {teacher_channels}")

        strategy_key = cfg.strategy.lower()

        if strategy_key in ('boundary', 'boundary_v1'):
            maybe_set('student_channels', list(student_channels))
            maybe_set('teacher_channels', list(teacher_channels) if teacher_channels is not None else None)
            maybe_set('num_classes', self.label_manager.num_segmentation_heads)
        elif strategy_key == 'reco':
            maybe_set('student_channels', list(student_channels))
            maybe_set('teacher_channels', list(teacher_channels) if teacher_channels is not None else None)
        elif strategy_key == 'skd':
            maybe_set('student_channels', list(student_channels))
            maybe_set('teacher_channels', list(teacher_channels) if teacher_channels is not None else None)
        elif strategy_key == 'frequency_v2_mean':
            maybe_set('student_channels', list(student_channels))
            maybe_set('teacher_channels', list(teacher_channels) if teacher_channels is not None else None)
        elif strategy_key in ('fitnet', 'cwd', 'rkd', 'cirkd', 'dprd'):
            maybe_set('student_channels', list(student_channels))
            maybe_set('teacher_channels', list(teacher_channels) if teacher_channels is not None else None)
        elif strategy_key == 'ifvd':
            maybe_set('student_channels', list(student_channels))
            maybe_set('teacher_channels', list(teacher_channels) if teacher_channels is not None else None)
            maybe_set('num_classes', self.label_manager.num_segmentation_heads)
        elif strategy_key == 'feature':
            maybe_set('student_channels', student_channels[-1] if student_channels else None)
            maybe_set('teacher_channels', teacher_channels[-1] if teacher_channels else None)
        else:
            maybe_set('student_channels', student_channels[-1] if student_channels else None)
            maybe_set('teacher_channels', teacher_channels[-1] if teacher_channels else None)

        # Create strategy
        strategy = build_method(cfg.strategy, **strategy_config)

        return strategy

    def _get_network_channels(self,
                              network: Optional[nn.Module] = None,
                              prefer_teacher: bool = False) -> list:
        """Extract feature channels from network architecture"""
        net = network or self.network

        def _to_channel_list(raw) -> Optional[List[int]]:
            if raw is None:
                return None
            if isinstance(raw, torch.Tensor):
                return [int(x) for x in raw.detach().cpu().view(-1).tolist()]
            if isinstance(raw, (list, tuple)):
                return [int(x) for x in raw]
            if isinstance(raw, int):
                return [int(raw)]
            return None

        # Unwrap DDP if needed
        if hasattr(net, 'module'):
            net = net.module

        # Try to get from encoder
        if hasattr(net, 'encoder') and hasattr(net.encoder, 'output_channels'):
            channels = _to_channel_list(net.encoder.output_channels)
            if channels:
                if len(channels) > 1:
                    return channels
                descriptor_channels = (
                    self._student_network_descriptor['arch_kwargs'].get('features_per_stage')
                    if self._student_network_descriptor is not None
                    else None
                )
                descriptor_channels = _to_channel_list(descriptor_channels)
                if descriptor_channels and len(descriptor_channels) > 1:
                    return descriptor_channels
                return channels

        # Fallback to plans manager (prefer teacher plans if requested and available)
        if prefer_teacher and self.distill_config.teacher_plans:
            teacher_plans_manager = PlansManager(self.distill_config.teacher_plans)
            teacher_cfg = teacher_plans_manager.get_configuration(self.configuration_name)
            arch_kwargs_teacher = teacher_cfg.network_arch_init_kwargs
            if 'features_per_stage' in arch_kwargs_teacher:
                channels = _to_channel_list(arch_kwargs_teacher['features_per_stage'])
                if channels:
                    return channels

        if network is None and self._student_network_descriptor is not None:
            descriptor_channels = self._student_network_descriptor['arch_kwargs'].get('features_per_stage')
            channels = _to_channel_list(descriptor_channels)
            if channels:
                return channels

        arch_kwargs = self.configuration_manager.network_arch_init_kwargs
        if 'features_per_stage' in arch_kwargs:
            channels = _to_channel_list(arch_kwargs['features_per_stage'])
            if channels:
                return channels

        raise ValueError("Cannot automatically determine network channels. "
                        "Please specify in strategy_config.")

    def _compute_stagewise_distillation(
        self,
        target: torch.Tensor,
        student_output: Optional[torch.Tensor],
        target_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not hasattr(self.distill_strategy, "get_stage_indices"):
            raise AttributeError("Stage-wise distillation requested but strategy lacks 'get_stage_indices'.")

        stage_indices = self.distill_strategy.get_stage_indices()
        if not stage_indices:
            raise ValueError("Stage-wise distillation strategy has no stages configured.")
        if isinstance(target_list, (list, tuple)) and getattr(self.distill_strategy, "use_mask", False):
            stage_indices = [idx for idx in stage_indices if idx < len(target_list)]
            if not stage_indices:
                raise ValueError(
                    f"No valid distillation stages after aligning with target_list length {len(target_list)}."
                )

        total_loss = torch.zeros((), device=self.device)
        loss_dict: Dict[str, float] = {}

        for idx in stage_indices:
            key = f"stage{idx}"
            student_feat = self.student_features.get(key)
            teacher_feat = self.teacher_features.get(key)

            if student_feat is None or teacher_feat is None:
                raise ValueError(
                    f"Missing features for '{key}'. "
                    f"Student keys: {list(self.student_features.keys())}, "
                    f"Teacher keys: {list(self.teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.distill_strategy.compute_stage_loss(
                stage_idx=idx,
                student_feat=student_feat,
                teacher_feat=teacher_feat,
                target=target,
                target_list=target_list,
                student_output=student_output,
            )
            total_loss = total_loss + stage_loss

            for name, value in stage_loss_dict.items():
                if isinstance(value, torch.Tensor):
                    value_float = float(value.detach().cpu())
                else:
                    value_float = float(value)
                loss_dict[f"stage{idx}_{name}"] = value_float

        return total_loss, loss_dict

    def train_step(self, batch: dict) -> dict:
        """
        Single training step with distillation

        Args:
            batch: Training batch

        Returns:
            Dict of losses and metrics
        """
        iter_start = time.time()
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target_for_distill = target[0]
            target_list = target
        else:
            target = target.to(self.device, non_blocking=True)
            target_for_distill = target
            target_list = None

        # Clear feature storage
        self.student_features.clear()
        self.teacher_features.clear()

        # Forward pass: student
        self.optimizer.zero_grad(set_to_none=True)

        amp_context = autocast(device_type='cuda', enabled=self.distill_config.mixed_precision) \
            if self.device.type == 'cuda' else dummy_context()

        with amp_context:
            # Student prediction
            output = self.network(data)
            student_primary_output = output[0] if isinstance(output, (list, tuple)) else output

            # Compute segmentation loss
            loss_seg = self.loss(output, target)

            # Teacher forward (no grad)
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)
            else:
                teacher_output = None

            # Compute distillation loss
            if self.distill_strategy is not None and self.current_epoch >= 0:
                kd_weight = self.kd_weight_scheduler(self.current_epoch)

                if kd_weight > 0:
                    if getattr(self.distill_strategy, "supports_stagewise", False):
                        loss_distill, distill_loss_dict = self._compute_stagewise_distillation(
                            target=target_for_distill,
                            student_output=student_primary_output,
                            target_list=target_list,
                        )
                    else:
                        loss_distill, distill_loss_dict = self.distill_strategy.forward(
                            student_features=self.student_features,
                            teacher_features=self.teacher_features,
                            target=target_for_distill,
                            student_output=student_primary_output,
                            teacher_output=teacher_output,
                        )
                    loss_total = loss_seg + kd_weight * loss_distill
                else:
                    loss_distill = torch.tensor(0.0, device=self.device)
                    distill_loss_dict = {}
                    loss_total = loss_seg
            else:
                loss_distill = torch.tensor(0.0, device=self.device)
                distill_loss_dict = {}
                loss_total = loss_seg
                kd_weight = 0.0

        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # Fail fast on non-finite gradients to localize the first bad iteration.
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 12, error_if_nonfinite=False
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss_total.backward()
            # Fail fast on non-finite gradients to localize the first bad iteration.
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 12, error_if_nonfinite=False
            )
            self.optimizer.step()

        if self.ema_helper is not None:
            self.ema_helper.update(self.network)

        self.iteration_in_epoch += 1

        loss_total_value = loss_total.item()
        loss_seg_value = loss_seg.item()
        loss_distill_value = loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0.0
        loss_distill_weighted = kd_weight * loss_distill_value
        component_entries = []
        for name, value in distill_loss_dict.items():
            raw = value.item() if isinstance(value, torch.Tensor) else float(value)
            component_entries.append((name, raw, raw * kd_weight))
        grad_norm = float(grad_norm_tensor) if isinstance(grad_norm_tensor, torch.Tensor) else float(grad_norm_tensor)

        if self.iteration_in_epoch % self._iter_log_interval == 0 or self.iteration_in_epoch == 1:
            timestamp_str = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            lr = self.optimizer.param_groups[0]['lr']
            batch_time = time.time() - iter_start
            if self.device.type == 'cuda':
                max_mem = int(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
            else:
                max_mem = 0
            message = (
                f"- nnunetv2 - Epoch(train) "
                f"[{self.current_epoch}][{self.iteration_in_epoch}/{self.num_iterations_per_epoch}]  "
                f"lr: {lr:.3e}  time: {batch_time:.3f}   "
                f"memory: {max_mem}  "
                f"loss: {loss_total_value:.3f}  loss_seg: {loss_seg_value:.3f}  "
                f"loss_kd: {loss_distill_weighted:.3f}  grad_norm: {grad_norm:.3f}"
            )
            if component_entries:
                comp_str = "  ".join(
                    f"{name}: {raw:.3f} (w:{weighted:.3f})" for name, raw, weighted in component_entries
                )
                message += f"  {comp_str}"
            self.print_to_log_file(message)

        # Return losses
        return {
            'loss': loss_total_value,
            'loss_seg': loss_seg_value,
            'loss_distill': loss_distill_value,
            'kd_weight': kd_weight,
            **distill_loss_dict
        }

    def on_train_epoch_end(self, train_loss: float):
        """
        Called at the end of each training epoch

        Args:
            train_loss: Average training loss for the epoch
        """
        super().on_train_epoch_end(train_loss)

        # Log current KD weight
        kd_weight = self.kd_weight_scheduler(self.current_epoch)
        if len(self.kd_weight_history) <= self.current_epoch:
            self.kd_weight_history.append(kd_weight)
        else:
            self.kd_weight_history[self.current_epoch] = kd_weight
        self.print_to_log_file(f"   KD weight this epoch: {kd_weight:.4f}")

    def on_epoch_end(self):
        current_epoch = self.current_epoch
        super().on_epoch_end()
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            snapshot_name = f"checkpoint_epoch_{current_epoch + 1}.pth"
            self.current_epoch = current_epoch
            try:
                self.save_checkpoint(join(self.output_folder, snapshot_name))
            finally:
                self.current_epoch = current_epoch + 1
        # Always save latest checkpoint at the end of each epoch (independent of save_interval).
        self.current_epoch = current_epoch
        try:
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        finally:
            self.current_epoch = current_epoch + 1

    def on_train_end(self):
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        self.current_epoch += 1

        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        if self.ema_helper is not None:
            self.ema_helper.apply_shadow(self.network)
            self._ema_active_during_validation = True
        else:
            self._ema_active_during_validation = False

    def save_checkpoint(self, filename: str):
        """
        Save checkpoint with distillation state

        Args:
            filename: Checkpoint filename
        """
        if self.ema_helper is not None:
            ema_already_applied = bool(self.ema_helper.backup_params)

            if not ema_already_applied:
                self.ema_helper.apply_shadow(self.network)

            try:
                super().save_checkpoint(filename)
            finally:
                if not ema_already_applied:
                    self.ema_helper.restore(self.network)
        else:
            super().save_checkpoint(filename)

        # Save distillation config snapshots (only keep last + final)
        checkpoint_path = Path(self.output_folder) / filename

        if self.ema_helper is not None and self.local_rank == 0 and not self.disable_checkpointing:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint['ema_state'] = self.ema_helper.state_dict()
            torch.save(checkpoint, checkpoint_path)

        ckpt_name = checkpoint_path.name
        if ckpt_name == "checkpoint_latest.pth":
            config_path = checkpoint_path.parent / "distill_config_last.yaml"
        elif ckpt_name == "checkpoint_final.pth":
            config_path = checkpoint_path.parent / "distill_config_final.yaml"
        else:
            config_path = None

        if config_path is not None:
            self.distill_config.to_yaml(str(config_path))
            if not self._distill_config_logged:
                self.print_to_log_file(f"Distillation config snapshot saved to {config_path}")
                self._distill_config_logged = True

    def load_checkpoint(self, filename_or_checkpoint):
        """
        Extend base checkpoint loading to restore EMA state if present.
        """
        temp_ckpt_path = None
        if isinstance(filename_or_checkpoint, str):
            checkpoint_path = filename_or_checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        else:
            # Accept in-memory checkpoint by writing a temp file so base loader can handle it
            import tempfile
            import os
            checkpoint = filename_or_checkpoint
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
                torch.save(checkpoint, tmp.name)
                checkpoint_path = tmp.name
                temp_ckpt_path = tmp.name

        ema_state = checkpoint.get('ema_state')

        # Use base loader (expects a file path)
        super().load_checkpoint(checkpoint_path)

        if ema_state is not None:
            if self.ema_helper is None and self._should_enable_ema():
                self.ema_helper = ExponentialMovingAverage(
                    self.network,
                    decay=self.distill_config.ema_decay,
                    device=torch.device('cpu')
                )
            if self.ema_helper is not None:
                self.ema_helper.load_state_dict(ema_state)

        # Cleanup temp file if we created one
        if temp_ckpt_path is not None:
            try:
                os.remove(temp_ckpt_path)
            except OSError:
                pass

    def on_validation_epoch_end(self, val_outputs: list):
        """
        Called after validation epoch, implements early stopping

        Args:
            val_outputs: List of validation outputs
        """
        super().on_validation_epoch_end(val_outputs)

        if self._ema_active_during_validation and self.ema_helper is not None:
            self.ema_helper.restore(self.network)
            self._ema_active_during_validation = False

        # Check early stopping
        if self.distill_config.early_stopping_patience > 0:
            # Get current validation metric (use mean foreground Dice)
            current_metric = self.logger.my_fantastic_logging['ema_fg_dice'][-1]

            # Initialize or update best
            if self.best_val_metric is None or current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = self.current_epoch
                self.print_to_log_file(f"✨ New best validation Dice: {current_metric:.4f} at epoch {self.current_epoch}")

            # Check if should stop
            epochs_no_improve = self.current_epoch - self.best_epoch
            if epochs_no_improve >= self.distill_config.early_stopping_patience:
                self.print_to_log_file(f"\n{'='*60}")
                self.print_to_log_file(f"⚠️  Early stopping triggered!")
                self.print_to_log_file(f"   No improvement for {epochs_no_improve} epochs")
                self.print_to_log_file(f"   Best metric: {self.best_val_metric:.4f} at epoch {self.best_epoch}")
                self.print_to_log_file(f"{'='*60}\n")
                self.early_stop_triggered = True

        if self.logger.my_fantastic_logging['mean_fg_dice']:
            mean_fg = float(self.logger.my_fantastic_logging['mean_fg_dice'][-1])
            ema_fg = float(self.logger.my_fantastic_logging['ema_fg_dice'][-1])
            self.print_to_log_file(
                f"   Validation pseudo Dice: mean={mean_fg:.4f}, ema={ema_fg:.4f}"
            )

    def run_training(self):
        """
        Run training loop with early stopping support

        Overrides base run_training to add early stopping check
        """
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            # Check early stopping
            if self.early_stop_triggered:
                self.print_to_log_file(f"Training stopped early at epoch {epoch}")
                break

            # Standard epoch loop
            self.on_epoch_start()

            # Training
            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            # Validation
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def __repr__(self) -> str:
        """Pretty print trainer info"""
        cfg = self.distill_config
        lines = [
            "DistillationTrainer",
            f"  Dataset: {self.dataset_json.get('name', 'Unknown')}",
            f"  Fold: {self.fold}",
            f"  Strategy: {cfg.strategy}",
            f"  KD weight: {cfg.kd_weight}",
            f"  Reduction factor: {cfg.reduction_factor}",
            f"  Device: {self.device}",
        ]
        return "\n".join(lines)
