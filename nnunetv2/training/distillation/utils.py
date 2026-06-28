"""
Utility functions for distillation training
"""
import inspect
import math
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List

import torch
import torch.nn as nn

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def load_teacher_model(checkpoint_path: str,
                      plans_path: str,
                      configuration: str,
                      num_input_channels: int,
                      num_output_channels: int,
                      device: torch.device,
                      deep_supervision: bool = True) -> Tuple[nn.Module, List[str], List[str]]:
    """
    Load teacher model from checkpoint

    Args:
        checkpoint_path: Path to teacher checkpoint (.pth)
        plans_path: Path to teacher plans.json
        configuration: Configuration name (e.g., '3d_fullres')
        num_input_channels: Number of input channels
        num_output_channels: Number of output channels
        device: Device to load model on
        deep_supervision: Whether teacher uses deep supervision

    Returns:
        Teacher model (in eval mode), missing keys, unexpected keys
    """
    # Load plans
    plans_manager = PlansManager(plans_path)
    teacher_cfg = plans_manager.get_configuration(configuration)

    # Build network architecture
    teacher_model = get_network_from_plans(
        teacher_cfg.network_arch_class_name,
        teacher_cfg.network_arch_init_kwargs,
        teacher_cfg.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=deep_supervision
    )

    # Load checkpoint
    load_kwargs = dict(map_location=device)
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = False  # allow loading legacy checkpoints with pickled objects
    checkpoint = torch.load(checkpoint_path, **load_kwargs)

    # Extract state dict (handle different checkpoint formats)
    if 'network_weights' in checkpoint:
        state_dict = checkpoint['network_weights']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load weights (collect compatibility information)
    load_result = teacher_model.load_state_dict(state_dict, strict=False)
    if hasattr(load_result, 'missing_keys'):
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
    else:
        missing_keys = []
        unexpected_keys = []

    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    return teacher_model, missing_keys, unexpected_keys


def register_feature_hooks(model: nn.Module,
                           feature_storage: Dict[str, torch.Tensor],
                           required_features: Dict[str, str],
                           prefix: str = '') -> list:
    """
    Register forward hooks to extract intermediate features

    Args:
        model: Model to register hooks on
        feature_storage: Dict to store extracted features
        required_features: Dict mapping feature names to layer paths
                          e.g., {'encoder_output': 'encoder.stages[-1]'}
        prefix: Prefix for feature names (e.g., 'student', 'teacher')

    Returns:
        List of hook handles (can be used to remove hooks later)

    Example:
        >>> features = {}
        >>> hooks = register_feature_hooks(
        ...     model=network,
        ...     feature_storage=features,
        ...     required_features={'encoder_output': 'encoder.stages.4'},
        ...     prefix='student'
        ... )
        >>> # After forward pass, features will contain:
        >>> # {'encoder_output': tensor(...)}
    """
    handles = []

    # Unwrap DDP if needed
    if hasattr(model, 'module'):
        model = model.module

    for feature_name, layer_path in required_features.items():
        # Parse layer path (e.g., 'encoder.stages.4' or 'encoder')
        layer = _get_layer_by_path(model, layer_path)

        # Create hook function
        def make_hook(name):
            def hook(module, input, output):
                # Store output in feature_storage
                feature_storage[name] = output
            return hook

        # Register hook
        handle = layer.register_forward_hook(make_hook(feature_name))
        handles.append(handle)

    return handles


def _get_layer_by_path(model: nn.Module, path: str) -> nn.Module:
    """
    Get a layer from model by path string

    Args:
        model: Root model
        path: Path to layer (e.g., 'encoder.stages.4' or 'encoder.stages[-1]')

    Returns:
        The target layer

    Example:
        >>> encoder = _get_layer_by_path(network, 'encoder')
        >>> stage4 = _get_layer_by_path(network, 'encoder.stages.4')
        >>> last_stage = _get_layer_by_path(network, 'encoder.stages[-1]')
    """
    parts = path.split('.')
    current = model

    for part in parts:
        # Handle indexing (e.g., 'stages[4]' or 'stages[-1]')
        if '[' in part and ']' in part:
            attr_name, index = part.split('[')
            index = int(index.rstrip(']'))
            current = getattr(current, attr_name)[index]
        else:
            current = getattr(current, part)

    return current


def get_kd_weight_scheduler(schedule: str,
                            max_weight: float,
                            warmup_epochs: int,
                            total_epochs: int,
                            warmup_start_epoch: int = 0) -> Callable[[int], float]:
    """
    Create KD weight scheduler

    Args:
        schedule: Schedule type ('constant', 'warmup', 'cosine', 'cosine_healing')
        max_weight: Maximum KD weight
        warmup_epochs: Number of warmup epochs (ramp duration)
        total_epochs: Total training epochs
        warmup_start_epoch: Epoch index when ramping should begin

    Returns:
        Scheduler function: epoch -> kd_weight

    Example:
        >>> scheduler = get_kd_weight_scheduler('warmup', max_weight=0.5,
        ...                                     warmup_epochs=50, total_epochs=500)
        >>> kd_weight = scheduler(current_epoch)
    """
    warmup_start = max(0, warmup_start_epoch)
    ramp_epochs = max(0, warmup_epochs)
    ramp_end = warmup_start + ramp_epochs

    if schedule == 'constant':
        def scheduler(epoch: int) -> float:
            return max_weight

    elif schedule == 'warmup':
        def scheduler(epoch: int) -> float:
            if epoch < warmup_start:
                return 0.0
            if ramp_epochs <= 0:
                return max_weight
            progress = min((epoch - warmup_start) / ramp_epochs, 1.0)
            return max_weight * progress

    elif schedule == 'cosine':
        def scheduler(epoch: int) -> float:
            if epoch < warmup_start:
                return 0.0
            if ramp_epochs > 0 and epoch < ramp_end:
                progress = (epoch - warmup_start) / ramp_epochs
                return max_weight * progress

            effective_total = max(total_epochs - ramp_end, 1)
            progress = min(max(epoch - ramp_end, 0) / effective_total, 1.0)
            return max_weight * 0.5 * (1 + math.cos(math.pi * progress))

    elif schedule == 'cosine_healing':
        def scheduler(epoch: int) -> float:
            if epoch < warmup_start:
                return 0.0
            effective_duration = ramp_epochs if ramp_epochs > 0 else max(total_epochs - warmup_start, 1)
            if effective_duration <= 0:
                return max_weight
            progress = min((epoch - warmup_start) / effective_duration, 1.0)
            # Cosine ramp from 0 -> max_weight
            return max_weight * 0.5 * (1 - math.cos(math.pi * progress))

    else:
        raise ValueError(f"Unknown schedule: {schedule}. "
                        f"Available: constant, warmup, cosine, cosine_healing")

    return scheduler


def setup_logging(config, trainer):
    """
    Setup Weights & Biases logging

    Args:
        config: DistillationConfig instance
        trainer: DistillationTrainer instance
    """
    try:
        import wandb
    except ImportError:
        print("⚠️  wandb not installed. Skipping W&B logging.")
        return

    # Initialize wandb
    run_name = config.wandb_name or f"{config.strategy}_fold{trainer.fold}"

    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=config.to_dict(),
        resume='allow'
    )

    # Log model architecture
    wandb.watch(trainer.network, log='all', log_freq=100)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dict with total, trainable, and frozen parameter counts

    Example:
        >>> param_counts = count_parameters(network)
        >>> print(f"Total: {param_counts['total']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen
    }


def print_model_comparison(student: nn.Module, teacher: nn.Module):
    """
    Print comparison of student and teacher models

    Args:
        student: Student model
        teacher: Teacher model
    """
    student_params = count_parameters(student)
    teacher_params = count_parameters(teacher)

    reduction_ratio = student_params['total'] / teacher_params['total']

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Teacher parameters: {teacher_params['total']:,}")
    print(f"Student parameters: {student_params['total']:,}")
    print(f"Reduction ratio: {reduction_ratio:.2%}")
    print("=" * 60 + "\n")


def verify_feature_extraction(model: nn.Module,
                              input_shape: tuple,
                              required_features: Dict[str, str],
                              device: torch.device = torch.device('cuda')) -> bool:
    """
    Verify that feature extraction works correctly

    Args:
        model: Model to test
        input_shape: Input tensor shape (e.g., (1, 1, 128, 128, 128))
        required_features: Features to extract
        device: Device to run on

    Returns:
        True if all features extracted successfully

    Example:
        >>> success = verify_feature_extraction(
        ...     model=network,
        ...     input_shape=(1, 1, 64, 128, 128),
        ...     required_features={'encoder_output': 'encoder'}
        ... )
    """
    features = {}
    hooks = register_feature_hooks(model, features, required_features)

    try:
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)

        # Check features
        success = True
        for name in required_features.keys():
            if name not in features:
                print(f"❌ Feature '{name}' not extracted")
                success = False
            else:
                print(f"✓ Feature '{name}' shape: {features[name].shape}")

        return success

    finally:
        # Remove hooks
        for handle in hooks:
            handle.remove()
