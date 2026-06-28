#!/usr/bin/env python3
"""
Simple training script for distillation

Usage:
    # From YAML config
    python train.py --config config.yaml --dataset Dataset018_BTCV --fold 0

    # With command-line overrides
    python train.py --config config.yaml --dataset Dataset018_BTCV --fold 0 \\
        --kd_weight 0.8 --num_epochs 300

    # Quick start with kd_method
    python train.py --kd_method boundary \\
        --teacher /path/to/teacher.pth \\
        --dataset Dataset018_BTCV --fold 0
"""
import argparse
import sys
import json
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.distillation.config import (
    DistillationConfig,
    get_baseline_config,
    get_boundary_config,
    get_boundary_v1_config,
    get_boundary_v2_config,
    get_cwd_config,
    get_fitnet_config,
    get_ifvd_config,
    get_cirkd_config,
    get_reco_config,
    get_relation_config,
    get_rkd_config,
    get_skd_config,
)


# Mapping of kd_method names to preset factory functions. Add new entries here
# when you want the CLI to have a one-line preset for a method. Otherwise the
# CLI falls back to a generic DistillationConfig with the given strategy.
KD_PRESET_BUILDERS = {
    "boundary": get_boundary_config,
    "boundary_v1": get_boundary_v1_config,
    "boundary_v2": get_boundary_v2_config,
    "reco": get_reco_config,
    "fitnet": get_fitnet_config,
    "cwd": get_cwd_config,
    "skd": get_skd_config,
    "ifvd": get_ifvd_config,
    "cirkd": get_cirkd_config,
    "relation": get_relation_config,
    "rkd": get_rkd_config,
    "none": get_baseline_config,
    "baseline": get_baseline_config,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train with knowledge distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Dataset018_BTCV)')
    parser.add_argument('--fold', type=int, required=True,
                       help='Training fold (0-4)')

    # Config source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--config', type=str,
                             help='Path to YAML config file')
    config_group.add_argument('--kd_method', type=str,
                             help='KD method name (preset alias or registered strategy name)')

    # Resume
    parser.add_argument('--resume_from', type=str,
                       help='Resume training from a checkpoint path')

    # Preset options
    parser.add_argument('--teacher', type=str,
                       help='Teacher checkpoint path (required for KD methods except baseline/none)')

    # Optional overrides
    parser.add_argument('--configuration', type=str, default='3d_fullres',
                       help='nnUNet configuration (default: 3d_fullres)')

    # Student model configuration
    parser.add_argument('--reduction_factor', type=int,
                       help='Channel reduction factor (1=same, 2=half, 4=quarter)')
    parser.add_argument('--student_channels', type=str,
                       help='Manual channel specification (comma-separated), e.g., "32,64,128,256,320"')
    parser.add_argument('--student_architecture', type=str,
                       help='Student architecture class name (e.g., PlainConvUNet)')
    parser.add_argument('--student_plans', type=str,
                       help='Path to student plans.json')

    # Training hyperparameters
    parser.add_argument('--kd_weight', type=float,
                       help='Override KD weight')
    parser.add_argument('--num_epochs', type=int,
                       help='Override number of epochs')
    parser.add_argument('--kd_schedule', type=str, choices=['constant', 'warmup', 'cosine', 'cosine_healing'],
                       help='KD weight schedule')
    parser.add_argument('--kd_warmup_epochs', type=int,
                       help='Number of warmup epochs')
    parser.add_argument('--kd_warmup_start_epoch', type=int,
                       help='Epoch when KD warmup begins (stays 0 before this)')

    # Strategy configuration
    parser.add_argument('--boundary_width', type=int,
                       help='Boundary width for boundary distillation')
    parser.add_argument('--num_classes', type=int,
                       help='Number of classes (including background)')
    parser.add_argument('--layer_indices', type=str,
                       help='Common encoder stages, e.g., "all" or "0,1,2"')
    parser.add_argument('--stage_weights', type=str,
                       help='Comma-separated per-stage weights')
    parser.add_argument('--skd_patch_size', type=int,
                       help='Patch size for SKD pooling kernel')
    parser.add_argument('--skd_layers', type=str,
                       help='SKD encoder stages (e.g., "all" or "0,1,2")')
    parser.add_argument('--skd_stage_weights', type=str,
                       help='Comma-separated SKD stage weights (must match selected layers)')
    parser.add_argument('--cwd_norm', type=str, choices=['none', 'channel', 'spatial', 'channel_mean'],
                       help='CWD normalization mode')
    parser.add_argument('--cwd_divergence', type=str, choices=['mse', 'kl'],
                       help='CWD divergence type')
    parser.add_argument('--cwd_temperature', type=float,
                       help='CWD temperature')

    # Logging
    parser.add_argument('--wandb_project', type=str,
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str,
                       help='Weights & Biases run name')

    # EMA
    parser.add_argument('--enable_ema', action='store_true',
                       help='Force-enable EMA tracking for the student (overrides auto behaviour)')
    parser.add_argument('--disable_ema', action='store_true',
                       help='Disable EMA tracking even when distillation is active')
    parser.add_argument('--ema_decay', type=float,
                       help='EMA decay factor (default: 0.999)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    # Debug
    parser.add_argument('--verify_features', action='store_true',
                       help='Verify feature extraction before training')

    # Evaluation
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip automatic validation after training')
    parser.add_argument('--eval_with_best', action='store_true',
                       help='Run validation using checkpoint_best.pth if available')
    parser.add_argument('--export_validation_probabilities', action='store_true',
                       help='Save validation softmax probabilities alongside segmentations')

    return parser.parse_args()


def load_dataset_info(dataset_name: str, configuration: str, plans_path_override: str | None = None):
    """
    Load dataset JSON and plans

    Args:
        dataset_name: Dataset name
        configuration: nnUNet configuration

    Returns:
        dataset_json, plans
    """
    preprocessed_dir = join(nnUNet_preprocessed, dataset_name)

    # Load dataset.json
    dataset_json_path = join(preprocessed_dir, 'dataset.json')
    if not Path(dataset_json_path).exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_path}")

    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)

    # Load plans
    plans_path = plans_path_override or join(preprocessed_dir, 'nnUNetPlans.json')
    if not Path(plans_path).exists():
        raise FileNotFoundError(f"Plans not found: {plans_path}")

    with open(plans_path, 'r') as f:
        plans = json.load(f)

    print(f"✓ Loaded dataset: {dataset_json.get('name', dataset_name)}")
    print(f"✓ Configuration: {configuration}")

    return dataset_json, plans


def normalize_kd_method(kd_method: str) -> str:
    method = kd_method.lower()
    if method == "baseline":
        return "none"
    return method


def create_config_from_kd_method(kd_method: str, teacher: str) -> DistillationConfig:
    """Create config from kd_method name, using preset builders or a generic fallback."""
    normalized = normalize_kd_method(kd_method)

    builder = KD_PRESET_BUILDERS.get(kd_method) or KD_PRESET_BUILDERS.get(normalized)

    if builder is not None:
        if normalized == "none":
            return builder()
        if not teacher:
            raise ValueError(f"--teacher is required for kd_method '{kd_method}'")
        return builder(teacher_checkpoint=teacher)

    # Fallback: generic config for arbitrary methods registered at runtime
    if normalized != "none" and not teacher:
        raise ValueError(f"--teacher is required for kd_method '{kd_method}'")

    return DistillationConfig(
        teacher_checkpoint=teacher if normalized != "none" else None,
        strategy=normalized,
        kd_weight=0.0 if normalized == "none" else 0.5,
    )


def apply_overrides(config: DistillationConfig, args):
    """Apply command-line overrides to config"""
    overrides = {}
    strategy_overrides = {}

    def parse_layer_indices(value: str):
        cleaned = value.strip()
        if cleaned.lower() in ('all', '*'):
            return 'all'
        return [int(x.strip()) for x in cleaned.split(',') if x.strip()]

    def parse_stage_weights(value: str):
        return [float(x.strip()) for x in value.split(',') if x.strip()]

    # Student model configuration
    if args.reduction_factor is not None:
        overrides['reduction_factor'] = args.reduction_factor

    if args.student_channels is not None:
        # Parse comma-separated string to list of ints
        channels = [int(c.strip()) for c in args.student_channels.split(',')]
        overrides['student_features_per_stage'] = channels

    if args.student_architecture is not None:
        overrides['student_architecture'] = args.student_architecture

    if args.student_plans is not None:
        overrides['student_plans'] = args.student_plans

    # Training hyperparameters
    if args.kd_weight is not None:
        overrides['kd_weight'] = args.kd_weight

    if args.num_epochs is not None:
        overrides['num_epochs'] = args.num_epochs

    if args.kd_schedule is not None:
        overrides['kd_schedule'] = args.kd_schedule

    if args.kd_warmup_epochs is not None:
        overrides['kd_warmup_epochs'] = args.kd_warmup_epochs
    if args.kd_warmup_start_epoch is not None:
        overrides['kd_warmup_start_epoch'] = args.kd_warmup_start_epoch

    # Strategy-specific parameters
    if args.boundary_width is not None:
        strategy_overrides['boundary_width'] = args.boundary_width

    if args.num_classes is not None:
        strategy_overrides['num_classes'] = args.num_classes

    if args.layer_indices is not None:
        strategy_overrides['layer_indices'] = parse_layer_indices(args.layer_indices)
    elif args.skd_layers is not None:
        strategy_overrides['layer_indices'] = parse_layer_indices(args.skd_layers)

    if args.skd_patch_size is not None:
        strategy_overrides['patch_size'] = args.skd_patch_size

    if args.stage_weights is not None:
        strategy_overrides['stage_weights'] = parse_stage_weights(args.stage_weights)
    elif args.skd_stage_weights is not None:
        strategy_overrides['stage_weights'] = parse_stage_weights(args.skd_stage_weights)

    if args.cwd_norm is not None:
        strategy_overrides['norm_type'] = args.cwd_norm

    if args.cwd_divergence is not None:
        strategy_overrides['divergence'] = args.cwd_divergence

    if args.cwd_temperature is not None:
        strategy_overrides['temperature'] = args.cwd_temperature

    # Logging
    if args.wandb_project is not None:
        overrides['wandb_project'] = args.wandb_project

    if args.wandb_name is not None:
        overrides['wandb_name'] = args.wandb_name

    # EMA
    if args.enable_ema and args.disable_ema:
        raise ValueError("Cannot specify both --enable_ema and --disable_ema")

    if args.enable_ema:
        overrides['use_ema'] = True
    elif args.disable_ema:
        overrides['use_ema'] = False

    if args.ema_decay is not None:
        overrides['ema_decay'] = args.ema_decay

    # Apply overrides
    if overrides:
        print(f"\n📝 Applying config overrides: {overrides}")
        config.update(**overrides)

    if strategy_overrides:
        print(f"📝 Applying strategy overrides: {strategy_overrides}")
        config.strategy_config.update(strategy_overrides)

    return config


def main():
    args = parse_args()

    print("=" * 70)
    print("🚀 Medical Image Segmentation Distillation Toolkit")
    print("=" * 70)

    # Create or load config
    if args.config:
        print(f"\n📄 Loading config from: {args.config}")
        config = DistillationConfig.from_yaml(args.config)
    else:
        print(f"\n🎯 Using kd_method: {args.kd_method}")
        config = create_config_from_kd_method(args.kd_method, args.teacher)

    # Apply overrides
    config = apply_overrides(config, args)
    if args.eval_with_best:
        config.eval_with_best = True

    plans_override = None
    if config.student_plans:
        plans_override = config.student_plans
        print(f"📌 Using student plans: {plans_override}")
    elif config.teacher_plans:
        plans_override = config.teacher_plans
        if config.student_architecture:
            print(f"📌 Using teacher plans for student architecture: {plans_override}")
        else:
            print(f"📌 Using teacher plans for student build (default): {plans_override}")

    # Load dataset info
    dataset_json, plans = load_dataset_info(args.dataset, args.configuration, plans_override)

    # Print config
    print("\n" + "=" * 70)
    print("Configuration:")
    print("=" * 70)
    print(config)
    print("=" * 70)

    # Create trainer
    print(f"\n🔧 Creating trainer...")
    from nnunetv2.training.distillation.distiller import DistillationTrainer

    # Convert device string to torch.device
    import torch
    device = torch.device(args.device)

    trainer = DistillationTrainer(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        distillation_config=config,
        device=device
    )

    # Initialize
    print(f"\n⚙️  Initializing...")
    trainer.initialize()

    # Resume from checkpoint if provided
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_from path not found: {ckpt_path}")
        print(f"\n📦 Resuming from checkpoint: {ckpt_path}")
        trainer.load_checkpoint(str(ckpt_path))

    # Optional: Verify feature extraction
    if args.verify_features:
        print(f"\n🔍 Verifying feature extraction...")
        from nnunetv2.training.distillation.utils import verify_feature_extraction

        # Get a sample input shape
        patch_size = trainer.configuration_manager.patch_size
        num_input_channels = trainer.num_input_channels

        success = verify_feature_extraction(
            model=trainer.network,
            input_shape=(1, num_input_channels, *patch_size),
            required_features=trainer.distill_strategy.get_required_features(),
            device=trainer.device
        )

        if not success:
            print("❌ Feature extraction verification failed!")
            sys.exit(1)

        print("✓ Feature extraction verified successfully!")

    # Train
    print(f"\n🏋️  Starting training...")
    print("=" * 70)
    trainer.run_training()

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)

    # Automatic validation (mirrors nnUNet run_training behaviour)
    if args.skip_eval:
        print("\n⚠️  Skipping automatic validation (--skip_eval specified)")
        return

    eval_with_best = config.eval_with_best

    checkpoint_description = 'final in-memory weights'
    if eval_with_best:
        best_checkpoint = Path(trainer.output_folder) / 'checkpoint_best.pth'
        if best_checkpoint.exists():
            print(f"\n📦 Loading best checkpoint for validation: {best_checkpoint}")
            trainer.load_checkpoint(str(best_checkpoint))
            checkpoint_description = best_checkpoint.name
        else:
            print(f"\n⚠️  Requested best-checkpoint evaluation but file not found at {best_checkpoint}. "
                  f"Falling back to final checkpoint.")
            eval_with_best = False
            checkpoint_description = 'final in-memory weights'
    else:
        print("\n📦 Using final in-memory weights for validation (matches nnU-Net run_training.py default)")

    print("\n🧪 Running validation...")
    from nnunetv2.training.distillation.validation.sync_validation import perform_actual_validation_sync
    perform_actual_validation_sync(
        trainer,
        save_probabilities=args.export_validation_probabilities,
        use_mirroring=False,
        tile_step_size=0.5,
    )
    print("✓ Validation complete!")
    print(f"   Results saved in: {trainer.output_folder}/validation")
    print(f"   Checkpoint evaluated: {checkpoint_description}")


if __name__ == '__main__':
    main()
