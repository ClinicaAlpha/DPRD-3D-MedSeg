#!/usr/bin/env python3
"""
Example: Boundary Distillation for BraTS2021 Tumor Segmentation

This example demonstrates boundary-aware distillation specifically for BraTS2021,
using ResEncUNet as the teacher model (stronger performance than plain UNet).

BraTS is ideal for boundary distillation because:
- Multiple tumor regions with ambiguous boundaries
- Small target regions (especially enhancing tumor)
- Complex multi-class segmentation task
"""
import sys
import json
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join

# Add nnUNet to path if needed
# sys.path.insert(0, '/path/to/nnUNet')

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.distillation import (
    DistillationTrainer,
    DistillationConfig,
    get_boundary_config
)


def main():
    # ==================== Configuration ====================

    # Dataset
    DATASET_NAME = 'Dataset019_BraTS2021'
    FOLD = 0  # Change to your desired fold (0-4 available)
    CONFIGURATION = '3d_fullres_LR4'  # Use LR4 configuration for BraTS

    # Teacher model: ResEncUNet baseline (stronger than plain UNet)
    # Available folds: 0, 3, 4 (based on checkpoint availability)
    TEACHER_CHECKPOINT = join(
        nnUNet_results,
        DATASET_NAME,
        'nnUNetTrainer_baseline__nnUNetResEncUNetLPlans__3d_fullres_LR4',
        f'fold_{FOLD}',
        'checkpoint_best.pth'
    )

    # Teacher plans file (ResEncUNet architecture)
    TEACHER_PLANS = join(
        nnUNet_results,
        DATASET_NAME,
        'nnUNetTrainer_baseline__nnUNetResEncUNetLPlans__3d_fullres_LR4',
        'plans.json'
    )

    # Check teacher exists
    if not Path(TEACHER_CHECKPOINT).exists():
        print(f"❌ Teacher checkpoint not found: {TEACHER_CHECKPOINT}")
        print(f"   Available folds: 0, 3, 4")
        print(f"   Please train a teacher model first or update the fold number.")
        sys.exit(1)

    # ==================== Create Config ====================

    # Option 1: Use helper config with defaults (Recommended for initial experiments)
    config = get_boundary_config(
        teacher_checkpoint=TEACHER_CHECKPOINT,
        teacher_plans=TEACHER_PLANS,
        num_classes=3,  # BraTS has 3 tumor regions
        kd_weight=0.5
    )

    # Option 2: Customize for BraTS-specific settings
    config.update(
        # Student model configuration
        reduction_factor=2,         # Student has half the channels (~75% smaller)

        # Boundary distillation settings (optimized for tumor boundaries)
        boundary_width=3,           # 3-pixel boundary band
        use_attention_loss=True,    # Enable attention loss for better feature alignment

        # Training configuration
        num_epochs=1000,            # BraTS typically needs more epochs
        kd_schedule='warmup',       # Gradually introduce distillation
        kd_warmup_epochs=100,       # Start distillation after 100 epochs

        # Early stopping
        early_stopping_patience=50, # Stop if no improvement for 50 epochs

        # Logging
        wandb_project='brats_boundary_distillation',
        wandb_name=f'boundary_resenc_fold{FOLD}_r{config.reduction_factor}'
    )

    # Update strategy-specific config for BraTS
    config.strategy_config.update({
        'boundary_width': 3,
        'use_boundary_loss': True,
        'use_attention_loss': True,  # Beneficial for complex boundaries
        'chunk_d': 16,  # Adjust based on GPU memory (reduce if OOM)
        'num_classes': 3  # BraTS regions: whole tumor, tumor core, enhancing tumor
    })

    # ==================== Load Dataset Info ====================

    preprocessed_dir = join(nnUNet_preprocessed, DATASET_NAME)

    with open(join(preprocessed_dir, 'dataset.json'), 'r') as f:
        dataset_json = json.load(f)

    with open(join(preprocessed_dir, 'nnUNetPlans.json'), 'r') as f:
        plans = json.load(f)

    # ==================== Print Configuration Summary ====================

    print("=" * 70)
    print("🧠 BraTS2021 Boundary Distillation")
    print("=" * 70)
    print(f"\n📊 Dataset Information:")
    print(f"   Dataset: {DATASET_NAME}")
    print(f"   Fold: {FOLD}")
    print(f"   Configuration: {CONFIGURATION}")
    print(f"   Tumor regions: {dataset_json['labels']}")
    print(f"   Training samples: {dataset_json.get('numTraining', 'N/A')}")

    print(f"\n👨‍🏫 Teacher Model:")
    print(f"   Architecture: ResEncUNet (stronger than plain UNet)")
    print(f"   Checkpoint: {TEACHER_CHECKPOINT}")
    print(f"   Plans: {TEACHER_PLANS}")

    print(f"\n👨‍🎓 Student Model:")
    print(f"   Architecture: Same as teacher (ResEncUNet)")
    print(f"   Reduction factor: {config.reduction_factor}x")
    print(f"   Expected size: ~{100 - (100 / config.reduction_factor**2):.0f}% smaller")

    print(f"\n🎯 Distillation Strategy:")
    print(f"   Strategy: Boundary-aware distillation")
    print(f"   Boundary width: {config.strategy_config['boundary_width']} pixels")
    print(f"   Use attention: {config.strategy_config['use_attention_loss']}")
    print(f"   KD weight: {config.kd_weight}")
    print(f"   KD schedule: {config.kd_schedule} (warmup: {config.kd_warmup_epochs} epochs)")

    print(f"\n⚙️  Training Configuration:")
    print(f"   Max epochs: {config.num_epochs}")
    print(f"   Early stopping patience: {config.early_stopping_patience}")
    print(f"   Initial LR: {config.initial_lr}")

    if config.wandb_project:
        print(f"\n📈 Logging:")
        print(f"   W&B project: {config.wandb_project}")
        print(f"   W&B run name: {config.wandb_name}")

    print("=" * 70)

    # ==================== Create Trainer ====================

    print("\n⚙️  Creating distillation trainer...")
    trainer = DistillationTrainer(
        plans=plans,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        distillation_config=config
    )

    # ==================== Initialize and Train ====================

    print("\n⚙️  Initializing trainer...")
    trainer.initialize()

    print("\n🏋️  Starting training...")
    print("   This may take several hours depending on your hardware.")
    print("   Training progress will be logged to the console and W&B (if configured).")
    print()

    trainer.run_training()

    print("\n✅ Training complete!")
    print(f"   Results saved to: {trainer.output_folder}")
    print(f"   Best checkpoint: {trainer.output_folder}/checkpoint_best.pth")


if __name__ == '__main__':
    main()
