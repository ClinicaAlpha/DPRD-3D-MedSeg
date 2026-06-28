# BraTS2021 Boundary Distillation

## Quick Start

### Config File: [config_BraTS_boundary.yaml](config_BraTS_boundary.yaml)

核心配置：
- **Dataset**: BraTS2021（肿瘤分割）
- **Teacher**: ResEncUNet (fold 0-4 all available)
- **Strategy**: Boundary distillation
- **Compression**: 2x (reduction_factor=2)

### Training Command

```bash
cd /bdm-das/ADSP_v1/H100/ADSP_v1/code_qlan/nnUNet

# Use config file (recommended)
python -m nnunetv2.training.distillation.train \
    --config nnunetv2/training/distillation/configs/config_BraTS_boundary.yaml \
    --dataset Dataset019_BraTS2021 \
    --fold 0

# Or use Python script directly
python nnunetv2/training/distillation/configs/example_BraTS.py
```

## Modify Configuration

Edit [config_BraTS_boundary.yaml](config_BraTS_boundary.yaml):

```yaml
# Change compression level
reduction_factor: 4  # Options: 1, 2, 4, 8

# Change fold (all 5 folds available: 0-4)
teacher_checkpoint: .../fold_3/checkpoint_best.pth

# Adjust boundary width
strategy_config:
  boundary_width: 5  # Increase from 3 to 5 pixels

# Change KD weight
kd_weight: 0.8  # Increase from 0.5
```

## Available Teacher Models

All 5 folds trained with ResEncUNet:
```
/bdm-das/ADSP_v1/H100/ADSP_v1/code_qlan/nnUNet_results/Dataset019_BraTS2021/
  nnUNetTrainer_baseline__nnUNetResEncUNetLPlans__3d_fullres_LR4/
    ├── fold_0/checkpoint_best.pth  ✅
    ├── fold_1/checkpoint_best.pth  ✅
    ├── fold_2/checkpoint_best.pth  ✅
    ├── fold_3/checkpoint_best.pth  ✅
    └── fold_4/checkpoint_best.pth  ✅
```

## Files

- **config_BraTS_boundary.yaml** - YAML配置文件（推荐使用）
- **example_BraTS.py** - Python训练脚本（可选）
- This README

That's it! 简洁明了。
