# DPRD-3D-MedSeg

Official implementation of **Displacement Preserving Relational Distillation for Robust Medical Segmentation**.

DPRD is a knowledge distillation framework for efficient and robust 3D medical image segmentation. Instead of directly matching dense voxel-wise activations, DPRD transfers teacher knowledge by aligning normalized pairwise displacement relations among ROI-pooled case-level embeddings. The method is implemented on top of nnU-Net v2 and supports heterogeneous teacher-student distillation, such as MedNeXt to MobileUNetV3.

## Highlights

- **Displacement-preserving relational alignment** for scale-invariant local relational supervision.
- **ROI-aware feature masking** to reduce background-dominated supervision in 3D volumes.
- **Multi-stage encoder distillation** for hierarchical anatomical consistency.
- **Efficient deployment**: teacher and projection layers are used only during training; inference uses the compact student model.

## Performance

The paper evaluates DPRD on ISLES 2022 and AMOS 2022 against representative KD baselines including Logits KD, FitNet, RKD, and CIRKD.

### ISLES 2022

| Method | Params (M) | FLOPs (G) | Dice (%) ↑ | NSD (2mm) ↑ | HD95 (mm) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher-MedNeXt | 12.10 | 974.58 | 76.88±2.84 | 85.64±2.55 | 14.12±3.11 |
| Baseline-PlainConvUNet | 1.95 | 497.79 | 72.92±3.09 | 83.34±2.87 | 17.21±3.57 |
| Logits KD | - | - | 72.02±3.22 | 82.61±3.12 | 17.49±3.74 |
| FitNet | - | - | 73.93±3.05 | 84.29±2.82 | 16.79±3.57 |
| RKD | 1.95 | 497.79 | 72.85±3.15 | 83.51±2.94 | 18.42±4.03 |
| CIRKD | - | - | 74.04±3.03 | 85.06±2.92 | 14.43±3.53 |
| **DPRD** | - | - | **75.16±3.03** | **85.92±2.90** | **13.08±3.37** |

### AMOS 2022

| Method | Params (M) | FLOPs (G) | Dice (%) ↑ | NSD (2mm) ↑ | HD95 (mm) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher-MedNeXt | 12.11 | 1360 | 85.37±2.03 | 84.93±2.14 | 14.52±2.12 |
| Baseline-MobileUNetV3 | 0.63 | 39.19 | 80.16±2.58 | 77.87±3.09 | 18.74±1.69 |
| Logits KD | - | - | 83.14±2.42 | 81.80±2.74 | 15.60±1.60 |
| FitNet | - | - | 81.80±2.74 | 80.01±2.68 | 15.02±1.89 |
| RKD | 0.63 | 39.19 | 83.49±2.29 | 82.77±2.42 | 17.15±2.43 |
| CIRKD | - | - | 82.09±2.54 | 80.13±2.67 | 14.75±1.46 |
| **DPRD** | - | - | **85.46±2.13** | **85.29±2.14** | **11.72±2.11** |

On AMOS, the MobileUNetV3 student uses roughly 5% of the teacher parameters and 3% of the teacher FLOPs while slightly exceeding the MedNeXt teacher in Dice.

## Method Overview

DPRD consists of two main components:

1. **Displacement-Preserving Relational Alignment (DPRA)**

   DPRD computes pairwise displacement vectors between ROI-pooled case-level embeddings in a mini-batch. Teacher and student displacements are normalized by their batch-wise relational scale, then aligned with a Smooth L1 trajectory loss. A distance consistency term further preserves relative displacement lengths.

2. **ROI-Aware Feature Masking (RAFM)**

   Ground-truth foreground labels are merged into an anatomical ROI mask during training. The mask is resized to feature-map resolution and used to pool task-relevant feature embeddings, reducing background dilution in 3D volumes.

Across encoder stages, DPRD applies stage-specific weights:

```text
[0.05, 0.10, 0.10, 0.15, 0.25, 0.35]
```

Core implementation:

```text
nnunetv2/training/distillation/methods/DPRD.py
```

Main AMOS configuration:

```text
cfgs/MobileUNetV3_kd_DPRD_amos.yaml
```

## Installation

Create a Python 3.10+ environment and install the package in editable mode:

```bash
git clone git@github.com:ClinicaAlpha/DPRD-3D-MedSeg.git
cd DPRD-3D-MedSeg
pip install -e .
```

Set the nnU-Net data paths:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## Data And Checkpoints

This repository does not include medical images, preprocessed data, or model checkpoints.

For the provided AMOS configuration, prepare:

```text
data/nnUNet_preprocessed/Dataset218_AMOS2022_postChallenge_task1/
data/nnUNet_results/Dataset218_AMOS2022_postChallenge_task1/
```

The AMOS DPRD config expects a trained MedNeXt teacher checkpoint and plan files:

```yaml
teacher_checkpoint: data/nnUNet_results/.../fold_3/checkpoint_best.pth
teacher_plans: data/nnUNet_results/.../student_plans.json
student_plans: data/nnUNet_results/.../plans.json
```

Update these paths if your data are stored elsewhere.

## Training

Train the MobileUNetV3 student with DPRD on AMOS:

```bash
python -m nnunetv2.training.distillation.train \
  --config cfgs/MobileUNetV3_kd_DPRD_amos.yaml \
  --dataset Dataset218_AMOS2022_postChallenge_task1 \
  --fold 3 \
  --configuration 3d_fullres
```

Baseline configs:

```text
cfgs/config_mednext_teacher.yaml
cfgs/config_MobileUNetV3_student_reduce4.yaml
```

## Repository Layout

```text
cfgs/                                      # Main experiment configs
nnunetv2/training/distillation/            # Distillation trainer and utilities
nnunetv2/training/distillation/methods/     # DPRD and KD methods
nnunetv2/nets/                             # Network definitions
scripts/                                   # Slurm and evaluation helpers
```

## Citation

Citation information will be added after the camera-ready version is finalized.

## Acknowledgements

This codebase is built on top of nnU-Net v2. We thank the nnU-Net authors and the open-source medical image segmentation community. The paper acknowledges computational resources from the Delta system at NCSA through ACCESS allocations and support from NIH and NSF awards.

## License

This project is released under the Apache-2.0 license.
