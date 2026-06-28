# nnUNet-KD Distillation Framework

An nnU-Net v2-based knowledge distillation subsystem that provides a unified
training entry point, flexible strategy configuration, and plug-and-play
implementations of common KD methods. After the recent refactor, all stage-wise
methods follow a single paradigm: the trainer handles scheduling, while each
method focuses on one stage. This makes logging and extensions cleaner.

If you use the distillation framework or ReCo-KD, please cite:

```
@misc{lan2026recokdregioncontextawareknowledge,
      title={ReCo-KD: Region- and Context-Aware Knowledge Distillation for Efficient 3D Medical Image Segmentation},
      author={Qizhen Lan and Yu-Chun Hsu and Nida Saddaf Khan and Xiaoqian Jiang},
      year={2026},
      eprint={2601.08301},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.08301},
}
```

---

## Directory Overview

```
nnunetv2/training/distillation/
├── __init__.py                  # DistillationTrainer / preset configs / registry
├── config.py                    # DistillationConfig dataclass + presets
├── distiller.py                 # DistillationTrainer: main loop & stage scheduling
├── ema.py                       # EMA utilities for student weights
├── evaluation/                  # evaluation helpers
├── train.py                     # CLI entry point (YAML or CLI args)
├── utils.py                     # teacher loading, hooks, KD schedules, W&B wrapper
├── methods/                     # KD strategies
│   ├── base.py                  # DistillationMethod abstract base class
│   ├── boundary.py              # classic boundary distillation
│   ├── boundary_v1.py           # Sobel + EMA + uncertainty weighting
│   ├── feature.py               # simple feature regression (MSE)
│   ├── fitnet.py                # FitNet feature regression
│   ├── cwd.py                   # channel-wise distillation
│   ├── ifvd.py                  # inter-class feature variance distillation
│   ├── reco.py                  # relation-based context distillation
│   ├── skd.py                   # structural KD
│   └── __init__.py              # registry & build_method factory
├── trainers/                    # nnUNetTrainer re-exports (reserved)
├── tools/                       # helper scripts
└── validation/                  # post-KD validation scripts
```

---

## Training Pipeline Highlights

1. Automatic student construction
- Supports `reduction_factor` or explicit `student_features_per_stage`.
- Writes `student_plans.json` for reproducibility.
- Automatically infers teacher/student channels per encoder stage and passes
  them into KD methods.

2. Heterogeneous KD
- Use `student_architecture` to specify a different student network class.
- Use `student_plans` (or `teacher_plans`) to align stage definitions.
- Suitable for MobileUNetV3 / ShuffleNet / MedNeXt student models.

3. Multi-scale target support
- If the dataloader returns deep supervision targets as a list, the trainer
  passes them as `target_list`.
- ReCo can directly use `target_list[stage_idx]` as the stage-level GT mask.
- If a list is not provided, GT is resized to each stage with nearest neighbor.

4. Unified stage-wise scheduling
- `DistillationTrainer` checks `supports_stagewise=True`.
- If supported, the trainer iterates `strategy.get_stage_indices()` and calls
  `compute_stage_loss` per stage.
- Each method returns atomic metrics (for example `{"boundary": ...}`) which
  are logged with a `stage{k}_` prefix.
- The legacy `forward` interface remains for quick tests or scripting.

5. Logging and monitoring
- Logs include `loss_seg`, `loss_kd`, and `stage{i}_component` (both raw and
  weighted).
- Weights & Biases is supported via `wandb_project` / `wandb_name`.
- EMA for student weights is controlled by `use_ema`.

6. Teacher management
- Automatically loads teacher checkpoint and plans, supports freezing and DDP.
- Encoder hooks are registered with `encoder.stages[i]` naming.

---

## DistillationConfig Quick Reference

| Field | Purpose |
| --- | --- |
| `teacher_checkpoint / teacher_plans` | teacher weights and plans paths |
| `strategy` | KD method name (boundary, boundary_v1, reco, fitnet, cwd, ifvd, skd, feature, none, etc.) |
| `strategy_config` | method-specific parameters (see below) |
| `kd_weight / kd_schedule / kd_warmup_epochs` | total KD weight and schedule (constant / warmup / cosine / cosine_healing) |
| `kd_warmup_start_epoch` | delay before KD starts ramping up (default: 0) |
| `reduction_factor` | student channel reduction factor (>1 applies) |
| `student_features_per_stage` | explicit student channels per stage (highest priority) |
| `student_architecture` | student network class path (heterogeneous KD) |
| `student_plans` | student plans path (align stage definitions) |
| `mednext_*` | optional MedNeXt kwargs passed into student initialization (YAML config only) |
| `num_epochs / initial_lr / weight_decay` | training config |
| `early_stopping_patience` | early stopping patience, 0 disables |
| `eval_with_best` | evaluate with `checkpoint_best` (includes EMA) |
| `mixed_precision / use_ema / ema_decay` | AMP and EMA parameters |
| `log_interval / wandb_project / wandb_name` | logging control |

`config.py` includes ready-to-use presets (for example `get_boundary_config()`)
that you can override.

### MedNeXt YAML keys

- `mednext_model_id`: `S`, `B`, `M`, `L`
- `mednext_exp_r`: int or list
- `mednext_block_counts`: list (length 9)
- `mednext_kernel_size`
- `mednext_enc_kernel_size`
- `mednext_dec_kernel_size`
- `mednext_checkpoint_style`
- `mednext_norm_type`: `group` or `layer`
- `mednext_grn`: bool
- `mednext_do_res`: bool
- `mednext_do_res_up_down`: bool

Note: current distillation CLI does not expose `--mednext_*` flags. Set these in YAML.

---

## Strategy Config Cheatsheet

| Strategy | Key Parameters (`strategy_config`) | Stage-wise | Log Keys |
| --- | --- | --- | --- |
| `boundary` | `boundary_width`, `layer_indices`, `use_attention_loss`, etc. | yes | `stage{k}_boundary` / `stage{k}_attention` |
| `boundary_v1` | `boundary_width`, `warmup_iters`, `uncertainty_scale`, `use_teacher_stats`, `class_weights`, etc. | yes | `stage{k}_boundary`, `stage{k}_attention`, `stage{k}_warmup_scale` |
| `boundary_v2` | `layer_indices`, `sobel_scale`, `softmax_temperature`, `eps` | yes | `stage{k}_boundary`, `stage{k}_mask_mean` |
| `reco` | `num_classes`, `chunk_d`, `temp`, `tau`, `coef_*`, `stage_weights`, `stage_alpha`, `drop_last_stage` | yes | `stage{k}_fg`, `stage{k}_bg`, ... |
| `fitnet` | `layer_indices`, `stage_weights`, `is_3d` | yes | `stage{k}_fitnet` |
| `cwd` | `layer_indices`, `stage_weights`, `norm_type`, `divergence`, `temperature` | yes | `stage{k}_cwd` |
| `ifvd` | `layer_indices`, `num_classes`, `stage_weights` | yes | `stage{k}_ifvd` |
| `skd` | `layer_indices`, `stage_weights`, `patch_size` | yes | `stage{k}_skd` |
| `feature` | `layer` (single stage) | no | `mse_loss` |
| `none` | - | returns 0 | - |

`layer_indices` can be an int, list, or string `"all"` / `"*"` and supports
negative indexing. If not set, the default is the last stage only.

ReCo notes:
- If `target` or `target_list` is single-channel, set `num_classes`.
- If the dataloader returns multi-scale `target_list`, ReCo will directly use
  `target_list[stage_idx]` and skip redundant resizing.

---

## Usage Examples

### 1. YAML config (recommended)

```yaml
# configs/config_BraTS_boundary.yaml
teacher_checkpoint: /path/to/teacher/fold_3/checkpoint_best.pth
teacher_plans: /path/to/teacher/plans.json
reduction_factor: 4

strategy: boundary_v1
strategy_config:
  layer_indices: [1, 2, 3, 4]
  boundary_width: 3
  warmup_iters: 12000
  teacher_stat_ema_decay: 0.97
  use_teacher_stats: true
  uncertainty_scale: 1.5

kd_weight: 0.45
kd_schedule: warmup
kd_warmup_epochs: 150

num_epochs: 1000
initial_lr: 0.01
weight_decay: 0.00003

log_interval: 10
use_ema: true
ema_decay: 0.999
```

Run:

```bash
python -m nnunetv2.training.distillation.train \
    --config nnunetv2/training/distillation/configs/config_BraTS_boundary.yaml \
    --dataset Dataset019_BraTS2021 \
    --fold 3 \
    --configuration 3d_fullres
```

Example log:

```
... loss_seg: 0.213  loss_kd: 0.041  grad_norm: 0.325  stage0_boundary: 0.112 (w:0.009) ...
```

### 2. Quick CLI run

```bash
python -m nnunetv2.training.distillation.train \
    --dataset Dataset018_BTCV --fold 0 --configuration 3d_fullres \
    --teacher /ckpt/teacher.pth \
    --kd_method fitnet \
    --layer_indices all \
    --kd_weight 1.0 --kd_schedule warmup --kd_warmup_epochs 100 \
    --reduction_factor 2
```

`train.py` builds a `DistillationConfig` internally and injects
`layer_indices` / `stage_weights` into the selected method.

### 3. Heterogeneous student

```yaml
# examples/config_reco_hetero.yaml
teacher_checkpoint: /path/to/teacher/checkpoint_best.pth
teacher_plans: /path/to/teacher/plans.json

# Student model (different from teacher)
student_architecture: nnunetv2.nets.mobile_unet_v3.MobileUNetV3
student_plans: /path/to/teacher/plans.json   # or provide explicit student plans
reduction_factor: 1

strategy: reco
strategy_config:
  num_classes: 13
  stage_alpha: 0.4
```

Run:

```bash
python -m nnunetv2.training.distillation.train \
    --config examples/config_reco_hetero.yaml \
    --dataset Dataset018_BTCV \
    --fold 0 \
    --configuration 3d_fullres
```

Additional examples:
- `nnunetv2/training/distillation/configs/Reco/config_reco_btcv_hetero.yaml`
- `nnunetv2/training/distillation/configs/Reco/config_reco_btcv_shufflenet.yaml`

### 4. Train a MedNeXt teacher (no KD, YAML-only)

Use distillation runner with `strategy: none` to train a standalone MedNeXt model
that can later be used as a teacher checkpoint.

```yaml
# examples/config_mednext_teacher.yaml
strategy: none
teacher_checkpoint: null
teacher_plans: null

student_architecture: nnunetv2.nets.mednext_v1.MedNeXtV1
mednext_model_id: B
mednext_kernel_size: 3
mednext_grn: false

num_epochs: 1000
initial_lr: 0.01
weight_decay: 0.00003
```

```bash
python -m nnunetv2.training.distillation.train \
    --config examples/config_mednext_teacher.yaml \
    --dataset Dataset018_BTCV \
    --fold 0 \
    --configuration 3d_fullres
```

### 5. Train MedNeXt student with KD

```yaml
# examples/config_mednext_student_kd.yaml
teacher_checkpoint: /path/to/teacher/checkpoint_best.pth
teacher_plans: /path/to/teacher/plans.json

strategy: boundary_v2
strategy_config:
  layer_indices: [0, 1, 2, 3]
  sobel_scale: 1.0
  softmax_temperature: 0.5

kd_weight: 0.5
kd_schedule: warmup
kd_warmup_epochs: 120

student_architecture: nnunetv2.nets.mednext_v1.MedNeXtV1
mednext_model_id: S
mednext_kernel_size: 3
```

```bash
python -m nnunetv2.training.distillation.train \
    --config examples/config_mednext_student_kd.yaml \
    --dataset Dataset018_BTCV \
    --fold 0 \
    --configuration 3d_fullres
```

### 6. Kernel size guidance for MedNeXt

- Start with `mednext_kernel_size: 3` for most 3D tasks.
- Use odd values (`3`, `5`, `7`) only.
- If memory is tight, keep kernel size at `3` and tune depth/width first.
- For anisotropic data, prefer `3` unless experiments show stable gains with larger kernels.

---

## Logging and Diagnostics

- Channel inference: initialization prints `Detected student/teacher channels`.
  If the length is 1, check the student reducer or set `student_channels` in
  `strategy_config`.
- KD visualization: stage-wise metrics are logged as `stage{k}_component` and
  are ready for TensorBoard / W&B.
- Common errors:
  - `layer index ... out of bounds`: `layer_indices` exceeds available stages.
  - `Missing features for stage{k}`: teacher or student structure mismatch.
  - `ModuleNotFoundError: hiddenlayer`: only affects network graph visuals;
    install `hiddenlayer` or ignore.

---

## Extension Guide

### Implement a new stage-wise method

1. Inherit from `DistillationMethod` and implement:

```python
class MyKD(DistillationMethod, nn.Module):
    supports_stagewise = True

    def __init__(self, **config):
        ...
        self.stage_indices = [...]

    def get_stage_indices(self):
        return list(self.stage_indices)

    def compute_stage_loss(self, stage_idx, student_feat, teacher_feat, target, **kwargs):
        loss, metrics = ...
        return loss, {"my_metric": loss}

    def forward(...):
        # optional, kept for compatibility
        ...
```

2. Register it in `methods/__init__.py`:

```python
from .my_kd import MyKD
METHOD_REGISTRY["my_kd"] = MyKD
```

3. Optional: add a preset in `config.py` (for example `get_my_kd_config`).

The trainer will automatically detect `supports_stagewise` and call
`compute_stage_loss` per stage.

### Single-stage methods

If your method uses a single feature (like `feature.py`), you can omit
`supports_stagewise` and just return `(loss, metrics)` in `forward`.

---

## Validation / Inference

`validation/run_validation.py` can load the best checkpoint (including EMA)
for evaluation. Example:

```bash
python -m nnunetv2.training.distillation.validation.run_validation \
    --results_folder /path/to/nnUNet_results \
    --config BraTS_boundary.yaml \
    --folds 0 1 2 3 4
```

Set `eval_with_best: true` in the config or pass `--eval_with_best` to enable
this behavior at the end of training.

---

## KD Weight Schedules

- `constant`: fixed at `kd_weight` for all epochs.
- `warmup`: stays at 0 until `kd_warmup_start_epoch`, then linearly ramps up
  to `kd_weight` over `kd_warmup_epochs`.
- `cosine`: linear warmup to `kd_weight`, then cosine decay over the remaining
  epochs.
- `cosine_healing`: half-cosine ramp from 0 to `kd_weight` after
  `kd_warmup_start_epoch`, then stays at the peak.

Tip: set `kd_warmup_start_epoch` to delay KD until the student stabilizes. If
`kd_warmup_epochs=0`, the weight jumps to the target at the start epoch.
