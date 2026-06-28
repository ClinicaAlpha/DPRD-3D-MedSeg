#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="$(dirname "$0")/batch_eval.yaml"

PYTHONPATH="/bdm-das/qlan/nnUNet-KD" \
python /bdm-das/qlan/nnUNet-KD/nnunetv2/training/distillation/evaluation/batch_eval.py \
  --config "${CONFIG_PATH}"
