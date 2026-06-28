#!/usr/bin/env bash
set -euo pipefail

# Edit this list for future batches.
EXPS=(
  /bdm-das/qlan/Results/nnUNet_results/Dataset018_BTCV/DistillationTrainer__reco__MobileUNetV3__kd-warmup__nnUNetResEncUNetLPlans__3d_fullres
  /bdm-das/qlan/Results/nnUNet_results/Dataset018_BTCV/DistillationTrainer__reco__ShuffleNetV2UNet__kd-warmup__nnUNetResEncUNetLPlans__3d_fullres
)

DATASET=Dataset018_BTCV
FOLDS=(3)
VALIDATION_DIR_NAME=validation
METRICS_FILENAME=nsd_hd95.csv
OUT_FILENAME=aggregate_report.csv

SCRIPT=/bdm-das/qlan/nnUNet-KD/nnunetv2/training/distillation/tools/aggregate_eval_report.py

for e in "${EXPS[@]}"; do
  for f in "${FOLDS[@]}"; do
    OUT_CSV="$e/fold_${f}/${OUT_FILENAME}"
    CMD=(
      python "$SCRIPT"
      --dataset "$DATASET"
      --exp "$e"
      --folds "$f"
      --validation-dir-name "$VALIDATION_DIR_NAME"
      --metrics-filename "$METRICS_FILENAME"
      --run-eval
      --output "$OUT_CSV"
    )
    echo "[run] ${CMD[*]}"
    "${CMD[@]}"
  done
done
