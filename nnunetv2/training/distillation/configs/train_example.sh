#!/bin/bash
# 完整的训练示例 - 直接可用！
# 使用方法: bash train_example.sh

# ============================================================================
#   配置区域 - 修改这里的路径
# ============================================================================

# 数据集信息
DATASET="Dataset018_BTCV"
FOLD=0
CONFIG="3d_fullres"

# 教师模型路径
TEACHER_CHECKPOINT="/path/to/teacher/fold_0/checkpoint_best.pth"

# 学生模型配置
REDUCTION_FACTOR=2        # 学生模型通道数缩减因子 (1=不缩减, 2=减半, 4=1/4)

# 训练超参数
KD_WEIGHT=0.5            # 蒸馏损失权重
NUM_EPOCHS=500           # 训练轮数
KD_SCHEDULE="warmup"     # KD权重调度: constant/warmup/cosine
KD_WARMUP_EPOCHS=50      # 预热轮数

# 策略配置
STRATEGY="boundary"      # 蒸馏策略: boundary/reco/fitnet/cwd/skd/ifvd/none
BOUNDARY_WIDTH=3         # 边界宽度 (仅boundary策略使用)
NUM_CLASSES=3            # 类别数 (包括背景)

# 日志配置
WANDB_PROJECT="medical_seg_distillation"
WANDB_NAME="boundary_2x_fold${FOLD}"

# GPU配置
DEVICE="cuda:0"

# ============================================================================
#   方式 1: 使用命令行参数 (推荐用于快速实验)
# ============================================================================

echo "=========================================="
echo "方式 1: 使用命令行参数训练"
echo "=========================================="

python -m nnunetv2.training.distillation.train \
    --kd_method ${STRATEGY} \
    --teacher ${TEACHER_CHECKPOINT} \
    --dataset ${DATASET} \
    --fold ${FOLD} \
    --configuration ${CONFIG} \
    --reduction_factor ${REDUCTION_FACTOR} \
    --kd_weight ${KD_WEIGHT} \
    --num_epochs ${NUM_EPOCHS} \
    --kd_schedule ${KD_SCHEDULE} \
    --kd_warmup_epochs ${KD_WARMUP_EPOCHS} \
    --boundary_width ${BOUNDARY_WIDTH} \
    --num_classes ${NUM_CLASSES} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_name ${WANDB_NAME} \
    --device ${DEVICE}


# ============================================================================
#   方式 2: 使用 YAML 配置文件 (推荐用于正式实验)
# ============================================================================

# 创建临时配置文件
cat > /tmp/distill_config.yaml <<EOF
# 自动生成的配置文件
teacher_checkpoint: ${TEACHER_CHECKPOINT}
reduction_factor: ${REDUCTION_FACTOR}

strategy: ${STRATEGY}
strategy_config:
  boundary_width: ${BOUNDARY_WIDTH}
  num_classes: ${NUM_CLASSES}
  use_boundary_loss: true
  use_attention_loss: false

kd_weight: ${KD_WEIGHT}
kd_schedule: ${KD_SCHEDULE}
kd_warmup_epochs: ${KD_WARMUP_EPOCHS}

num_epochs: ${NUM_EPOCHS}
mixed_precision: true

wandb_project: ${WANDB_PROJECT}
wandb_name: ${WANDB_NAME}
EOF

echo ""
echo "=========================================="
echo "方式 2: 使用 YAML 配置文件训练"
echo "=========================================="
echo "配置文件已保存到: /tmp/distill_config.yaml"
echo ""

# 使用 YAML 配置训练 (注释掉，避免重复训练)
# python -m nnunetv2.training.distillation.train \
#     --config /tmp/distill_config.yaml \
#     --dataset ${DATASET} \
#     --fold ${FOLD} \
#     --configuration ${CONFIG} \
#     --device ${DEVICE}


# ============================================================================
#   其他示例配置
# ============================================================================

echo ""
echo "=========================================="
echo "其他常用配置示例"
echo "=========================================="

# 示例 1: 更激进的压缩 (4x)
echo ""
echo "# 示例 1: 4x 通道压缩"
echo "python -m nnunetv2.training.distillation.train \\"
echo "    --kd_method boundary \\"
echo "    --teacher ${TEACHER_CHECKPOINT} \\"
echo "    --dataset ${DATASET} \\"
echo "    --fold ${FOLD} \\"
echo "    --reduction_factor 4 \\"
echo "    --kd_weight 0.8"

# 示例 2: 手动指定通道数
echo ""
echo "# 示例 2: 手动指定通道数 (非对称压缩)"
echo "python -m nnunetv2.training.distillation.train \\"
echo "    --kd_method boundary \\"
echo "    --teacher ${TEACHER_CHECKPOINT} \\"
echo "    --dataset ${DATASET} \\"
echo "    --fold ${FOLD} \\"
echo "    --student_channels \"32,64,64,128,160\""

# 示例 3: 使用 ReCo 策略
echo ""
echo "# 示例 3: 使用 ReCo 蒸馏策略"
echo "python -m nnunetv2.training.distillation.train \\"
echo "    --kd_method reco \\"
echo "    --teacher ${TEACHER_CHECKPOINT} \\"
echo "    --dataset ${DATASET} \\"
echo "    --fold ${FOLD} \\"
echo "    --kd_weight 0.01"

# 示例 4: Baseline (不用蒸馏)
echo ""
echo "# 示例 4: Baseline (不用蒸馏，用于对比)"
echo "python -m nnunetv2.training.distillation.train \\"
echo "    --kd_method baseline \\"
echo "    --dataset ${DATASET} \\"
echo "    --fold ${FOLD}"

echo ""
echo "=========================================="
echo "训练完成！结果保存在 nnUNet_results/${DATASET}/"
echo "=========================================="
