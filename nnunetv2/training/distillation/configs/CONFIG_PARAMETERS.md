# Config Parameters Mapping

## Config → SimpleBoundaryDistillation 参数映射

### YAML Config 文件结构

```yaml
# config_BraTS_boundary.yaml

# ==================== Strategy Config (传递给 SimpleBoundaryDistillation) ====================
strategy: boundary

strategy_config:
  # 必需参数 (Required)
  num_classes: 3                 # 类别数 → SimpleBoundaryDistillation.num_classes
  student_channels: auto         # 自动从网络获取
  teacher_channels: auto         # 自动从网络获取

  # 边界相关 (Boundary)
  boundary_width: 3              # 边界带宽度 → SimpleBoundaryDistillation.boundary_width

  # Loss 开关 (Loss switches)
  use_boundary_loss: true        # 是否使用边界loss → SimpleBoundaryDistillation.use_boundary_loss
  use_attention_loss: true       # 是否使用attention loss → SimpleBoundaryDistillation.use_attention_loss

  # 性能优化 (Performance)
  chunk_d: 16                    # 深度分块 → SimpleBoundaryDistillation.chunk_d

  # 温度参数 (Temperature)
  temp: 0.5                      # attention温度 → SimpleBoundaryDistillation.temp
```

### 参数详细说明

| YAML Path | Python Parameter | Type | Default | Description |
|-----------|-----------------|------|---------|-------------|
| `strategy_config.num_classes` | `num_classes` | int | **Required** | 分割类别数（BraTS=3） |
| `strategy_config.student_channels` | `student_channels` | int | **Auto** | 学生特征通道数（自动获取） |
| `strategy_config.teacher_channels` | `teacher_channels` | int | **Auto** | 教师特征通道数（自动获取） |
| `strategy_config.boundary_width` | `boundary_width` | int | 3 | 边界带宽度（像素） |
| `strategy_config.use_boundary_loss` | `use_boundary_loss` | bool | True | 是否使用边界高频蒸馏 |
| `strategy_config.use_attention_loss` | `use_attention_loss` | bool | False | 是否使用attention一致性 |
| `strategy_config.chunk_d` | `chunk_d` | int | 16 | 深度方向分块大小（显存优化） |
| `strategy_config.temp` | `temp` | float | 0.5 | Attention softmax温度 |

### 代码流程

```
config_BraTS_boundary.yaml
    ↓
DistillationConfig.from_yaml()
    ↓
DistillationTrainer.__init__(distillation_config=config)
    ↓
DistillationTrainer.initialize()
    ↓
    _create_distillation_strategy()
        ↓
        create_strategy('boundary', strategy_config=cfg.strategy_config)
            ↓
            BoundaryDistillation(**strategy_config)
                ↓
                SimpleBoundaryDistillation(
                    student_channels=config['student_channels'],
                    teacher_channels=config['teacher_channels'],
                    boundary_width=config.get('boundary_width', 3),
                    num_classes=config['num_classes'],
                    chunk_d=config.get('chunk_d', 16),
                    use_boundary_loss=config.get('use_boundary_loss', True),
                    use_attention_loss=config.get('use_attention_loss', False),
                    temp=config.get('temp', 0.5)
                )
```

### 特殊说明

#### 1. 自动获取的参数

`student_channels` 和 `teacher_channels` 不需要在config中指定，会在运行时自动从网络架构中获取：

```python
# In distiller.py _create_distillation_strategy()
student_channels = self.network.encoder.stages[-1].output_channels
teacher_channels = self.teacher_model.encoder.stages[-1].output_channels

config['strategy_config']['student_channels'] = student_channels
config['strategy_config']['teacher_channels'] = teacher_channels
```

#### 2. Loss 组合

可以通过开关控制使用哪些loss：

```yaml
# 只使用边界loss（最简单）
use_boundary_loss: true
use_attention_loss: false

# 同时使用两种loss
use_boundary_loss: true
use_attention_loss: true

# 都不使用（相当于no distillation）
use_boundary_loss: false
use_attention_loss: false
```

#### 3. 内存优化

如果显存不足，调整 `chunk_d`:

```yaml
chunk_d: 8   # 减小分块，降低显存使用（但可能稍慢）
chunk_d: 16  # 默认值
chunk_d: 32  # 增大分块，加快速度（需要更多显存）
```

#### 4. Temperature 调节

`temp` 控制attention的平滑程度：

```yaml
temp: 0.1   # 更尖锐的attention（集中在少数位置）
temp: 0.5   # 默认值（平衡）
temp: 1.0   # 更平滑的attention（分布更均匀）
```

### 其他 Config 参数

除了 `strategy_config`，还有其他训练相关参数：

```yaml
# 蒸馏权重
kd_weight: 0.5                   # 蒸馏loss的权重
kd_schedule: warmup              # KD权重调度策略
kd_warmup_epochs: 100            # warmup轮数
use_ema: null                    # null=蒸馏时自动启用, true/false=手动控制
ema_decay: 0.999                 # EMA 衰减系数，越接近1越平滑

# 训练设置
num_epochs: 1000
initial_lr: 0.01
weight_decay: 0.00003

# Early Stopping ✨
early_stopping_patience: 50      # N个epoch没提升则停止训练

# 学生模型
reduction_factor: 2              # 通道缩减倍数
```

- `use_ema` 默认为 `null`，代表在有教师蒸馏时自动启用 EMA；显式设为 `true/false` 可以手动覆盖。
- `ema_decay` 控制 EMA 平滑强度，0.99～0.999 为常见取值，越接近 1 历史权重占比越高。

#### Early Stopping 说明

训练会监控验证集的 **mean foreground Dice**：

- 如果连续 `early_stopping_patience` 个epoch没有提升，自动停止训练
- 设置为 `0` 禁用early stopping
- 会在日志中打印最佳epoch和metric

```
✨ New best validation Dice: 0.8523 at epoch 245

⚠️  Early stopping triggered!
   No improvement for 50 epochs
   Best metric: 0.8523 at epoch 245
```

这些参数在 `distiller.py` 中被应用：

```python
# In _apply_config_overrides()
if cfg.num_epochs is not None:
    self.num_epochs = cfg.num_epochs

# In initialize()
self.kd_weight_scheduler = get_kd_weight_scheduler(
    schedule=cfg.kd_schedule,
    max_weight=cfg.kd_weight,
    warmup_epochs=cfg.kd_warmup_epochs,
    total_epochs=cfg.num_epochs
)
```

## 验证参数是否正确传递

在训练开始时，会打印参数信息：

```
🎯 Setting up distillation strategy: boundary
   ✓ Strategy config: {
       'boundary_width': 3,
       'num_classes': 3,
       'student_channels': 160,
       'teacher_channels': 320,
       'chunk_d': 16,
       'use_boundary_loss': True,
       'use_attention_loss': True,
       'temp': 0.5
   }
```

检查这个输出确保所有参数都正确传递。

## 总结

✅ **所有 SimpleBoundaryDistillation 需要的参数都已经在 config 中**
✅ **参数通过 strategy_config 传递**
✅ **student_channels 和 teacher_channels 自动获取**
✅ **所有参数都有合理的默认值**
✅ **参数流程清晰：YAML → Config → Trainer → Strategy → KD Method**
