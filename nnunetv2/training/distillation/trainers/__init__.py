# Re-export the primary trainer so downstream users can do:
# from nnunetv2.training.distillation.trainers import DistillationTrainer
from ..distiller import DistillationTrainer

__all__ = ["DistillationTrainer"]

