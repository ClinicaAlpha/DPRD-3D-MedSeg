#!/usr/bin/env python3
"""
Validate a previously trained distillation run without re-training.

This helper mirrors nnU-Net's `--val` behaviour and defaults to the
BraTS boundary distillation results directory provided in the request.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.training.distillation.config import DistillationConfig
from nnunetv2.training.distillation.distiller import DistillationTrainer
from nnunetv2.training.distillation.train import load_dataset_info
from nnunetv2.training.distillation.validation.sync_validation import perform_actual_validation_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation on an existing distillation checkpoint (no re-training)."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(
            "/bdm-das/ADSP_v1/H100/ADSP_v1/code_qlan/nnUNet_results/"
            "Dataset019_BraTS2021/DistillationTrainer__boundary__R4__kd-warmup__nnUNetPlans__3d_fullres"
        ),
        help="Root directory of the trained distillation experiment.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset019_BraTS2021",
        help="nnU-Net dataset identifier (e.g. Dataset019_BraTS2021).",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        help="nnU-Net configuration (default: 3d_fullres).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        help="Specific folds to validate. If omitted, all fold_* directories are used.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoint_best.pth",
        help="Checkpoint filename to evaluate (default: checkpoint_best.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run validation on (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--export_probabilities",
        action="store_true",
        help="Also save softmax probabilities during validation.",
    )
    parser.add_argument(
        "--validation_folder",
        type=str,
        default="validation",
        help="Subfolder under each fold directory for validation outputs (default: validation).",
    )
    return parser.parse_args()


def discover_folds(results_dir: Path) -> List[int]:
    folds: List[int] = []
    for child in results_dir.iterdir():
        if child.is_dir() and child.name.startswith("fold_"):
            try:
                folds.append(int(child.name.split("_")[1]))
            except ValueError:
                continue
    return sorted(folds)


def resolve_checkpoint(fold_dir: Path, checkpoint_name: str) -> Path:
    checkpoint_path = fold_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_config(checkpoint_path: Path) -> Path:
    legacy = checkpoint_path.with_name(f"{checkpoint_path.stem}_distill_config.yaml")
    if legacy.is_file():
        return legacy
    if checkpoint_path.name == "checkpoint_final.pth":
        cfg_path = checkpoint_path.with_name("distill_config_final.yaml")
    else:
        cfg_path = checkpoint_path.with_name("distill_config_last.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Distillation config not found next to checkpoint: {cfg_path}")
    return cfg_path


def stabilize_transient_validation_paths(
    plans: dict,
    distill_config: DistillationConfig,
    validation_folder: str,
) -> tuple[dict, DistillationConfig]:
    if validation_folder == "validation":
        return plans, distill_config

    plans = copy.deepcopy(plans)
    distill_config = copy.deepcopy(distill_config)

    plans["plans_name"] = "nnUNetPlans_validation_tmp"
    distill_config.experiment_tag = "validation_tmp"
    distill_config.experiment_tag_mode = "override"

    return plans, distill_config


def run_validation_for_fold(
    dataset: str,
    configuration: str,
    device: torch.device,
    fold: int,
    fold_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    export_probabilities: bool,
    validation_folder: str,
) -> None:
    print("=" * 80)
    print(f"🔍 Validating fold {fold} using {checkpoint_path.name}")
    print("=" * 80)

    distill_config = DistillationConfig.from_yaml(str(config_path))
    plans_override = None
    student_plans_file = fold_dir.parent / "student_plans.json"
    if student_plans_file.is_file():
        plans_override = str(student_plans_file)
        print(f"📌 Using student plans from results: {plans_override}")
        # Important: student_plans.json already encodes the reduced channels.
        # If we keep reduction_factor>1 here, the model would be reduced twice
        # (for example 80 -> 20), causing checkpoint shape mismatches.
        if distill_config.reduction_factor and distill_config.reduction_factor > 1:
            print(
                f"⚠️  student_plans already reduced; overriding reduction_factor "
                f"{distill_config.reduction_factor} -> 1 for validation."
            )
            distill_config.reduction_factor = 1
    elif distill_config.student_plans:
        plans_override = distill_config.student_plans
        print(f"📌 Using student plans from config: {plans_override}")
    elif distill_config.teacher_plans:
        plans_override = distill_config.teacher_plans
        print(f"📌 Using teacher plans from config: {plans_override}")

    dataset_json, plans = load_dataset_info(dataset, configuration, plans_override)
    plans, distill_config = stabilize_transient_validation_paths(plans, distill_config, validation_folder)

    trainer = DistillationTrainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        distillation_config=distill_config,
        device=device,
    )
    trainer.initialize()
    trainer.load_checkpoint(str(checkpoint_path))

    # Ensure validation writes into the existing experiment directory rather than
    # a recomputed folder name derived from plans_name.
    trainer.output_folder_base = str(fold_dir.parent)
    trainer.output_folder = str(fold_dir)
    maybe_mkdir_p(trainer.output_folder)
    validation_output_dir = fold_dir / validation_folder
    print(f"📁 Writing validation outputs to: {validation_output_dir}")

    # Speed-focused validation: disable mirroring/TTA.
    # nnUNetTrainer.perform_actual_validation always builds the predictor with
    # use_mirroring=True, but passing None axes effectively disables TTA.
    trainer.inference_allowed_mirroring_axes = None
    print("⚡ Disabling mirroring/TTA for faster validation.")

    perform_actual_validation_sync(
        trainer,
        save_probabilities=export_probabilities,
        use_mirroring=False,
        tile_step_size=0.5,
        validation_folder=validation_folder,
    )

    print(f"✅ Validation finished for fold {fold}")
    print(f"   Results written to: {validation_output_dir}\n")


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir.resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if args.folds:
        folds_to_run: Iterable[int] = args.folds
    else:
        folds_to_run = discover_folds(results_dir)
        if not folds_to_run:
            raise RuntimeError(f"No fold_* directories found under {results_dir}")

    device = torch.device(args.device)

    for fold in folds_to_run:
        fold_dir = results_dir / f"fold_{fold}"
        if not fold_dir.is_dir():
            print(f"⚠️  Fold directory missing: {fold_dir}, skipping.")
            continue

        try:
            checkpoint_path = resolve_checkpoint(fold_dir, args.checkpoint_name)
            config_path = resolve_config(checkpoint_path)
        except FileNotFoundError as exc:
            print(f"⚠️  {exc}")
            continue

        run_validation_for_fold(
            dataset=args.dataset,
            configuration=args.configuration,
            device=device,
            fold=fold,
            fold_dir=fold_dir,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            export_probabilities=args.export_probabilities,
            validation_folder=args.validation_folder,
        )


if __name__ == "__main__":
    main()
