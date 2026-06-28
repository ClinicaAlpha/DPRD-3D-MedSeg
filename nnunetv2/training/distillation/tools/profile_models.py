#!/usr/bin/env python3
"""
Profile nnUNet architectures for params/FLOPs/peak memory.

Examples:
  # Profile teacher + student from a distillation config
  python profile_models.py \
    --config ../configs/Reco/config_reco_btcv_hetero.yaml \
    --dataset Dataset018_BTCV --configuration 3d_fullres

  # Profile explicit architectures
  python profile_models.py \
    --plans /path/to/nnUNetPlans.json \
    --dataset Dataset018_BTCV --configuration 3d_fullres \
    --architectures dynamic_network_architectures.architectures.unet.ResidualEncoderUNet \
    --architectures nnunetv2.nets.mobile_unet_v3.MobileUNetV3
"""
from __future__ import annotations

import argparse
import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.distillation.config import DistillationConfig


def _load_dataset_json(dataset_name: Optional[str], dataset_json_path: Optional[str]) -> dict:
    if dataset_json_path is None and dataset_name is None:
        raise ValueError("Provide --dataset or --dataset-json")
    if dataset_json_path is None:
        dataset_json_path = Path(nnUNet_preprocessed) / dataset_name / "dataset.json"
    with open(dataset_json_path, "r") as f:
        return json.load(f)


def _format_count(value: Optional[float], unit: str = "") -> str:
    if value is None:
        return "N/A"
    suffixes = ["", "K", "M", "G", "T"]
    idx = 0
    val = float(value)
    while val >= 1000.0 and idx < len(suffixes) - 1:
        val /= 1000.0
        idx += 1
    return f"{val:.3f}{suffixes[idx]}{unit}"


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    val = float(value)
    while val >= 1024.0 and idx < len(suffixes) - 1:
        val /= 1024.0
        idx += 1
    return f"{val:.2f}{suffixes[idx]}"


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        num = int(p.numel())
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


def _compute_flops_thop(model: torch.nn.Module, sample: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    try:
        from thop import profile
    except Exception:
        return None, None

    try:
        macs, params = profile(model, inputs=(sample,), verbose=False)
        macs = float(macs) if macs is not None else None
        params = float(params) if params is not None else None
        flops = None if macs is None else 2.0 * macs
        return flops, macs
    except Exception:
        return None, None


def _compute_flops_fx(model: torch.nn.Module, sample: torch.Tensor) -> Optional[float]:
    try:
        from torch_flops import TorchFLOPsByFX
    except Exception:
        return None

    try:
        flops_counter = TorchFLOPsByFX(model)
        flops_counter.propagate(sample)
        total_flops = None
        if hasattr(flops_counter, "get_total_flops"):
            total_flops = flops_counter.get_total_flops()
        elif hasattr(flops_counter, "print_total_flops"):
            signature = inspect.signature(flops_counter.print_total_flops)
            if "show" in signature.parameters:
                total_flops = flops_counter.print_total_flops(show=False)
            else:
                total_flops = flops_counter.print_total_flops()
        if total_flops is None:
            return None
        return float(total_flops)
    except Exception:
        return None


def _parse_int_list(value: str) -> List[int]:
    cleaned = value.strip().replace("x", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return [int(p) for p in parts]


def _profile_model(model: torch.nn.Module,
                   input_shape: Tuple[int, ...],
                   device: torch.device) -> Dict[str, object]:
    model = model.to(device)
    model.eval()

    sample = torch.zeros(input_shape, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        flops_thop, thop_macs = _compute_flops_thop(model, sample)
        flops_fx = _compute_flops_fx(model, sample)
        if flops_thop is None and flops_fx is None:
            model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device)
    else:
        peak_mem = None

    total_params, trainable_params = _count_params(model)
    return {
        "params": total_params,
        "trainable_params": trainable_params,
        "flops_thop": flops_thop,
        "flops_fx": flops_fx,
        "thop_macs": thop_macs,
        "peak_mem_bytes": peak_mem,
    }


def _build_model(arch_class_name: str,
                 arch_kwargs: dict,
                 arch_kwargs_req_import: List[str],
                 num_input_channels: int,
                 num_output_channels: int,
                 deep_supervision: bool,
                 student_features: Optional[List[int]] = None,
                 reduction_factor: Optional[int] = None) -> torch.nn.Module:
    modified_kwargs = deepcopy(arch_kwargs)
    if student_features is not None:
        modified_kwargs["features_per_stage"] = student_features
    elif reduction_factor is not None and reduction_factor > 1:
        features = list(modified_kwargs.get("features_per_stage", []))
        if features:
            modified_kwargs["features_per_stage"] = [max(int(f // reduction_factor), 8) for f in features]

    return get_network_from_plans(
        arch_class_name,
        modified_kwargs,
        arch_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=deep_supervision,
    )


def _load_config(config_path: Optional[str]) -> Optional[DistillationConfig]:
    if config_path is None:
        return None
    return DistillationConfig.from_yaml(config_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile nnUNet architectures (params/FLOPs/peak memory)")
    parser.add_argument("--config", type=str, help="Distillation config YAML (optional)")
    parser.add_argument("--plans", type=str, help="Plans JSON path (optional if --config is set)")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., Dataset018_BTCV)")
    parser.add_argument("--dataset-json", type=str, dest="dataset_json", help="Path to dataset.json")
    parser.add_argument("--configuration", type=str, default="3d_fullres", help="Configuration name")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--patch-size", type=str,
                        help="Override patch size, e.g. '80,256,256' or '80x256x256'")
    parser.add_argument("--no-deep-supervision", dest="deep_supervision", action="store_false",
                        help="Disable deep supervision outputs")
    parser.add_argument("--architectures", type=str, action="append", default=[],
                        help="Architecture class path (repeatable)")
    parser.add_argument("--include-student", action="store_true", help="Include student from config")
    parser.add_argument("--include-teacher", action="store_true", help="Include teacher from config")
    parser.add_argument("--output", type=str, help="Write results to JSON")
    parser.set_defaults(deep_supervision=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    plans_path = args.plans or (cfg.teacher_plans if cfg and cfg.teacher_plans else None)
    if plans_path is None:
        raise ValueError("Provide --plans or a config with teacher_plans")

    plans_manager = PlansManager(plans_path)
    configuration = plans_manager.get_configuration(args.configuration)
    arch_class_name = configuration.network_arch_class_name
    arch_kwargs = configuration.network_arch_init_kwargs
    arch_kwargs_req_import = configuration.network_arch_init_kwargs_req_import
    patch_size = configuration.patch_size

    dataset_json = _load_dataset_json(args.dataset, args.dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_output_channels = label_manager.num_segmentation_heads

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if args.patch_size:
        patch_size = _parse_int_list(args.patch_size)
    input_shape = (args.batch_size, num_input_channels, *patch_size)

    architectures = list(args.architectures)

    models: List[Tuple[str, torch.nn.Module]] = []

    if cfg and not args.include_teacher and not args.include_student and not architectures:
        args.include_teacher = True
        args.include_student = True

    if cfg and args.include_teacher:
        models.append(("teacher", _build_model(
            arch_class_name,
            arch_kwargs,
            arch_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            args.deep_supervision,
        )))

    if cfg and args.include_student:
        student_arch = cfg.student_architecture or arch_class_name
        models.append((f"student ({student_arch})", _build_model(
            student_arch,
            arch_kwargs,
            arch_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            args.deep_supervision,
            student_features=cfg.student_features_per_stage,
            reduction_factor=cfg.reduction_factor,
        )))

    for arch in architectures:
        models.append((arch, _build_model(
            arch,
            arch_kwargs,
            arch_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            args.deep_supervision,
        )))

    if cfg and args.include_teacher:
        models = [(f"teacher ({arch_class_name})", model) if name == "teacher" else (name, model)
                  for name, model in models]

    results = []
    for name, model in models:
        metrics = _profile_model(model, input_shape, device)
        results.append({"name": name, **metrics})
        if device.type == "cuda":
            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    print(f"Input shape: {input_shape} | device={device.type}")
    name_width = max(len("model"), max(len(item["name"]) for item in results))
    header = (
        f"{'model':<{name_width}}  "
        f"{'params':>10}  "
        f"{'flops_thop':>12}  "
        f"{'flops_fx':>12}  "
        f"{'peak_mem':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['name']:<{name_width}}  "
            f"{_format_count(item['params']):>10}  "
            f"{_format_count(item['flops_thop'], unit='F'):>12}  "
            f"{_format_count(item['flops_fx'], unit='F'):>12}  "
            f"{_format_bytes(item['peak_mem_bytes']):>10}"
        )
    print("Note: flops_thop is 2x MACs from thop when available.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "input_shape": input_shape,
                    "device": device.type,
                    "results": results,
                },
                f,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
