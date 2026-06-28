#!/usr/bin/env python3
"""
Profile external models (params/FLOPs/peak memory) with dummy inputs.

Examples:
  # MONAI UNETR (if monai is installed)
  python profile_external_models.py \
    --model monai.networks.nets.UNETR \
    --model-kwargs '{"in_channels":1,"out_channels":3,"img_size":[96,96,96],"feature_size":16}' \
    --input-shape 1,1,96,96,96 \
    --device cuda

  # Multiple models with per-model kwargs
  python profile_external_models.py \
    --model monai.networks.nets.UNETR \
    --model monai.networks.nets.SwinUNETR \
    --model-kwargs '{"in_channels":1,"out_channels":3,"img_size":[96,96,96],"feature_size":16}' \
    --model-kwargs '{"in_channels":1,"out_channels":3,"img_size":[96,96,96]}' \
    --input-shape 1,1,96,96,96
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


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


def _load_json_arg(value: str) -> dict:
    if not value:
        return {}
    if value.startswith("@"):
        path = Path(value[1:])
        with path.open("r") as f:
            return json.load(f)
    return json.loads(value)


def _import_class(path: str):
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _profile_model(model: torch.nn.Module,
                   input_shape: Tuple[int, ...],
                   device: torch.device,
                   amp: bool,
                   warmup: int) -> Dict[str, object]:
    model = model.to(device)
    model.eval()

    sample = torch.zeros(input_shape, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    autocast_kwargs = {"enabled": amp} if device.type == "cuda" else {"enabled": False}

    with torch.no_grad():
        with autocast_ctx(**autocast_kwargs):
            for _ in range(max(0, warmup)):
                model(sample)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile external models (params/FLOPs/peak memory)")
    parser.add_argument("--model", action="append", required=True,
                        help="Fully-qualified class path (repeatable)")
    parser.add_argument("--model-kwargs", action="append", default=[],
                        help="JSON dict or @path.json, repeat to match --model")
    parser.add_argument("--input-shape", required=True,
                        help="Input shape, e.g. '1,1,96,96,96'")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--amp", action="store_true", help="Use autocast (cuda only)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forwards before measuring")
    parser.add_argument("--output", type=str, help="Write results to JSON")
    args = parser.parse_args()

    input_shape = tuple(_parse_int_list(args.input_shape))
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    kwargs_list = [_load_json_arg(v) for v in args.model_kwargs]
    if not kwargs_list:
        kwargs_list = [{} for _ in args.model]
    elif len(kwargs_list) == 1 and len(args.model) > 1:
        kwargs_list = kwargs_list * len(args.model)
    elif len(kwargs_list) != len(args.model):
        raise ValueError("Number of --model-kwargs must be 1 or match number of --model entries")

    results = []
    for model_path, model_kwargs in zip(args.model, kwargs_list):
        try:
            cls = _import_class(model_path)
        except Exception as e:
            print(f"❌ Failed to import {model_path}: {e}")
            continue

        try:
            model = cls(**model_kwargs)
        except Exception as e:
            print(f"❌ Failed to init {model_path} with kwargs={model_kwargs}: {e}")
            continue

        metrics = _profile_model(model, input_shape, device, args.amp, args.warmup)
        results.append({"name": model_path, "kwargs": model_kwargs, **metrics})
        if device.type == "cuda":
            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    if not results:
        print("No models were profiled.")
        return 1

    print(f"Input shape: {input_shape} | device={device.type} | amp={args.amp}")
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
                    "amp": args.amp,
                    "results": results,
                },
                f,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
