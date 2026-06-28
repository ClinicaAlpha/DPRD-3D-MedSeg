#!/usr/bin/env python3
"""
Evaluate teacher/student displacement alignment on nnU-Net preprocessed cases.

This script is designed for rebuttal analysis rather than training. It loads a
teacher checkpoint and one student checkpoint, extracts encoder-stage features,
samples foreground voxels, and writes per-case plus summary CSV metrics.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-plans", type=Path, required=True)
    parser.add_argument("--student-plans", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-input-channels", type=int)
    parser.add_argument("--num-output-channels", type=int)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        help="Spatial crop size D H W. Defaults to the nnU-Net plan patch_size.",
    )
    parser.add_argument(
        "--full-volume",
        action="store_true",
        help="Forward the full preprocessed case. This can require very large memory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_b2nd(path: Path) -> np.ndarray:
    try:
        import blosc2
    except ImportError as e:
        raise ImportError(
            f"Cannot read {path} because blosc2 is not installed in this environment"
        ) from e

    array = blosc2.open(str(path), mode="r")
    return np.asarray(array[:])


def infer_num_input_channels(dataset_json: dict[str, Any]) -> int:
    channel_names = dataset_json.get("channel_names") or dataset_json.get("modality")
    if isinstance(channel_names, dict):
        return len(channel_names)
    if isinstance(channel_names, list):
        return len(channel_names)
    raise ValueError("Cannot infer input channels from dataset.json; pass --num-input-channels")


def infer_num_output_channels(dataset_json: dict[str, Any]) -> int:
    labels = dataset_json.get("labels")
    if isinstance(labels, dict):
        values: list[int] = []
        for value in labels.values():
            if isinstance(value, int):
                values.append(value)
            elif isinstance(value, list):
                values.extend(int(v) for v in value)
        if values:
            return max(values) + 1
    raise ValueError("Cannot infer output channels from dataset.json; pass --num-output-channels")


def build_model(
    plans_path: Path,
    configuration: str,
    checkpoint_path: Path,
    num_input_channels: int,
    num_output_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    plans_manager = PlansManager(str(plans_path))
    cfg = plans_manager.get_configuration(configuration)
    model = get_network_from_plans(
        cfg.network_arch_class_name,
        cfg.network_arch_init_kwargs,
        cfg.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=False,
    )

    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(checkpoint_path, **load_kwargs)
    state_dict = (
        checkpoint.get("network_weights")
        or checkpoint.get("state_dict")
        or checkpoint.get("model_state_dict")
        or checkpoint
    )
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def infer_patch_size(plans_path: Path, configuration: str) -> tuple[int, int, int]:
    plans_manager = PlansManager(str(plans_path))
    cfg = plans_manager.get_configuration(configuration)
    patch_size = getattr(cfg, "patch_size", None)
    if patch_size is None:
        raise ValueError("Cannot infer patch_size from plans; pass --patch-size D H W")
    patch_size = tuple(int(i) for i in patch_size)
    if len(patch_size) != 3:
        raise ValueError(f"Expected 3D patch_size, got {patch_size}")
    return patch_size


def register_encoder_hooks(model: torch.nn.Module, stages: list[int]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    storage: dict[int, torch.Tensor] = {}
    handles = []
    encoder = getattr(model, "encoder", None)
    stage_modules = getattr(encoder, "stages", None)
    if stage_modules is None:
        raise ValueError("Model does not expose encoder.stages; add a model-specific hook mapping")

    for stage in stages:
        if stage >= len(stage_modules):
            continue

        def make_hook(stage_idx: int):
            def hook(_module, _inputs, output):
                storage[stage_idx] = output
            return hook

        handles.append(stage_modules[stage].register_forward_hook(make_hook(stage)))
    return storage, handles


def load_case(preprocessed_dir: Path, case_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    npz_path = preprocessed_dir / "nnUNetPlans_3d_fullres" / f"{case_id}.npz"
    b2nd_path = preprocessed_dir / "nnUNetPlans_3d_fullres" / f"{case_id}.b2nd"
    b2nd_seg_path = preprocessed_dir / "nnUNetPlans_3d_fullres" / f"{case_id}_seg.b2nd"
    pkl_path = preprocessed_dir / "nnUNetPlans_3d_fullres" / f"{case_id}.pkl"

    if npz_path.is_file():
        with np.load(npz_path) as npz:
            data = npz["data"]

        seg = None
        if pkl_path.is_file():
            with pkl_path.open("rb") as f:
                props = pickle.load(f)
            seg = props.get("seg")

        if seg is None:
            # Older nnU-Net layouts can store image channels followed by segmentation.
            image = data[:-1]
            seg = data[-1:]
        else:
            image = data
            if seg.ndim == 3:
                seg = seg[None]
        return torch.from_numpy(image).float(), torch.from_numpy(seg).float()

    if b2nd_path.is_file() and b2nd_seg_path.is_file():
        image = load_b2nd(b2nd_path)
        seg = load_b2nd(b2nd_seg_path)
        if image.ndim == 3:
            image = image[None]
        if seg.ndim == 3:
            seg = seg[None]
        return torch.from_numpy(image).float(), torch.from_numpy(seg).float()

    raise FileNotFoundError(
        "Missing preprocessed case. Expected either "
        f"{npz_path} or both {b2nd_path} and {b2nd_seg_path}"
    )


def pad_to_patch(
    image: torch.Tensor,
    seg: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_shape = image.shape[1:]
    pads = []
    for dim, target in zip(reversed(spatial_shape), reversed(patch_size)):
        missing = max(0, target - dim)
        before = missing // 2
        after = missing - before
        pads.extend([before, after])
    if any(pads):
        image = F.pad(image, pads, mode="constant", value=0)
        seg = F.pad(seg, pads, mode="constant", value=0)
    return image, seg


def crop_foreground_patch(
    image: torch.Tensor,
    seg: torch.Tensor,
    patch_size: tuple[int, int, int],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, seg = pad_to_patch(image, seg, patch_size)
    spatial_shape = image.shape[1:]
    foreground = (seg[0] > 0).nonzero(as_tuple=False)

    starts = []
    if foreground.numel() > 0:
        center_idx = torch.randint(foreground.shape[0], (1,), generator=generator).item()
        center = foreground[center_idx]
        for axis, patch_dim in enumerate(patch_size):
            max_start = spatial_shape[axis] - patch_dim
            start = int(center[axis]) - patch_dim // 2
            starts.append(max(0, min(start, max_start)))
    else:
        for axis, patch_dim in enumerate(patch_size):
            max_start = spatial_shape[axis] - patch_dim
            if max_start > 0:
                starts.append(int(torch.randint(max_start + 1, (1,), generator=generator).item()))
            else:
                starts.append(0)

    z, y, x = starts
    dz, dy, dx = patch_size
    image = image[:, z : z + dz, y : y + dy, x : x + dx]
    seg = seg[:, z : z + dz, y : y + dy, x : x + dx]
    return image.contiguous(), seg.contiguous()


def resize_feature_pair(teacher: torch.Tensor, student: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher.shape[2:] == student.shape[2:]:
        return teacher, student
    target = tuple(min(a, b) for a, b in zip(teacher.shape[2:], student.shape[2:]))
    teacher = F.interpolate(teacher, size=target, mode="trilinear", align_corners=False)
    student = F.interpolate(student, size=target, mode="trilinear", align_corners=False)
    return teacher, student


def sample_embeddings(
    teacher_feat: torch.Tensor,
    student_feat: torch.Tensor,
    seg: torch.Tensor,
    num_points: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_feat, student_feat = resize_feature_pair(teacher_feat, student_feat)
    mask = F.interpolate(seg.to(teacher_feat.device), size=teacher_feat.shape[2:], mode="nearest") > 0
    coords = mask[0, 0].nonzero(as_tuple=False)
    if coords.numel() == 0:
        return teacher_feat.new_empty((0, teacher_feat.shape[1])), student_feat.new_empty((0, student_feat.shape[1]))
    if coords.shape[0] > num_points:
        idx = torch.randperm(coords.shape[0], generator=generator, device=coords.device)[:num_points]
        coords = coords[idx]
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    emb_t = teacher_feat[0, :, z, y, x].transpose(0, 1).contiguous()
    emb_s = student_feat[0, :, z, y, x].transpose(0, 1).contiguous()
    return emb_t, emb_s


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return math.nan
    return float(torch.dot(x, y) / denom)


def rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def compute_metrics(emb_t: torch.Tensor, emb_s: torch.Tensor) -> dict[str, float]:
    n = emb_t.shape[0]
    if n < 2:
        return {}
    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=emb_t.device)
    delta_t = emb_t[idx_i] - emb_t[idx_j]
    delta_s = emb_s[idx_i] - emb_s[idx_j]
    dist_t = torch.linalg.vector_norm(delta_t, ord=2, dim=1)
    dist_s = torch.linalg.vector_norm(delta_s, ord=2, dim=1)

    metrics = {
        "num_points": float(n),
        "num_pairs": float(delta_t.shape[0]),
        "distance_correlation_pearson": pearson(dist_t, dist_s),
        "distance_correlation_spearman": pearson(rankdata(dist_t), rankdata(dist_s)),
    }
    if emb_t.shape[1] == emb_s.shape[1]:
        metrics["cosine_alignment"] = float(F.cosine_similarity(delta_t, delta_s, dim=1, eps=1e-8).mean())
        nt = F.normalize(delta_t, p=2, dim=1, eps=1e-8)
        ns = F.normalize(delta_s, p=2, dim=1, eps=1e-8)
        metrics["normalized_displacement_error"] = float(torch.linalg.vector_norm(nt - ns, ord=2, dim=1).mean())
    else:
        metrics["cosine_alignment"] = math.nan
        metrics["normalized_displacement_error"] = math.nan
    return metrics


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan


def nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr)) if np.isfinite(arr).any() else math.nan


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Run this inside a GPU Slurm allocation, or pass --device cpu for a small debug run."
        )
    device = torch.device(args.device)
    print(f"[displacement-alignment] method={args.method}", flush=True)
    print(f"[displacement-alignment] device={device}", flush=True)
    print(f"[displacement-alignment] teacher_checkpoint={args.teacher_checkpoint}", flush=True)
    print(f"[displacement-alignment] student_checkpoint={args.student_checkpoint}", flush=True)
    print(f"[displacement-alignment] preprocessed_dir={args.preprocessed_dir}", flush=True)
    print(f"[displacement-alignment] output={args.output}", flush=True)
    dataset_json = load_json(args.dataset_json)
    num_input = args.num_input_channels or infer_num_input_channels(dataset_json)
    num_output = args.num_output_channels or infer_num_output_channels(dataset_json)
    patch_size = None if args.full_volume else tuple(args.patch_size or infer_patch_size(args.teacher_plans, args.configuration))
    if patch_size is not None:
        print(f"[displacement-alignment] crop_patch_size={patch_size}", flush=True)
    else:
        print("[displacement-alignment] full_volume=True", flush=True)

    print("[displacement-alignment] loading teacher model...", flush=True)
    teacher = build_model(args.teacher_plans, args.configuration, args.teacher_checkpoint, num_input, num_output, device)
    print("[displacement-alignment] loading student model...", flush=True)
    student = build_model(args.student_plans, args.configuration, args.student_checkpoint, num_input, num_output, device)
    teacher_features, teacher_handles = register_encoder_hooks(teacher, args.stages)
    student_features, student_handles = register_encoder_hooks(student, args.stages)

    cases = [line.strip() for line in args.val_list.read_text().splitlines() if line.strip()]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(
        f"[displacement-alignment] evaluating {len(cases)} cases, stages={args.stages}, "
        f"num_points={args.num_points}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for case_index, case_id in enumerate(cases, start=1):
            print(f"[displacement-alignment] case {case_index}/{len(cases)}: {case_id}", flush=True)
            image, seg = load_case(args.preprocessed_dir, case_id)
            if patch_size is not None:
                image, seg = crop_foreground_patch(image, seg, patch_size, crop_generator)
            image = image.unsqueeze(0).to(device)
            seg = seg.unsqueeze(0).to(device)
            teacher_features.clear()
            student_features.clear()
            teacher(image)
            student(image)
            missing_stages = [stage for stage in args.stages if stage not in teacher_features or stage not in student_features]
            if missing_stages:
                print(f"[displacement-alignment] warning: missing hooked stages {missing_stages}", flush=True)
            for stage in args.stages:
                if stage not in teacher_features or stage not in student_features:
                    continue
                emb_t, emb_s = sample_embeddings(
                    teacher_features[stage],
                    student_features[stage],
                    seg,
                    args.num_points,
                    generator,
                )
                metrics = compute_metrics(emb_t, emb_s)
                if metrics:
                    rows.append({"method": args.method, "case_id": case_id, "stage": stage, **metrics})
                else:
                    print(
                        f"[displacement-alignment] warning: skipped case={case_id} stage={stage} "
                        f"with {emb_t.shape[0]} sampled points",
                        flush=True,
                    )

    for handle in teacher_handles + student_handles:
        handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "case_id",
        "stage",
        "num_points",
        "num_pairs",
        "cosine_alignment",
        "normalized_displacement_error",
        "distance_correlation_spearman",
        "distance_correlation_pearson",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.csv")
    metric_names = fieldnames[5:]
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *[f"{m}_mean" for m in metric_names], *[f"{m}_std" for m in metric_names], "num_rows"])
        means = [nanmean([float(r[m]) for r in rows]) for m in metric_names]
        stds = [nanstd([float(r[m]) for r in rows]) for m in metric_names]
        writer.writerow([args.method, *means, *stds, len(rows)])
    print(f"[displacement-alignment] wrote {len(rows)} rows to {args.output}", flush=True)
    print(f"[displacement-alignment] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
