#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate foreground Dice/HD95/NSD for a prediction folder.

Output format is aligned with local `metrics.csv` style:
- DiceNNUNet_*
- (optional) DiceMONAI_*
- HD95_*_vox
- HD95_*_mm
- NSD_*
- per-case means + NSD_tau/NSD_unit
- dataset-level OVERALL_MEAN and OVERALL_STD rows
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nb
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from tqdm import tqdm

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:
    pd = None
    HAS_PANDAS = False

from nnunetv2.paths import nnUNet_preprocessed


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Dice/NSD/HD95 for a prediction folder (foreground by default)."
    )
    p.add_argument("--pred_dir", required=True, help="Folder containing predicted segmentations.")
    p.add_argument("--dataset", default=None, help="Dataset name (e.g., Dataset018_BTCV).")
    p.add_argument("--dataset_json", default=None, help="Path to dataset.json (optional).")
    p.add_argument("--gt_dir", default=None, help="Folder of GT segmentations (optional).")
    p.add_argument("--save_csv", default=None, help="Output CSV path (default: <pred_dir>/nsd_hd95.csv).")
    p.add_argument(
        "--pred_prefix_to_strip",
        default="",
        help="Optional prefix to drop from prediction filenames when looking up GT (e.g., 'TestSet_').",
    )
    p.add_argument(
        "--include_background",
        action="store_true",
        help="If set, also evaluate background class. Default: foreground only.",
    )
    p.add_argument(
        "--nsd_both_empty_as_one",
        action="store_true",
        help=(
            "NSD empty-set handling: both-empty->1 and pred-only->0. "
            "Default is local legacy style (both-empty->0, pred-only->1)."
        ),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of case-level worker processes. Default: 1.",
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Optional local cache directory for copied pred/gt files (for repeated runs on slow NFS). "
            "If set, files are copied once and evaluated from cache."
        ),
    )
    p.add_argument(
        "--with_monai_dice",
        action="store_true",
        help=(
            "Also compute/report MONAI-style Dice. Default is off for speed because "
            "nnU-Net and MONAI Dice are typically identical in this setup."
        ),
    )
    p.add_argument(
        "--metric_mode",
        choices=("auto", "legacy", "brats", "fast"),
        default="auto",
        help=(
            "Metric behavior mode. "
            "'legacy' matches old eval_effidec3d-style Dice/HD95 empty-case handling and HD95 aggregation "
            "(recommended for BTCV). "
            "'brats' uses nnU-Net-style Dice empty-case semantics and non-legacy boundary behavior "
            "(recommended for BraTS). "
            "'fast' keeps the previous non-legacy implementation. "
            "'auto' selects 'brats' for datasets containing 'brats' in name/path, else 'legacy'."
        ),
    )
    return p.parse_args(argv)


def load_dataset_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_label_spec(dj: dict) -> "OrderedDict[str, List[int]]":
    labels = dj.get("labels", {})
    spec = []
    for name, val in labels.items():
        if isinstance(val, int):
            ids = [int(val)]
        elif isinstance(val, list):
            ids = [int(x) for x in val]
        else:
            continue
        if all(x == 0 for x in ids):
            spec.append((name, [0]))
            continue
        spec.append((name, sorted(ids)))
    spec.sort(key=lambda x: min(x[1]))
    if not any(min(v) == 0 for _, v in spec):
        spec.insert(0, ("background", [0]))
    return OrderedDict(spec)


def filter_class_map(
    class_map: "OrderedDict[str, List[int]]", include_background: bool
) -> "OrderedDict[str, List[int]]":
    if include_background:
        return class_map
    return OrderedDict(
        (name, ids) for name, ids in class_map.items() if not (len(ids) == 1 and ids[0] == 0)
    )


def get_file_ending(dj: dict) -> str:
    return dj.get("file_ending", ".nii.gz")


def get_nsd_tolerance(dj: dict, class_names: List[str], default_mm: float = 2.0) -> "OrderedDict[str, float]":
    per_class = {k: float(default_mm) for k in class_names}
    if isinstance(dj.get("nsd_tolerance_mm"), (int, float)):
        val = float(dj["nsd_tolerance_mm"])
        per_class = {k: val for k in class_names}
    if isinstance(dj.get("nsd_tolerance_mm_per_class"), dict):
        for k, v in dj["nsd_tolerance_mm_per_class"].items():
            try:
                per_class[k] = float(v)
            except Exception:
                pass
    return OrderedDict(per_class)


def filenames_in(dir_path: str, ending: str) -> List[str]:
    fs = [f for f in os.listdir(dir_path) if f.endswith(ending)]
    fs.sort()
    return fs


def make_mask(vol: np.ndarray, id_list: List[int]) -> np.ndarray:
    if len(id_list) == 1:
        return vol == id_list[0]
    return np.isin(vol, id_list)


def surface_mask(bin_mask: np.ndarray) -> np.ndarray:
    if not np.any(bin_mask):
        return np.zeros_like(bin_mask, dtype=bool)
    st = generate_binary_structure(3, 1)
    er = binary_erosion(bin_mask, structure=st, border_value=0)
    return np.logical_and(bin_mask, np.logical_not(er))


def dice_standard(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    tp = np.logical_and(pred_bin, gt_bin).sum(dtype=np.float64)
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum(dtype=np.float64)
    den = 2.0 * tp + fp + fn
    return float(2.0 * tp / den) if den > 0.0 else np.nan


def dice_nnunet_rule(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    gs = int(gt_bin.sum())
    ps = int(pred_bin.sum())
    if gs == 0 and ps == 0:
        return np.nan
    if gs == 0 and ps > 0:
        return 0.0
    return dice_standard(pred_bin, gt_bin)


def dice_legacy_rule(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    ps = int(pred_bin.sum())
    gs = int(gt_bin.sum())
    if ps > 0 and gs > 0:
        return dice_standard(pred_bin, gt_bin)
    # Do not reward false positives on empty GT.
    if ps > 0 and gs == 0:
        return 0.0
    # Keep legacy behavior for both-empty as 0.0 (instead of NaN).
    return 0.0


def dice_monai_rule(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    gs = int(gt_bin.sum())
    if gs == 0:
        return np.nan
    return dice_standard(pred_bin, gt_bin)


def hd95_from_distances(dist_a_to_b: np.ndarray, dist_b_to_a: np.ndarray) -> float:
    if dist_a_to_b.size == 0 or dist_b_to_a.size == 0:
        return np.nan
    return float(max(np.percentile(dist_a_to_b, 95.0), np.percentile(dist_b_to_a, 95.0)))


def hd95_from_distances_legacy(dist_a_to_b: np.ndarray, dist_b_to_a: np.ndarray) -> float:
    """
    Legacy-style HD95 aggregation: percentile over concatenated bidirectional distances.
    This mirrors medpy-style behavior more closely than max(p95(a->b), p95(b->a)).
    """
    if dist_a_to_b.size == 0 or dist_b_to_a.size == 0:
        return np.nan
    return float(np.percentile(np.concatenate([dist_a_to_b, dist_b_to_a]), 95.0))


def boundary_metrics(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing_mm: tuple[float, float, float],
    tau_mm: float,
    both_empty_as_one: bool,
    metric_mode: str,
) -> Tuple[float, float, float]:
    """Compute HD95(vox), HD95(mm), NSD in one pass per class."""
    ps = int(pred_bin.sum())
    gs = int(gt_bin.sum())

    legacy_mode = metric_mode == "legacy"

    if ps == 0 and gs == 0:
        nsd = 1.0 if both_empty_as_one else 0.0
        return (0.0, 0.0, nsd) if legacy_mode else (np.nan, np.nan, nsd)
    if ps > 0 and gs == 0:
        nsd = 0.0 if both_empty_as_one else 1.0
        return (0.0, 0.0, nsd) if legacy_mode else (np.nan, np.nan, nsd)
    if ps == 0 and gs > 0:
        return (0.0, 0.0, 0.0) if legacy_mode else (np.nan, np.nan, 0.0)

    surf_p = surface_mask(pred_bin)
    surf_g = surface_mask(gt_bin)
    n_p = int(surf_p.sum())
    n_g = int(surf_g.sum())
    if n_p + n_g == 0:
        nsd = 1.0 if both_empty_as_one else 0.0
        return (0.0, 0.0, nsd) if legacy_mode else (np.nan, np.nan, nsd)

    dt_mm_to_g = distance_transform_edt(~surf_g, sampling=spacing_mm)
    dt_mm_to_p = distance_transform_edt(~surf_p, sampling=spacing_mm)
    dist_p_to_g_mm = dt_mm_to_g[surf_p]
    dist_g_to_p_mm = dt_mm_to_p[surf_g]

    within_p = (dist_p_to_g_mm <= tau_mm).sum()
    within_g = (dist_g_to_p_mm <= tau_mm).sum()
    nsd = float((within_p + within_g) / float(n_p + n_g))
    hd95_fn = hd95_from_distances_legacy if legacy_mode else hd95_from_distances
    hd95_mm = hd95_fn(dist_p_to_g_mm, dist_g_to_p_mm)

    dt_vox_to_g = distance_transform_edt(~surf_g)
    dt_vox_to_p = distance_transform_edt(~surf_p)
    hd95_vox = hd95_fn(dt_vox_to_g[surf_p], dt_vox_to_p[surf_g])
    return hd95_vox, hd95_mm, nsd


def resolve_metric_mode(args: argparse.Namespace, dataset_json_path: str) -> str:
    if args.metric_mode != "auto":
        return args.metric_mode
    name_tokens = []
    if args.dataset:
        name_tokens.append(args.dataset.lower())
    if dataset_json_path:
        name_tokens.append(dataset_json_path.lower())
    joined = " ".join(name_tokens)
    return "brats" if "brats" in joined else "legacy"


def resolve_nsd_empty_policy(metric_mode: str, nsd_both_empty_as_one_flag: bool) -> bool:
    # BraTS mode defaults to mathematical empty-set handling for NSD.
    if metric_mode == "brats":
        return True
    return nsd_both_empty_as_one_flag


def round_or_nan(v: float, ndigits: int = 4) -> float:
    if not np.isfinite(v):
        return np.nan
    return round(float(v), ndigits)


def mean_or_nan(values: List[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def std_or_nan(values: List[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1))


def evaluate_case(
    gt_img_path: str,
    seg_img_path: str,
    class_map: "OrderedDict[str, List[int]]",
    tol_map: Dict[str, float],
    nsd_both_empty_as_one: bool,
    with_monai_dice: bool,
    metric_mode: str,
) -> dict:
    row: dict = {}
    gt_img = nb.load(gt_img_path)
    gt = np.asanyarray(gt_img.dataobj)
    seg = np.asanyarray(nb.load(seg_img_path).dataobj)

    spacing_mm = tuple(float(s) for s in gt_img.header.get_zooms()[:3])

    d_nu_list: List[float] = []
    d_mn_list: List[float] = [] if with_monai_dice else None
    h_vox_list: List[float] = []
    h_mm_list: List[float] = []
    nsd_list: List[float] = []

    for cname, id_list in class_map.items():
        gt_mask = make_mask(gt, id_list)
        seg_mask = make_mask(seg, id_list)

        d_nu = dice_legacy_rule(seg_mask, gt_mask) if metric_mode == "legacy" else dice_nnunet_rule(seg_mask, gt_mask)
        h_vox, h_mm, nsd = boundary_metrics(
            seg_mask,
            gt_mask,
            spacing_mm=spacing_mm,
            tau_mm=tol_map[cname],
            both_empty_as_one=nsd_both_empty_as_one,
            metric_mode=metric_mode,
        )

        row[f"DiceNNUNet_{cname}"] = round_or_nan(d_nu)
        if with_monai_dice:
            d_mn = dice_monai_rule(seg_mask, gt_mask)
            row[f"DiceMONAI_{cname}"] = round_or_nan(d_mn)
            d_mn_list.append(d_mn)
        row[f"HD95_{cname}_vox"] = round_or_nan(h_vox)
        row[f"HD95_{cname}_mm"] = round_or_nan(h_mm)
        row[f"NSD_{cname}"] = round_or_nan(nsd)

        d_nu_list.append(d_nu)
        h_vox_list.append(h_vox)
        h_mm_list.append(h_mm)
        nsd_list.append(nsd)

    row["Mean_DiceNNUNet"] = round_or_nan(mean_or_nan(d_nu_list))
    if with_monai_dice:
        row["Mean_DiceMONAI"] = round_or_nan(mean_or_nan(d_mn_list))
    row["Mean_HD95_vox"] = round_or_nan(mean_or_nan(h_vox_list))
    row["Mean_HD95_mm"] = round_or_nan(mean_or_nan(h_mm_list))
    row["Mean_NSD"] = round_or_nan(mean_or_nan(nsd_list))
    return row


def resolve_dataset_json(args: argparse.Namespace) -> str:
    if args.dataset_json:
        return args.dataset_json
    if not args.dataset:
        raise ValueError("--dataset or --dataset_json is required.")
    return str(Path(nnUNet_preprocessed) / args.dataset / "dataset.json")


def resolve_gt_dir(args: argparse.Namespace) -> str:
    if args.gt_dir:
        return args.gt_dir
    if not args.dataset:
        raise ValueError("--dataset or --gt_dir is required.")
    return str(Path(nnUNet_preprocessed) / args.dataset / "gt_segmentations")


def tau_repr_for_csv(tol_map: Dict[str, float]) -> str:
    values = [float(v) for v in tol_map.values()]
    if not values:
        return "nan"
    if all(abs(v - values[0]) < 1e-8 for v in values):
        return str(values[0])
    return "per_class"


def maybe_cache_pairs(
    pred_dir: str,
    gt_dir: str,
    seg_files: List[str],
    mapped_gt: Dict[str, str],
    cache_dir: str | None,
) -> Tuple[str, str]:
    if not cache_dir:
        return pred_dir, gt_dir
    cache_root = Path(cache_dir)
    pred_cache = cache_root / "pred"
    gt_cache = cache_root / "gt"
    pred_cache.mkdir(parents=True, exist_ok=True)
    gt_cache.mkdir(parents=True, exist_ok=True)

    for fn in seg_files:
        src_p = Path(pred_dir) / fn
        dst_p = pred_cache / fn
        if not dst_p.exists() or dst_p.stat().st_size != src_p.stat().st_size:
            shutil.copy2(src_p, dst_p)
        gt_fn = mapped_gt[fn]
        src_g = Path(gt_dir) / gt_fn
        dst_g = gt_cache / gt_fn
        if not dst_g.exists() or dst_g.stat().st_size != src_g.stat().st_size:
            shutil.copy2(src_g, dst_g)
    return str(pred_cache), str(gt_cache)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    pred_dir = args.pred_dir
    dataset_json = resolve_dataset_json(args)
    gt_dir = resolve_gt_dir(args)
    save_csv = args.save_csv or str(Path(pred_dir) / "nsd_hd95.csv")
    metric_mode = resolve_metric_mode(args, dataset_json)
    nsd_both_empty_as_one = resolve_nsd_empty_policy(metric_mode, args.nsd_both_empty_as_one)

    dj = load_dataset_json(dataset_json)
    class_map_all = build_label_spec(dj)
    class_map = filter_class_map(class_map_all, include_background=args.include_background)
    if len(class_map) == 0:
        raise RuntimeError("No classes selected for evaluation.")

    file_ending = get_file_ending(dj)
    tol_map = get_nsd_tolerance(dj, list(class_map.keys()), default_mm=2.0)
    tau_repr = tau_repr_for_csv(tol_map)
    print(f"[metric_mode] requested={args.metric_mode}, effective={metric_mode}, nsd_both_empty_as_one={nsd_both_empty_as_one}")

    seg_files = filenames_in(pred_dir, file_ending)
    if not seg_files:
        raise FileNotFoundError(f"No prediction files ending with '{file_ending}' in {pred_dir}")

    def map_pred_to_gt(fn: str) -> str:
        if args.pred_prefix_to_strip and fn.startswith(args.pred_prefix_to_strip):
            return fn[len(args.pred_prefix_to_strip) :]
        return fn

    mapped_gt = {fn: map_pred_to_gt(fn) for fn in seg_files}
    missing = [f for f, gt_fn in mapped_gt.items() if not os.path.exists(os.path.join(gt_dir, gt_fn))]
    if missing:
        raise FileNotFoundError("Missing GT for files:\n" + "\n".join(missing))

    pred_dir_eval, gt_dir_eval = maybe_cache_pairs(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        seg_files=seg_files,
        mapped_gt=mapped_gt,
        cache_dir=args.cache_dir,
    )

    class_names = list(class_map.keys())
    cols = (
        ["Name"]
        + [f"DiceNNUNet_{c}" for c in class_names]
        + ["Mean_DiceNNUNet"]
        + [f"HD95_{c}_vox" for c in class_names]
        + ["Mean_HD95_vox"]
        + [f"HD95_{c}_mm" for c in class_names]
        + ["Mean_HD95_mm"]
        + [f"NSD_{c}" for c in class_names]
        + ["Mean_NSD", "NSD_tau", "NSD_unit"]
    )
    if args.with_monai_dice:
        insert_idx = cols.index(f"HD95_{class_names[0]}_vox")
        monai_cols = [f"DiceMONAI_{c}" for c in class_names] + ["Mean_DiceMONAI"]
        cols = cols[:insert_idx] + monai_cols + cols[insert_idx:]

    rows = []
    t0 = time.perf_counter()
    if args.num_workers <= 1:
        for fn in tqdm(seg_files, desc="Evaluating Dice, NSD & HD95"):
            gt_fn = mapped_gt[fn]
            gt_p = os.path.join(gt_dir_eval, gt_fn)
            seg_p = os.path.join(pred_dir_eval, fn)
            row = {"Name": fn}
            row.update(
                evaluate_case(
                    gt_p,
                    seg_p,
                    class_map,
                    tol_map,
                    nsd_both_empty_as_one=nsd_both_empty_as_one,
                    with_monai_dice=args.with_monai_dice,
                    metric_mode=metric_mode,
                )
            )
            row["NSD_tau"] = tau_repr
            row["NSD_unit"] = "mm"
            rows.append(row)
    else:
        tasks = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            for idx, fn in enumerate(seg_files):
                gt_fn = mapped_gt[fn]
                gt_p = os.path.join(gt_dir_eval, gt_fn)
                seg_p = os.path.join(pred_dir_eval, fn)
                fut = ex.submit(
                    evaluate_case,
                    gt_p,
                    seg_p,
                    class_map,
                    tol_map,
                    nsd_both_empty_as_one,
                    args.with_monai_dice,
                    metric_mode,
                )
                tasks.append((idx, fn, fut))

            done_rows = {}
            future_to_meta = {fut: (idx, fn) for idx, fn, fut in tasks}
            for fut in tqdm(as_completed(future_to_meta), total=len(tasks), desc="Evaluating Dice, NSD & HD95"):
                idx, fn = future_to_meta[fut]
                row = {"Name": fn}
                row.update(fut.result())
                row["NSD_tau"] = tau_repr
                row["NSD_unit"] = "mm"
                done_rows[idx] = row
        rows = [done_rows[i] for i in range(len(seg_files))]
    elapsed = time.perf_counter() - t0

    means = {}
    stds = {}
    for c in cols:
        if c in ("Name", "NSD_tau", "NSD_unit"):
            continue
        values = [r.get(c, np.nan) for r in rows]
        means[c] = mean_or_nan(values)
        stds[c] = std_or_nan(values)

    overall_mean_row = {c: np.nan for c in cols}
    overall_mean_row["Name"] = "OVERALL_MEAN"
    overall_mean_row["NSD_tau"] = tau_repr
    overall_mean_row["NSD_unit"] = "mm"
    for c, v in means.items():
        overall_mean_row[c] = v

    overall_std_row = {c: np.nan for c in cols}
    overall_std_row["Name"] = "OVERALL_STD"
    overall_std_row["NSD_tau"] = tau_repr
    overall_std_row["NSD_unit"] = "mm"
    for c, v in stds.items():
        overall_std_row[c] = v

    Path(os.path.dirname(save_csv)).mkdir(parents=True, exist_ok=True)
    if HAS_PANDAS:
        df = pd.DataFrame(rows + [overall_mean_row, overall_std_row], columns=cols)
        df.to_csv(save_csv, index=False)
    else:
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(overall_mean_row)
            writer.writerow(overall_std_row)

    print(">" * 22)
    for cname in class_names:
        md_nu = means.get(f"DiceNNUNet_{cname}", np.nan)
        sd_nu = stds.get(f"DiceNNUNet_{cname}", np.nan)
        mh_v = means.get(f"HD95_{cname}_vox", np.nan)
        sh_v = stds.get(f"HD95_{cname}_vox", np.nan)
        mh_m = means.get(f"HD95_{cname}_mm", np.nan)
        sh_m = stds.get(f"HD95_{cname}_mm", np.nan)
        mn = means.get(f"NSD_{cname}", np.nan)
        sn = stds.get(f"NSD_{cname}", np.nan)
        if args.with_monai_dice:
            md_mn = means.get(f"DiceMONAI_{cname}", np.nan)
            sd_mn = stds.get(f"DiceMONAI_{cname}", np.nan)
            print(
                f"{cname:20s}  Dice(nnU-Net): {md_nu:.4f}±{sd_nu:.4f}   Dice(MONAI): {md_mn:.4f}±{sd_mn:.4f}   "
                f"HD95(vox): {mh_v:.4f}±{sh_v:.4f}   HD95(mm): {mh_m:.4f}±{sh_m:.4f}   NSD: {mn:.4f}±{sn:.4f}"
            )
        else:
            print(
                f"{cname:20s}  Dice(nnU-Net): {md_nu:.4f}±{sd_nu:.4f}   "
                f"HD95(vox): {mh_v:.4f}±{sh_v:.4f}   HD95(mm): {mh_m:.4f}±{sh_m:.4f}   NSD: {mn:.4f}±{sn:.4f}"
            )

    overall_d_nu = means.get("Mean_DiceNNUNet", np.nan)
    overall_sd_nu = stds.get("Mean_DiceNNUNet", np.nan)
    overall_h_v = means.get("Mean_HD95_vox", np.nan)
    overall_sh_v = stds.get("Mean_HD95_vox", np.nan)
    overall_h_m = means.get("Mean_HD95_mm", np.nan)
    overall_sh_m = stds.get("Mean_HD95_mm", np.nan)
    overall_n = means.get("Mean_NSD", np.nan)
    overall_sn = stds.get("Mean_NSD", np.nan)
    tau_print = f"{float(tau_repr):g}" if tau_repr != "per_class" else "per-class"
    if args.with_monai_dice:
        overall_d_mn = means.get("Mean_DiceMONAI", np.nan)
        overall_sd_mn = stds.get("Mean_DiceMONAI", np.nan)
        print(
            f"{'OVERALL':20s}  Dice(nnU-Net): {overall_d_nu:.4f}±{overall_sd_nu:.4f}   Dice(MONAI): {overall_d_mn:.4f}±{overall_sd_mn:.4f}   "
            f"HD95(vox): {overall_h_v:.4f}±{overall_sh_v:.4f}   HD95(mm): {overall_h_m:.4f}±{overall_sh_m:.4f}   NSD@{tau_print}mm: {overall_n:.4f}±{overall_sn:.4f}"
        )
    else:
        print(
            f"{'OVERALL':20s}  Dice(nnU-Net): {overall_d_nu:.4f}±{overall_sd_nu:.4f}   "
            f"HD95(vox): {overall_h_v:.4f}±{overall_sh_v:.4f}   HD95(mm): {overall_h_m:.4f}±{overall_sh_m:.4f}   NSD@{tau_print}mm: {overall_n:.4f}±{overall_sn:.4f}"
        )
    print(f"{'Elapsed(sec)':20s}  {elapsed:.2f}")
    print("<" * 22)


if __name__ == "__main__":
    main()
