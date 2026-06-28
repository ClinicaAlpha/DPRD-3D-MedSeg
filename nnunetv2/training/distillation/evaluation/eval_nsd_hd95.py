#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate foreground Dice/HD95/NSD for a prediction folder.

Output format is aligned with local `metrics.csv` style:
- DiceNNUNet_*
- DiceMONAI_*
- HD95_*_vox
- HD95_*_mm
- NSD_*
- per-case means + NSD_tau/NSD_unit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

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
    m = np.zeros_like(vol, dtype=bool)
    for i in id_list:
        m |= vol == i
    return m


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


def dice_monai_rule(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    gs = int(gt_bin.sum())
    if gs == 0:
        return np.nan
    return dice_standard(pred_bin, gt_bin)


def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: tuple[float, float, float]) -> float:
    ps = int(pred_bin.sum())
    gs = int(gt_bin.sum())
    if ps == 0 or gs == 0:
        return np.nan
    surf_p = surface_mask(pred_bin)
    surf_g = surface_mask(gt_bin)
    if surf_p.sum() == 0 or surf_g.sum() == 0:
        return np.nan
    dt_to_g = distance_transform_edt(~surf_g, sampling=spacing)
    dt_to_p = distance_transform_edt(~surf_p, sampling=spacing)
    dist_p_to_g = dt_to_g[surf_p]
    dist_g_to_p = dt_to_p[surf_g]
    if dist_p_to_g.size == 0 or dist_g_to_p.size == 0:
        return np.nan
    return float(max(np.percentile(dist_p_to_g, 95.0), np.percentile(dist_g_to_p, 95.0)))


def nsd_at_tolerance(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing: tuple[float, float, float],
    tau_mm: float,
    both_empty_as_one: bool,
) -> float:
    ps = int(pred_bin.sum())
    gs = int(gt_bin.sum())

    if ps == 0 and gs == 0:
        return 1.0 if both_empty_as_one else 0.0
    if ps > 0 and gs == 0:
        return 0.0 if both_empty_as_one else 1.0
    if ps == 0 and gs > 0:
        return 0.0

    surf_p = surface_mask(pred_bin)
    surf_g = surface_mask(gt_bin)
    n_p = int(surf_p.sum())
    n_g = int(surf_g.sum())
    if n_p + n_g == 0:
        return 1.0 if both_empty_as_one else 0.0

    dt_to_g = distance_transform_edt(~surf_g, sampling=spacing)
    dt_to_p = distance_transform_edt(~surf_p, sampling=spacing)
    within_p = (dt_to_g[surf_p] <= tau_mm).sum()
    within_g = (dt_to_p[surf_g] <= tau_mm).sum()
    return float((within_p + within_g) / float(n_p + n_g))


def round_or_nan(v: float, ndigits: int = 4) -> float:
    if not np.isfinite(v):
        return np.nan
    return round(float(v), ndigits)


def mean_or_nan(values: List[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def evaluate_case(
    gt_img_path: str,
    seg_img_path: str,
    class_map: "OrderedDict[str, List[int]]",
    tol_map: Dict[str, float],
    nsd_both_empty_as_one: bool,
) -> dict:
    row: dict = {}
    gt_img = nb.load(gt_img_path)
    gt = gt_img.get_fdata().astype(np.uint8)
    seg = nb.load(seg_img_path).get_fdata().astype(np.uint8)

    spacing_mm = tuple(float(s) for s in gt_img.header.get_zooms()[:3])
    spacing_vox = (1.0, 1.0, 1.0)

    d_nu_list: List[float] = []
    d_mn_list: List[float] = []
    h_vox_list: List[float] = []
    h_mm_list: List[float] = []
    nsd_list: List[float] = []

    for cname, id_list in class_map.items():
        gt_mask = make_mask(gt, id_list)
        seg_mask = make_mask(seg, id_list)

        d_nu = dice_nnunet_rule(seg_mask, gt_mask)
        d_mn = dice_monai_rule(seg_mask, gt_mask)
        h_vox = hd95(seg_mask, gt_mask, spacing_vox)
        h_mm = hd95(seg_mask, gt_mask, spacing_mm)
        nsd = nsd_at_tolerance(seg_mask, gt_mask, spacing_mm, tol_map[cname], nsd_both_empty_as_one)

        row[f"DiceNNUNet_{cname}"] = round_or_nan(d_nu)
        row[f"DiceMONAI_{cname}"] = round_or_nan(d_mn)
        row[f"HD95_{cname}_vox"] = round_or_nan(h_vox)
        row[f"HD95_{cname}_mm"] = round_or_nan(h_mm)
        row[f"NSD_{cname}"] = round_or_nan(nsd)

        d_nu_list.append(d_nu)
        d_mn_list.append(d_mn)
        h_vox_list.append(h_vox)
        h_mm_list.append(h_mm)
        nsd_list.append(nsd)

    row["Mean_DiceNNUNet"] = round_or_nan(mean_or_nan(d_nu_list))
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


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    pred_dir = args.pred_dir
    dataset_json = resolve_dataset_json(args)
    gt_dir = resolve_gt_dir(args)
    save_csv = args.save_csv or str(Path(pred_dir) / "nsd_hd95.csv")

    dj = load_dataset_json(dataset_json)
    class_map_all = build_label_spec(dj)
    class_map = filter_class_map(class_map_all, include_background=args.include_background)
    if len(class_map) == 0:
        raise RuntimeError("No classes selected for evaluation.")

    file_ending = get_file_ending(dj)
    tol_map = get_nsd_tolerance(dj, list(class_map.keys()), default_mm=2.0)
    tau_repr = tau_repr_for_csv(tol_map)

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

    class_names = list(class_map.keys())
    cols = (
        ["Name"]
        + [f"DiceNNUNet_{c}" for c in class_names]
        + ["Mean_DiceNNUNet"]
        + [f"DiceMONAI_{c}" for c in class_names]
        + ["Mean_DiceMONAI"]
        + [f"HD95_{c}_vox" for c in class_names]
        + ["Mean_HD95_vox"]
        + [f"HD95_{c}_mm" for c in class_names]
        + ["Mean_HD95_mm"]
        + [f"NSD_{c}" for c in class_names]
        + ["Mean_NSD", "NSD_tau", "NSD_unit"]
    )

    rows = []
    for fn in tqdm(seg_files, desc="Evaluating Dice, NSD & HD95"):
        gt_fn = mapped_gt[fn]
        gt_p = os.path.join(gt_dir, gt_fn)
        seg_p = os.path.join(pred_dir, fn)
        row = {"Name": fn}
        row.update(
            evaluate_case(
                gt_p,
                seg_p,
                class_map,
                tol_map,
                nsd_both_empty_as_one=args.nsd_both_empty_as_one,
            )
        )
        row["NSD_tau"] = tau_repr
        row["NSD_unit"] = "mm"
        rows.append(row)

    means = {}
    for c in cols:
        if c in ("Name", "NSD_tau", "NSD_unit"):
            continue
        means[c] = mean_or_nan([r.get(c, np.nan) for r in rows])

    overall_row = {c: np.nan for c in cols}
    overall_row["Name"] = "OVERALL_MEAN"
    overall_row["NSD_tau"] = tau_repr
    overall_row["NSD_unit"] = "mm"
    for c, v in means.items():
        overall_row[c] = v

    Path(os.path.dirname(save_csv)).mkdir(parents=True, exist_ok=True)
    if HAS_PANDAS:
        df = pd.DataFrame(rows + [overall_row], columns=cols)
        df.to_csv(save_csv, index=False)
    else:
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(overall_row)

    print(">" * 22)
    for cname in class_names:
        md_nu = means.get(f"DiceNNUNet_{cname}", np.nan)
        md_mn = means.get(f"DiceMONAI_{cname}", np.nan)
        mh_v = means.get(f"HD95_{cname}_vox", np.nan)
        mh_m = means.get(f"HD95_{cname}_mm", np.nan)
        mn = means.get(f"NSD_{cname}", np.nan)
        print(
            f"{cname:20s}  Dice(nnU-Net): {md_nu:.4f}   Dice(MONAI): {md_mn:.4f}   "
            f"HD95(vox): {mh_v:.4f}   HD95(mm): {mh_m:.4f}   NSD: {mn:.4f}"
        )

    overall_d_nu = means.get("Mean_DiceNNUNet", np.nan)
    overall_d_mn = means.get("Mean_DiceMONAI", np.nan)
    overall_h_v = means.get("Mean_HD95_vox", np.nan)
    overall_h_m = means.get("Mean_HD95_mm", np.nan)
    overall_n = means.get("Mean_NSD", np.nan)
    tau_print = f"{float(tau_repr):g}" if tau_repr != "per_class" else "per-class"
    print(
        f"{'OVERALL':20s}  Dice(nnU-Net): {overall_d_nu:.4f}   Dice(MONAI): {overall_d_mn:.4f}   "
        f"HD95(vox): {overall_h_v:.4f}   HD95(mm): {overall_h_m:.4f}   NSD@{tau_print}mm: {overall_n:.4f}"
    )
    print("<" * 22)


if __name__ == "__main__":
    main()
