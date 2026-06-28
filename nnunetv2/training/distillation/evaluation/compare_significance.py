#!/usr/bin/env python3
"""
Paired statistical tests for per-case evaluation CSV files.

Expected inputs are CSVs produced by eval_nsd_hd95_dice_gpu.py or related
evaluation scripts. Rows are paired by the "Name" column; OVERALL_* rows are
ignored. For each requested metric, the script reports paired Wilcoxon and
paired t-test p-values.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRICS = ("Mean_DiceNNUNet", "Mean_HD95_mm", "Mean_NSD")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two per-case metrics CSVs with paired tests.")
    p.add_argument("--baseline_csv", required=True, help="CSV for the baseline/reference method.")
    p.add_argument("--proposed_csv", required=True, help="CSV for the proposed/new method.")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to compare. Default: Mean_DiceNNUNet Mean_HD95_mm Mean_NSD.",
    )
    p.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help=(
            "Alternative hypothesis for proposed - baseline. "
            "Use 'greater' for metrics where higher is better, 'less' for HD95 if proposed should be lower."
        ),
    )
    p.add_argument(
        "--save_csv",
        default=None,
        help="Output CSV path. Default: <proposed_csv parent>/significance_vs_baseline.csv.",
    )
    return p.parse_args()


def load_cases(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Name" not in df.columns:
        raise ValueError(f"Missing 'Name' column in {path}")
    df = df[~df["Name"].astype(str).str.startswith("OVERALL_")].copy()
    df = df.set_index("Name", drop=False)
    return df


def finite_pairs(base: pd.Series, prop: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    b = pd.to_numeric(base, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(prop, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(b) & np.isfinite(p)
    return b[mask], p[mask]


def safe_wilcoxon(diff: np.ndarray, alternative: str) -> float:
    if diff.size == 0 or np.allclose(diff, 0.0):
        return math.nan
    return float(stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox").pvalue)


def safe_ttest(base: np.ndarray, prop: np.ndarray, alternative: str) -> float:
    if base.size < 2 or np.allclose(prop - base, 0.0):
        return math.nan
    return float(stats.ttest_rel(prop, base, alternative=alternative).pvalue)


def compare_metrics(base_df: pd.DataFrame, prop_df: pd.DataFrame, metrics: Iterable[str], alternative: str) -> List[dict]:
    common_names = base_df.index.intersection(prop_df.index)
    if len(common_names) == 0:
        raise ValueError("No paired case names found between the two CSVs.")

    rows = []
    for metric in metrics:
        if metric not in base_df.columns:
            raise ValueError(f"Metric '{metric}' not found in baseline CSV.")
        if metric not in prop_df.columns:
            raise ValueError(f"Metric '{metric}' not found in proposed CSV.")

        base_vals, prop_vals = finite_pairs(base_df.loc[common_names, metric], prop_df.loc[common_names, metric])
        diff = prop_vals - base_vals
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(diff.size),
                "baseline_mean": float(np.mean(base_vals)) if diff.size else math.nan,
                "baseline_std": float(np.std(base_vals, ddof=1)) if diff.size > 1 else math.nan,
                "proposed_mean": float(np.mean(prop_vals)) if diff.size else math.nan,
                "proposed_std": float(np.std(prop_vals, ddof=1)) if diff.size > 1 else math.nan,
                "mean_diff_proposed_minus_baseline": float(np.mean(diff)) if diff.size else math.nan,
                "median_diff_proposed_minus_baseline": float(np.median(diff)) if diff.size else math.nan,
                "wilcoxon_p": safe_wilcoxon(diff, alternative),
                "paired_ttest_p": safe_ttest(base_vals, prop_vals, alternative),
                "alternative": alternative,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    base_df = load_cases(args.baseline_csv)
    prop_df = load_cases(args.proposed_csv)
    rows = compare_metrics(base_df, prop_df, args.metrics, args.alternative)

    save_csv = args.save_csv or str(Path(args.proposed_csv).with_name("significance_vs_baseline.csv"))
    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(save_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved significance results to: {save_csv}")
    for row in rows:
        print(
            f"{row['metric']}: n={row['n_pairs']}, diff={row['mean_diff_proposed_minus_baseline']:.6f}, "
            f"Wilcoxon p={row['wilcoxon_p']:.6g}, paired t-test p={row['paired_ttest_p']:.6g}"
        )


if __name__ == "__main__":
    main()
