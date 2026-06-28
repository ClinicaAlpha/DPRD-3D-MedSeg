#!/usr/bin/env python3
"""
Aggregate distillation validation metrics into one CSV.

For each experiment/fold, this script can:
1) Optionally run eval_nsd_hd95 to generate metrics CSV
2) Parse per-sample metrics from nsd_hd95.csv
3) Parse per-sample validation timing from training logs
4) Emit one combined report CSV containing sample rows + summary rows

Summary includes mean/std validation time across validation samples.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s+(.*)$")
PREDICT_RE = re.compile(r"^predicting\s+(.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-sample NSD/HD95 + validation timing for multiple experiments."
    )
    p.add_argument(
        "--exp",
        action="append",
        required=True,
        help="Experiment root folder (repeatable), e.g. .../DistillationTrainer__...__3d_fullres",
    )
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. Dataset018_BTCV")
    p.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[3],
        help="Fold indices to process. Default: 3",
    )
    p.add_argument(
        "--validation-dir-name",
        default="validation",
        help="Validation directory name under fold_x. Default: validation",
    )
    p.add_argument(
        "--metrics-filename",
        default="nsd_hd95.csv",
        help="Metrics CSV filename under validation dir. Default: nsd_hd95.csv",
    )
    p.add_argument(
        "--run-eval",
        action="store_true",
        help="Run eval_nsd_hd95 before aggregation (if CSV exists, it will be overwritten only with --force-eval).",
    )
    p.add_argument(
        "--force-eval",
        action="store_true",
        help="Force rerun eval_nsd_hd95 even if metrics CSV already exists.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for subprocess eval command. Default: current interpreter",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output aggregated CSV path",
    )
    p.add_argument(
        "--timing-log",
        action="append",
        default=[],
        help=(
            "Optional explicit timing log mapping: exp_path=log_path . "
            "Repeatable. If omitted, auto-detect latest training_log_*.txt in fold dir."
        ),
    )
    return p.parse_args()


def parse_timing_log_map(items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --timing-log format: {item}. Expected exp_path=log_path")
        exp, log = item.split("=", 1)
        out[os.path.abspath(exp)] = log
    return out


def run_eval_if_needed(
    py_exec: str,
    exp_dir: Path,
    fold: int,
    validation_dir_name: str,
    dataset: str,
    metrics_filename: str,
    run_eval: bool,
    force_eval: bool,
) -> Path:
    pred_dir = exp_dir / f"fold_{fold}" / validation_dir_name
    metrics_csv = pred_dir / metrics_filename

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Validation folder not found: {pred_dir}")

    should_run = run_eval and (force_eval or (not metrics_csv.is_file()))
    if should_run:
        cmd = [
            py_exec,
            "-m",
            "nnunetv2.training.distillation.evaluation.eval_nsd_hd95",
            "--pred_dir",
            str(pred_dir),
            "--dataset",
            dataset,
            "--save_csv",
            str(metrics_csv),
        ]
        print(f"[eval] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    if not metrics_csv.is_file():
        raise FileNotFoundError(
            f"Metrics CSV not found: {metrics_csv}. Use --run-eval or check the path."
        )
    return metrics_csv


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def split_sample_and_overall_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    sample_rows = []
    overall = {}
    for r in rows:
        if r.get("Name") == "OVERALL_MEAN":
            overall = r
        else:
            sample_rows.append(r)
    return sample_rows, overall


def detect_latest_training_log(fold_dir: Path) -> Optional[Path]:
    logs = sorted(fold_dir.glob("training_log_*.txt"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_datetime(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def parse_segments_from_log(log_path: Path) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    events: List[Tuple[dt.datetime, str, str]] = []
    complete_time: Optional[dt.datetime] = None

    with log_path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = TS_RE.match(line)
            if not m:
                continue
            ts = parse_datetime(m.group(1))
            msg = m.group(2)

            pm = PREDICT_RE.match(msg)
            if pm:
                events.append((ts, "predict", pm.group(1).strip()))
                continue

            if msg.startswith("Validation complete"):
                complete_time = ts
                if events:
                    segments.append({"events": events[:], "complete_time": complete_time})
                events = []
                complete_time = None

    return segments


def choose_segment(
    segments: List[Dict[str, object]], sample_names: List[str]
) -> Optional[Dict[str, object]]:
    target = set(sample_names)
    if not target:
        return None

    candidates = []
    for seg in segments:
        events = seg["events"]  # type: ignore[index]
        names = {name for _, typ, name in events if typ == "predict"}
        if target.issubset(names):
            comp = seg.get("complete_time")
            if isinstance(comp, dt.datetime):
                candidates.append((comp, seg))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def durations_from_segment(seg: Dict[str, object]) -> Dict[str, float]:
    events: List[Tuple[dt.datetime, str, str]] = seg["events"]  # type: ignore[index]
    complete: dt.datetime = seg["complete_time"]  # type: ignore[index]
    predict_events = [(t, name) for (t, typ, name) in events if typ == "predict"]
    out: Dict[str, float] = {}
    for idx, (t0, name) in enumerate(predict_events):
        if idx + 1 < len(predict_events):
            t1 = predict_events[idx + 1][0]
        else:
            t1 = complete
        out[name] = max(0.0, (t1 - t0).total_seconds())
    return out


def to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def summarize_times(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return float(m), float(s)


def main() -> int:
    args = parse_args()
    exp_paths = [Path(e).resolve() for e in args.exp]
    timing_log_map = parse_timing_log_map(args.timing_log)

    all_rows: List[Dict[str, object]] = []

    for exp_dir in exp_paths:
        exp_name = exp_dir.name
        for fold in args.folds:
            fold_dir = exp_dir / f"fold_{fold}"
            metrics_csv = run_eval_if_needed(
                py_exec=args.python,
                exp_dir=exp_dir,
                fold=fold,
                validation_dir_name=args.validation_dir_name,
                dataset=args.dataset,
                metrics_filename=args.metrics_filename,
                run_eval=args.run_eval,
                force_eval=args.force_eval,
            )

            fieldnames, rows = read_csv_rows(metrics_csv)
            sample_rows, overall_row = split_sample_and_overall_rows(rows)
            sample_names = [r.get("Name", "") for r in sample_rows]

            exp_key = os.path.abspath(str(exp_dir))
            log_override = timing_log_map.get(exp_key)
            timing_log = Path(log_override) if log_override else detect_latest_training_log(fold_dir)

            timing_per_sample: Dict[str, float] = {}
            if timing_log and timing_log.is_file():
                segments = parse_segments_from_log(timing_log)
                seg = choose_segment(segments, sample_names)
                if seg is not None:
                    timing_per_sample = durations_from_segment(seg)
            else:
                timing_log = None

            timed_vals = [v for k, v in timing_per_sample.items() if k in sample_names]
            t_mean, t_std = summarize_times(timed_vals)

            for r in sample_rows:
                name = r.get("Name", "")
                row: Dict[str, object] = {
                    "row_type": "sample",
                    "exp_path": str(exp_dir),
                    "exp_name": exp_name,
                    "fold": fold,
                    "sample_name": name,
                    "validation_dir": str(exp_dir / f"fold_{fold}" / args.validation_dir_name),
                    "metrics_csv": str(metrics_csv),
                    "timing_log": str(timing_log) if timing_log else "",
                    "validation_time_sec": timing_per_sample.get(name),
                    "validation_time_mean_sec": t_mean,
                    "validation_time_std_sec": t_std,
                    "num_samples_timed": len(timed_vals),
                }
                for c in fieldnames:
                    if c == "Name":
                        continue
                    row[c] = r.get(c)
                all_rows.append(row)

            summary: Dict[str, object] = {
                "row_type": "summary",
                "exp_path": str(exp_dir),
                "exp_name": exp_name,
                "fold": fold,
                "sample_name": "OVERALL_MEAN",
                "validation_dir": str(exp_dir / f"fold_{fold}" / args.validation_dir_name),
                "metrics_csv": str(metrics_csv),
                "timing_log": str(timing_log) if timing_log else "",
                "validation_time_sec": "",
                "validation_time_mean_sec": t_mean,
                "validation_time_std_sec": t_std,
                "num_samples_timed": len(timed_vals),
            }

            if overall_row:
                for c in fieldnames:
                    if c == "Name":
                        continue
                    summary[c] = overall_row.get(c)
            else:
                mean_nsd_vals = [to_float(r.get("Mean_NSD", "")) for r in sample_rows]
                mean_nsd_vals = [v for v in mean_nsd_vals if v is not None]
                summary["Mean_NSD"] = mean(mean_nsd_vals) if mean_nsd_vals else None
            all_rows.append(summary)

    base_cols = [
        "row_type",
        "exp_path",
        "exp_name",
        "fold",
        "sample_name",
        "validation_dir",
        "metrics_csv",
        "timing_log",
        "validation_time_sec",
        "validation_time_mean_sec",
        "validation_time_std_sec",
        "num_samples_timed",
    ]
    extra_cols = sorted({k for r in all_rows for k in r.keys() if k not in base_cols})
    out_cols = base_cols + extra_cols

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"[ok] wrote report: {out_path}")
    print(f"[ok] rows: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
