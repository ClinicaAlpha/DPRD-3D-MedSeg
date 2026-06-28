#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from nnunetv2.training.distillation.evaluation.eval_nsd_hd95 import main as eval_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate NSD/HD95 across multiple experiment folders.")
    p.add_argument("--config", required=True, help="YAML or JSON config with experiments.")
    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML not available. Install pyyaml or use JSON config.") from exc
    return yaml.safe_load(path.read_text())


def build_argv(exp: Dict[str, Any]) -> List[str]:
    argv = ["--pred_dir", exp["pred_dir"]]
    if exp.get("dataset"):
        argv += ["--dataset", exp["dataset"]]
    if exp.get("dataset_json"):
        argv += ["--dataset_json", exp["dataset_json"]]
    if exp.get("gt_dir"):
        argv += ["--gt_dir", exp["gt_dir"]]
    if exp.get("save_csv"):
        argv += ["--save_csv", exp["save_csv"]]
    if exp.get("pred_prefix_to_strip"):
        argv += ["--pred_prefix_to_strip", exp["pred_prefix_to_strip"]]
    return argv


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    experiments = cfg.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in config.")

    print(f"Found {len(experiments)} experiments.")
    for idx, exp in enumerate(experiments, start=1):
        name = exp.get("name", f"exp_{idx}")
        print("=" * 80)
        print(f"[{idx}/{len(experiments)}] {name}")
        print("=" * 80)
        argv = build_argv(exp)
        eval_main(argv)


if __name__ == "__main__":
    main()
