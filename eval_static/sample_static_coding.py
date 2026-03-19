#!/usr/bin/env python3
"""
Sample and save static coding evaluation items.

This script builds a fixed sampled item file that can later be consumed by
the coding evaluator, so selection and evaluation remain separated.

Usage examples:
    python eval_static/sample_static_coding.py --profile paper
    python eval_static/sample_static_coding.py --config eval_static/config_static_coding.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm
import yaml

from eval_static_coding import (
    DATASET_LOOKUP,
    DATASET_SPECS,
    build_sample_plan,
    build_eval_prompt,
    load_env_file,
    load_raw_rows,
    sample_rows,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and save a fixed static coding item set.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Sampling settings are read from the 'sampling' section.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=[spec.name for spec in DATASET_SPECS],
        help="Datasets to sample from. Defaults to all supported coding datasets.",
    )
    parser.add_argument(
        "--profile",
        choices=["pilot", "paper"],
        default=None,
        help="Sampling profile. 'paper' uses the larger built-in counts.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=None,
        help="Override and sample the same number of items from every selected dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save sampled_items.jsonl and sampling_config.json.",
    )
    return parser.parse_args()


def resolve_env_path() -> Path:
    here = Path(__file__).resolve()
    local_env = here.with_name(".env")
    if local_env.exists():
        return local_env
    return here.parent.parent / ".env"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _expand_env(raw)


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    section = config.get("sampling", {})

    if args.datasets is None:
        args.datasets = section.get("datasets", [spec.name for spec in DATASET_SPECS])
    if args.profile is None:
        args.profile = section.get("profile", "pilot")
    if args.samples_per_dataset is None and "samples_per_dataset" in section:
        args.samples_per_dataset = int(section["samples_per_dataset"])
    if args.seed is None:
        args.seed = int(section.get("seed", 0))
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")
    return args


def build_output_dir(user_output_dir: str | None) -> Path:
    if user_output_dir:
        return Path(user_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "static_samples" / timestamp


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)

    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_plan = build_sample_plan(args.datasets, args.profile, args.samples_per_dataset)

    sampled_items: list[dict[str, Any]] = []
    dataset_progress = tqdm(args.datasets, desc="Sampling datasets", unit="dataset")
    for dataset_name in dataset_progress:
        spec = DATASET_LOOKUP[dataset_name]
        dataset_progress.set_postfix_str(f"{dataset_name} -> {sample_plan[dataset_name]} items", refresh=False)
        raw_rows = load_raw_rows(spec)
        if spec.kind == "livecodebench":
            stratify_field = "difficulty"
        else:
            stratify_field = None

        sampled_rows = sample_rows(
            raw_rows,
            n=sample_plan[dataset_name],
            seed=args.seed + sum(ord(ch) for ch in dataset_name),
            stratify_field=stratify_field,
        )

        for idx, item in enumerate(sampled_rows):
            sampled_items.append(
                {
                    "dataset": dataset_name,
                    "sample_index": idx,
                    "prompt": build_eval_prompt(spec, item),
                    "raw_item": item,
                }
            )
    dataset_progress.close()

    write_jsonl(output_dir / "sampled_items.jsonl", sampled_items)

    sampling_config = {
        "datasets": args.datasets,
        "profile": args.profile,
        "sample_plan": sample_plan,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "num_items": len(sampled_items),
    }
    (output_dir / "sampling_config.json").write_text(
        json.dumps(sampling_config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(sampled_items)} sampled items to {output_dir / 'sampled_items.jsonl'}")


if __name__ == "__main__":
    main()
