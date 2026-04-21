#!/usr/bin/env python3
"""
Sample and save static math evaluation items.

This script builds a fixed sampled item file that can later be consumed by
`eval_static_math.py`, so selection and evaluation are separated.

Usage examples:
    python eval_static/sample_static_math.py --profile paper
    python eval_static/sample_static_math.py --profile paper --output-dir results/static_samples/full_v1
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

from eval_static_math import (
    DATASET_LOOKUP,
    DATASET_SPECS,
    build_eval_prompt,
    build_gold_answer,
    build_raw_question,
    build_sample_plan,
    get_item_metadata,
    load_env_file,
    load_jsonl,
    load_raw_rows,
    sample_rows,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and save a fixed static math item set.",
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
        help="Datasets to sample from. Defaults to all supported datasets.",
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
    parent_env = here.parent.parent / ".env"
    return parent_env


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

    # copy_from: datasets to copy verbatim from an existing sampled_items.jsonl
    copy_from = section.get("copy_from", {})
    args.copy_from_file = copy_from.get("file")
    args.copy_from_datasets = list(copy_from.get("datasets", []))
    args.copy_from_responses_file = copy_from.get("responses_file")

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

    # --- Copy items from an existing sampled_items.jsonl (e.g. v0) ---
    copied_items: list[dict[str, Any]] = []
    if args.copy_from_file and args.copy_from_datasets:
        copy_path = Path(args.copy_from_file)
        if not copy_path.exists():
            raise FileNotFoundError(f"copy_from file not found: {copy_path}")
        copy_ds = set(args.copy_from_datasets)
        all_v0 = load_jsonl(copy_path)
        copied_items = [item for item in all_v0 if item.get("dataset") in copy_ds]
        found_ds = {item["dataset"] for item in copied_items}
        missing = copy_ds - found_ds
        if missing:
            raise ValueError(f"copy_from datasets not found in {copy_path}: {missing}")
        print(
            f"Copied {len(copied_items)} items from {copy_path} "
            f"({', '.join(sorted(found_ds))})"
        )

    # --- Sample fresh datasets ---
    sample_plan = build_sample_plan(args.datasets, args.profile, args.samples_per_dataset)

    fresh_items: list[dict[str, Any]] = []
    dataset_progress = tqdm(args.datasets, desc="Sampling datasets", unit="dataset")
    for dataset_name in dataset_progress:
        spec = DATASET_LOOKUP[dataset_name]
        dataset_progress.set_postfix_str(f"{dataset_name} -> {sample_plan[dataset_name]} items", refresh=False)
        raw_rows = load_raw_rows(spec)
        if spec.kind == "math":
            stratify_field = "level"
        elif spec.kind == "olympiad":
            stratify_field = "subfield"
        else:
            stratify_field = None

        sampled_rows = sample_rows(
            raw_rows,
            n=sample_plan[dataset_name],
            seed=args.seed + sum(ord(ch) for ch in dataset_name),
            stratify_field=stratify_field,
        )

        for idx, item in enumerate(sampled_rows):
            metadata = get_item_metadata(spec, item)
            fresh_items.append(
                {
                    "dataset": dataset_name,
                    "dataset_kind": spec.kind,
                    "sample_index": idx,
                    "question": build_raw_question(spec, item),
                    "prompt": build_eval_prompt(spec, item),
                    "gold_answer": build_gold_answer(spec, item),
                    **metadata,
                    "raw_item": item,
                }
            )
    dataset_progress.close()

    # Copied items come first so aime/olympiad appear before hle-math
    all_items = copied_items + fresh_items
    write_jsonl(output_dir / "sampled_items.jsonl", all_items)

    # --- Copy responses for --resume support ---
    # Pre-populate responses.jsonl with already-evaluated records so that
    # eval_static_math.py --resume skips them without re-running.
    if args.copy_from_responses_file and args.copy_from_datasets:
        resp_path = Path(args.copy_from_responses_file)
        if not resp_path.exists():
            print(f"Warning: copy_from responses_file not found, skipping: {resp_path}")
        else:
            copy_ds = set(args.copy_from_datasets)
            all_responses = load_jsonl(resp_path)
            relevant = [r for r in all_responses if r.get("dataset") in copy_ds]
            write_jsonl(output_dir / "responses.jsonl", relevant)
            print(f"Copied {len(relevant)} response records to {output_dir / 'responses.jsonl'} (for --resume)")

    sampling_config = {
        "datasets_sampled": args.datasets,
        "datasets_copied": args.copy_from_datasets,
        "copy_from_file": str(args.copy_from_file) if args.copy_from_file else None,
        "profile": args.profile,
        "sample_plan": sample_plan,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "num_items": len(all_items),
        "num_copied": len(copied_items),
        "num_fresh": len(fresh_items),
    }
    (output_dir / "sampling_config.json").write_text(
        json.dumps(sampling_config, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved {len(all_items)} items to {output_dir / 'sampled_items.jsonl'} "
        f"({len(copied_items)} copied, {len(fresh_items)} freshly sampled)"
    )


if __name__ == "__main__":
    main()
