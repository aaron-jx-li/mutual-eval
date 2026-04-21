#!/usr/bin/env python3
"""
Sample the fixed misc v1 item set (HLE strata + SimpleQA) into sampled_items.jsonl.

    python eval_static/sample_static_misc.py --config eval_static/config_static_misc.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval_static_misc import build_misc_sampled_items
from eval_static_math import load_env_file, resolve_env_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample static misc (HLE + SimpleQA) items.")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config; sampling settings under 'sampling'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override sampling.output_dir from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override sampling.seed.",
    )
    return parser.parse_args()


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


def apply_sampling_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        raise SystemExit("Error: --config is required (misc v1 strata are defined in YAML).")
    config = load_yaml_config(args.config)
    section = config.get("sampling", {})

    if args.seed is None:
        args.seed = int(section.get("seed", 0))
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")
    args.sampling_section = section
    return args


def build_output_dir(user_output_dir: str | None) -> Path:
    if user_output_dir:
        return Path(user_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "static_samples" / f"misc_{timestamp}"


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_sampling_defaults(args)

    section = dict(args.sampling_section)
    section["seed"] = args.seed
    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items, meta = build_misc_sampled_items(section)
    write_jsonl(output_dir / "sampled_items.jsonl", items)

    sampling_config = {
        "config": args.config,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "num_items": len(items),
        **meta,
    }
    (output_dir / "sampling_config.json").write_text(json.dumps(sampling_config, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {output_dir / 'sampled_items.jsonl'}")


if __name__ == "__main__":
    main()
