#!/usr/bin/env python3
"""
Static coding evaluation helpers and dataset definitions.

This module currently provides the shared sampling utilities used by
`sample_static_coding.py`, including dataset metadata, prompt builders,
and fixed benchmark loading logic for the coding-domain static suite.

The execution-based evaluator is intentionally left for a follow-up pass;
for now, running this file directly exits with a short explanatory message.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from model_api_smoke_test import load_env_file


CODE_INSTRUCTION = (
    "Write a correct Python 3 solution. "
    "Return only executable Python code in a single ```python``` block, with no explanation."
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str
    default_pilot_samples: int
    default_paper_samples: int
    source_filename: str | None = None


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        "humaneval-plus",
        "evalplus/humanevalplus",
        None,
        "test",
        "humaneval",
        30,
        80,
    ),
    DatasetSpec(
        "mbpp-plus-sanitized",
        "evalplus/mbppplus",
        None,
        "test",
        "mbpp",
        30,
        140,
    ),
    DatasetSpec(
        "livecodebench-v6",
        "livecodebench/code_generation_lite",
        None,
        "test",
        "livecodebench",
        40,
        160,
        source_filename="test6.jsonl",
    ),
]

DATASET_LOOKUP = {spec.name: spec for spec in DATASET_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static coding evaluator placeholder.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Reserved for parity with the math evaluator.",
    )
    return parser.parse_args()


def build_sample_plan(selected_datasets: list[str], profile: str, uniform_n: int | None) -> dict[str, int]:
    plan: dict[str, int] = {}
    for name in selected_datasets:
        spec = DATASET_LOOKUP[name]
        if uniform_n is not None:
            plan[name] = uniform_n
        elif profile == "paper":
            plan[name] = spec.default_paper_samples
        else:
            plan[name] = spec.default_pilot_samples
    return plan


def _load_livecodebench_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    if not spec.source_filename:
        raise ValueError(f"Dataset '{spec.name}' requires a source filename.")
    path = hf_hub_download(
        repo_id=spec.hf_path,
        filename=spec.source_filename,
        repo_type="dataset",
    )
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_raw_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    if spec.kind == "livecodebench":
        return _load_livecodebench_rows(spec)

    if spec.hf_config is None:
        dataset = load_dataset(spec.hf_path)[spec.split]
    else:
        dataset = load_dataset(spec.hf_path, spec.hf_config)[spec.split]
    return [dict(row) for row in dataset]


def sample_rows(rows: list[dict], *, n: int, seed: int, stratify_field: str | None = None) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    if not stratify_field:
        return rng.sample(rows, n)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(stratify_field, "unknown"))].append(row)

    buckets = list(grouped.values())
    if not buckets:
        return rng.sample(rows, n)

    per_bucket = max(1, n // len(buckets))
    sampled: list[dict] = []
    leftovers: list[dict] = []
    for bucket in buckets:
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        take = min(per_bucket, len(shuffled))
        sampled.extend(shuffled[:take])
        leftovers.extend(shuffled[take:])

    if len(sampled) < n:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: n - len(sampled)])
    elif len(sampled) > n:
        rng.shuffle(sampled)
        sampled = sampled[:n]
    return sampled


def _decode_livecodebench_private_tests(raw_value: str) -> list[dict[str, Any]]:
    decoded = zlib.decompress(__import__("base64").b64decode(raw_value))
    parsed = pickle.loads(decoded)
    if isinstance(parsed, bytes):
        parsed = parsed.decode("utf-8")
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return parsed


def build_raw_question(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "humaneval":
        return (
            "Complete the following Python function.\n\n"
            f"```python\n{item['prompt'].rstrip()}\n```"
        )
    if dataset_spec.kind == "mbpp":
        return f"Write a Python function for this task:\n\n{item['prompt'].strip()}"
    if dataset_spec.kind == "livecodebench":
        starter_code = item.get("starter_code", "").rstrip()
        sections = [
            f"Title: {item.get('question_title', '').strip()}",
            "",
            item.get("question_content", "").strip(),
        ]
        if starter_code:
            sections.extend(
                [
                    "",
                    "Starter code:",
                    f"```python\n{starter_code}\n```",
                ]
            )
        public_cases = item.get("public_test_cases")
        if public_cases:
            sections.extend(
                [
                    "",
                    "Public tests:",
                    public_cases,
                ]
            )
        return "\n".join(part for part in sections if part is not None)
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def build_eval_prompt(dataset_spec: DatasetSpec, item: dict) -> str:
    return f"{build_raw_question(dataset_spec, item)}\n\n{CODE_INSTRUCTION}"


def build_gold_answer(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "humaneval":
        return f"{item['prompt'].rstrip()}\n{item['canonical_solution'].rstrip()}"
    if dataset_spec.kind == "mbpp":
        return item["code"]
    if dataset_spec.kind == "livecodebench":
        private_tests = _decode_livecodebench_private_tests(item["private_test_cases"])
        return json.dumps(
            {
                "public_tests": json.loads(item["public_test_cases"]),
                "private_test_count": len(private_tests),
            },
            ensure_ascii=False,
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def get_item_metadata(dataset_spec: DatasetSpec, item: dict) -> dict[str, Any]:
    if dataset_spec.kind == "humaneval":
        return {"level": "function", "subject": "python_function_synthesis"}
    if dataset_spec.kind == "mbpp":
        return {"level": "basic", "subject": "python_programming"}
    if dataset_spec.kind == "livecodebench":
        return {
            "level": item.get("difficulty") or "unknown",
            "subject": item.get("platform") or "competitive_programming",
        }
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parse_args()
    raise SystemExit(
        "Sampling helpers are implemented in this module, but the execution-based "
        "coding evaluator is not wired up yet. Use `python eval_static/sample_static_coding.py ...` for now."
    )


if __name__ == "__main__":
    main()
