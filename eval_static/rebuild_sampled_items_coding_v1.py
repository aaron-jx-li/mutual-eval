#!/usr/bin/env python3
"""
Rebuild sampled_items.jsonl for coding_v1 non-agentic questions.

Reads the existing responses_nonagentic.jsonl (which contains prompt/question/metadata
per response), collects unique (dataset, sample_index) items, then matches each
item's prompt back to its original raw dataset row so we can store the full
raw_item — mirroring the sampled_items.jsonl format used in coding_v0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval_static_coding import (
    DATASET_LOOKUP,
    build_eval_prompt,
    load_env_file,
    load_raw_rows,
    write_jsonl,
)

NONAGENTIC_DATASETS = {"mbpp-plus-sanitized", "livecodebench-v6"}

V1_DIR = Path("results/static_eval/coding_v1")
RESPONSES_FILE = V1_DIR / "responses_nonagentic.jsonl"
OUTPUT_FILE = V1_DIR / "sampled_items.jsonl"


def load_env() -> None:
    here = Path(__file__).resolve()
    for candidate in [here.with_name(".env"), here.parent.parent / ".env"]:
        if candidate.exists():
            load_env_file(candidate)
            return


def collect_unique_items(responses_file: Path) -> list[dict]:
    """Return one representative row per unique (dataset, sample_index)."""
    seen: set[tuple[str, int]] = set()
    items: list[dict] = []
    with responses_file.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["dataset"], row["sample_index"])
            if key not in seen and row["dataset"] in NONAGENTIC_DATASETS:
                seen.add(key)
                items.append(row)
    items.sort(key=lambda r: (r["dataset"], r["sample_index"]))
    return items


def build_prompt_to_raw_map(dataset_name: str) -> dict[str, dict]:
    """Build a map from prompt string → raw_item for every row in a dataset."""
    spec = DATASET_LOOKUP[dataset_name]
    raw_rows = load_raw_rows(spec)
    return {build_eval_prompt(spec, row): row for row in raw_rows}


def main() -> None:
    load_env()

    unique_items = collect_unique_items(RESPONSES_FILE)
    print(f"Found {len(unique_items)} unique non-agentic items in responses_nonagentic.jsonl")

    # Build prompt→raw maps only for the datasets we need
    datasets_needed = {item["dataset"] for item in unique_items}
    prompt_maps: dict[str, dict[str, dict]] = {}
    for ds in sorted(datasets_needed):
        print(f"Loading raw rows for {ds} ...")
        prompt_maps[ds] = build_prompt_to_raw_map(ds)
        print(f"  {len(prompt_maps[ds])} rows loaded")

    sampled_items: list[dict] = []
    missing = 0
    for item in unique_items:
        ds = item["dataset"]
        prompt = item["prompt"]
        raw_item = prompt_maps[ds].get(prompt)
        if raw_item is None:
            print(f"WARNING: no raw_item match for {ds} sample_index={item['sample_index']}")
            missing += 1
            continue
        sampled_items.append(
            {
                "dataset": ds,
                "sample_index": item["sample_index"],
                "prompt": prompt,
                "raw_item": raw_item,
            }
        )

    print(f"\nMatched {len(sampled_items)}/{len(unique_items)} items ({missing} unmatched)")
    write_jsonl(OUTPUT_FILE, sampled_items)
    print(f"Wrote {len(sampled_items)} items to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
