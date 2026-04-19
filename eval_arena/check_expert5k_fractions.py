#!/usr/bin/env python3
"""
Informal test: print category/label distributions in arena-expert-5k.

No LLM calls — just aggregates whatever label fields exist in the dataset.

Usage:
  python eval_arena/check_expert5k_math_fraction.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


CANDIDATE_LABEL_FIELDS = [
    "category", "categories", "topic", "domain", "subject",
    "task_type", "is_code", "is_math", "language",
    "occupational_tags", "judge_label", "human_label", "winner",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lmarena-ai/arena-expert-5k")
    p.add_argument("--split", default="train")
    p.add_argument("--top-n", type=int, default=30, help="Max values to show per field.")
    return p.parse_args()


def fmt_counter(counter: Counter, total: int, top_n: int) -> None:
    for val, cnt in counter.most_common(top_n):
        bar = "#" * int(40 * cnt / total)
        print(f"    {str(val):<40s}  {cnt:5d}  ({100*cnt/total:5.1f}%)  {bar}")
    if len(counter) > top_n:
        print(f"    ... ({len(counter) - top_n} more values)")


def main() -> None:
    args = parse_args()

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    rows = [dict(r) for r in ds]
    n = len(rows)
    print(f"  Total rows: {n}\n")

    # Discover all fields
    all_fields = list(rows[0].keys()) if rows else []
    print(f"Fields in dataset: {all_fields}\n")

    # Language breakdown first (always useful)
    lang_counter: Counter = Counter(str(r.get("language", "missing")) for r in rows)
    n_en = lang_counter.get("en", 0)
    print(f"{'='*60}")
    print(f"language  (total={n})")
    fmt_counter(lang_counter, n, args.top_n)
    print()

    # Print distribution for every candidate label field that exists
    for field in CANDIDATE_LABEL_FIELDS:
        if field == "language":
            continue
        if field not in all_fields:
            continue

        vals = [r.get(field) for r in rows]

        # occupational_tags: dict with True/None values — extract True keys only
        if field == "occupational_tags":
            flat_tags: list[str] = []
            for v in vals:
                if isinstance(v, dict):
                    flat_tags.extend(k for k, flag in v.items() if flag is True)
                elif isinstance(v, str) and v.startswith("{"):
                    try:
                        import ast
                        parsed = ast.literal_eval(v)
                        if isinstance(parsed, dict):
                            flat_tags.extend(k for k, flag in parsed.items() if flag is True)
                    except Exception:
                        pass
            counter = Counter(flat_tags)
            total = len(flat_tags) if flat_tags else 1
        # Flatten list-type fields (e.g. categories as a list)
        elif any(isinstance(v, list) for v in vals if v is not None):
            flat: list = []
            for v in vals:
                if isinstance(v, list):
                    flat.extend(str(x) for x in v)
                elif v is not None:
                    flat.append(str(v))
            counter = Counter(flat)
            total = len(flat)
        else:
            counter = Counter(str(v) for v in vals)
            total = n

        print(f"{'='*60}")
        print(f"{field}  (n={total})")
        fmt_counter(counter, total, args.top_n)
        print()

    # English-only breakdown for key fields
    en_rows = [r for r in rows if str(r.get("language", "")).strip().lower() == "en"]
    if en_rows and len(en_rows) < n:
        print(f"{'='*60}")
        print(f"English-only ({n_en} rows) — key fields")
        for field in CANDIDATE_LABEL_FIELDS:
            if field in ("language",) or field not in all_fields:
                continue
            vals = [r.get(field) for r in en_rows]
            if all(v is None for v in vals):
                continue
            if field == "occupational_tags":
                flat_tags = []
                for v in vals:
                    if isinstance(v, dict):
                        flat_tags.extend(k for k, flag in v.items() if flag is True)
                    elif isinstance(v, str) and v.startswith("{"):
                        try:
                            import ast
                            parsed = ast.literal_eval(v)
                            if isinstance(parsed, dict):
                                flat_tags.extend(k for k, flag in parsed.items() if flag is True)
                        except Exception:
                            pass
                counter = Counter(flat_tags)
            else:
                counter = Counter(str(v) for v in vals)
            if len(counter) <= 1:
                continue
            print(f"\n  {field}:")
            for val, cnt in counter.most_common(args.top_n):
                print(f"    {str(val):<40s}  {cnt:5d}  ({100*cnt/n_en:5.1f}%)")


if __name__ == "__main__":
    main()
