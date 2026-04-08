"""Shared argparse base parser for all robustness experiment scripts."""

from __future__ import annotations

import argparse


def base_parser(description: str = "IRT Robustness Experiment") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--static-csv", default=None, help="Static benchmark CSV (overrides --static-jsonl when provided)")
    p.add_argument("--arena-csv", default=None, help="Arena pairwise CSV (overrides --arena-reward-jsonl when provided)")
    p.add_argument("--static-jsonl", default="data/new/static_math_v0.jsonl", help="Static JSONL from static_eval pipeline (default)")
    p.add_argument("--arena-jsonl", default="data/new/arena_math_v0.jsonl", help="Arena reward JSONL from arena_eval pipeline (default)")
    p.add_argument("--out-dir", default="robustness/results", help="Directory for CSV/plot outputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Random seeds to use")
    p.add_argument("--num-epochs", type=int, default=2000, help="IRT training epochs")
    p.add_argument("--quiet", action="store_true", help="Suppress per-epoch training output")
    return p
