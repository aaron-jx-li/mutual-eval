"""Shared argparse base parser for all robustness experiment scripts."""

from __future__ import annotations

import argparse


def base_parser(description: str = "IRT Robustness Experiment") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--static-csv", default="data/static_10_models.csv", help="Path to static benchmark CSV")
    p.add_argument("--arena-csv", default="data/pairwise_results_900.csv", help="Path to arena pairwise CSV")
    p.add_argument("--out-dir", default="robustness/results", help="Directory for CSV/plot outputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Random seeds to use")
    p.add_argument("--num-epochs", type=int, default=2000, help="IRT training epochs")
    p.add_argument("--quiet", action="store_true", help="Suppress per-epoch training output")
    return p
