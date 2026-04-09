#!/usr/bin/env python3
"""
Compare a naive Bradley-Terry arena baseline against the repo's arena method.

This script is designed for the response-level arena JSONL files in `data/hf/`,
such as:

- `arena_math_v0.jsonl`
- `arena_coding_v0.jsonl`
- `arena_generic_v0.jsonl`

Each row is expected to contain at least:
    status, item_id, model_label, reward

Protocol
--------
1. Load one or more arena reward JSONL files.
2. Split model responses within each question, separately for each source, so
   every question remains present in both train and test while some
   `(question, model)` rows are held out.
3. Fit:
   - `naive_bt`: a global Bradley-Terry model on hard pairwise wins derived
     from reward differences.
   - `reward_irt`: the existing arena-only reward estimator from `ranking.py`
     (`fit_reward_irt`), which uses the full reward magnitude plus question
     difficulty/discrimination.
4. Evaluate all methods on held-out response pairs by pairwise accuracy against
   raw reward differences.

Notes
-----
- The Chatbot Arena paper fits BT on observed pairwise outcomes. In Eq. (7),
  the authors use inverse propensity weighting because their online system has
  non-uniform pair sampling probabilities. Those propensities are not available
  in these offline JSONL files, so this script uses the unweighted BT maximum-
  likelihood fit as a paper-inspired "naive BT" baseline.
- Reward ties are defined by `abs(reward_a - reward_b) <= --pair-tie-eps` and
  are excluded from BT training and from held-out pairwise accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranking import fit_reward_irt  # noqa: E402


DEFAULT_ARENA_JSONLS = [
    REPO_ROOT / "data" / "hf" / "arena_math_v0.jsonl",
    REPO_ROOT / "data" / "hf" / "arena_coding_v0.jsonl",
    REPO_ROOT / "data" / "hf" / "arena_generic_v0.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a naive Bradley-Terry arena baseline against the repo's "
            "arena-only reward estimator on one or more arena JSONL files."
        )
    )
    parser.add_argument(
        "--arena-jsonl",
        type=Path,
        nargs="+",
        default=DEFAULT_ARENA_JSONLS,
        help=(
            "One or more arena response JSONLs with reward scores. Defaults to "
            "the math, coding, and generic files in data/hf/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "reward_compare" / "arena_bt_vs_ours",
        help="Directory for rankings and the JSON summary.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of model responses to hold out within each question, per source.",
    )
    parser.add_argument(
        "--pair-tie-eps",
        type=float,
        default=0.0,
        help=(
            "Treat reward pairs with abs(delta) <= this threshold as ties. "
            "These pairs are excluded from held-out accuracy."
        ),
    )
    parser.add_argument(
        "--theta-tie-eps",
        type=float,
        default=0.0,
        help="Treat predicted theta differences within this threshold as ties.",
    )
    parser.add_argument(
        "--normalize-rewards",
        choices=("none", "global", "per_source"),
        default="per_source",
        help=(
            "How to normalize train rewards before fitting reward_irt. "
            "BT uses only reward ordering, so it is unaffected."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model-label subset.",
    )
    parser.add_argument(
        "--max-questions-per-source",
        type=int,
        default=None,
        help="Optional cap on the number of questions kept from each source.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2000,
        help="Training epochs for reward_irt.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for reward_irt.",
    )
    parser.add_argument(
        "--bt-num-epochs",
        type=int,
        default=2000,
        help="Training epochs for the naive BT baseline.",
    )
    parser.add_argument(
        "--bt-lr",
        type=float,
        default=0.05,
        help="Learning rate for the naive BT baseline.",
    )
    parser.add_argument(
        "--bt-reg-lambda",
        type=float,
        default=0,
        help="L2 regularization coefficient for the naive BT baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for question train/test splitting.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress logs.",
    )
    return parser.parse_args()


def source_tag_for_path(path: Path) -> str:
    return path.stem


def summarize_arena_df(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "questions": 0,
            "models": 0,
        }
    return {
        "rows": int(len(df)),
        "questions": int(df["question_id"].nunique()),
        "models": int(df["model_name"].nunique()),
    }


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_arena_reward_jsonl(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source_tag = source_tag_for_path(path)
        if not path.exists():
            raise SystemExit(f"Arena JSONL not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") != "ok" or record.get("reward") is None:
                    continue
                question_id = f"{source_tag}::{record['item_id']}"
                rows.append(
                    {
                        "source": source_tag,
                        "question_id": question_id,
                        "model_name": str(record["model_label"]),
                        "reward_raw": float(record["reward"]),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(
        subset=["source", "question_id", "model_name"],
        keep="last",
    ).reset_index(drop=True)
    return df


def restrict_questions_per_source(
    df: pd.DataFrame,
    *,
    max_questions_per_source: int | None,
) -> pd.DataFrame:
    if max_questions_per_source is None:
        return df

    allowed_question_ids: list[str] = []
    for source, group in df.groupby("source", sort=True):
        question_ids = sorted(group["question_id"].unique())[:max_questions_per_source]
        allowed_question_ids.extend(question_ids)
    return df[df["question_id"].isin(allowed_question_ids)].copy()


def filter_min_models_per_question(df: pd.DataFrame, *, min_models: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df.groupby("question_id")["model_name"].nunique()
    keep_questions = counts[counts >= min_models].index
    return df[df["question_id"].isin(keep_questions)].copy()


def split_models_within_question_by_source(
    df: pd.DataFrame,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_fraction <= 0:
        return df.copy(), df.iloc[0:0].copy()

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for source, source_group in df.groupby("source", sort=True):
        for question_id, question_group in source_group.groupby("question_id", sort=True):
            group = question_group.sort_values("model_name").reset_index(drop=True)
            n_rows = len(group)
            if n_rows < 4:
                raise SystemExit(
                    "Within-question holdout requires at least 4 model responses per "
                    f"question. Source={source}, question_id={question_id}, rows={n_rows}."
                )

            n_test = int(round(n_rows * test_fraction))
            n_test = max(2, min(n_test, n_rows - 2))

            order = list(range(n_rows))
            rng = random.Random(f"{seed}::{source}::{question_id}")
            rng.shuffle(order)
            test_positions = set(order[:n_test])

            train_frames.append(
                group.iloc[[idx for idx in range(n_rows) if idx not in test_positions]].copy()
            )
            test_frames.append(
                group.iloc[[idx for idx in range(n_rows) if idx in test_positions]].copy()
            )

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else df.iloc[0:0].copy()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else df.iloc[0:0].copy()
    return train_df, test_df


def apply_reward_normalization(
    train_df: pd.DataFrame,
    *,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = train_df.copy()
    stats: dict[str, dict[str, float]] = {}

    if out.empty:
        out["reward"] = pd.Series(dtype=float)
        return out, stats

    def _safe_std(series: pd.Series) -> float:
        std = float(series.std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            return 1.0
        return std

    if mode == "none":
        out["reward"] = out["reward_raw"].astype(float)
        for source, group in out.groupby("source", sort=True):
            stats[source] = {
                "mean": float(group["reward_raw"].mean()),
                "std": _safe_std(group["reward_raw"]),
            }
        return out, stats

    if mode == "global":
        mean = float(out["reward_raw"].mean())
        std = _safe_std(out["reward_raw"])
        out["reward"] = (out["reward_raw"] - mean) / std
        stats["global"] = {"mean": mean, "std": std}
        return out, stats

    out["reward"] = out["reward_raw"].astype(float)
    for source, group in out.groupby("source", sort=True):
        mean = float(group["reward_raw"].mean())
        std = _safe_std(group["reward_raw"])
        stats[source] = {"mean": mean, "std": std}
        mask = out["source"] == source
        out.loc[mask, "reward"] = (out.loc[mask, "reward_raw"] - mean) / std
    return out, stats


def build_hard_pairwise_df(
    df: pd.DataFrame,
    *,
    pair_tie_eps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_ties = 0
    decisive_pairs = 0
    question_counter = 0

    for question_id, group in df.groupby("question_id", sort=False):
        question_counter += 1
        group = group.sort_values("model_name").reset_index(drop=True)
        source = str(group.iloc[0]["source"])

        for idx1, idx2 in combinations(range(len(group)), 2):
            left = group.iloc[idx1]
            right = group.iloc[idx2]
            delta = float(left["reward_raw"]) - float(right["reward_raw"])

            if abs(delta) <= pair_tie_eps:
                skipped_ties += 1
                continue

            decisive_pairs += 1
            rows.append(
                {
                    "source": source,
                    "question_id": question_id,
                    "model_1": str(left["model_name"]),
                    "model_2": str(right["model_name"]),
                    "label": 0 if delta > 0 else 1,
                }
            )

    pairwise_df = pd.DataFrame(rows)
    stats = {
        "questions": question_counter,
        "rows": int(len(pairwise_df)),
        "decisive_pairs": decisive_pairs,
        "skipped_ties": skipped_ties,
    }
    return pairwise_df, stats


def fit_bradley_terry(
    pairwise_df: pd.DataFrame,
    *,
    num_epochs: int,
    lr: float,
    reg_lambda: float,
    verbose: bool,
) -> pd.DataFrame:
    if pairwise_df.empty:
        raise ValueError("Cannot fit Bradley-Terry with zero decisive pairwise rows.")

    df = pairwise_df.copy()
    df["model_1"] = df["model_1"].astype(str)
    df["model_2"] = df["model_2"].astype(str)
    df["y"] = (df["label"] == 0).astype(float)

    model_ids = pd.Index(
        pd.unique(pd.concat([df["model_1"], df["model_2"]], ignore_index=True)),
        name="model_name",
    )
    model_to_idx = {model_name: idx for idx, model_name in enumerate(model_ids)}

    df["m1_idx"] = df["model_1"].map(model_to_idx)
    df["m2_idx"] = df["model_2"].map(model_to_idx)

    device = _get_device()
    m1 = torch.tensor(df["m1_idx"].values, dtype=torch.long, device=device)
    m2 = torch.tensor(df["m2_idx"].values, dtype=torch.long, device=device)
    y = torch.tensor(df["y"].values, dtype=torch.float32, device=device)

    theta = nn.Embedding(len(model_ids), 1, device=device)
    nn.init.zeros_(theta.weight)

    optimizer = optim.Adam(theta.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        logits = theta(m1).squeeze(-1) - theta(m2).squeeze(-1)
        loss = bce(logits, y) + reg_lambda * theta.weight.pow(2).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            theta.weight.sub_(theta.weight.mean())

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(f"  BT epoch {epoch:5d} | loss = {loss.item():.4f}")

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    return (
        pd.DataFrame({"model_name": model_ids, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )


def theta_map_from_ranking(model_ranking: pd.DataFrame) -> dict[str, float]:
    return {
        str(row["model_name"]): float(row["theta"])
        for row in model_ranking.to_dict(orient="records")
    }


def evaluate_pairwise_accuracy(
    test_df: pd.DataFrame,
    theta_map: dict[str, float],
    *,
    pair_tie_eps: float,
    theta_tie_eps: float,
) -> dict[str, Any]:
    correct = 0
    total = 0
    skipped_reward_ties = 0
    skipped_theta_ties = 0
    skipped_missing_theta = 0
    question_counter = 0

    for question_id, group in test_df.groupby("question_id", sort=False):
        question_counter += 1
        rows = (
            group[["model_name", "reward_raw"]]
            .dropna()
            .sort_values("model_name")
            .to_dict(orient="records")
        )
        for left, right in combinations(rows, 2):
            reward_delta = float(left["reward_raw"]) - float(right["reward_raw"])
            if abs(reward_delta) <= pair_tie_eps:
                skipped_reward_ties += 1
                continue

            theta_left = theta_map.get(str(left["model_name"]))
            theta_right = theta_map.get(str(right["model_name"]))
            if theta_left is None or theta_right is None:
                skipped_missing_theta += 1
                continue

            theta_delta = theta_left - theta_right
            if abs(theta_delta) <= theta_tie_eps:
                skipped_theta_ties += 1
                continue

            pred_left_wins = theta_delta > 0
            gold_left_wins = reward_delta > 0
            correct += int(pred_left_wins == gold_left_wins)
            total += 1

    return {
        "accuracy": (correct / total) if total else None,
        "correct_pairs": correct,
        "total_pairs": total,
        "questions": question_counter,
        "skipped_reward_ties": skipped_reward_ties,
        "skipped_theta_ties": skipped_theta_ties,
        "skipped_missing_theta": skipped_missing_theta,
    }


def evaluate_pairwise_accuracy_by_source(
    test_df: pd.DataFrame,
    theta_map: dict[str, float],
    *,
    pair_tie_eps: float,
    theta_tie_eps: float,
) -> dict[str, dict[str, Any]]:
    return {
        str(source): evaluate_pairwise_accuracy(
            group,
            theta_map,
            pair_tie_eps=pair_tie_eps,
            theta_tie_eps=theta_tie_eps,
        )
        for source, group in test_df.groupby("source", sort=True)
    }


def compute_model_rank_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> dict[str, float | None]:
    merged = left[["model_name", "theta"]].merge(
        right[["model_name", "theta"]],
        on="model_name",
        suffixes=("_left", "_right"),
    )
    if len(merged) < 2:
        return {"spearman_rho": None, "p_value": None}
    rho, p_value = spearmanr(merged["theta_left"], merged["theta_right"])
    return {
        "spearman_rho": None if pd.isna(rho) else float(rho),
        "p_value": None if pd.isna(p_value) else float(p_value),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.test_fraction < 1.0):
        raise SystemExit("--test-fraction must satisfy 0 <= value < 1.")
    if args.max_questions_per_source is not None and args.max_questions_per_source < 1:
        raise SystemExit("--max-questions-per-source must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    arena_df = load_arena_reward_jsonl(args.arena_jsonl)
    if arena_df.empty:
        raise SystemExit("No usable arena reward rows were loaded.")

    if args.models:
        requested = set(args.models)
        arena_df = arena_df[arena_df["model_name"].isin(requested)].copy()
        if arena_df.empty:
            raise SystemExit("No rows remain after applying --models.")

    arena_df = restrict_questions_per_source(
        arena_df,
        max_questions_per_source=args.max_questions_per_source,
    )
    arena_df = filter_min_models_per_question(arena_df, min_models=4)
    if arena_df.empty:
        raise SystemExit("No arena questions remain with at least four models.")

    arena_train_df, arena_test_df = split_models_within_question_by_source(
        arena_df,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    if arena_train_df.empty:
        raise SystemExit("Arena train split is empty.")
    if arena_test_df.empty:
        raise SystemExit("Arena test split is empty.")
    if args.test_fraction > 0:
        all_questions = set(arena_df["question_id"].unique())
        train_questions = set(arena_train_df["question_id"].unique())
        test_questions = set(arena_test_df["question_id"].unique())
        if train_questions != all_questions or test_questions != all_questions:
            raise RuntimeError(
                "Within-question holdout invariant failed: train/test do not both "
                "cover every question."
            )

    arena_train_norm, norm_stats = apply_reward_normalization(
        arena_train_df,
        mode=args.normalize_rewards,
    )

    bt_train_pairs, bt_pair_stats = build_hard_pairwise_df(
        arena_train_df,
        pair_tie_eps=args.pair_tie_eps,
    )
    if bt_train_pairs.empty:
        raise SystemExit(
            "Naive BT training pairs are empty. Consider lowering --pair-tie-eps "
            "or using more questions."
        )

    methods: dict[str, dict[str, Any]] = {}

    bt_ranking = fit_bradley_terry(
        bt_train_pairs,
        num_epochs=args.bt_num_epochs,
        lr=args.bt_lr,
        reg_lambda=args.bt_reg_lambda,
        verbose=not args.quiet,
    )
    bt_theta = theta_map_from_ranking(bt_ranking)
    methods["naive_bt"] = {
        "train_rows": int(len(bt_train_pairs)),
        "train_questions": int(arena_train_df["question_id"].nunique()),
        "model_ranking": bt_ranking.to_dict(orient="records"),
        "test": {
            "overall": evaluate_pairwise_accuracy(
                arena_test_df,
                bt_theta,
                pair_tie_eps=args.pair_tie_eps,
                theta_tie_eps=args.theta_tie_eps,
            ),
            "by_source": evaluate_pairwise_accuracy_by_source(
                arena_test_df,
                bt_theta,
                pair_tie_eps=args.pair_tie_eps,
                theta_tie_eps=args.theta_tie_eps,
            ),
        },
        "train_pairwise_stats": bt_pair_stats,
    }

    reward_irt_ranking, _reward_irt_question_params = fit_reward_irt(
        arena_train_norm[["model_name", "question_id", "reward"]],
        num_epochs=args.num_epochs,
        lr=args.lr,
        verbose=not args.quiet,
    )
    reward_irt_theta = theta_map_from_ranking(reward_irt_ranking)
    methods["reward_irt"] = {
        "train_rows": int(len(arena_train_norm)),
        "train_questions": int(arena_train_norm["question_id"].nunique()),
        "model_ranking": reward_irt_ranking.to_dict(orient="records"),
        "test": {
            "overall": evaluate_pairwise_accuracy(
                arena_test_df,
                reward_irt_theta,
                pair_tie_eps=args.pair_tie_eps,
                theta_tie_eps=args.theta_tie_eps,
            ),
            "by_source": evaluate_pairwise_accuracy_by_source(
                arena_test_df,
                reward_irt_theta,
                pair_tie_eps=args.pair_tie_eps,
                theta_tie_eps=args.theta_tie_eps,
            ),
        },
    }

    comparisons: dict[str, Any] = {}
    if "naive_bt" in methods and "reward_irt" in methods:
        comparisons["naive_bt_vs_reward_irt_rank_corr"] = compute_model_rank_correlation(
            pd.DataFrame(methods["naive_bt"]["model_ranking"]),
            pd.DataFrame(methods["reward_irt"]["model_ranking"]),
        )

    per_source_counts = {
        str(source): summarize_arena_df(group)
        for source, group in arena_df.groupby("source", sort=True)
    }
    train_source_counts = {
        str(source): summarize_arena_df(group)
        for source, group in arena_train_df.groupby("source", sort=True)
    }
    test_source_counts = {
        str(source): summarize_arena_df(group)
        for source, group in arena_test_df.groupby("source", sort=True)
    }
    question_model_counts = dict(
        Counter(arena_df.groupby("question_id")["model_name"].nunique())
    )
    train_question_model_counts = dict(
        Counter(arena_train_df.groupby("question_id")["model_name"].nunique())
    )
    test_question_model_counts = dict(
        Counter(arena_test_df.groupby("question_id")["model_name"].nunique())
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "arena_jsonl": [str(path) for path in args.arena_jsonl],
            "output_dir": str(args.output_dir),
            "test_fraction": args.test_fraction,
            "pair_tie_eps": args.pair_tie_eps,
            "theta_tie_eps": args.theta_tie_eps,
            "normalize_rewards": args.normalize_rewards,
            "models": args.models,
            "max_questions_per_source": args.max_questions_per_source,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "bt_num_epochs": args.bt_num_epochs,
            "bt_lr": args.bt_lr,
            "bt_reg_lambda": args.bt_reg_lambda,
            "seed": args.seed,
        },
        "notes": {
            "naive_bt": (
                "Unweighted Bradley-Terry MLE on reward-derived hard pairwise "
                "wins/losses. Eq. (7) propensity weighting from the Arena paper "
                "is not applied because the offline JSONLs do not include pair "
                "sampling probabilities."
            )
        },
        "normalization": norm_stats,
        "data": {
            "all": summarize_arena_df(arena_df),
            "train": summarize_arena_df(arena_train_df),
            "test": summarize_arena_df(arena_test_df),
            "per_source_all": per_source_counts,
            "per_source_train": train_source_counts,
            "per_source_test": test_source_counts,
            "question_model_counts": question_model_counts,
            "train_question_model_counts": train_question_model_counts,
            "test_question_model_counts": test_question_model_counts,
            "bt_train_pairwise": bt_pair_stats,
        },
        "methods": methods,
        "comparisons": comparisons,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for method_name, payload in methods.items():
        ranking_path = args.output_dir / f"{method_name}_model_ranking.csv"
        pd.DataFrame(payload["model_ranking"]).to_csv(ranking_path, index=False)

    print("\nHeld-out arena evaluation")
    print("-" * 96)
    print(
        f"{'method':18} {'overall_acc':>12} {'test_pairs':>12} "
        f"{'train_q':>8} {'train_rows':>10}"
    )

    def fmt(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    for method_name, payload in methods.items():
        overall = payload["test"]["overall"]
        print(
            f"{method_name:18} {fmt(overall['accuracy']):>12} "
            f"{overall['total_pairs']:12d} {payload['train_questions']:8d} "
            f"{payload['train_rows']:10d}"
        )
    print("-" * 96)

    if arena_test_df["source"].nunique() > 1:
        print("\nHeld-out accuracy by source")
        print("-" * 96)
        print(f"{'method':18} {'source':24} {'accuracy':>12} {'test_pairs':>12}")
        for method_name, payload in methods.items():
            for source, metrics in payload["test"]["by_source"].items():
                print(
                    f"{method_name:18} {source:24} {fmt(metrics['accuracy']):>12} "
                    f"{metrics['total_pairs']:12d}"
                )
        print("-" * 96)

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
