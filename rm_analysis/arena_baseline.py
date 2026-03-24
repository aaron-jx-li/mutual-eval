#!/usr/bin/env python3
"""
Compute Arena baselines and reward-aware IRT accuracy against human labels.

This script reports three metrics:
1. A ranking-only baseline built from human pairwise win rates.
2. A reward-only global ranking baseline built from observed reward values.
3. An arena-only reward-distilled pairwise IRT model fit from reward scores.

For hard-label evaluation, human `tie` and `both_bad` examples are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RANKING_DIR = REPO_ROOT / "ranking"
if str(RANKING_DIR) not in sys.path:
    sys.path.insert(0, str(RANKING_DIR))

from rank_rm import fit_joint_reward_distilled_irt


VALID_LABELS = {"model_a", "model_b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Arena ranking-only and reward-aware IRT accuracies.",
    )
    parser.add_argument(
        "--human-data",
        type=Path,
        default=REPO_ROOT / "data" / "arena_math_900.json",
        help="Path to the Arena JSON file with human labels.",
    )
    parser.add_argument(
        "--reward-data",
        type=Path,
        default=REPO_ROOT / "results" / "arena_900" / "judge_human_rm.json",
        help="Path to the JSON file with pairwise rewards.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5000,
        help="Training epochs for reward-aware IRT.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Adam learning rate for reward-aware IRT.",
    )
    parser.add_argument(
        "--lambda-arena",
        type=float,
        default=1.0,
        help="Weight for arena soft-pairwise loss.",
    )
    parser.add_argument(
        "--lambda-bb",
        type=float,
        default=0.3,
        help="Weight for both-bad anchoring loss.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1e-4,
        help="L2 regularization coefficient.",
    )
    parser.add_argument(
        "--both-bad-threshold",
        type=float,
        default=-1.0,
        help="Threshold tau for identifying reward-based both-bad pairs.",
    )
    parser.add_argument(
        "--both-bad-use-raw",
        action="store_true",
        help="Threshold both-bad pairs on raw rewards instead of z-scores.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress logs.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}.")
    return data


def build_model_ranking(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    wins: Counter[str] = Counter()
    losses: Counter[str] = Counter()
    models: set[str] = set()

    for row in records:
        label = row.get("human_label")
        if label not in VALID_LABELS:
            continue

        model_a = row["model_a"]
        model_b = row["model_b"]
        models.update((model_a, model_b))

        if label == "model_a":
            wins[model_a] += 1
            losses[model_b] += 1
        else:
            wins[model_b] += 1
            losses[model_a] += 1

    ranking_rows: list[dict[str, Any]] = []
    for model in models:
        total = wins[model] + losses[model]
        win_rate = wins[model] / total if total else 0.0
        ranking_rows.append(
            {
                "model": model,
                "wins": wins[model],
                "losses": losses[model],
                "total": total,
                "win_rate": win_rate,
            }
        )

    ranking_rows.sort(
        key=lambda row: (-row["win_rate"], -row["wins"], row["losses"], row["model"]),
    )
    rank_by_model = {row["model"]: idx for idx, row in enumerate(ranking_rows)}
    return ranking_rows, rank_by_model


def compute_ranking_only_accuracy(
    records: list[dict[str, Any]],
    rank_by_model: dict[str, int],
) -> tuple[float, int, int]:
    correct = 0
    total = 0

    for row in records:
        label = row.get("human_label")
        if label not in VALID_LABELS:
            continue

        model_a = row["model_a"]
        model_b = row["model_b"]
        if model_a not in rank_by_model or model_b not in rank_by_model:
            continue

        predicted = "model_a" if rank_by_model[model_a] < rank_by_model[model_b] else "model_b"
        correct += int(predicted == label)
        total += 1

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total


def build_reward_only_ranking(
    reward_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    reward_sums: Counter[str] = Counter()
    reward_counts: Counter[str] = Counter()
    models: set[str] = set()

    for row in reward_records:
        reward_a = row.get("reward_a")
        reward_b = row.get("reward_b")
        model_a = row.get("model_a")
        model_b = row.get("model_b")
        if reward_a is None or reward_b is None or model_a is None or model_b is None:
            continue

        model_a = str(model_a)
        model_b = str(model_b)
        reward_a = float(reward_a)
        reward_b = float(reward_b)
        models.update((model_a, model_b))

        reward_sums[model_a] += reward_a
        reward_sums[model_b] += reward_b
        reward_counts[model_a] += 1
        reward_counts[model_b] += 1

    ranking_rows: list[dict[str, Any]] = []
    for model in models:
        count = reward_counts[model]
        mean_reward = reward_sums[model] / count if count else 0.0
        ranking_rows.append(
            {
                "model": model,
                "mean_reward": mean_reward,
                "reward_sum": reward_sums[model],
                "count": count,
            }
        )

    ranking_rows.sort(
        key=lambda row: (-row["mean_reward"], -row["count"], row["model"]),
    )
    rank_by_model = {row["model"]: idx for idx, row in enumerate(ranking_rows)}
    return ranking_rows, rank_by_model


def compute_reward_only_accuracy(
    human_records: list[dict[str, Any]],
    rank_by_model: dict[str, int],
) -> tuple[float, int, int]:
    correct = 0
    total = 0

    for row in human_records:
        label = row.get("human_label")
        if label not in VALID_LABELS:
            continue

        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        if model_a not in rank_by_model or model_b not in rank_by_model:
            continue

        predicted = "model_a" if rank_by_model[model_a] < rank_by_model[model_b] else "model_b"
        correct += int(predicted == label)
        total += 1

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total


def build_reward_pairwise_df(
    reward_records: list[dict[str, Any]],
    *,
    both_bad_threshold: float,
    both_bad_use_zscore: bool,
) -> pd.DataFrame:
    reward_values: list[float] = []
    for row in reward_records:
        if row.get("reward_a") is not None:
            reward_values.append(float(row["reward_a"]))
        if row.get("reward_b") is not None:
            reward_values.append(float(row["reward_b"]))

    if not reward_values:
        return pd.DataFrame()

    reward_array = np.asarray(reward_values, dtype=float)
    reward_mean = float(reward_array.mean())
    reward_std = float(reward_array.std(ddof=0))
    if not math.isfinite(reward_std) or reward_std < 1e-8:
        reward_std = 1.0

    rows: list[dict[str, Any]] = []
    for row in reward_records:
        reward_a = row.get("reward_a")
        reward_b = row.get("reward_b")
        if reward_a is None or reward_b is None:
            continue

        reward_a = float(reward_a)
        reward_b = float(reward_b)
        z_a = (reward_a - reward_mean) / reward_std
        z_b = (reward_b - reward_mean) / reward_std
        soft_pref = 1.0 / (1.0 + math.exp(-(z_a - z_b)))

        score_a = z_a if both_bad_use_zscore else reward_a
        score_b = z_b if both_bad_use_zscore else reward_b
        rows.append(
            {
                "source": "judge_human_rm",
                "benchmark": "Arena",
                "question_id": str(row["id"]),
                "model_1": str(row["model_a"]),
                "model_2": str(row["model_b"]),
                "reward_raw_1": reward_a,
                "reward_raw_2": reward_b,
                "reward_z_1": z_a,
                "reward_z_2": z_b,
                "target_prob": soft_pref,
                "both_bad": bool(score_a < both_bad_threshold and score_b < both_bad_threshold),
            }
        )
    return pd.DataFrame(rows)


def compute_reward_irt_hard_accuracy(
    human_records: list[dict[str, Any]],
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    learned_gamma: float,
) -> tuple[float, int, int]:
    theta_map = model_params.set_index("model_name")["theta"]
    question_map = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]

    correct = 0
    total = 0

    for row in human_records:
        label = row.get("human_label")
        if label not in VALID_LABELS:
            continue

        question_id = str(row["id"])
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        if (
            question_id not in question_map.index
            or model_a not in theta_map.index
            or model_b not in theta_map.index
        ):
            continue

        theta_1 = float(theta_map[model_a])
        theta_2 = float(theta_map[model_b])
        difficulty_b = float(question_map.loc[question_id, "difficulty_b"])
        a_q = float(question_map.loc[question_id, "discrimination_exp_k"])

        pi_1 = 1.0 / (1.0 + math.exp(-(theta_1 - difficulty_b)))
        pi_2 = 1.0 / (1.0 + math.exp(-(theta_2 - difficulty_b)))
        logit = learned_gamma * a_q * (pi_1 - pi_2)
        pred_prob = 1.0 / (1.0 + math.exp(-logit))
        if math.isclose(pred_prob, 0.5, abs_tol=1e-12):
            continue

        predicted = "model_a" if pred_prob > 0.5 else "model_b"
        correct += int(predicted == label)
        total += 1

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total


def main() -> None:
    args = parse_args()

    human_records = load_records(args.human_data)
    reward_records = load_records(args.reward_data)

    ranking_rows, rank_by_model = build_model_ranking(human_records)
    ranking_accuracy, ranking_correct, ranking_total = compute_ranking_only_accuracy(
        human_records,
        rank_by_model,
    )
    reward_ranking_rows, reward_rank_by_model = build_reward_only_ranking(reward_records)
    reward_only_accuracy, reward_only_correct, reward_only_total = compute_reward_only_accuracy(
        human_records,
        reward_rank_by_model,
    )

    pairwise_df = build_reward_pairwise_df(
        reward_records,
        both_bad_threshold=args.both_bad_threshold,
        both_bad_use_zscore=not args.both_bad_use_raw,
    )
    if pairwise_df.empty:
        raise SystemExit("No usable reward rows found in reward data.")

    model_params, question_params, fit_meta = fit_joint_reward_distilled_irt(
        static_df=None,
        pairwise_df=pairwise_df,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=0.0,
        lambda_arena=args.lambda_arena,
        lambda_bb=args.lambda_bb,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )
    irt_accuracy, irt_correct, irt_total = compute_reward_irt_hard_accuracy(
        human_records,
        model_params,
        question_params,
        learned_gamma=float(fit_meta["learned_gamma"]),
    )

    print(f"Loaded {len(human_records)} human-labeled rows from: {args.human_data}")
    print(f"Loaded {len(reward_records)} reward rows from: {args.reward_data}")
    print("Hard-label evaluation excludes human labels: tie, both_bad")
    print()
    print(
        f"Ranking-only hard-label accuracy: {ranking_accuracy:.6f} "
        f"({ranking_correct}/{ranking_total})"
    )
    print(
        f"Reward-only global ranking accuracy: {reward_only_accuracy:.6f} "
        f"({reward_only_correct}/{reward_only_total})"
    )
    print(
        f"Reward-aware pairwise IRT hard-label accuracy: {irt_accuracy:.6f} "
        f"({irt_correct}/{irt_total})"
    )
    print(
        f"Reward-aware pairwise rows used for fitting: {len(pairwise_df)} "
        f"(both_bad={int(pairwise_df['both_bad'].sum())})"
    )
    print()
    print("Reward-only model ranking by average reward:")
    for idx, row in enumerate(reward_ranking_rows[:15], start=1):
        print(
            f"{idx:>2}. {row['model']:<40} "
            f"mean_reward={row['mean_reward']:.4f} "
            f"count={row['count']:>4}"
        )
    print()
    print("Model ranking by pairwise win rate:")
    for idx, row in enumerate(ranking_rows, start=1):
        print(
            f"{idx:>2}. {row['model']:<40} "
            f"win_rate={row['win_rate']:.4f} "
            f"wins={row['wins']:>4} "
            f"losses={row['losses']:>4}"
        )


if __name__ == "__main__":
    main()
