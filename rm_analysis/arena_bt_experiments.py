#!/usr/bin/env python3
"""
Arena-only experiments comparing reward-distilled IRT against Bradley-Terry.

This script focuses on out-of-sample evaluation rather than in-sample fit:

1. Question generalization with prompt-level support:
   - Hold out a subset of Arena questions.
   - Bradley-Terry (BT) trains on decisive human pairs from the remaining questions.
   - Reward-distilled IRT trains on reward responses from train questions plus
     `k` support-model rewards on each held-out question.
   - Evaluation is on the held-out human pair for each test question.

2. Leave-model-pair-out generalization:
   - Hold out entire model-pair identities from BT training.
   - Reward-distilled IRT does not train on the held-out pair's own reward
     responses on the evaluation question, but may use other models on that
     question to estimate question difficulty.

3. Sample efficiency:
   - Vary the number of support models on held-out questions.
   - Vary the number of training questions available.

4. Stability:
   - Bootstrap low-data training questions and compare how stable the induced
     rankings are for BT vs reward-distilled IRT.

The evaluation target can come from either:
- human pair labels (`--human-data`, original protocol), or
- reward-derived pairwise targets when only reward values are available.

Outputs:
  - Raw CSVs for each experiment
  - Aggregated summary tables
  - Plot files (unless --no-plots)

Example:
  python rm_analysis/arena_bt_experiments.py \
      --arena-reward-jsonl results/arena_eval/math_v0/responses.jsonl \
      --output-dir results/arena_bt_experiments/math_v0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RANKING_DIR = REPO_ROOT / "ranking"
if str(RANKING_DIR) not in sys.path:
    sys.path.insert(0, str(RANKING_DIR))

from rank_rm import build_soft_pairwise_targets, fit_joint_reward_distilled_irt

DECISIVE_LABELS = {"model_a", "model_b"}
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_SUPPORT_SIZES = [2, 4, 6, 8]
DEFAULT_TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
DEFAULT_ECE_BINS = 10


def progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool,
) -> Iterable[Any]:
    if not enabled:
        return iterable

    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            return iterable

    if total <= 0:
        return iterable

    def generator() -> Iterable[Any]:
        bar_width = 24
        for idx, item in enumerate(iterable, start=1):
            filled = min(bar_width, int(round(bar_width * idx / total)))
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\r[{bar}] {idx}/{total} {desc}",
                end="",
                file=sys.stderr,
                flush=True,
            )
            yield item
        print(file=sys.stderr, flush=True)

    return generator()


@dataclass(frozen=True)
class RewardIrtConfig:
    num_epochs: int
    lr: float
    lambda_arena: float
    lambda_bb: float
    reg_lambda: float
    both_bad_threshold: float
    both_bad_use_zscore: bool
    verbose: bool


@dataclass(frozen=True)
class BtConfig:
    num_epochs: int
    lr: float
    reg_lambda: float
    verbose: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run arena-only BT vs reward-distilled IRT experiments.",
    )
    parser.add_argument(
        "--human-data",
        type=Path,
        default=None,
        help="Optional JSON file containing Arena human pair labels.",
    )
    parser.add_argument(
        "--target-source",
        choices=["auto", "human", "reward"],
        default="auto",
        help=(
            "Source of held-out pair targets. "
            "'human' uses human Arena labels, 'reward' derives pair targets from reward differences, "
            "'auto' uses human when --human-data is provided and reward otherwise."
        ),
    )
    parser.add_argument(
        "--arena-reward-jsonl",
        nargs="+",
        type=Path,
        required=True,
        help="One or more arena_eval responses.jsonl files with reward scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "arena_bt_experiments" / "default",
        help="Directory for result tables and plots.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2000,
        help="Training epochs for both BT and reward-distilled IRT.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Adam learning rate for both BT and reward-distilled IRT.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1e-4,
        help="L2 regularization for both BT and reward-distilled IRT.",
    )
    parser.add_argument(
        "--lambda-arena",
        type=float,
        default=1.0,
        help="Arena loss weight for reward-distilled IRT.",
    )
    parser.add_argument(
        "--lambda-bb",
        type=float,
        default=0.3,
        help="Both-bad loss weight for reward-distilled IRT.",
    )
    parser.add_argument(
        "--both-bad-threshold",
        type=float,
        default=-1.0,
        help="Threshold used to tag both-bad reward pairs.",
    )
    parser.add_argument(
        "--both-bad-mode",
        choices=["zscore", "raw"],
        default="zscore",
        help="Whether both-bad thresholding uses reward z-scores or raw rewards.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of eligible questions used as held-out test questions.",
    )
    parser.add_argument(
        "--pair-holdout-fraction",
        type=float,
        default=0.25,
        help="Fraction of unique model-pair identities held out in pair generalization.",
    )
    parser.add_argument(
        "--support-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SUPPORT_SIZES,
        help="Support-model counts for held-out question experiments; must be >= 2.",
    )
    parser.add_argument(
        "--train-fractions",
        nargs="+",
        type=float,
        default=DEFAULT_TRAIN_FRACTIONS,
        help="Fractions of training questions used for learning-curve experiments.",
    )
    parser.add_argument(
        "--learning-curve-support-size",
        type=int,
        default=4,
        help="Support size used during global train-fraction learning curves.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=50,
        help="Bootstrap replicates for low-data stability analysis.",
    )
    parser.add_argument(
        "--bootstrap-train-fraction",
        type=float,
        default=0.1,
        help="Training-question fraction used during stability bootstrapping.",
    )
    parser.add_argument(
        "--bootstrap-support-size",
        type=int,
        default=4,
        help="Support size used during stability bootstrapping.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k cutoff for rank retention in stability analysis.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=DEFAULT_ECE_BINS,
        help="Number of bins used to compute expected calibration error (ECE).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds for repeated experiments.",
    )
    parser.add_argument(
        "--max-test-questions",
        type=int,
        default=None,
        help="Optional cap on held-out questions per seed for faster sweeps.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress logs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable high-level progress bars for experiment sweeps.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_epochs < 1:
        raise SystemExit("--num-epochs must be at least 1.")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive.")
    if args.reg_lambda < 0:
        raise SystemExit("--reg-lambda must be non-negative.")
    if not 0 < args.test_fraction < 1:
        raise SystemExit("--test-fraction must be in (0, 1).")
    if not 0 < args.pair_holdout_fraction < 1:
        raise SystemExit("--pair-holdout-fraction must be in (0, 1).")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be at least 1.")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")
    if args.ece_bins < 2:
        raise SystemExit("--ece-bins must be at least 2.")
    if any(size < 2 for size in args.support_sizes):
        raise SystemExit("All --support-sizes must be at least 2 for pairwise distillation.")
    if args.learning_curve_support_size < 2:
        raise SystemExit("--learning-curve-support-size must be at least 2.")
    if args.bootstrap_support_size < 2:
        raise SystemExit("--bootstrap-support-size must be at least 2.")
    if any(fraction <= 0 or fraction > 1 for fraction in args.train_fractions):
        raise SystemExit("All --train-fractions must lie in (0, 1].")
    if not 0 < args.bootstrap_train_fraction <= 1:
        raise SystemExit("--bootstrap-train-fraction must lie in (0, 1].")


def load_human_pairs(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON array in {path}.")

    rows: list[dict[str, Any]] = []
    for record in raw:
        label = str(record.get("human_label", ""))
        rows.append(
            {
                "question_id": str(record["id"]),
                "model_1": str(record["model_a"]),
                "model_2": str(record["model_b"]),
                "label": label,
                "target": 1.0 if label == "model_a" else 0.0 if label == "model_b" else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    decisive = df[df["label"].isin(DECISIVE_LABELS)].copy().reset_index(drop=True)
    decisive["pair_key"] = decisive.apply(
        lambda row: canonical_pair_key(str(row["model_1"]), str(row["model_2"])),
        axis=1,
    )
    return decisive


def load_reward_responses(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Reward JSONL not found: {path}")
        source_tag = path.parent.name.strip() or path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") != "ok" or record.get("reward") is None:
                    continue
                rows.append(
                    {
                        "source": source_tag,
                        "benchmark": "Arena",
                        "question_id": str(record["item_id"]),
                        "model_name": str(record["model_label"]),
                        "reward_raw": float(record["reward"]),
                    }
                )

    reward_df = pd.DataFrame(rows)
    if reward_df.empty:
        raise SystemExit("No usable reward rows found in the provided arena_eval JSONL files.")

    reward_df = reward_df.drop_duplicates(
        subset=["question_id", "model_name"],
        keep="last",
    ).reset_index(drop=True)
    reward_mean = float(reward_df["reward_raw"].mean())
    reward_std = float(reward_df["reward_raw"].std(ddof=0))
    if not math.isfinite(reward_std) or reward_std < 1e-8:
        reward_std = 1.0
    reward_df["reward_z"] = (reward_df["reward_raw"] - reward_mean) / reward_std
    return reward_df


def build_reward_pair_targets(reward_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question_id, group in reward_df.groupby("question_id", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values("model_name").reset_index(drop=True)
        for idx1, idx2 in combinations(range(len(group)), 2):
            row_1 = group.iloc[idx1]
            row_2 = group.iloc[idx2]
            reward_gap = float(row_1["reward_z"] - row_2["reward_z"])
            target_prob = 1.0 / (1.0 + math.exp(-reward_gap))
            rows.append(
                {
                    "question_id": str(question_id),
                    "model_1": str(row_1["model_name"]),
                    "model_2": str(row_2["model_name"]),
                    "pair_key": canonical_pair_key(str(row_1["model_name"]), str(row_2["model_name"])),
                    "label": "model_1" if target_prob > 0.5 else "model_2" if target_prob < 0.5 else "tie",
                    "target": target_prob,
                    "target_hard": 1.0 if target_prob > 0.5 else 0.0 if target_prob < 0.5 else 0.5,
                    "reward_gap_z": reward_gap,
                    "reward_raw_1": float(row_1["reward_raw"]),
                    "reward_raw_2": float(row_2["reward_raw"]),
                }
            )
    return pd.DataFrame(rows)


def canonical_pair_key(model_1: str, model_2: str) -> str:
    left, right = sorted((model_1, model_2))
    return f"{left}__{right}"


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_bt_model(
    decisive_pairs: pd.DataFrame,
    *,
    num_epochs: int,
    lr: float,
    reg_lambda: float,
    verbose: bool,
) -> pd.DataFrame:
    if decisive_pairs.empty:
        raise ValueError("BT training requires at least one decisive pair.")

    train_df = decisive_pairs.copy().reset_index(drop=True)
    model_ids = pd.Index(
        pd.unique(pd.concat([train_df["model_1"], train_df["model_2"]], ignore_index=True)),
        name="model_name",
    )
    model_to_idx = {model: idx for idx, model in enumerate(model_ids)}
    train_df["m1_idx"] = train_df["model_1"].map(model_to_idx)
    train_df["m2_idx"] = train_df["model_2"].map(model_to_idx)

    device = _get_device()
    m1_idx = torch.tensor(train_df["m1_idx"].values, dtype=torch.long, device=device)
    m2_idx = torch.tensor(train_df["m2_idx"].values, dtype=torch.long, device=device)
    targets = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)

    theta = nn.Embedding(len(model_ids), 1, device=device)
    nn.init.zeros_(theta.weight)
    optimizer = optim.Adam(theta.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = theta(m1_idx).squeeze(-1) - theta(m2_idx).squeeze(-1)
        loss = bce(logits, targets) + reg_lambda * theta.weight.pow(2).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            theta.weight.sub_(theta.weight.mean())

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(f"  [BT] Epoch {epoch:5d} | loss={loss.item():.4f}")

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    return (
        pd.DataFrame({"model_name": model_ids, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )


def compute_binary_metrics(
    prediction_df: pd.DataFrame,
    *,
    method: str,
    experiment: str,
    seed: int,
    budget_name: str,
    budget_value: float | int,
    eval_group_count: int,
    skipped_count: int,
    ece_bins: int,
) -> dict[str, Any]:
    if prediction_df.empty:
        return {
            "method": method,
            "experiment": experiment,
            "seed": seed,
            "budget_name": budget_name,
            "budget_value": budget_value,
            "accuracy": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "n_eval": 0,
            "coverage": 0.0,
            "eval_group_count": int(eval_group_count),
            "skipped_count": int(skipped_count),
        }

    clipped = prediction_df["pred_prob"].clip(1e-6, 1 - 1e-6)
    target = prediction_df["target"]
    accuracy = float(((prediction_df["pred_prob"] >= 0.5) == (target >= 0.5)).mean())
    logloss = float(-(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)).mean())
    brier = float(((prediction_df["pred_prob"] - target) ** 2).mean())
    ece = compute_expected_calibration_error(
        prediction_df["pred_prob"].to_numpy(dtype=float),
        target.to_numpy(dtype=float),
        n_bins=ece_bins,
    )
    coverage = float(len(prediction_df) / eval_group_count) if eval_group_count else 0.0
    return {
        "method": method,
        "experiment": experiment,
        "seed": seed,
        "budget_name": budget_name,
        "budget_value": budget_value,
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "ece": ece,
        "n_eval": int(len(prediction_df)),
        "coverage": coverage,
        "eval_group_count": int(eval_group_count),
        "skipped_count": int(skipped_count),
    }


def compute_expected_calibration_error(
    pred_prob: np.ndarray,
    target: np.ndarray,
    *,
    n_bins: int,
) -> float:
    if pred_prob.size == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(pred_prob.size)
    for bin_idx in range(n_bins):
        left = bin_edges[bin_idx]
        right = bin_edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (pred_prob >= left) & (pred_prob <= right)
        else:
            mask = (pred_prob >= left) & (pred_prob < right)
        if not np.any(mask):
            continue
        bin_prob = pred_prob[mask]
        bin_target = target[mask]
        ece += float(mask.mean()) * abs(float(bin_prob.mean()) - float(bin_target.mean()))
    return ece


def predict_bt(eval_pairs: pd.DataFrame, model_params: pd.DataFrame) -> pd.DataFrame:
    theta_map = model_params.set_index("model_name")["theta"]
    rows: list[dict[str, Any]] = []
    for row in eval_pairs.itertuples(index=False):
        if row.model_1 not in theta_map.index or row.model_2 not in theta_map.index:
            continue
        logit = float(theta_map[row.model_1] - theta_map[row.model_2])
        pred_prob = 1.0 / (1.0 + math.exp(-logit))
        rows.append(
            {
                "question_id": row.question_id,
                "pair_key": row.pair_key,
                "model_1": row.model_1,
                "model_2": row.model_2,
                "target": float(row.target),
                "pred_prob": pred_prob,
            }
        )
    return pd.DataFrame(rows)


def fit_reward_irt_model(
    reward_train_df: pd.DataFrame,
    config: RewardIrtConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if reward_train_df.empty:
        raise ValueError("Reward-distilled IRT training requires reward responses.")

    pairwise_df = build_soft_pairwise_targets(
        reward_train_df,
        both_bad_threshold=config.both_bad_threshold,
        both_bad_use_zscore=config.both_bad_use_zscore,
    )
    if pairwise_df.empty:
        raise ValueError("Reward-distilled IRT needs at least one question with >= 2 reward responses.")

    model_params, question_params, fit_meta = fit_joint_reward_distilled_irt(
        static_df=None,
        pairwise_df=pairwise_df,
        reward_df=None,
        arena_mode="soft_pairwise",
        num_epochs=config.num_epochs,
        lr=config.lr,
        lambda_static=0.0,
        lambda_arena=config.lambda_arena,
        lambda_bb=config.lambda_bb,
        reg_lambda=config.reg_lambda,
        verbose=config.verbose,
    )
    return model_params, question_params, fit_meta, pairwise_df


def predict_reward_irt(
    eval_pairs: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    learned_gamma: float,
) -> pd.DataFrame:
    theta_map = model_params.set_index("model_name")["theta"]
    question_map = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]

    rows: list[dict[str, Any]] = []
    for row in eval_pairs.itertuples(index=False):
        if (
            row.model_1 not in theta_map.index
            or row.model_2 not in theta_map.index
            or row.question_id not in question_map.index
        ):
            continue

        theta_1 = float(theta_map[row.model_1])
        theta_2 = float(theta_map[row.model_2])
        difficulty_b = float(question_map.loc[row.question_id, "difficulty_b"])
        a_q = float(question_map.loc[row.question_id, "discrimination_exp_k"])
        pi_1 = 1.0 / (1.0 + math.exp(-(theta_1 - difficulty_b)))
        pi_2 = 1.0 / (1.0 + math.exp(-(theta_2 - difficulty_b)))
        logit = learned_gamma * a_q * (pi_1 - pi_2)
        pred_prob = 1.0 / (1.0 + math.exp(-logit))
        rows.append(
            {
                "question_id": row.question_id,
                "pair_key": row.pair_key,
                "model_1": row.model_1,
                "model_2": row.model_2,
                "target": float(row.target),
                "pred_prob": pred_prob,
            }
        )
    return pd.DataFrame(rows)


def subset_reward_by_questions(reward_df: pd.DataFrame, question_ids: Iterable[str]) -> pd.DataFrame:
    question_ids = set(question_ids)
    return reward_df[reward_df["question_id"].isin(question_ids)].copy().reset_index(drop=True)


def sample_train_questions(
    train_question_ids: list[str],
    *,
    train_fraction: float,
    rng: np.random.Generator,
) -> list[str]:
    if not train_question_ids:
        return []
    sample_size = max(1, int(round(train_fraction * len(train_question_ids))))
    sample_size = min(sample_size, len(train_question_ids))
    sampled = rng.choice(np.asarray(train_question_ids), size=sample_size, replace=False)
    return sorted(str(question_id) for question_id in sampled.tolist())


def build_question_generalization_split(
    human_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seed: int,
    support_size: int,
    test_fraction: float,
    max_test_questions: int | None = None,
    train_fraction: float = 1.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reward_models_by_q = reward_df.groupby("question_id")["model_name"].apply(set).to_dict()

    eligible_question_ids: list[str] = []
    for row in human_df.itertuples(index=False):
        available_models = reward_models_by_q.get(row.question_id, set())
        support_candidates = available_models.difference({row.model_1, row.model_2})
        if len(support_candidates) >= support_size:
            eligible_question_ids.append(str(row.question_id))

    eligible_question_ids = sorted(set(eligible_question_ids))
    if len(eligible_question_ids) < 2:
        raise ValueError(
            "Not enough eligible questions for question generalization. "
            "Need at least 2 questions with support candidates."
        )

    test_size = max(1, int(round(test_fraction * len(eligible_question_ids))))
    test_size = min(test_size, len(eligible_question_ids) - 1)
    test_question_ids = rng.choice(np.asarray(eligible_question_ids), size=test_size, replace=False).tolist()
    test_question_ids = [str(question_id) for question_id in test_question_ids]
    if max_test_questions is not None:
        test_question_ids = test_question_ids[:max_test_questions]
    test_question_set = set(test_question_ids)
    train_question_ids = sorted(set(eligible_question_ids) - test_question_set)
    sampled_train_questions = sample_train_questions(
        train_question_ids,
        train_fraction=train_fraction,
        rng=rng,
    )
    sampled_train_set = set(sampled_train_questions)

    support_rows: list[pd.DataFrame] = []
    support_map: dict[str, list[str]] = {}
    for question_id in test_question_ids:
        human_row = human_df[human_df["question_id"] == question_id].iloc[0]
        reward_group = reward_df[reward_df["question_id"] == question_id].copy()
        support_candidates = sorted(
            set(reward_group["model_name"]).difference({human_row["model_1"], human_row["model_2"]})
        )
        chosen_support = rng.choice(
            np.asarray(support_candidates),
            size=support_size,
            replace=False,
        ).tolist()
        support_map[question_id] = [str(model_name) for model_name in chosen_support]
        support_rows.append(reward_group[reward_group["model_name"].isin(chosen_support)])

    reward_train = subset_reward_by_questions(reward_df, sampled_train_set)
    if support_rows:
        reward_train = pd.concat([reward_train, *support_rows], ignore_index=True)
    reward_train = reward_train.drop_duplicates(
        subset=["question_id", "model_name"],
        keep="last",
    ).reset_index(drop=True)

    bt_train = human_df[human_df["question_id"].isin(sampled_train_set)].copy().reset_index(drop=True)
    eval_df = human_df[human_df["question_id"].isin(test_question_set)].copy().reset_index(drop=True)

    return {
        "bt_train": bt_train,
        "reward_train": reward_train,
        "eval_df": eval_df,
        "train_question_ids": sampled_train_questions,
        "all_train_question_ids": train_question_ids,
        "test_question_ids": test_question_ids,
        "support_map": support_map,
    }


def build_reward_only_question_generalization_split(
    pair_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seed: int,
    support_size: int,
    test_fraction: float,
    max_test_questions: int | None = None,
    train_fraction: float = 1.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    models_by_question = reward_df.groupby("question_id")["model_name"].apply(lambda col: sorted(set(col))).to_dict()
    eligible_question_ids = sorted(
        question_id
        for question_id, model_names in models_by_question.items()
        if len(model_names) >= support_size + 2
    )
    if len(eligible_question_ids) < 2:
        raise ValueError(
            "Not enough reward-only questions for question generalization. "
            "Need at least 2 questions with support_size + 2 or more models."
        )

    test_size = max(1, int(round(test_fraction * len(eligible_question_ids))))
    test_size = min(test_size, len(eligible_question_ids) - 1)
    test_question_ids = rng.choice(np.asarray(eligible_question_ids), size=test_size, replace=False).tolist()
    test_question_ids = [str(question_id) for question_id in test_question_ids]
    if max_test_questions is not None:
        test_question_ids = test_question_ids[:max_test_questions]
    test_question_set = set(test_question_ids)
    train_question_ids = sorted(set(eligible_question_ids) - test_question_set)
    sampled_train_questions = sample_train_questions(
        train_question_ids,
        train_fraction=train_fraction,
        rng=rng,
    )
    sampled_train_set = set(sampled_train_questions)

    support_rows: list[pd.DataFrame] = []
    eval_rows: list[pd.DataFrame] = []
    support_map: dict[str, list[str]] = {}
    query_map: dict[str, list[str]] = {}
    for question_id in test_question_ids:
        model_names = models_by_question[question_id]
        chosen_support = rng.choice(np.asarray(model_names), size=support_size, replace=False).tolist()
        chosen_support = [str(model_name) for model_name in chosen_support]
        support_map[question_id] = chosen_support
        query_models = [model_name for model_name in model_names if model_name not in chosen_support]
        query_map[question_id] = query_models
        support_rows.append(
            reward_df[
                (reward_df["question_id"] == question_id) & (reward_df["model_name"].isin(chosen_support))
            ].copy()
        )
        eval_rows.append(
            pair_df[
                (pair_df["question_id"] == question_id)
                & (~pair_df["model_1"].isin(chosen_support))
                & (~pair_df["model_2"].isin(chosen_support))
            ].copy()
        )

    reward_train = subset_reward_by_questions(reward_df, sampled_train_set)
    if support_rows:
        reward_train = pd.concat([reward_train, *support_rows], ignore_index=True)
    reward_train = reward_train.drop_duplicates(
        subset=["question_id", "model_name"],
        keep="last",
    ).reset_index(drop=True)

    bt_train = pair_df[pair_df["question_id"].isin(sampled_train_set)].copy().reset_index(drop=True)
    eval_df = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame(columns=pair_df.columns)

    return {
        "bt_train": bt_train,
        "reward_train": reward_train,
        "eval_df": eval_df.reset_index(drop=True),
        "train_question_ids": sampled_train_questions,
        "all_train_question_ids": train_question_ids,
        "test_question_ids": test_question_ids,
        "support_map": support_map,
        "query_map": query_map,
    }


def build_pair_generalization_split(
    human_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seed: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reward_models_by_q = reward_df.groupby("question_id")["model_name"].apply(set).to_dict()

    eligible_df = human_df.copy()
    eligible_df["support_count_after_holdout"] = eligible_df.apply(
        lambda row: len(
            reward_models_by_q.get(str(row["question_id"]), set()).difference({str(row["model_1"]), str(row["model_2"])})
        ),
        axis=1,
    )
    eligible_df = eligible_df[eligible_df["support_count_after_holdout"] >= 2].copy().reset_index(drop=True)
    if eligible_df.empty:
        raise ValueError("No questions have at least two non-eval support models for pair generalization.")

    pair_keys = sorted(eligible_df["pair_key"].unique())
    holdout_size = max(1, int(round(holdout_fraction * len(pair_keys))))
    holdout_size = min(holdout_size, len(pair_keys) - 1)
    heldout_pairs = rng.choice(np.asarray(pair_keys), size=holdout_size, replace=False).tolist()
    heldout_pair_set = set(str(pair_key) for pair_key in heldout_pairs)

    bt_train = human_df[~human_df["pair_key"].isin(heldout_pair_set)].copy().reset_index(drop=True)
    eval_df = eligible_df[eligible_df["pair_key"].isin(heldout_pair_set)].copy().reset_index(drop=True)

    heldout_rows = eval_df[["question_id", "model_1", "model_2"]].copy()
    heldout_long = pd.concat(
        [
            heldout_rows.rename(columns={"model_1": "model_name"})[["question_id", "model_name"]],
            heldout_rows.rename(columns={"model_2": "model_name"})[["question_id", "model_name"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    reward_train = reward_df.merge(
        heldout_long.assign(_drop=1),
        on=["question_id", "model_name"],
        how="left",
    )
    reward_train = reward_train[reward_train["_drop"].isna()].drop(columns="_drop").reset_index(drop=True)

    return {
        "bt_train": bt_train,
        "reward_train": reward_train,
        "eval_df": eval_df,
        "heldout_pairs": sorted(heldout_pair_set),
    }


def build_reward_only_pair_generalization_split(
    pair_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seed: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reward_models_by_q = reward_df.groupby("question_id")["model_name"].apply(set).to_dict()

    eligible_df = pair_df.copy()
    eligible_df["support_count_after_holdout"] = eligible_df.apply(
        lambda row: len(
            reward_models_by_q.get(str(row["question_id"]), set()).difference({str(row["model_1"]), str(row["model_2"])})
        ),
        axis=1,
    )
    eligible_df = eligible_df[eligible_df["support_count_after_holdout"] >= 2].copy().reset_index(drop=True)
    if eligible_df.empty:
        raise ValueError("No reward-only pairs have at least two non-eval support models.")

    question_ids = sorted(eligible_df["question_id"].unique())
    holdout_size = max(1, int(round(holdout_fraction * len(question_ids))))
    holdout_size = min(holdout_size, len(question_ids))
    heldout_question_ids = rng.choice(np.asarray(question_ids), size=holdout_size, replace=False).tolist()

    eval_parts: list[pd.DataFrame] = []
    heldout_pairs: list[str] = []
    for question_id in heldout_question_ids:
        question_pairs = eligible_df[eligible_df["question_id"] == question_id].copy().reset_index(drop=True)
        pair_idx = int(rng.integers(len(question_pairs)))
        heldout_row = question_pairs.iloc[[pair_idx]].copy()
        eval_parts.append(heldout_row)
        heldout_pairs.append(f"{question_id}::{heldout_row.iloc[0]['pair_key']}")

    eval_df = pd.concat(eval_parts, ignore_index=True) if eval_parts else pd.DataFrame(columns=pair_df.columns)
    if eval_df.empty:
        raise ValueError("No reward-only held-out pair instances selected.")

    bt_train = pair_df.merge(
        eval_df[["question_id", "pair_key"]].assign(_drop=1),
        on=["question_id", "pair_key"],
        how="left",
    )
    bt_train = bt_train[bt_train["_drop"].isna()].drop(columns="_drop").reset_index(drop=True)

    heldout_rows = eval_df[["question_id", "model_1", "model_2"]].copy()
    heldout_long = pd.concat(
        [
            heldout_rows.rename(columns={"model_1": "model_name"})[["question_id", "model_name"]],
            heldout_rows.rename(columns={"model_2": "model_name"})[["question_id", "model_name"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    reward_train = reward_df.merge(
        heldout_long.assign(_drop=1),
        on=["question_id", "model_name"],
        how="left",
    )
    reward_train = reward_train[reward_train["_drop"].isna()].drop(columns="_drop").reset_index(drop=True)

    return {
        "bt_train": bt_train,
        "reward_train": reward_train,
        "eval_df": eval_df,
        "heldout_pairs": sorted(heldout_pairs),
    }


def run_bt_and_reward_eval(
    bt_train: pd.DataFrame,
    reward_train: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    experiment: str,
    seed: int,
    budget_name: str,
    budget_value: float | int,
    ece_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    result_rows: list[dict[str, Any]] = []
    extra_outputs: dict[str, pd.DataFrame] = {}

    bt_model_params = fit_bt_model(
        bt_train,
        num_epochs=bt_config.num_epochs,
        lr=bt_config.lr,
        reg_lambda=bt_config.reg_lambda,
        verbose=bt_config.verbose,
    )
    bt_preds = predict_bt(eval_df, bt_model_params)
    bt_row = compute_binary_metrics(
        bt_preds,
        method="bt",
        experiment=experiment,
        seed=seed,
        budget_name=budget_name,
        budget_value=budget_value,
        eval_group_count=len(eval_df),
        skipped_count=len(eval_df) - len(bt_preds),
        ece_bins=ece_bins,
    )
    result_rows.append(bt_row)
    extra_outputs[f"{experiment}_bt_ranking_seed_{seed}_{budget_name}_{budget_value}"] = bt_model_params

    reward_model_params, reward_question_params, fit_meta, pairwise_df = fit_reward_irt_model(
        reward_train,
        reward_config,
    )
    reward_preds = predict_reward_irt(
        eval_df,
        reward_model_params,
        reward_question_params,
        learned_gamma=float(fit_meta["learned_gamma"]),
    )
    reward_row = compute_binary_metrics(
        reward_preds,
        method="reward_distilled_irt",
        experiment=experiment,
        seed=seed,
        budget_name=budget_name,
        budget_value=budget_value,
        eval_group_count=len(eval_df),
        skipped_count=len(eval_df) - len(reward_preds),
        ece_bins=ece_bins,
    )
    reward_row["pairwise_train_rows"] = int(len(pairwise_df))
    reward_row["train_questions"] = int(reward_train["question_id"].nunique())
    result_rows.append(reward_row)
    extra_outputs[f"{experiment}_reward_ranking_seed_{seed}_{budget_name}_{budget_value}"] = reward_model_params

    return result_rows, extra_outputs


def run_question_generalization_experiment(
    human_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    support_sizes: list[int],
    seeds: list[int],
    test_fraction: float,
    max_test_questions: int | None,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
    ece_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings = list(product(seeds, support_sizes))
    iterator = progress_iter(
        settings,
        total=len(settings),
        desc="Question generalization",
        enabled=show_progress,
    )
    for seed, support_size in iterator:
        split = build_question_generalization_split(
            human_df,
            reward_df,
            seed=seed,
            support_size=support_size,
            test_fraction=test_fraction,
            max_test_questions=max_test_questions,
            train_fraction=1.0,
        )
        result_rows, _ = run_bt_and_reward_eval(
            split["bt_train"],
            split["reward_train"],
            split["eval_df"],
            bt_config=bt_config,
            reward_config=reward_config,
            experiment="question_generalization",
            seed=seed,
            budget_name="support_size",
            budget_value=support_size,
            ece_bins=ece_bins,
        )
        for row in result_rows:
            row["n_train_questions"] = len(split["train_question_ids"])
            row["n_test_questions"] = len(split["test_question_ids"])
            row["support_size"] = support_size
        rows.extend(result_rows)
    return pd.DataFrame(rows)


def run_reward_only_question_generalization_experiment(
    pair_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    support_sizes: list[int],
    seeds: list[int],
    test_fraction: float,
    max_test_questions: int | None,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
    ece_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings = list(product(seeds, support_sizes))
    iterator = progress_iter(
        settings,
        total=len(settings),
        desc="Question generalization",
        enabled=show_progress,
    )
    for seed, support_size in iterator:
        split = build_reward_only_question_generalization_split(
            pair_df,
            reward_df,
            seed=seed,
            support_size=support_size,
            test_fraction=test_fraction,
            max_test_questions=max_test_questions,
            train_fraction=1.0,
        )
        result_rows, _ = run_bt_and_reward_eval(
            split["bt_train"],
            split["reward_train"],
            split["eval_df"],
            bt_config=bt_config,
            reward_config=reward_config,
            experiment="question_generalization",
            seed=seed,
            budget_name="support_size",
            budget_value=support_size,
            ece_bins=ece_bins,
        )
        for row in result_rows:
            row["n_train_questions"] = len(split["train_question_ids"])
            row["n_test_questions"] = len(split["test_question_ids"])
            row["support_size"] = support_size
            row["avg_query_models_per_test_question"] = float(
                np.mean([len(models) for models in split["query_map"].values()])
            ) if split["query_map"] else float("nan")
        rows.extend(result_rows)
    return pd.DataFrame(rows)


def run_pair_generalization_experiment(
    human_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seeds: list[int],
    holdout_fraction: float,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
    ece_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in progress_iter(
        seeds,
        total=len(seeds),
        desc="Pair generalization",
        enabled=show_progress,
    ):
        split = build_pair_generalization_split(
            human_df,
            reward_df,
            seed=seed,
            holdout_fraction=holdout_fraction,
        )
        result_rows, _ = run_bt_and_reward_eval(
            split["bt_train"],
            split["reward_train"],
            split["eval_df"],
            bt_config=bt_config,
            reward_config=reward_config,
            experiment="pair_generalization",
            seed=seed,
            budget_name="pair_holdout_fraction",
            budget_value=holdout_fraction,
            ece_bins=ece_bins,
        )
        for row in result_rows:
            row["n_heldout_pairs"] = len(split["heldout_pairs"])
        rows.extend(result_rows)
    return pd.DataFrame(rows)


def run_reward_only_pair_generalization_experiment(
    pair_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seeds: list[int],
    holdout_fraction: float,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
    ece_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in progress_iter(
        seeds,
        total=len(seeds),
        desc="Pair generalization",
        enabled=show_progress,
    ):
        split = build_reward_only_pair_generalization_split(
            pair_df,
            reward_df,
            seed=seed,
            holdout_fraction=holdout_fraction,
        )
        result_rows, _ = run_bt_and_reward_eval(
            split["bt_train"],
            split["reward_train"],
            split["eval_df"],
            bt_config=bt_config,
            reward_config=reward_config,
            experiment="pair_generalization",
            seed=seed,
            budget_name="pair_holdout_fraction",
            budget_value=holdout_fraction,
            ece_bins=ece_bins,
        )
        for row in result_rows:
            row["n_heldout_pairs"] = len(split["heldout_pairs"])
        rows.extend(result_rows)
    return pd.DataFrame(rows)


def run_train_fraction_experiment(
    human_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    train_fractions: list[float],
    seeds: list[int],
    test_fraction: float,
    support_size: int,
    max_test_questions: int | None,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
    ece_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings = list(product(seeds, train_fractions))
    iterator = progress_iter(
        settings,
        total=len(settings),
        desc="Train-fraction learning curve",
        enabled=show_progress,
    )
    for seed, train_fraction in iterator:
        split = build_question_generalization_split(
            human_df,
            reward_df,
            seed=seed,
            support_size=support_size,
            test_fraction=test_fraction,
            max_test_questions=max_test_questions,
            train_fraction=train_fraction,
        )
        result_rows, _ = run_bt_and_reward_eval(
            split["bt_train"],
            split["reward_train"],
            split["eval_df"],
            bt_config=bt_config,
            reward_config=reward_config,
            experiment="train_fraction_learning_curve",
            seed=seed,
            budget_name="train_fraction",
            budget_value=float(train_fraction),
            ece_bins=ece_bins,
        )
        for row in result_rows:
            row["n_train_questions"] = len(split["train_question_ids"])
            row["n_test_questions"] = len(split["test_question_ids"])
            row["support_size"] = support_size
        rows.extend(result_rows)
    return pd.DataFrame(rows)


def run_reward_only_train_fraction_experiment(
    pair_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    train_fractions: list[float],
    seeds: list[int],
    test_fraction: float,
    support_size: int,
    max_test_questions: int | None,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
    ece_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings = list(product(seeds, train_fractions))
    iterator = progress_iter(
        settings,
        total=len(settings),
        desc="Train-fraction learning curve",
        enabled=show_progress,
    )
    for seed, train_fraction in iterator:
        split = build_reward_only_question_generalization_split(
            pair_df,
            reward_df,
            seed=seed,
            support_size=support_size,
            test_fraction=test_fraction,
            max_test_questions=max_test_questions,
            train_fraction=train_fraction,
        )
        result_rows, _ = run_bt_and_reward_eval(
            split["bt_train"],
            split["reward_train"],
            split["eval_df"],
            bt_config=bt_config,
            reward_config=reward_config,
            experiment="train_fraction_learning_curve",
            seed=seed,
            budget_name="train_fraction",
            budget_value=float(train_fraction),
            ece_bins=ece_bins,
        )
        for row in result_rows:
            row["n_train_questions"] = len(split["train_question_ids"])
            row["n_test_questions"] = len(split["test_question_ids"])
            row["support_size"] = support_size
            row["avg_query_models_per_test_question"] = float(
                np.mean([len(models) for models in split["query_map"].values()])
            ) if split["query_map"] else float("nan")
        rows.extend(result_rows)
    return pd.DataFrame(rows)


def rank_models(model_params: pd.DataFrame) -> dict[str, int]:
    ordered = model_params.sort_values("theta", ascending=False)["model_name"].tolist()
    return {model_name: rank for rank, model_name in enumerate(ordered, start=1)}


def run_bootstrap_stability_experiment(
    human_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float,
    support_size: int,
    train_fraction: float,
    bootstrap_replicates: int,
    max_test_questions: int | None,
    top_k: int,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = build_question_generalization_split(
        human_df,
        reward_df,
        seed=seed,
        support_size=support_size,
        test_fraction=test_fraction,
        max_test_questions=max_test_questions,
        train_fraction=train_fraction,
    )

    question_reward_full = subset_reward_by_questions(reward_df, split["all_train_question_ids"])
    support_reward = split["reward_train"][~split["reward_train"]["question_id"].isin(split["train_question_ids"])].copy()
    bt_full = human_df[human_df["question_id"].isin(split["train_question_ids"])].copy().reset_index(drop=True)
    reward_full = subset_reward_by_questions(reward_df, split["train_question_ids"])
    if not support_reward.empty:
        reward_full = pd.concat([reward_full, support_reward], ignore_index=True)

    bt_reference = fit_bt_model(
        bt_full,
        num_epochs=bt_config.num_epochs,
        lr=bt_config.lr,
        reg_lambda=bt_config.reg_lambda,
        verbose=bt_config.verbose,
    )
    reward_reference, _, _, _ = fit_reward_irt_model(reward_full, reward_config)
    reference_rankings = {
        "bt": rank_models(bt_reference),
        "reward_distilled_irt": rank_models(reward_reference),
    }

    train_question_ids = list(split["train_question_ids"])
    rng = np.random.default_rng(seed + 1000)
    ranking_rows: list[dict[str, Any]] = []
    inversion_rows: list[dict[str, Any]] = []

    for replicate in progress_iter(
        range(bootstrap_replicates),
        total=bootstrap_replicates,
        desc="Bootstrap stability",
        enabled=show_progress,
    ):
        boot_questions = rng.choice(np.asarray(train_question_ids), size=len(train_question_ids), replace=True).tolist()
        boot_question_series = pd.Series(boot_questions, name="question_id")
        boot_counts = boot_question_series.value_counts().to_dict()

        bt_boot = bt_full.merge(
            pd.DataFrame({"question_id": list(boot_counts), "weight": list(boot_counts.values())}),
            on="question_id",
            how="inner",
        )
        bt_boot = bt_boot.loc[bt_boot.index.repeat(bt_boot["weight"])].drop(columns="weight").reset_index(drop=True)

        reward_boot_parts: list[pd.DataFrame] = []
        for question_id, count in boot_counts.items():
            group = question_reward_full[question_reward_full["question_id"] == question_id]
            if group.empty:
                continue
            reward_boot_parts.extend([group.copy()] * int(count))
        reward_boot = pd.concat(reward_boot_parts, ignore_index=True) if reward_boot_parts else pd.DataFrame()
        if not support_reward.empty:
            reward_boot = pd.concat([reward_boot, support_reward], ignore_index=True)
        reward_boot = reward_boot.reset_index(drop=True)

        bt_model = fit_bt_model(
            bt_boot,
            num_epochs=bt_config.num_epochs,
            lr=bt_config.lr,
            reg_lambda=bt_config.reg_lambda,
            verbose=False,
        )
        reward_model, _, _, _ = fit_reward_irt_model(
            reward_boot,
            RewardIrtConfig(
                num_epochs=reward_config.num_epochs,
                lr=reward_config.lr,
                lambda_arena=reward_config.lambda_arena,
                lambda_bb=reward_config.lambda_bb,
                reg_lambda=reward_config.reg_lambda,
                both_bad_threshold=reward_config.both_bad_threshold,
                both_bad_use_zscore=reward_config.both_bad_use_zscore,
                verbose=False,
            ),
        )

        for method, model_params in {
            "bt": bt_model,
            "reward_distilled_irt": reward_model,
        }.items():
            rank_map = rank_models(model_params)
            reference_rank_map = reference_rankings[method]
            ordered_reference = sorted(reference_rank_map, key=reference_rank_map.get)
            common_models = [model_name for model_name in ordered_reference if model_name in rank_map]

            for model_name in common_models:
                ranking_rows.append(
                    {
                        "method": method,
                        "replicate": replicate,
                        "model_name": model_name,
                        "rank": rank_map[model_name],
                        "theta": float(model_params.set_index("model_name").loc[model_name, "theta"]),
                        "reference_rank": reference_rank_map[model_name],
                        "in_top_k": int(rank_map[model_name] <= top_k),
                    }
                )

            pair_total = 0
            pair_inverted = 0
            for model_a, model_b in combinations(common_models, 2):
                pair_total += 1
                ref_order = reference_rank_map[model_a] < reference_rank_map[model_b]
                boot_order = rank_map[model_a] < rank_map[model_b]
                pair_inverted += int(ref_order != boot_order)
            inversion_rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "pair_inversion_rate": pair_inverted / pair_total if pair_total else float("nan"),
                }
            )

    ranking_df = pd.DataFrame(ranking_rows)
    inversion_df = pd.DataFrame(inversion_rows)

    stability_summary = (
        ranking_df.groupby(["method", "model_name"], as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            top_k_retention=("in_top_k", "mean"),
            reference_rank=("reference_rank", "first"),
        )
        .sort_values(["method", "reference_rank"])
        .reset_index(drop=True)
    )
    inversion_summary = (
        inversion_df.groupby("method", as_index=False)
        .agg(
            mean_pair_inversion_rate=("pair_inversion_rate", "mean"),
            std_pair_inversion_rate=("pair_inversion_rate", "std"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return stability_summary, inversion_summary


def run_reward_only_bootstrap_stability_experiment(
    pair_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float,
    support_size: int,
    train_fraction: float,
    bootstrap_replicates: int,
    max_test_questions: int | None,
    top_k: int,
    bt_config: BtConfig,
    reward_config: RewardIrtConfig,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = build_reward_only_question_generalization_split(
        pair_df,
        reward_df,
        seed=seed,
        support_size=support_size,
        test_fraction=test_fraction,
        max_test_questions=max_test_questions,
        train_fraction=train_fraction,
    )

    question_reward_full = subset_reward_by_questions(reward_df, split["all_train_question_ids"])
    support_reward = split["reward_train"][~split["reward_train"]["question_id"].isin(split["train_question_ids"])].copy()
    bt_full = pair_df[pair_df["question_id"].isin(split["train_question_ids"])].copy().reset_index(drop=True)
    reward_full = subset_reward_by_questions(reward_df, split["train_question_ids"])
    if not support_reward.empty:
        reward_full = pd.concat([reward_full, support_reward], ignore_index=True)

    bt_reference = fit_bt_model(
        bt_full,
        num_epochs=bt_config.num_epochs,
        lr=bt_config.lr,
        reg_lambda=bt_config.reg_lambda,
        verbose=bt_config.verbose,
    )
    reward_reference, _, _, _ = fit_reward_irt_model(reward_full, reward_config)
    reference_rankings = {
        "bt": rank_models(bt_reference),
        "reward_distilled_irt": rank_models(reward_reference),
    }

    train_question_ids = list(split["train_question_ids"])
    rng = np.random.default_rng(seed + 1000)
    ranking_rows: list[dict[str, Any]] = []
    inversion_rows: list[dict[str, Any]] = []

    for replicate in progress_iter(
        range(bootstrap_replicates),
        total=bootstrap_replicates,
        desc="Bootstrap stability",
        enabled=show_progress,
    ):
        boot_questions = rng.choice(np.asarray(train_question_ids), size=len(train_question_ids), replace=True).tolist()
        boot_question_series = pd.Series(boot_questions, name="question_id")
        boot_counts = boot_question_series.value_counts().to_dict()

        bt_boot = bt_full.merge(
            pd.DataFrame({"question_id": list(boot_counts), "weight": list(boot_counts.values())}),
            on="question_id",
            how="inner",
        )
        bt_boot = bt_boot.loc[bt_boot.index.repeat(bt_boot["weight"])].drop(columns="weight").reset_index(drop=True)

        reward_boot_parts: list[pd.DataFrame] = []
        for question_id, count in boot_counts.items():
            group = question_reward_full[question_reward_full["question_id"] == question_id]
            if group.empty:
                continue
            reward_boot_parts.extend([group.copy()] * int(count))
        reward_boot = pd.concat(reward_boot_parts, ignore_index=True) if reward_boot_parts else pd.DataFrame()
        if not support_reward.empty:
            reward_boot = pd.concat([reward_boot, support_reward], ignore_index=True)
        reward_boot = reward_boot.reset_index(drop=True)

        bt_model = fit_bt_model(
            bt_boot,
            num_epochs=bt_config.num_epochs,
            lr=bt_config.lr,
            reg_lambda=bt_config.reg_lambda,
            verbose=False,
        )
        reward_model, _, _, _ = fit_reward_irt_model(
            reward_boot,
            RewardIrtConfig(
                num_epochs=reward_config.num_epochs,
                lr=reward_config.lr,
                lambda_arena=reward_config.lambda_arena,
                lambda_bb=reward_config.lambda_bb,
                reg_lambda=reward_config.reg_lambda,
                both_bad_threshold=reward_config.both_bad_threshold,
                both_bad_use_zscore=reward_config.both_bad_use_zscore,
                verbose=False,
            ),
        )

        for method, model_params in {
            "bt": bt_model,
            "reward_distilled_irt": reward_model,
        }.items():
            rank_map = rank_models(model_params)
            reference_rank_map = reference_rankings[method]
            ordered_reference = sorted(reference_rank_map, key=reference_rank_map.get)
            common_models = [model_name for model_name in ordered_reference if model_name in rank_map]

            for model_name in common_models:
                ranking_rows.append(
                    {
                        "method": method,
                        "replicate": replicate,
                        "model_name": model_name,
                        "rank": rank_map[model_name],
                        "theta": float(model_params.set_index("model_name").loc[model_name, "theta"]),
                        "reference_rank": reference_rank_map[model_name],
                        "in_top_k": int(rank_map[model_name] <= top_k),
                    }
                )

            pair_total = 0
            pair_inverted = 0
            for model_a, model_b in combinations(common_models, 2):
                pair_total += 1
                ref_order = reference_rank_map[model_a] < reference_rank_map[model_b]
                boot_order = rank_map[model_a] < rank_map[model_b]
                pair_inverted += int(ref_order != boot_order)
            inversion_rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "pair_inversion_rate": pair_inverted / pair_total if pair_total else float("nan"),
                }
            )

    ranking_df = pd.DataFrame(ranking_rows)
    inversion_df = pd.DataFrame(inversion_rows)

    stability_summary = (
        ranking_df.groupby(["method", "model_name"], as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            top_k_retention=("in_top_k", "mean"),
            reference_rank=("reference_rank", "first"),
        )
        .sort_values(["method", "reference_rank"])
        .reset_index(drop=True)
    )
    inversion_summary = (
        inversion_df.groupby("method", as_index=False)
        .agg(
            mean_pair_inversion_rate=("pair_inversion_rate", "mean"),
            std_pair_inversion_rate=("pair_inversion_rate", "std"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return stability_summary, inversion_summary


def aggregate_results(df: pd.DataFrame, budget_name: str) -> pd.DataFrame:
    aggregated = (
        df.groupby(["experiment", "method", "budget_value"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            logloss_mean=("logloss", "mean"),
            logloss_std=("logloss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            n_eval_mean=("n_eval", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    aggregated[budget_name] = aggregated["budget_value"]
    return aggregated


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_metric_lines(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in df.groupby("method", sort=False):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], marker="o", linewidth=2, label=method)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pair_generalization_bars(df: pd.DataFrame, save_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    methods = df["method"].tolist()
    axes[0].bar(methods, df["accuracy_mean"])
    axes[0].set_title("Pair Generalization Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(methods, df["logloss_mean"])
    axes[1].set_title("Pair Generalization Log Loss")
    axes[1].set_ylabel("Log Loss")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(methods, df["ece_mean"])
    axes[2].set_title("Pair Generalization ECE")
    axes[2].set_ylabel("ECE")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stability(stability_df: pd.DataFrame, inversion_df: pd.DataFrame, save_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    stability_summary = stability_df.groupby("method", as_index=False)["std_rank"].mean()
    axes[0].bar(stability_summary["method"], stability_summary["std_rank"])
    axes[0].set_title("Mean Rank Std Dev")
    axes[0].set_ylabel("Std(rank)")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(inversion_df["method"], inversion_df["mean_pair_inversion_rate"])
    axes[1].set_title("Pairwise Inversion Rate")
    axes[1].set_ylabel("Inversion Rate")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    output_dir: Path,
    *,
    evaluation_source: str,
    question_summary: pd.DataFrame,
    train_fraction_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    inversion_summary: pd.DataFrame,
) -> None:
    def frame_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows_"
        return df.to_markdown(index=False, floatfmt=".4f")

    report = "\n".join(
        [
            "# Arena-Only BT Experiments",
            "",
            f"Evaluation source: `{evaluation_source}`",
            "",
            "## Question Generalization",
            frame_to_markdown(question_summary),
            "",
            "## Train Fraction Learning Curve",
            frame_to_markdown(train_fraction_summary),
            "",
            "## Pair Generalization",
            frame_to_markdown(pair_summary),
            "",
            "## Bootstrap Stability",
            frame_to_markdown(stability_summary),
            "",
            "## Pairwise Inversion",
            frame_to_markdown(inversion_summary),
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    show_progress = not args.no_progress and sys.stderr.isatty()

    bt_config = BtConfig(
        num_epochs=args.num_epochs,
        lr=args.lr,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )
    reward_config = RewardIrtConfig(
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_arena=args.lambda_arena,
        lambda_bb=args.lambda_bb,
        reg_lambda=args.reg_lambda,
        both_bad_threshold=args.both_bad_threshold,
        both_bad_use_zscore=args.both_bad_mode == "zscore",
        verbose=not args.quiet,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reward responses from {len(args.arena_reward_jsonl)} JSONL files")
    reward_df = load_reward_responses(args.arena_reward_jsonl)
    print(
        f"  reward rows={len(reward_df)} "
        f"questions={reward_df['question_id'].nunique()} "
        f"models={reward_df['model_name'].nunique()}"
    )

    if args.target_source == "auto":
        evaluation_source = "human" if args.human_data is not None else "reward"
    else:
        evaluation_source = args.target_source
    if evaluation_source == "human" and args.human_data is None:
        raise SystemExit("--target-source human requires --human-data.")

    human_df: pd.DataFrame | None = None
    reward_pair_df: pd.DataFrame | None = None
    if evaluation_source == "human":
        print(f"Loading human Arena data from {args.human_data}")
        human_df = load_human_pairs(args.human_data)
        print(
            f"  decisive human pairs={len(human_df)} "
            f"questions={human_df['question_id'].nunique()} "
            f"pairs={human_df['pair_key'].nunique()}"
        )
    else:
        reward_pair_df = build_reward_pair_targets(reward_df)
        print(
            f"Built reward-derived pair targets: rows={len(reward_pair_df)} "
            f"questions={reward_pair_df['question_id'].nunique()} "
            f"pairs={reward_pair_df['pair_key'].nunique()}"
        )

    if evaluation_source == "human":
        assert human_df is not None
        question_results = run_question_generalization_experiment(
            human_df,
            reward_df,
            support_sizes=sorted(set(args.support_sizes)),
            seeds=args.seeds,
            test_fraction=args.test_fraction,
            max_test_questions=args.max_test_questions,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
            ece_bins=args.ece_bins,
        )
    else:
        assert reward_pair_df is not None
        question_results = run_reward_only_question_generalization_experiment(
            reward_pair_df,
            reward_df,
            support_sizes=sorted(set(args.support_sizes)),
            seeds=args.seeds,
            test_fraction=args.test_fraction,
            max_test_questions=args.max_test_questions,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
            ece_bins=args.ece_bins,
        )
    save_csv(question_results, output_dir / "raw" / "question_generalization.csv")
    question_summary = aggregate_results(question_results, "support_size")
    save_csv(question_summary, output_dir / "summary_question_generalization.csv")

    if evaluation_source == "human":
        assert human_df is not None
        pair_results = run_pair_generalization_experiment(
            human_df,
            reward_df,
            seeds=args.seeds,
            holdout_fraction=args.pair_holdout_fraction,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
            ece_bins=args.ece_bins,
        )
    else:
        assert reward_pair_df is not None
        pair_results = run_reward_only_pair_generalization_experiment(
            reward_pair_df,
            reward_df,
            seeds=args.seeds,
            holdout_fraction=args.pair_holdout_fraction,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
            ece_bins=args.ece_bins,
        )
    save_csv(pair_results, output_dir / "raw" / "pair_generalization.csv")
    pair_summary = aggregate_results(pair_results, "pair_holdout_fraction")
    save_csv(pair_summary, output_dir / "summary_pair_generalization.csv")

    if evaluation_source == "human":
        assert human_df is not None
        train_fraction_results = run_train_fraction_experiment(
            human_df,
            reward_df,
            train_fractions=sorted(set(args.train_fractions)),
            seeds=args.seeds,
            test_fraction=args.test_fraction,
            support_size=args.learning_curve_support_size,
            max_test_questions=args.max_test_questions,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
            ece_bins=args.ece_bins,
        )
    else:
        assert reward_pair_df is not None
        train_fraction_results = run_reward_only_train_fraction_experiment(
            reward_pair_df,
            reward_df,
            train_fractions=sorted(set(args.train_fractions)),
            seeds=args.seeds,
            test_fraction=args.test_fraction,
            support_size=args.learning_curve_support_size,
            max_test_questions=args.max_test_questions,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
            ece_bins=args.ece_bins,
        )
    save_csv(train_fraction_results, output_dir / "raw" / "train_fraction_learning_curve.csv")
    train_fraction_summary = aggregate_results(train_fraction_results, "train_fraction")
    save_csv(train_fraction_summary, output_dir / "summary_train_fraction_learning_curve.csv")

    if evaluation_source == "human":
        assert human_df is not None
        stability_summary, inversion_summary = run_bootstrap_stability_experiment(
            human_df,
            reward_df,
            seed=args.seeds[0],
            test_fraction=args.test_fraction,
            support_size=args.bootstrap_support_size,
            train_fraction=args.bootstrap_train_fraction,
            bootstrap_replicates=args.bootstrap_replicates,
            max_test_questions=args.max_test_questions,
            top_k=args.top_k,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
        )
    else:
        assert reward_pair_df is not None
        stability_summary, inversion_summary = run_reward_only_bootstrap_stability_experiment(
            reward_pair_df,
            reward_df,
            seed=args.seeds[0],
            test_fraction=args.test_fraction,
            support_size=args.bootstrap_support_size,
            train_fraction=args.bootstrap_train_fraction,
            bootstrap_replicates=args.bootstrap_replicates,
            max_test_questions=args.max_test_questions,
            top_k=args.top_k,
            bt_config=bt_config,
            reward_config=reward_config,
            show_progress=show_progress,
        )
    save_csv(stability_summary, output_dir / "summary_bootstrap_stability.csv")
    save_csv(inversion_summary, output_dir / "summary_bootstrap_inversion.csv")

    write_markdown_report(
        output_dir,
        evaluation_source=evaluation_source,
        question_summary=question_summary,
        train_fraction_summary=train_fraction_summary,
        pair_summary=pair_summary,
        stability_summary=stability_summary,
        inversion_summary=inversion_summary,
    )

    if not args.no_plots:
        question_accuracy = question_summary[question_summary["experiment"] == "question_generalization"]
        plot_metric_lines(
            question_accuracy,
            x_col="support_size",
            y_col="accuracy_mean",
            title="Held-Out Question Generalization Accuracy",
            xlabel="Support models per held-out question",
            ylabel="Accuracy",
            save_path=output_dir / "plots" / "question_generalization_accuracy.png",
        )
        plot_metric_lines(
            question_accuracy,
            x_col="support_size",
            y_col="logloss_mean",
            title="Held-Out Question Generalization Log Loss",
            xlabel="Support models per held-out question",
            ylabel="Log Loss",
            save_path=output_dir / "plots" / "question_generalization_logloss.png",
        )
        plot_metric_lines(
            question_accuracy,
            x_col="support_size",
            y_col="brier_mean",
            title="Held-Out Question Generalization Brier Score",
            xlabel="Support models per held-out question",
            ylabel="Brier",
            save_path=output_dir / "plots" / "question_generalization_brier.png",
        )
        plot_metric_lines(
            question_accuracy,
            x_col="support_size",
            y_col="ece_mean",
            title="Held-Out Question Generalization ECE",
            xlabel="Support models per held-out question",
            ylabel="ECE",
            save_path=output_dir / "plots" / "question_generalization_ece.png",
        )

        train_fraction_plot_df = train_fraction_summary[
            train_fraction_summary["experiment"] == "train_fraction_learning_curve"
        ]
        plot_metric_lines(
            train_fraction_plot_df,
            x_col="train_fraction",
            y_col="accuracy_mean",
            title="Sample Efficiency by Train Fraction",
            xlabel="Training-question fraction",
            ylabel="Accuracy",
            save_path=output_dir / "plots" / "train_fraction_accuracy.png",
        )
        plot_metric_lines(
            train_fraction_plot_df,
            x_col="train_fraction",
            y_col="logloss_mean",
            title="Sample Efficiency by Train Fraction",
            xlabel="Training-question fraction",
            ylabel="Log Loss",
            save_path=output_dir / "plots" / "train_fraction_logloss.png",
        )
        plot_metric_lines(
            train_fraction_plot_df,
            x_col="train_fraction",
            y_col="brier_mean",
            title="Sample Efficiency by Train Fraction",
            xlabel="Training-question fraction",
            ylabel="Brier",
            save_path=output_dir / "plots" / "train_fraction_brier.png",
        )
        plot_metric_lines(
            train_fraction_plot_df,
            x_col="train_fraction",
            y_col="ece_mean",
            title="Sample Efficiency by Train Fraction",
            xlabel="Training-question fraction",
            ylabel="ECE",
            save_path=output_dir / "plots" / "train_fraction_ece.png",
        )

        pair_plot_df = pair_summary[pair_summary["experiment"] == "pair_generalization"]
        plot_pair_generalization_bars(
            pair_plot_df,
            output_dir / "plots" / "pair_generalization.png",
        )
        plot_stability(
            stability_summary,
            inversion_summary,
            output_dir / "plots" / "bootstrap_stability.png",
        )

    print(f"Saved arena-only BT experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
