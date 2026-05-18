#!/usr/bin/env python3
"""
Ablate DualEval pair treatment: both-bad anchoring and tie filtering.

Variants:
  - full:             configured bb_ratio, tie_ratio, and lambda_bb
  - no_bb_loss:       configured bb_ratio/tie_ratio, lambda_bb = 0
  - no_bb_flagging:   bb_ratio = 0
  - no_tie_filter:    tie_ratio = 0
  - no_bb_no_tie:     bb_ratio = 0, tie_ratio = 0

All variants are evaluated on the same held-out pairwise target set built with
the full configured thresholds, so reported arena metrics are comparable.  By
default this script uses row-level holdout because item-parameter metrics need
the fitted model to contain the evaluation question IDs.

Example:
  python analysis/pair_treatment_ablation.py \
      --config ranking/config_dualeval.yaml \
      --output-dir results/dualeval_ablations/coding_v1_pair_treatment \
      --mode both \
      --test-fraction 0.2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DUALEVAL_PATH = REPO_ROOT / "ranking" / "dualeval.py"
dualeval_spec = importlib.util.spec_from_file_location("dualeval_module", DUALEVAL_PATH)
if dualeval_spec is None or dualeval_spec.loader is None:
    raise ImportError(f"Could not load DualEval module from {DUALEVAL_PATH}")
dualeval = importlib.util.module_from_spec(dualeval_spec)
dualeval_spec.loader.exec_module(dualeval)


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_paths(paths: list[str]) -> list[str]:
    return [str(resolve_path(path)) for path in paths]


def load_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = dualeval.load_yaml_config(args.config)
    input_cfg = cfg.get("input", {})
    training_cfg = cfg.get("training", {})

    if args.static_jsonl is None:
        args.static_jsonl = ensure_list(input_cfg.get("static_jsonl"))
    if args.arena_reward_jsonl is None:
        args.arena_reward_jsonl = ensure_list(input_cfg.get("arena_reward_jsonl"))
    if args.mode is None:
        mode = str(training_cfg.get("mode", "both")).strip()
        args.mode = mode if mode in {"arena", "both"} else "both"
    if args.num_epochs is None:
        args.num_epochs = int(training_cfg.get("num_epochs", 2000))
    if args.lr is None:
        args.lr = float(training_cfg.get("lr", 0.02))
    if args.lambda_static is None:
        args.lambda_static = float(training_cfg.get("lambda_static", 1.0))
    if args.lambda_arena is None:
        args.lambda_arena = float(training_cfg.get("lambda_arena", 1.0))
    if args.lambda_bb is None:
        args.lambda_bb = float(training_cfg.get("lambda_bb", 0.2))
    if args.reg_lambda is None:
        args.reg_lambda = float(training_cfg.get("reg_lambda", 1e-3))
    if args.bb_ratio is None:
        args.bb_ratio = float(training_cfg.get("bb_ratio", 0.15))
    if args.tie_ratio is None:
        args.tie_ratio = float(training_cfg.get("tie_ratio", 0.10))

    args.static_jsonl = resolve_paths(args.static_jsonl)
    args.arena_reward_jsonl = resolve_paths(args.arena_reward_jsonl)
    args.output_dir = resolve_path(args.output_dir)
    return args


def split_questions(
    df: pd.DataFrame,
    *,
    question_col: str,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    if df.empty or test_fraction <= 0.0:
        questions = set(df[question_col].astype(str).unique()) if question_col in df.columns else set()
        return df.copy().reset_index(drop=True), df.copy().reset_index(drop=True), questions
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"--test-fraction must be in [0, 1), got {test_fraction}")

    questions = np.array(sorted(df[question_col].astype(str).unique()))
    if len(questions) < 2:
        return df.copy().reset_index(drop=True), df.copy().reset_index(drop=True), set(questions.tolist())

    rng = np.random.default_rng(seed)
    n_eval = max(1, min(len(questions) - 1, int(round(test_fraction * len(questions)))))
    eval_questions = set(rng.choice(questions, size=n_eval, replace=False).tolist())
    eval_mask = df[question_col].astype(str).isin(eval_questions)
    return (
        df.loc[~eval_mask].reset_index(drop=True),
        df.loc[eval_mask].reset_index(drop=True),
        eval_questions,
    )


def split_rows(df: pd.DataFrame, *, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    if df.empty or test_fraction <= 0.0:
        questions = set(df["question_id"].astype(str).unique()) if "question_id" in df.columns else set()
        return df.copy().reset_index(drop=True), df.copy().reset_index(drop=True), questions
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"--test-fraction must be in [0, 1), got {test_fraction}")

    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_fraction
    if not mask.any() and len(df) > 1:
        mask[rng.integers(0, len(df))] = True
    if mask.all() and len(df) > 1:
        mask[rng.integers(0, len(df))] = False

    eval_df = df.loc[mask].reset_index(drop=True)
    train_df = df.loc[~mask].reset_index(drop=True)
    questions = set(eval_df["question_id"].astype(str).unique()) if "question_id" in eval_df.columns else set()
    return train_df, eval_df, questions


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    pos_rank_sum = float(ranks[y_true].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    true_cumsum = np.cumsum(sorted_true)
    ranks = np.arange(1, len(sorted_true) + 1)
    precision = true_cumsum / ranks
    return float(precision[sorted_true].sum() / n_pos)


def static_pairwise_accuracy(static_df: pd.DataFrame, model_params: pd.DataFrame) -> tuple[float, int]:
    if static_df.empty:
        return float("nan"), 0
    theta = model_params.set_index("model_name")["theta"]
    eval_df = static_df.copy()
    eval_df["theta"] = eval_df["model_name"].map(theta)
    eval_df = eval_df.dropna(subset=["theta", "judge_result"])
    correct = 0.0
    total = 0
    for _, group in eval_df.groupby("question_id", sort=False):
        pos = group[group["judge_result"].astype(int) == 1]["theta"].to_numpy()
        neg = group[group["judge_result"].astype(int) == 0]["theta"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        comparisons = pos[:, None] - neg[None, :]
        correct += float((comparisons > 0).sum()) + 0.5 * float((comparisons == 0).sum())
        total += int(comparisons.size)
    return (correct / total if total else float("nan")), total


def add_item_predictions(
    pairwise_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    learned_gamma: float | None,
) -> pd.DataFrame:
    theta = model_params.set_index("model_name")["theta"]
    q_params = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
    eval_df = pairwise_df.copy()
    eval_df["theta_1"] = eval_df["model_1"].map(theta)
    eval_df["theta_2"] = eval_df["model_2"].map(theta)
    eval_df["difficulty_b"] = eval_df["question_id"].map(q_params["difficulty_b"])
    eval_df["a_q"] = eval_df["question_id"].map(q_params["discrimination_exp_k"])
    eval_df = eval_df.dropna(subset=["theta_1", "theta_2", "difficulty_b", "a_q", "target_prob"])
    if eval_df.empty:
        return eval_df
    a_q = eval_df["a_q"].to_numpy(dtype=float)
    b_q = eval_df["difficulty_b"].to_numpy(dtype=float)
    p1 = 1.0 / (1.0 + np.exp(-a_q * (eval_df["theta_1"].to_numpy(dtype=float) - b_q)))
    p2 = 1.0 / (1.0 + np.exp(-a_q * (eval_df["theta_2"].to_numpy(dtype=float) - b_q)))
    gamma = learned_gamma if learned_gamma is not None else 1.0
    eval_df["pred_pair_prob"] = 1.0 / (1.0 + np.exp(-gamma * (p1 - p2)))
    eval_df["pred_both_bad_score"] = (1.0 - p1) * (1.0 - p2)
    return eval_df


def arena_hard_pair_accuracy(
    pairwise_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    learned_gamma: float | None,
) -> tuple[float, int]:
    eval_df = add_item_predictions(
        pairwise_df,
        model_params,
        question_params,
        learned_gamma=learned_gamma,
    )
    if eval_df.empty:
        return float("nan"), 0
    hard = ~eval_df["tie"].astype(bool) & ~eval_df["both_bad"].astype(bool)
    eval_df = eval_df[hard]
    if eval_df.empty:
        return float("nan"), 0
    pred = eval_df["pred_pair_prob"] >= 0.5
    target = eval_df["target_prob"] >= 0.5
    return float(pred.eq(target).mean()), int(len(eval_df))


def both_bad_detection_metrics(
    pairwise_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    learned_gamma: float | None,
) -> dict[str, float]:
    eval_df = add_item_predictions(
        pairwise_df,
        model_params,
        question_params,
        learned_gamma=learned_gamma,
    )
    if eval_df.empty or "both_bad" not in eval_df.columns:
        return {
            "both_bad_auc": float("nan"),
            "both_bad_average_precision": float("nan"),
            "both_bad_positive_rate": float("nan"),
            "n_both_bad_eval_pairs": 0.0,
        }
    y_true = eval_df["both_bad"].astype(bool).to_numpy()
    scores = eval_df["pred_both_bad_score"].to_numpy(dtype=float)
    return {
        "both_bad_auc": float(binary_auc(y_true, scores)),
        "both_bad_average_precision": float(average_precision(y_true, scores)),
        "both_bad_positive_rate": float(y_true.mean()),
        "n_both_bad_eval_pairs": float(len(eval_df)),
    }


def build_pairwise(reward_df: pd.DataFrame, *, bb_ratio: float, tie_ratio: float) -> tuple[pd.DataFrame, float, float]:
    both_bad_threshold, tie_delta = dualeval.resolve_pairwise_thresholds(
        reward_df,
        bb_ratio=bb_ratio,
        tie_ratio=tie_ratio,
    )
    pairwise = dualeval.build_soft_pairwise_targets(
        reward_df,
        both_bad_threshold=both_bad_threshold,
        tie_delta=tie_delta,
    )
    return pairwise, both_bad_threshold, tie_delta


def fit_variant(
    *,
    variant_name: str,
    static_train: pd.DataFrame,
    reward_train: pd.DataFrame,
    static_eval: pd.DataFrame,
    pairwise_eval_common: pd.DataFrame,
    bb_ratio: float,
    tie_ratio: float,
    lambda_bb: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pairwise_train, both_bad_threshold, tie_delta = build_pairwise(
        reward_train,
        bb_ratio=bb_ratio,
        tie_ratio=tie_ratio,
    )
    if pairwise_train.empty:
        raise SystemExit(f"No pairwise training rows for variant {variant_name}.")

    model_params, question_params, fit_meta = dualeval.fit_irt(
        static_train if args.mode == "both" and not static_train.empty else None,
        pairwise_train,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=args.lambda_static,
        lambda_arena=args.lambda_arena,
        lambda_bb=lambda_bb,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )
    static_acc, n_static_pairs = static_pairwise_accuracy(
        static_eval if args.mode == "both" and not static_eval.empty else pd.DataFrame(),
        model_params,
    )
    arena_acc, n_arena_hard_pairs = arena_hard_pair_accuracy(
        pairwise_eval_common,
        model_params,
        question_params,
        learned_gamma=fit_meta["learned_gamma"],
    )
    bb_metrics = both_bad_detection_metrics(
        pairwise_eval_common,
        model_params,
        question_params,
        learned_gamma=fit_meta["learned_gamma"],
    )

    train_tie = pairwise_train["tie"].astype(bool)
    train_bb = pairwise_train["both_bad"].astype(bool)
    row = {
        "variant": variant_name,
        "train_bb_ratio": bb_ratio,
        "train_tie_ratio": tie_ratio,
        "train_lambda_bb": lambda_bb,
        "resolved_both_bad_threshold": both_bad_threshold,
        "resolved_tie_delta": tie_delta,
        "n_train_pairs": int(len(pairwise_train)),
        "n_train_hard_pairs": int((~train_tie & ~train_bb).sum()),
        "n_train_both_bad_pairs": int(train_bb.sum()),
        "n_train_tie_pairs": int(train_tie.sum()),
        "static_pairwise_accuracy": static_acc,
        "n_static_eval_pairs": n_static_pairs,
        "arena_hard_pair_accuracy": arena_acc,
        "n_arena_hard_eval_pairs": n_arena_hard_pairs,
        **bb_metrics,
        **fit_meta,
    }
    return row, model_params, question_params, fit_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ranking/config_dualeval.yaml")
    parser.add_argument("--static-jsonl", nargs="*", default=None)
    parser.add_argument("--arena-reward-jsonl", nargs="*", default=None)
    parser.add_argument("--mode", choices=["arena", "both"], default=None)
    parser.add_argument("--output-dir", default="results/dualeval_pair_treatment_ablation")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-static", type=float, default=None)
    parser.add_argument("--lambda-arena", type=float, default=None)
    parser.add_argument("--lambda-bb", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--bb-ratio", type=float, default=None)
    parser.add_argument("--tie-ratio", type=float, default=None)
    parser.add_argument(
        "--split-level",
        choices=["row", "question"],
        default="row",
        help=(
            "Use row-level holdout by default so eval questions have fitted item parameters. "
            "Question-level holdout leaves item-dependent metrics undefined."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = load_config_defaults(parse_args())
    if args.mode not in {"arena", "both"}:
        raise SystemExit("--mode must be 'arena' or 'both'.")
    if args.mode == "both" and not args.static_jsonl:
        raise SystemExit("Mode 'both' requires static_jsonl.")
    if not args.arena_reward_jsonl:
        raise SystemExit("Need arena_reward_jsonl.")

    static_df = dualeval.load_static_jsonl(args.static_jsonl) if args.static_jsonl else pd.DataFrame()
    reward_df = dualeval.load_arena_reward_jsonl(args.arena_reward_jsonl)
    if reward_df.empty:
        raise SystemExit("No usable arena reward rows loaded.")

    if args.mode == "both":
        if args.split_level == "question":
            static_train, static_eval, static_eval_questions = split_questions(
                static_df,
                question_col="question_id",
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
        else:
            static_train, static_eval, static_eval_questions = split_rows(
                static_df,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
    else:
        static_train, static_eval, static_eval_questions = pd.DataFrame(), pd.DataFrame(), set()
    if args.split_level == "question":
        reward_train, reward_eval, arena_eval_questions = split_questions(
            reward_df,
            question_col="question_id",
            test_fraction=args.test_fraction,
            seed=args.seed + 1,
        )
    else:
        reward_train, reward_eval, arena_eval_questions = split_rows(
            reward_df,
            test_fraction=args.test_fraction,
            seed=args.seed + 1,
        )

    pairwise_eval_common, eval_bb_threshold, eval_tie_delta = build_pairwise(
        reward_eval,
        bb_ratio=args.bb_ratio,
        tie_ratio=args.tie_ratio,
    )
    if pairwise_eval_common.empty:
        raise SystemExit("No usable pairwise eval rows built.")

    variants = [
        ("full", args.bb_ratio, args.tie_ratio, args.lambda_bb),
        ("no_bb_loss", args.bb_ratio, args.tie_ratio, 0.0),
        ("no_bb_flagging", 0.0, args.tie_ratio, args.lambda_bb),
        ("no_tie_filter", args.bb_ratio, 0.0, args.lambda_bb),
        ("no_bb_no_tie", 0.0, 0.0, args.lambda_bb),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fit_metadata: dict[str, Any] = {}
    for name, bb_ratio, tie_ratio, lambda_bb in variants:
        if not args.quiet:
            print(
                f"\n=== {name}: bb_ratio={bb_ratio}, tie_ratio={tie_ratio}, "
                f"lambda_bb={lambda_bb} ===",
                flush=True,
            )
        row, model_params, question_params, meta = fit_variant(
            variant_name=name,
            static_train=static_train,
            reward_train=reward_train,
            static_eval=static_eval,
            pairwise_eval_common=pairwise_eval_common,
            bb_ratio=bb_ratio,
            tie_ratio=tie_ratio,
            lambda_bb=lambda_bb,
            args=args,
        )
        rows.append(row)
        fit_metadata[name] = meta
        model_params.to_csv(args.output_dir / f"{name}_model_ranking.csv", index=False)
        question_params.to_csv(args.output_dir / f"{name}_question_ranking.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "pair_treatment_ablation_metrics.csv", index=False)
    summary = {
        "config": args.config,
        "mode": args.mode,
        "static_jsonl": args.static_jsonl,
        "arena_reward_jsonl": args.arena_reward_jsonl,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "lambda_static": args.lambda_static,
        "lambda_arena": args.lambda_arena,
        "lambda_bb": args.lambda_bb,
        "reg_lambda": args.reg_lambda,
        "full_bb_ratio": args.bb_ratio,
        "full_tie_ratio": args.tie_ratio,
        "eval_both_bad_threshold": eval_bb_threshold,
        "eval_tie_delta": eval_tie_delta,
        "split_level": args.split_level,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "n_static_train": int(len(static_train)),
        "n_static_eval": int(len(static_eval)),
        "n_static_eval_questions": int(len(static_eval_questions)),
        "n_reward_train": int(len(reward_train)),
        "n_reward_eval": int(len(reward_eval)),
        "n_arena_eval_questions": int(len(arena_eval_questions)),
        "n_pairwise_eval_common": int(len(pairwise_eval_common)),
        "fit_metadata": fit_metadata,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False, float_format=lambda x: "nan" if math.isnan(x) else f"{x:.4f}"))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
