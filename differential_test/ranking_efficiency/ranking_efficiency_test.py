#!/usr/bin/env python3
"""
Compares exactly three prompt groups:
1) top 10% by discrimination_exp_k
2) bottom 10% by discrimination_exp_k
3) random prompts with the same count as the 10% groups

Additionally, when --run-convergence-sweep is set:
  * Sweeps N from a small value up to the full pool size for each strategy
    (high_discrimination, random_baseline, low_discrimination).
  * At each N, fits a ranking (with multiple random seeds for error bars) and
    measures Spearman / Kendall / top-k vs. the reference ranking.
  * Identifies the *convergence point* — the smallest N where mean Spearman
    meets --stability-threshold — and reports how many questions / total model
    queries each strategy needs relative to the random baseline.

When --model-coverage-fractions is also supplied:
  * Sweeps the fraction of models evaluated per question (over the high-
    discrimination pool at its convergence point).  Shows that sparse model
    coverage still yields stable rankings, enabling a second axis of cost
    reduction beyond question selection.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RANKING_DIR = REPO_ROOT / "ranking"
RANK_V1_PATH = RANKING_DIR / "rank_v1.py"

if str(RANKING_DIR) not in sys.path:
    sys.path.insert(0, str(RANKING_DIR))


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _configure_torch_runtime() -> None:
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def _load_rank_v1_fit_fn() -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location("rank_v1_module", str(RANK_V1_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load rank_v1 module from {RANK_V1_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_irt_v1


@dataclass(frozen=True)
class ScoreSet:
    name: str
    question_ids: np.ndarray


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ranking efficiency experiment.")
    parser.add_argument(
        "--question-ranking",
        required=True,
        help=(
            "Path to question params input. Accepts either a CSV containing "
            "{question_id, discrimination_exp_k} or a directory containing "
            "static_question_params.csv and/or arena_question_params.csv."
        ),
    )
    parser.add_argument(
        "--arena-jsonl",
        required=True,
        help="Arena reward JSONL file used to estimate rankings from sampled prompts.",
    )
    parser.add_argument(
        "--static-jsonl",
        default=None,
        help=(
            "Optional static eval JSONL. If provided, each group ranking is fit jointly "
            "with static data plus sampled arena prompts."
        ),
    )
    parser.add_argument(
        "--reference-model-ranking",
        required=True,
        help="Reference model ranking CSV (columns: model_name, theta).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.10,
        help="Top/bottom percentile for high/low discrimination prompt pools.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random-group prompt selection.",
    )
    parser.add_argument(
        "--exclude-from-random",
        action="store_true",
        help="If set, random group excludes questions already selected in top/bottom groups.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for metrics outputs.",
    )
    parser.add_argument(
        "--rank-rm-num-epochs",
        type=int,
        default=1500,
        help="Epochs used by rank_v1 IRT fitting for each prompt group.",
    )
    parser.add_argument(
        "--rank-rm-lr",
        type=float,
        default=0.02,
        help="Learning rate used by rank_v1 IRT fitting.",
    )
    parser.add_argument(
        "--rank-rm-lambda-arena",
        type=float,
        default=1.0,
        help="Arena/reward loss weight used by rank_v1 IRT fitting.",
    )
    parser.add_argument(
        "--rank-rm-lambda-static",
        type=float,
        default=0.395,
        help="Static loss weight used by rank_v1 IRT fitting when --static-jsonl is provided.",
    )
    parser.add_argument(
        "--rank-rm-reg-lambda",
        type=float,
        default=1e-4,
        help="L2 regularization used by rank_v1 IRT fitting.",
    )
    parser.add_argument(
        "--rank-rm-quiet",
        action="store_true",
        help="Suppress rank_v1 training logs.",
    )

    # ------------------------------------------------------------------
    # Convergence sweep
    # ------------------------------------------------------------------
    parser.add_argument(
        "--run-convergence-sweep",
        action="store_true",
        help=(
            "Sweep N questions from small to the full pool for each strategy "
            "(high_discrimination, random_baseline, low_discrimination) and "
            "measure ranking stability.  Outputs convergence_sweep.csv and "
            "convergence_summary.json to {output-dir}/sweep/."
        ),
    )
    parser.add_argument(
        "--sweep-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of question-count values for the convergence sweep. "
            "If omitted, values are auto-generated on a log scale from 5 to pool size."
        ),
    )
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=3,
        help="Number of independent random samples per (strategy, N) point (for error bars).",
    )
    parser.add_argument(
        "--sweep-epochs",
        type=int,
        default=300,
        help=(
            "Epochs for rank_v1 fitting during the convergence sweep. "
            "Kept lower than --rank-rm-num-epochs for speed."
        ),
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.95,
        help="Spearman rho threshold used to declare ranking convergence.",
    )

    # ------------------------------------------------------------------
    # Model-coverage sweep
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model-coverage-fractions",
        type=str,
        default=None,
        help=(
            "Comma-separated fractions in (0, 1] specifying what fraction of models "
            "are evaluated per question in the model-coverage sweep.  "
            "Example: '0.2,0.4,0.6,0.8,1.0'.  "
            "Requires --run-convergence-sweep to determine the question count to use. "
            "If omitted, the model-coverage sweep is skipped."
        ),
    )
    parser.add_argument(
        "--coverage-sweep-questions",
        type=int,
        default=None,
        help=(
            "Fix the number of high-discrimination questions used in the model-coverage "
            "sweep.  If omitted, the convergence point found by --run-convergence-sweep "
            "is used (falling back to half the pool)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_question_ranking_input(question_ranking_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if question_ranking_path.is_dir():
        for fname in ("static_question_params.csv", "arena_question_params.csv"):
            f = question_ranking_path / fname
            if f.exists():
                frames.append(pd.read_csv(f))
        if not frames:
            raise ValueError(
                f"No question parameter CSVs found in directory {question_ranking_path}. "
                "Expected static_question_params.csv and/or arena_question_params.csv."
            )
    else:
        frames.append(pd.read_csv(question_ranking_path))
    return pd.concat(frames, ignore_index=True)


def _load_question_buckets(
    question_ranking_path: Path, percentile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qdf = _load_question_ranking_input(question_ranking_path)
    needed = {"question_id", "discrimination_exp_k"}
    missing = needed.difference(qdf.columns)
    if missing:
        raise ValueError(f"question_ranking.csv missing columns: {sorted(missing)}")

    qdf = qdf.copy()
    qdf["item_id"] = qdf["question_id"].astype(str).str.split("::", n=1).str[-1]
    qdf = qdf.drop_duplicates(subset=["item_id"], keep="last")

    if not (0.0 < percentile <= 1.00):
        raise ValueError("--percentile must be in (0, 1.00).")
    n = len(qdf)
    k = max(1, int(np.floor(n * percentile)))

    sorted_q = qdf.sort_values("discrimination_exp_k", ascending=False).reset_index(drop=True)
    high_ids = sorted_q.head(k)["item_id"].to_numpy()
    low_ids = sorted_q.tail(k)["item_id"].to_numpy()
    all_ids = sorted_q["item_id"].to_numpy()
    return all_ids, high_ids, low_ids


def _load_arena_rewards(arena_jsonl_path: Path) -> pd.DataFrame:
    df = pd.read_json(arena_jsonl_path, lines=True)
    needed = {"item_id", "model_label", "reward", "status"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Arena JSONL missing columns: {sorted(missing)}")
    df = df[(df["status"] == "ok") & df["reward"].notna()].copy()
    df["item_id"] = df["item_id"].astype(str)
    df["model_name"] = df["model_label"].astype(str)
    df["reward"] = df["reward"].astype(float)
    return df[["item_id", "model_name", "reward"]]


def _to_rank_rm_reward_df(reward_df: pd.DataFrame) -> pd.DataFrame:
    out = reward_df.copy()
    out["question_id"] = out["item_id"].astype(str)
    out["source"] = "efficiency_test"
    out["benchmark"] = "Arena"
    out["reward_raw"] = out["reward"].astype(float)
    mean = float(out["reward_raw"].mean())
    std = float(out["reward_raw"].std(ddof=0))
    if std < 1e-12:
        # Degenerate slice: no variance in rewards, so use a zero z-signal.
        out["reward_z"] = 0.0
    else:
        out["reward_z"] = (out["reward_raw"] - mean) / std
    return out[["source", "benchmark", "model_name", "question_id", "reward_raw", "reward_z"]]


def _load_static_jsonl(static_jsonl_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with static_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("status") != "ok" or d.get("correct") is None:
                continue
            dataset = str(d.get("dataset", "unknown"))
            sample_index = d.get("sample_index")
            question_id = f"{dataset}_{sample_index}"
            rows.append(
                {
                    "source": "efficiency_test_static",
                    "benchmark": dataset,
                    "model_name": str(d["model_label"]),
                    "question_id": question_id,
                    "judge_result": int(bool(d["correct"])),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["model_name", "question_id"], keep="last").reset_index(drop=True)


def _load_reference_ranking(reference_path: Path) -> pd.Series:
    ref = pd.read_csv(reference_path)
    needed = {"model_name", "theta"}
    missing = needed.difference(ref.columns)
    if missing:
        raise ValueError(f"reference model ranking missing columns: {sorted(missing)}")
    ref = ref.sort_values("theta", ascending=False).reset_index(drop=True)
    return pd.Series(np.arange(len(ref), dtype=float), index=ref["model_name"].astype(str))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _spearman_from_ranks(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    if rank_a.size < 2:
        return float("nan")
    corr = np.corrcoef(rank_a, rank_b)[0, 1]
    return float(corr)


def _kendall_tau_from_ranks(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    n = rank_a.size
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        da = rank_a[i] - rank_a[i + 1 :]
        db = rank_b[i] - rank_b[i + 1 :]
        prod = da * db
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
    denom = n * (n - 1) / 2
    return float((concordant - discordant) / denom) if denom else float("nan")


def _topk_overlap(pred_order: pd.Index, ref_order: pd.Index, k: int) -> float:
    k = min(k, len(pred_order), len(ref_order))
    if k <= 0:
        return float("nan")
    pred_top = set(pred_order[:k].tolist())
    ref_top = set(ref_order[:k].tolist())
    return float(len(pred_top & ref_top) / k)


# ---------------------------------------------------------------------------
# Ranking estimation
# ---------------------------------------------------------------------------

def _fallback_theta_from_rewards_and_static(
    rank_rm_reward_df: pd.DataFrame,
    static_df: pd.DataFrame,
    lambda_static: float,
) -> pd.Series:
    arena_theta = rank_rm_reward_df.groupby("model_name", sort=False)["reward_raw"].mean()
    if arena_theta.empty and not static_df.empty:
        return static_df.groupby("model_name", sort=False)["judge_result"].mean().sort_values(ascending=False)
    if static_df.empty or lambda_static <= 0:
        return arena_theta.sort_values(ascending=False)

    static_theta = static_df.groupby("model_name", sort=False)["judge_result"].mean()
    common = arena_theta.index.intersection(static_theta.index)
    if len(common) < 2:
        return arena_theta.sort_values(ascending=False)

    arena_common = arena_theta.loc[common]
    static_common = static_theta.loc[common]

    arena_std = float(arena_common.std(ddof=0))
    static_std = float(static_common.std(ddof=0))
    if arena_std < 1e-12:
        arena_std = 1.0
    if static_std < 1e-12:
        static_std = 1.0

    arena_z = (arena_common - float(arena_common.mean())) / arena_std
    static_z = (static_common - float(static_common.mean())) / static_std

    mixed = arena_z + float(lambda_static) * static_z
    out = arena_theta.copy()
    out.loc[common] = mixed
    return out.sort_values(ascending=False)


def _estimate_ranking_with_rank_v1(
    reward_df: pd.DataFrame,
    sampled_item_ids: np.ndarray,
    static_df: pd.DataFrame,
    fit_fn: Callable[..., Any],
    args: argparse.Namespace,
    num_epochs_override: int | None = None,
) -> pd.Series:
    num_epochs = num_epochs_override if num_epochs_override is not None else args.rank_rm_num_epochs
    sampled_id_set = set(sampled_item_ids.tolist())
    sampled_reward = reward_df[reward_df["item_id"].isin(sampled_id_set)].copy()
    sampled_static = (
        static_df[static_df["question_id"].isin(sampled_id_set)].copy()
        if not static_df.empty
        else pd.DataFrame()
    )
    if sampled_reward.empty and sampled_static.empty:
        return pd.Series(dtype=float)

    reward_fmt = _to_rank_rm_reward_df(sampled_reward) if not sampled_reward.empty else pd.DataFrame()
    try:
        model_params, _, _, _ = fit_fn(
            static_df=sampled_static if not sampled_static.empty else None,
            pairwise_df=None,
            reward_df=reward_fmt if not reward_fmt.empty else None,
            arena_mode="pairwise+regression",
            num_epochs=num_epochs,
            lr=args.rank_rm_lr,
            lambda_static=args.rank_rm_lambda_static if not sampled_static.empty else 0.0,
            lambda_arena=args.rank_rm_lambda_arena if not reward_fmt.empty else 0.0,
            lambda_reg=1.0,
            reg_lambda=args.rank_rm_reg_lambda,
            verbose=not args.rank_rm_quiet,
        )
        return model_params.set_index("model_name")["theta"].sort_values(ascending=False)
    except (MemoryError, RuntimeError) as exc:
        if not args.rank_rm_quiet:
            print(f"[ranking_efficiency_test] rank_v1 torch fit failed ({type(exc).__name__}); using lightweight fallback.")
        lambda_static = args.rank_rm_lambda_static if not sampled_static.empty else 0.0
        return _fallback_theta_from_rewards_and_static(reward_fmt, sampled_static, lambda_static)


def _ranking_to_df(group_name: str, pred_scores: pd.Series) -> pd.DataFrame:
    out = pred_scores.reset_index()
    out.columns = ["model_name", "theta"]
    out["group"] = group_name
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["group", "rank", "model_name", "theta"]]


# ---------------------------------------------------------------------------
# Convergence-sweep helpers
# ---------------------------------------------------------------------------

def _auto_sweep_sizes(max_n: int, n_points: int = 14) -> list[int]:
    """Return N values on a roughly log scale from 5 up to max_n."""
    min_n = min(5, max_n)
    if max_n <= min_n:
        return [max_n]
    raw = np.geomspace(min_n, max_n, n_points)
    sizes = sorted(set(int(round(x)) for x in raw))
    return [s for s in sizes if s <= max_n]


def _models_per_question_stats(reward_df: pd.DataFrame, item_ids: np.ndarray) -> dict:
    """Return response-count statistics for the given question IDs."""
    sub = reward_df[reward_df["item_id"].isin(set(item_ids.tolist()))]
    counts = sub.groupby("item_id")["model_name"].nunique()
    if counts.empty:
        return {"mean_models_per_question": 0.0, "median_models_per_question": 0.0, "total_responses": 0}
    return {
        "mean_models_per_question": float(counts.mean()),
        "median_models_per_question": float(counts.median()),
        "total_responses": int(len(sub)),
    }


def _compute_ranking_metrics(
    pred_scores: pd.Series,
    ref_rank_pos: pd.Series,
) -> dict | None:
    """Return a dict of ranking quality metrics, or None if insufficient overlap."""
    common = pred_scores.index.intersection(ref_rank_pos.index)
    if len(common) < 2:
        return None
    pred_rank_pos = pd.Series(np.arange(len(pred_scores), dtype=float), index=pred_scores.index)
    pred_common = pred_rank_pos.loc[common].to_numpy()
    ref_common = ref_rank_pos.loc[common].to_numpy()
    pred_order = pred_scores.index
    ref_order = ref_rank_pos.sort_values().index
    return {
        "spearman": _spearman_from_ranks(pred_common, ref_common),
        "kendall": _kendall_tau_from_ranks(pred_common, ref_common),
        "top3_overlap": _topk_overlap(pred_order, ref_order, k=3),
        "top5_overlap": _topk_overlap(pred_order, ref_order, k=5),
        "top1_match": float(pred_order[0] == ref_order[0]),
        "n_models_compared": int(len(common)),
    }


# ---------------------------------------------------------------------------
# Convergence sweep
# ---------------------------------------------------------------------------

def run_convergence_sweep(
    reward_df: pd.DataFrame,
    static_df: pd.DataFrame,
    high_pool: np.ndarray,
    low_pool: np.ndarray,
    all_pool: np.ndarray,
    ref_rank_pos: pd.Series,
    fit_fn: Callable[..., Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    """
    For each strategy (high_discrimination, random_baseline, low_discrimination),
    sweep N questions from small to the full pool and measure ranking quality.

    Returns
    -------
    sweep_df : pd.DataFrame
        One row per (strategy, n_questions) with mean/std metrics across repeats.
    convergence_summary : dict
        Convergence points and reduction factors vs. the random baseline.
    """
    # Parse sweep sizes
    if args.sweep_sizes:
        sweep_sizes_raw = sorted(
            set(int(x.strip()) for x in args.sweep_sizes.split(",") if x.strip())
        )
    else:
        max_pool = min(len(high_pool), len(low_pool))
        sweep_sizes_raw = _auto_sweep_sizes(max_pool)

    max_allowed_n = min(len(high_pool), len(low_pool))
    sweep_sizes_raw = [n for n in sweep_sizes_raw if n <= max_allowed_n]

    repeats = args.sweep_repeats
    threshold = args.stability_threshold
    num_epochs = args.sweep_epochs

    strategies: dict[str, np.ndarray] = {
        "high_discrimination": high_pool,
        "random_baseline": all_pool,
        "low_discrimination": low_pool,
    }

    master_rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for strategy_name, pool in strategies.items():
        if len(pool) == 0:
            continue
        # Clamp sweep sizes to pool size
        valid_sizes = [n for n in sweep_sizes_raw if n <= len(pool)]
        if not valid_sizes:
            valid_sizes = [len(pool)]

        for n in valid_sizes:
            spearman_vals: list[float] = []
            kendall_vals: list[float] = []
            top3_vals: list[float] = []
            top5_vals: list[float] = []
            total_queries_vals: list[int] = []
            n_models_compared_vals: list[int] = []

            for _ in range(repeats):
                sample_ids = master_rng.choice(pool, size=n, replace=False)

                pred_scores = _estimate_ranking_with_rank_v1(
                    reward_df,
                    sample_ids,
                    static_df,
                    fit_fn,
                    args,
                    num_epochs_override=num_epochs,
                )
                if pred_scores.empty:
                    continue

                m = _compute_ranking_metrics(pred_scores, ref_rank_pos)
                if m is None:
                    continue

                spearman_vals.append(m["spearman"])
                kendall_vals.append(m["kendall"])
                top3_vals.append(m["top3_overlap"])
                top5_vals.append(m["top5_overlap"])
                n_models_compared_vals.append(m["n_models_compared"])

                stats = _models_per_question_stats(reward_df, sample_ids)
                total_queries_vals.append(stats["total_responses"])

            if not spearman_vals:
                continue

            rows.append(
                {
                    "strategy": strategy_name,
                    "n_questions": int(n),
                    "spearman_mean": float(np.nanmean(spearman_vals)),
                    "spearman_std": float(np.nanstd(spearman_vals)),
                    "kendall_mean": float(np.nanmean(kendall_vals)),
                    "kendall_std": float(np.nanstd(kendall_vals)),
                    "top3_mean": float(np.nanmean(top3_vals)),
                    "top5_mean": float(np.nanmean(top5_vals)),
                    "total_queries_mean": float(np.nanmean(total_queries_vals)),
                    "total_queries_std": float(np.nanstd(total_queries_vals)),
                    "n_models_compared": int(np.nanmean(n_models_compared_vals)),
                    "repeats_used": int(len(spearman_vals)),
                }
            )

    sweep_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Convergence points
    # ------------------------------------------------------------------
    convergence_points: dict[str, dict] = {}
    for strategy_name in strategies:
        sub = sweep_df[sweep_df["strategy"] == strategy_name].sort_values("n_questions")
        if sub.empty:
            continue
        above = sub[sub["spearman_mean"] >= threshold]
        if above.empty:
            conv_n: int | None = None
            conv_queries: int | None = None
            conv_spearman: float | None = None
        else:
            first = above.iloc[0]
            conv_n = int(first["n_questions"])
            conv_queries = int(round(first["total_queries_mean"]))
            conv_spearman = float(first["spearman_mean"])
        convergence_points[strategy_name] = {
            "convergence_n_questions": conv_n,
            "convergence_total_queries": conv_queries,
            "spearman_at_convergence": conv_spearman,
            "max_spearman": float(sub["spearman_mean"].max()),
            "full_pool_size": int(len(strategies[strategy_name])),
            "stability_threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Reduction ratios vs. random baseline
    # ------------------------------------------------------------------
    reduction: dict = {}
    ref_cp = convergence_points.get("random_baseline", {})
    our_cp = convergence_points.get("high_discrimination", {})
    ref_n = ref_cp.get("convergence_n_questions")
    our_n = our_cp.get("convergence_n_questions")
    ref_q = ref_cp.get("convergence_total_queries")
    our_q = our_cp.get("convergence_total_queries")

    if ref_n is not None and our_n is not None:
        question_ratio = our_n / ref_n
        reduction["question_reduction"] = {
            "high_discrimination_n": our_n,
            "random_baseline_n": ref_n,
            "ratio": round(question_ratio, 3),
            "reduction_pct": round((1.0 - question_ratio) * 100.0, 1),
            "interpretation": (
                f"Our approach reaches stable ranking (Spearman≥{threshold}) "
                f"with {our_n} questions vs. {ref_n} for random selection "
                f"({round((1.0-question_ratio)*100,1)}% fewer questions)."
            ),
        }
    if ref_q is not None and our_q is not None:
        query_ratio = our_q / ref_q
        reduction["query_reduction"] = {
            "high_discrimination_total_queries": our_q,
            "random_baseline_total_queries": ref_q,
            "ratio": round(query_ratio, 3),
            "reduction_pct": round((1.0 - query_ratio) * 100.0, 1),
            "interpretation": (
                f"Total model queries at convergence: {our_q} (ours) vs. {ref_q} (random). "
                f"{round((1.0-query_ratio)*100,1)}% fewer model queries."
            ),
        }

    convergence_summary: dict = {
        "per_strategy": convergence_points,
        "reduction_vs_random_baseline": reduction,
        "sweep_config": {
            "repeats": repeats,
            "sweep_epochs": num_epochs,
            "stability_threshold": threshold,
            "sweep_sizes": sweep_sizes_raw,
        },
    }
    return sweep_df, convergence_summary


# ---------------------------------------------------------------------------
# Model-coverage sweep
# ---------------------------------------------------------------------------

def run_model_coverage_sweep(
    reward_df: pd.DataFrame,
    high_pool: np.ndarray,
    ref_rank_pos: pd.Series,
    fit_fn: Callable[..., Any],
    args: argparse.Namespace,
    coverage_fractions: list[float],
    n_questions_fixed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    For a fixed number of high-discrimination questions, vary the fraction of
    models evaluated per question and measure ranking quality.

    Parameters
    ----------
    n_questions_fixed : int | None
        Number of high-discrimination questions to use.  If None, uses
        --coverage-sweep-questions or half the pool size as fallback.

    Returns
    -------
    coverage_df : pd.DataFrame
    coverage_summary : dict
    """
    all_models = sorted(reward_df["model_name"].unique().tolist())
    n_models_total = len(all_models)

    n_q = n_questions_fixed
    if n_q is None:
        n_q = args.coverage_sweep_questions if args.coverage_sweep_questions else max(10, len(high_pool) // 2)
    n_q = min(n_q, len(high_pool))

    rng = np.random.default_rng(args.seed + 1000)
    sample_ids = rng.choice(high_pool, size=n_q, replace=False)

    rows: list[dict] = []
    for frac in sorted(coverage_fractions):
        n_m = max(2, int(round(n_models_total * frac)))
        n_m = min(n_m, n_models_total)

        # Sub-sample models independently per question
        sub_parts: list[pd.DataFrame] = []
        for qid in sample_ids:
            q_rows = reward_df[reward_df["item_id"] == qid].copy()
            models_here = q_rows["model_name"].unique()
            if len(models_here) <= n_m:
                sub_parts.append(q_rows)
            else:
                chosen = rng.choice(models_here, size=n_m, replace=False)
                sub_parts.append(q_rows[q_rows["model_name"].isin(chosen)])

        if not sub_parts:
            continue
        sub_df = pd.concat(sub_parts, ignore_index=True)
        total_queries = int(len(sub_df))

        rr_df = _to_rank_rm_reward_df(sub_df)
        try:
            model_params, _, _, _ = fit_fn(
                static_df=None,
                pairwise_df=None,
                reward_df=rr_df,
                arena_mode="pairwise+regression",
                num_epochs=args.sweep_epochs,
                lr=args.rank_rm_lr,
                lambda_static=0.0,
                lambda_arena=args.rank_rm_lambda_arena,
                lambda_reg=1.0,
                reg_lambda=args.rank_rm_reg_lambda,
                verbose=False,
            )
            pred_scores = model_params.set_index("model_name")["theta"].sort_values(ascending=False)
        except (MemoryError, RuntimeError) as exc:
            if not args.rank_rm_quiet:
                print(f"[coverage_sweep] fit failed ({type(exc).__name__}); using mean-reward fallback.")
            pred_scores = sub_df.groupby("model_name")["reward"].mean().sort_values(ascending=False)

        m = _compute_ranking_metrics(pred_scores, ref_rank_pos)
        if m is None:
            continue

        rows.append(
            {
                "coverage_fraction": round(float(frac), 4),
                "n_models_per_question": n_m,
                "n_models_total": n_models_total,
                "n_questions": n_q,
                "total_queries": total_queries,
                "spearman": m["spearman"],
                "kendall": m["kendall"],
                "top3_overlap": m["top3_overlap"],
                "top5_overlap": m["top5_overlap"],
                "top1_match": m["top1_match"],
                "n_models_compared": m["n_models_compared"],
            }
        )

    coverage_df = pd.DataFrame(rows)

    # Build summary: queries vs. full coverage
    full_row = coverage_df[coverage_df["n_models_per_question"] == n_models_total]
    full_queries = int(full_row["total_queries"].iloc[0]) if not full_row.empty else n_q * n_models_total
    full_spearman = float(full_row["spearman"].iloc[0]) if not full_row.empty else float("nan")

    # Find minimum fraction that still achieves ≥ 0.90 spearman
    threshold_90 = 0.90
    above_90 = coverage_df[coverage_df["spearman"] >= threshold_90].sort_values("coverage_fraction")
    min_frac_90 = float(above_90.iloc[0]["coverage_fraction"]) if not above_90.empty else None
    min_n_90 = int(above_90.iloc[0]["n_models_per_question"]) if not above_90.empty else None
    min_q_90 = int(above_90.iloc[0]["total_queries"]) if not above_90.empty else None

    coverage_summary = {
        "n_questions_used": n_q,
        "n_models_total": n_models_total,
        "full_coverage_queries": full_queries,
        "full_coverage_spearman": full_spearman,
        "min_fraction_for_spearman_0.90": min_frac_90,
        "min_models_per_question_for_spearman_0.90": min_n_90,
        "total_queries_at_spearman_0.90": min_q_90,
        "query_reduction_at_0.90": (
            round((1.0 - min_q_90 / full_queries) * 100.0, 1)
            if (min_q_90 and full_queries)
            else None
        ),
    }
    return coverage_df, coverage_summary


# ---------------------------------------------------------------------------
# Original 3-group experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(args.seed)
    _configure_torch_runtime()
    fit_fn = _load_rank_v1_fit_fn()

    question_path = Path(args.question_ranking)
    arena_path = Path(args.arena_jsonl)
    reference_path = Path(args.reference_model_ranking)

    all_ids, high_ids, low_ids = _load_question_buckets(question_path, args.percentile)
    reward_df = _load_arena_rewards(arena_path)
    static_df = _load_static_jsonl(Path(args.static_jsonl)) if args.static_jsonl else pd.DataFrame()
    ref_rank_pos = _load_reference_ranking(reference_path)
    ref_order = ref_rank_pos.sort_values().index

    # Restrict candidate pools to prompts observed in provided inputs.
    observed_ids = set(reward_df["item_id"].unique().tolist())
    if not static_df.empty:
        observed_ids.update(static_df["question_id"].astype(str).unique().tolist())
    all_pool = np.array([qid for qid in all_ids if qid in observed_ids], dtype=object)
    high_pool = np.array([qid for qid in high_ids if qid in observed_ids], dtype=object)
    low_pool = np.array([qid for qid in low_ids if qid in observed_ids], dtype=object)

    if len(high_pool) == 0 or len(low_pool) == 0:
        raise RuntimeError("High/low discrimination pools are empty. Check percentile and inputs.")

    fixed_n = min(len(high_pool), len(low_pool))
    random_candidates = all_pool
    if args.exclude_from_random:
        selected_ids = set(high_pool[:fixed_n].tolist()) | set(low_pool[:fixed_n].tolist())
        random_candidates = np.array([qid for qid in all_pool if qid not in selected_ids], dtype=object)
    if len(random_candidates) < fixed_n:
        raise RuntimeError(
            "Not enough questions available for random sampling after exclusions. "
            f"Needed {fixed_n}, but only {len(random_candidates)} candidates remain."
        )
    random_pool = rng.choice(random_candidates, size=fixed_n, replace=False)

    sampled_groups = [
        ScoreSet("high_discrimination_top", high_pool[:fixed_n]),
        ScoreSet("low_discrimination_bottom", low_pool[:fixed_n]),
        ScoreSet("random_equal_size", random_pool),
    ]

    metric_rows: list[dict] = []
    ranking_frames: list[pd.DataFrame] = []
    for group in sampled_groups:
        pred_scores = _estimate_ranking_with_rank_v1(reward_df, group.question_ids, static_df, fit_fn, args)
        common_models = pred_scores.index.intersection(ref_rank_pos.index)
        if len(common_models) < 2:
            continue
        pred_rank_pos = pd.Series(np.arange(len(pred_scores), dtype=float), index=pred_scores.index)
        pred_common = pred_rank_pos.loc[common_models].to_numpy()
        ref_common = ref_rank_pos.loc[common_models].to_numpy()
        pred_order = pred_scores.index

        metric_rows.append(
            {
                "group": group.name,
                "sample_size": int(len(group.question_ids)),
                "n_models_compared": int(len(common_models)),
                "ranking_method": "rank_v1_pairwise_regression_irt_with_static" if not static_df.empty else "rank_v1_pairwise_regression_irt",
                "spearman_rho": _spearman_from_ranks(pred_common, ref_common),
                "kendall_tau": _kendall_tau_from_ranks(pred_common, ref_common),
                "top1_match": float(pred_order[0] == ref_order[0]),
                "top3_overlap": _topk_overlap(pred_order, ref_order, k=3),
                "top5_overlap": _topk_overlap(pred_order, ref_order, k=5),
            }
        )
        ranking_frames.append(_ranking_to_df(group.name, pred_scores))

    summary_df = pd.DataFrame(metric_rows).sort_values("group").reset_index(drop=True)
    rankings_df = pd.concat(ranking_frames, ignore_index=True).sort_values(["group", "rank"]).reset_index(drop=True)

    meta = {
        "question_ranking": str(question_path),
        "arena_jsonl": str(arena_path),
        "static_jsonl": str(args.static_jsonl) if args.static_jsonl else None,
        "reference_model_ranking": str(reference_path),
        "percentile": args.percentile,
        "fixed_sample_size_per_group": int(fixed_n),
        "seed": args.seed,
        "exclude_from_random": bool(args.exclude_from_random),
        "ranking_method": (
            "rank_v1.fit_irt_v1(arena_mode=pairwise+regression, static+arena)"
            if not static_df.empty
            else "rank_v1.fit_irt_v1(arena_mode=pairwise+regression, arena-only)"
        ),
        "rank_rm_hparams": {
            "num_epochs": args.rank_rm_num_epochs,
            "lr": args.rank_rm_lr,
            "lambda_static": args.rank_rm_lambda_static if not static_df.empty else 0.0,
            "lambda_arena": args.rank_rm_lambda_arena,
            "reg_lambda": args.rank_rm_reg_lambda,
            "quiet": args.rank_rm_quiet,
            "allow_torch_dynamo": False,
            "allow_fallback": True,
        },
        "pool_sizes": {
            "all_questions": int(len(all_pool)),
            "high_discrimination_top": int(len(high_pool)),
            "low_discrimination_bottom": int(len(low_pool)),
            "random_candidates": int(len(random_candidates)),
        },
    }
    return summary_df, rankings_df, meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    suffix = f"percentile_{int(args.percentile * 100)}"
    out_dir = Path(args.output_dir) / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    _configure_torch_runtime()
    fit_fn = _load_rank_v1_fit_fn()

    question_path = Path(args.question_ranking)
    arena_path = Path(args.arena_jsonl)
    reference_path = Path(args.reference_model_ranking)

    all_ids, high_ids, low_ids = _load_question_buckets(question_path, args.percentile)
    reward_df = _load_arena_rewards(arena_path)
    static_df = _load_static_jsonl(Path(args.static_jsonl)) if args.static_jsonl else pd.DataFrame()
    ref_rank_pos = _load_reference_ranking(reference_path)

    observed_ids = set(reward_df["item_id"].unique().tolist())
    if not static_df.empty:
        observed_ids.update(static_df["question_id"].astype(str).unique().tolist())
    all_pool = np.array([qid for qid in all_ids if qid in observed_ids], dtype=object)
    high_pool = np.array([qid for qid in high_ids if qid in observed_ids], dtype=object)
    low_pool = np.array([qid for qid in low_ids if qid in observed_ids], dtype=object)

    # ------------------------------------------------------------------
    # Original 3-group experiment (always run)
    # ------------------------------------------------------------------
    summary_df, rankings_df, meta = run_experiment(args)
    summary_path = out_dir / "summary_metrics.csv"
    rankings_path = out_dir / "group_model_rankings.csv"
    meta_path = out_dir / "run_config.json"

    summary_df.to_csv(summary_path, index=False)
    rankings_df.to_csv(rankings_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved discrimination usefulness outputs:")
    print(f"  summary: {summary_path}")
    print(f"  rankings:{rankings_path}")
    print(f"  config:  {meta_path}")
    print("\nSummary preview:")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Convergence sweep (opt-in)
    # ------------------------------------------------------------------
    if not args.run_convergence_sweep:
        return

    print("\n" + "=" * 72)
    print("CONVERGENCE SWEEP")
    print("=" * 72)

    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_df, convergence_summary = run_convergence_sweep(
        reward_df=reward_df,
        static_df=static_df,
        high_pool=high_pool,
        low_pool=low_pool,
        all_pool=all_pool,
        ref_rank_pos=ref_rank_pos,
        fit_fn=fit_fn,
        args=args,
    )

    sweep_csv = sweep_dir / "convergence_sweep.csv"
    summary_json = sweep_dir / "convergence_summary.json"
    sweep_df.to_csv(sweep_csv, index=False)
    summary_json.write_text(json.dumps(convergence_summary, indent=2), encoding="utf-8")

    print(f"\nConvergence sweep saved to {sweep_dir}/")
    print(f"  sweep table: {sweep_csv}")
    print(f"  summary:     {summary_json}")
    print("\nConvergence points:")
    threshold = args.stability_threshold
    per_strat = convergence_summary.get("per_strategy", {})
    for strat, cp in per_strat.items():
        n = cp.get("convergence_n_questions")
        q = cp.get("convergence_total_queries")
        s = cp.get("spearman_at_convergence")
        pool_sz = cp.get("full_pool_size")
        n_str = f"{n} / {pool_sz}" if n is not None else f"not reached (max Spearman={cp.get('max_spearman', float('nan')):.3f})"
        q_str = f"{q}" if q is not None else "N/A"
        s_str = f"{s:.3f}" if s is not None else "N/A"
        print(f"  {strat:<28s}  N_questions={n_str}  total_queries={q_str}  spearman={s_str}")

    red = convergence_summary.get("reduction_vs_random_baseline", {})
    if "question_reduction" in red:
        r = red["question_reduction"]
        print(f"\nQuestion reduction (high_disc vs random):  {r['reduction_pct']}%  ({r['high_discrimination_n']} vs {r['random_baseline_n']} questions)")
    if "query_reduction" in red:
        r = red["query_reduction"]
        print(f"Query reduction    (high_disc vs random):  {r['reduction_pct']}%  ({r['high_discrimination_total_queries']} vs {r['random_baseline_total_queries']} queries)")

    # ------------------------------------------------------------------
    # Model-coverage sweep (opt-in via --model-coverage-fractions)
    # ------------------------------------------------------------------
    if not args.model_coverage_fractions:
        return

    fracs_raw = [float(x.strip()) for x in args.model_coverage_fractions.split(",") if x.strip()]
    if not fracs_raw:
        return

    print("\n" + "=" * 72)
    print("MODEL-COVERAGE SWEEP")
    print("=" * 72)

    # Use convergence point from sweep as the fixed question count
    our_conv_n: int | None = per_strat.get("high_discrimination", {}).get("convergence_n_questions")
    n_q_fixed = our_conv_n if our_conv_n is not None else None  # fallback handled inside

    coverage_df, coverage_summary = run_model_coverage_sweep(
        reward_df=reward_df,
        high_pool=high_pool,
        ref_rank_pos=ref_rank_pos,
        fit_fn=fit_fn,
        args=args,
        coverage_fractions=fracs_raw,
        n_questions_fixed=n_q_fixed,
    )

    coverage_csv = sweep_dir / "model_coverage_sweep.csv"
    coverage_json = sweep_dir / "model_coverage_summary.json"
    coverage_df.to_csv(coverage_csv, index=False)
    coverage_json.write_text(json.dumps(coverage_summary, indent=2), encoding="utf-8")

    print(f"\nModel-coverage sweep saved to {sweep_dir}/")
    print(f"  coverage table: {coverage_csv}")
    print(f"  summary:        {coverage_json}")
    print("\nModel-coverage results:")
    print(coverage_df[["coverage_fraction", "n_models_per_question", "total_queries", "spearman", "top3_overlap"]].to_string(index=False))

    # Combined reduction (question selection × model sparsity)
    min_frac = coverage_summary.get("min_fraction_for_spearman_0.90")
    min_q_90 = coverage_summary.get("total_queries_at_spearman_0.90")
    full_q = coverage_summary.get("full_coverage_queries")
    n_q_used = coverage_summary.get("n_questions_used")

    q_red = red.get("query_reduction", {})
    random_full_q = q_red.get("random_baseline_total_queries")  # queries at convergence with full models

    if min_q_90 is not None and random_full_q is not None:
        combined_ratio = min_q_90 / random_full_q
        print(
            f"\nCombined total-query reduction vs. random baseline at full model coverage:\n"
            f"  Our approach: {n_q_used} questions × {coverage_summary['min_models_per_question_for_spearman_0.90']} "
            f"models/question = {min_q_90} queries\n"
            f"  Random baseline: {random_full_q} queries\n"
            f"  Combined reduction: {round((1-combined_ratio)*100,1)}%  (ratio {round(combined_ratio,3)})"
        )


if __name__ == "__main__":
    main()
