#!/usr/bin/env python3
"""
Ablation: difficulty vs differential are not redundant.

Prompt groups:
1) high_difficulty_low_differential
2) low_difficulty_high_differential
3) high_both
4) low_both

Metrics per group:
- Ranking stability vs reference ranking (Spearman, Kendall, top-k overlap)
- Model separation for adjacent model pairs (distance from tie in win-rate)
"""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class ModelPair:
    model_a: str
    model_b: str
    rank_a: int
    rank_b: int
    theta_gap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Difficulty vs differential ablation.")
    parser.add_argument("--question-ranking", required=True, help="Path to question_ranking.csv.")
    parser.add_argument("--arena-jsonl", required=True, help="Arena reward JSONL path.")
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
        help="Reference ranking CSV with columns: model_name, theta.",
    )
    parser.add_argument(
        "--difficulty-quantile",
        type=float,
        default=0.5,
        help="Quantile threshold for high/low difficulty split (default median).",
    )
    parser.add_argument(
        "--differential-quantile",
        type=float,
        default=0.5,
        help="Quantile threshold for high/low differential split (default median).",
    )
    parser.add_argument(
        "--max-adjacent-pairs",
        type=int,
        default=8,
        help="Max adjacent model pairs for separation metric.",
    )
    parser.add_argument(
        "--max-theta-gap",
        type=float,
        default=None,
        help="Optional max theta gap for pair filtering.",
    )
    parser.add_argument(
        "--rank-v1-num-epochs",
        type=int,
        default=1500,
        help="Epochs for rank_v1 fitting per ablation group.",
    )
    parser.add_argument(
        "--rank-v1-lr",
        type=float,
        default=0.02,
        help="Learning rate for rank_v1 fitting.",
    )
    parser.add_argument(
        "--rank-v1-lambda-arena",
        type=float,
        default=1.0,
        help="Arena pairwise BCE loss weight for rank_v1 fitting.",
    )
    parser.add_argument(
        "--rank-v1-lambda-reg",
        type=float,
        default=1.0,
        help="Arena regression MSE loss weight for rank_v1 fitting (both-pr mode).",
    )
    parser.add_argument(
        "--rank-v1-lambda-static",
        type=float,
        default=0.395,
        help="Static loss weight for rank_v1 fitting when --static-jsonl is provided.",
    )
    parser.add_argument(
        "--rank-v1-reg-lambda",
        type=float,
        default=1e-4,
        help="L2 regularization for rank_v1 fitting.",
    )
    parser.add_argument(
        "--rank-v1-both-bad-threshold",
        type=float,
        default=-0.5,
        help="Z-score threshold below which both responses are considered bad (for pairwise targets).",
    )
    parser.add_argument(
        "--rank-v1-quiet",
        action="store_true",
        help="Suppress rank_v1 training logs.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _configure_torch_runtime() -> None:
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def _load_rank_v1_functions() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    spec = importlib.util.spec_from_file_location("rank_v1_module", RANK_V1_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load rank_v1 module from {RANK_V1_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_irt_v1, mod.build_soft_pairwise_targets, mod.resolve_pairwise_thresholds


def _load_question_df(path: Path) -> pd.DataFrame:
    qdf = pd.read_csv(path)
    needed = {"question_id", "difficulty_b", "discrimination_exp_k"}
    missing = needed.difference(qdf.columns)
    if missing:
        raise ValueError(f"question_ranking.csv missing columns: {sorted(missing)}")
    qdf = qdf.copy()
    qdf["item_id"] = qdf["question_id"].astype(str).str.split("::", n=1).str[-1]
    qdf = qdf.drop_duplicates(subset=["item_id"], keep="last")
    return qdf


def _load_arena(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    needed = {"item_id", "model_label", "reward", "status"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Arena JSONL missing columns: {sorted(missing)}")
    df = df[(df["status"] == "ok") & df["reward"].notna()].copy()
    df["item_id"] = df["item_id"].astype(str)
    df["model_name"] = df["model_label"].astype(str)
    df["reward"] = df["reward"].astype(float)
    return df[["item_id", "model_name", "reward"]]


def _load_static_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
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
                    "source": "ablation_static",
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


def _load_reference(path: Path) -> pd.Series:
    ref = pd.read_csv(path)
    needed = {"model_name", "theta"}
    missing = needed.difference(ref.columns)
    if missing:
        raise ValueError(f"reference ranking missing columns: {sorted(missing)}")
    ref = ref.sort_values("theta", ascending=False).reset_index(drop=True)
    return pd.Series(np.arange(len(ref), dtype=float), index=ref["model_name"].astype(str))


def _adjacent_pairs(path: Path, max_pairs: int, max_theta_gap: float | None) -> list[ModelPair]:
    ref = pd.read_csv(path).sort_values("theta", ascending=False).reset_index(drop=True)
    pairs: list[ModelPair] = []
    for i in range(len(ref) - 1):
        a = ref.iloc[i]
        b = ref.iloc[i + 1]
        gap = float(abs(float(a["theta"]) - float(b["theta"])))
        if max_theta_gap is not None and gap > max_theta_gap:
            continue
        pairs.append(
            ModelPair(
                model_a=str(a["model_name"]),
                model_b=str(b["model_name"]),
                rank_a=i + 1,
                rank_b=i + 2,
                theta_gap=gap,
            )
        )
        if len(pairs) >= max_pairs:
            break
    return pairs


def _kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
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
    return float(len(set(pred_order[:k]) & set(ref_order[:k])) / k)


def _group_item_ids(qdf: pd.DataFrame, observed_ids: set[str], d_q: float, a_q: float) -> dict[str, np.ndarray]:
    d_thr = float(qdf["difficulty_b"].quantile(d_q))
    a_thr = float(qdf["discrimination_exp_k"].quantile(a_q))
    sub = qdf[qdf["item_id"].isin(observed_ids)].copy()
    high_d = sub["difficulty_b"] >= d_thr
    high_a = sub["discrimination_exp_k"] >= a_thr
    groups = {
        "high_difficulty_low_differential": sub[high_d & (~high_a)]["item_id"].to_numpy(dtype=object),
        "low_difficulty_high_differential": sub[(~high_d) & high_a]["item_id"].to_numpy(dtype=object),
        "high_both": sub[high_d & high_a]["item_id"].to_numpy(dtype=object),
        "low_both": sub[(~high_d) & (~high_a)]["item_id"].to_numpy(dtype=object),
    }
    return groups


def _to_rank_v1_reward_df(arena_df: pd.DataFrame) -> pd.DataFrame:
    out = arena_df.copy()
    out["source"] = "ablation"
    out["benchmark"] = "Arena"
    out["question_id"] = out["item_id"].astype(str)
    out["reward_raw"] = out["reward"].astype(float)
    mean = float(out["reward_raw"].mean())
    std = float(out["reward_raw"].std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    out["reward_z"] = (out["reward_raw"] - mean) / std
    return out[["source", "benchmark", "model_name", "question_id", "reward_raw", "reward_z"]]


def _ranking_from_group(
    arena_df: pd.DataFrame,
    static_df: pd.DataFrame,
    item_ids: np.ndarray,
    fit_fn: Callable[..., Any],
    build_pairwise_fn: Callable[..., Any],
    resolve_thresholds_fn: Callable[..., Any],
    args: argparse.Namespace,
) -> pd.Series:
    item_id_set = set(item_ids.tolist())
    sampled_reward = arena_df[arena_df["item_id"].isin(item_id_set)]
    sampled_static = (
        static_df[static_df["question_id"].isin(item_id_set)].copy()
        if not static_df.empty
        else pd.DataFrame()
    )
    if sampled_reward.empty and sampled_static.empty:
        return pd.Series(dtype=float)
    rank_v1_reward = _to_rank_v1_reward_df(sampled_reward) if not sampled_reward.empty else pd.DataFrame()
    pairwise_df = pd.DataFrame()
    if not rank_v1_reward.empty:
        both_bad_threshold, tie_delta = resolve_thresholds_fn(
            rank_v1_reward,
            bb_ratio=None,
            tie_ratio=None,
            both_bad_threshold=args.rank_v1_both_bad_threshold,
            both_bad_use_zscore=True,
        )
        pairwise_df = build_pairwise_fn(
            rank_v1_reward,
            both_bad_threshold=both_bad_threshold,
            both_bad_use_zscore=True,
            tie_delta=tie_delta,
        )
    model_params, _, _, _ = fit_fn(
        sampled_static if not sampled_static.empty else None,
        pairwise_df if not pairwise_df.empty else None,
        rank_v1_reward if not rank_v1_reward.empty else None,
        arena_mode="pairwise+regression",
        num_epochs=args.rank_v1_num_epochs,
        lr=args.rank_v1_lr,
        lambda_static=args.rank_v1_lambda_static if not sampled_static.empty else 0.0,
        lambda_arena=args.rank_v1_lambda_arena if not pairwise_df.empty else 0.0,
        lambda_reg=args.rank_v1_lambda_reg if not rank_v1_reward.empty else 0.0,
        reg_lambda=args.rank_v1_reg_lambda,
        verbose=not args.rank_v1_quiet,
    )
    return model_params.set_index("model_name")["theta"].sort_values(ascending=False)


def _pair_separation(arena_df: pd.DataFrame, item_ids: np.ndarray, pairs: list[ModelPair]) -> tuple[float, int]:
    item_set = set(item_ids.tolist())
    vals: list[float] = []
    for p in pairs:
        sub = arena_df[
            arena_df["item_id"].isin(item_set) & arena_df["model_name"].isin([p.model_a, p.model_b])
        ].copy()
        if sub.empty:
            continue
        pv = sub.pivot_table(index="item_id", columns="model_name", values="reward", aggfunc="mean")
        if p.model_a not in pv.columns or p.model_b not in pv.columns:
            continue
        comp = pv[[p.model_a, p.model_b]].dropna()
        if len(comp) < 2:
            continue
        wins = float((comp[p.model_a] > comp[p.model_b]).sum()) + 0.5 * float(
            (comp[p.model_a] == comp[p.model_b]).sum()
        )
        win_rate = wins / len(comp)
        vals.append(abs(win_rate - 0.5))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), int(len(vals))


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _configure_torch_runtime()
    fit_fn, build_pairwise_fn, resolve_thresholds_fn = _load_rank_v1_functions()
    qdf = _load_question_df(Path(args.question_ranking))
    arena_df = _load_arena(Path(args.arena_jsonl))
    static_df = _load_static_jsonl(Path(args.static_jsonl)) if args.static_jsonl else pd.DataFrame()
    ref_rank_pos = _load_reference(Path(args.reference_model_ranking))
    ref_order = ref_rank_pos.sort_values().index
    pairs = _adjacent_pairs(Path(args.reference_model_ranking), args.max_adjacent_pairs, args.max_theta_gap)

    observed_ids = set(arena_df["item_id"].unique().tolist())
    if not static_df.empty:
        observed_ids.update(static_df["question_id"].astype(str).unique().tolist())
    groups = _group_item_ids(qdf, observed_ids, args.difficulty_quantile, args.differential_quantile)

    rng = np.random.default_rng(args.seed)
    min_group_size = min(len(v) for v in groups.values())

    groups = {
        name: rng.choice(item_ids, size=min_group_size, replace=False)
        for name, item_ids in groups.items()
    }

    rows: list[dict] = []
    ranking_rows: list[dict] = []
    for name, item_ids in groups.items():
        pred_scores = _ranking_from_group(arena_df, static_df, item_ids, fit_fn, build_pairwise_fn, resolve_thresholds_fn, args)
        common_models = pred_scores.index.intersection(ref_rank_pos.index)
        if len(common_models) < 2:
            continue
        pred_rank_pos = pd.Series(np.arange(len(pred_scores), dtype=float), index=pred_scores.index)
        pred_common = pred_rank_pos.loc[common_models].to_numpy()
        ref_common = ref_rank_pos.loc[common_models].to_numpy()
        spearman = float(np.corrcoef(pred_common, ref_common)[0, 1]) if len(common_models) > 1 else float("nan")
        kendall = _kendall_tau(pred_common, ref_common)
        pred_order = pred_scores.index
        sep_mean, sep_pairs = _pair_separation(arena_df, item_ids, pairs)
        rows.append(
            {
                "group": name,
                "n_prompts": int(len(item_ids)),
                "n_models_compared": int(len(common_models)),
                "spearman_rho_vs_ref": spearman,
                "kendall_tau_vs_ref": kendall,
                "top1_match": float(pred_order[0] == ref_order[0]),
                "top3_overlap": _topk_overlap(pred_order, ref_order, 3),
                "top5_overlap": _topk_overlap(pred_order, ref_order, 5),
                "mean_adjacent_pair_separation": sep_mean,
                "n_pairs_used_for_separation": sep_pairs,
            }
        )
        for r, (model, score) in enumerate(pred_scores.items(), start=1):
            ranking_rows.append({"group": name, "rank": r, "model_name": model, "theta": float(score)})

    if not rows:
        raise RuntimeError("No groups had enough overlap for ranking evaluation.")

    metric_df = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    ranking_df = pd.DataFrame(ranking_rows).sort_values(["group", "rank"]).reset_index(drop=True)
    summary = {
        "question_ranking": str(Path(args.question_ranking)),
        "arena_jsonl": str(Path(args.arena_jsonl)),
        "static_jsonl": str(Path(args.static_jsonl)) if args.static_jsonl else None,
        "reference_model_ranking": str(Path(args.reference_model_ranking)),
        "difficulty_quantile": args.difficulty_quantile,
        "differential_quantile": args.differential_quantile,
        "max_adjacent_pairs": args.max_adjacent_pairs,
        "max_theta_gap": args.max_theta_gap,
        "ranking_method": (
            "rank_v1.fit_irt_v1(mode=both-pr, static+arena)"
            if not static_df.empty
            else "rank_v1.fit_irt_v1(mode=both-pr, arena-only)"
        ),
        "rank_v1_hparams": {
            "num_epochs": args.rank_v1_num_epochs,
            "lr": args.rank_v1_lr,
            "lambda_static": args.rank_v1_lambda_static if not static_df.empty else 0.0,
            "lambda_arena": args.rank_v1_lambda_arena,
            "lambda_reg": args.rank_v1_lambda_reg,
            "reg_lambda": args.rank_v1_reg_lambda,
            "both_bad_threshold": args.rank_v1_both_bad_threshold,
            "quiet": args.rank_v1_quiet,
        },
        "n_adjacent_pairs_requested": len(pairs),
        "n_groups_evaluated": int(len(metric_df)),
        "pool_sizes": {k: int(len(v)) for k, v in groups.items()},
    }
    return metric_df, ranking_df, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_df, ranking_df, summary = run_experiment(args)
    metric_path = out_dir / "ablation_group_metrics.csv"
    ranking_path = out_dir / "ablation_group_rankings.csv"
    summary_path = out_dir / "run_summary.json"

    metric_df.to_csv(metric_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved ablation outputs:")
    print(f"  metrics: {metric_path}")
    print(f"  rankings:{ranking_path}")
    print(f"  summary: {summary_path}")
    print("\nMetrics preview:")
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
