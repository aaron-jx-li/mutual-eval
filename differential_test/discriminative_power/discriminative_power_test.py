#!/usr/bin/env python3
"""
Experiment: Do high-differential prompts better distinguish similar-quality models?

For adjacent model pairs in a reference ranking:
1) Evaluate pairwise win-rate on top differential prompts.
2) Evaluate pairwise win-rate on random prompts (same size).
3) Report Wilson confidence intervals and significance of win-rate differences.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RANK_RM_PATH = REPO_ROOT / "ranking" / "rank_rm.py"


@dataclass(frozen=True)
class ModelPair:
    model_a: str
    model_b: str
    rank_a: int
    rank_b: int
    theta_gap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close-model separation experiment.")
    parser.add_argument("--question-ranking", required=True, help="Path to question_ranking.csv.")
    parser.add_argument("--arena-jsonl", required=True, help="Arena reward JSONL path.")
    parser.add_argument(
        "--static-jsonl",
        default=None,
        help=(
            "Optional static eval JSONL. If provided, each sampled-group ranking is fit jointly "
            "with static data plus sampled arena prompts."
        ),
    )
    parser.add_argument(
        "--reference-model-ranking",
        required=True,
        help="Reference ranking CSV with columns: model_name, theta.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.10,
        help="Top percentile used as high-differential prompt set.",
    )
    parser.add_argument(
        "--max-adjacent-pairs",
        type=int,
        default=8,
        help="Maximum number of adjacent model pairs to evaluate.",
    )
    parser.add_argument(
        "--max-theta-gap",
        type=float,
        default=None,
        help="Optional max theta gap for adjacent pairs; skip wider gaps if set.",
    )
    parser.add_argument(
        "--rank-rm-num-epochs",
        type=int,
        default=1500,
        help="Epochs for rank_rm fitting per sampled set.",
    )
    parser.add_argument(
        "--rank-rm-lr",
        type=float,
        default=0.02,
        help="Learning rate for rank_rm fitting.",
    )
    parser.add_argument(
        "--rank-rm-lambda-arena",
        type=float,
        default=1.0,
        help="Arena/reward loss weight for rank_rm fitting.",
    )
    parser.add_argument(
        "--rank-rm-lambda-static",
        type=float,
        default=0.395,
        help="Static loss weight for rank_rm fitting when --static-jsonl is provided.",
    )
    parser.add_argument(
        "--rank-rm-reg-lambda",
        type=float,
        default=1e-4,
        help="L2 regularization for rank_rm fitting.",
    )
    parser.add_argument(
        "--rank-rm-quiet",
        action="store_true",
        help="Suppress rank_rm training logs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20,
        help="Bootstrap resamples for theta-gap confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def _configure_torch_runtime() -> None:
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def _load_rank_rm_fit_fn() -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location("rank_rm_module", RANK_RM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load rank_rm module from {RANK_RM_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_joint_reward_distilled_irt


def _load_question_ranking(path: Path, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    qdf = pd.read_csv(path)
    needed = {"question_id", "discrimination_exp_k"}
    missing = needed.difference(qdf.columns)
    if missing:
        raise ValueError(f"question_ranking.csv missing columns: {sorted(missing)}")
    if not (0.0 < percentile <= 1.0):
        raise ValueError("--percentile must be in (0, 1].")
    qdf = qdf.copy()
    qdf["item_id"] = qdf["question_id"].astype(str).str.split("::", n=1).str[-1]
    qdf = qdf.drop_duplicates(subset=["item_id"], keep="last")
    n = len(qdf)
    k = max(1, int(np.floor(n * percentile)))
    sorted_q = qdf.sort_values("discrimination_exp_k", ascending=False).reset_index(drop=True)
    high_ids = sorted_q.head(k)["item_id"].to_numpy(dtype=object)
    all_ids = sorted_q["item_id"].to_numpy(dtype=object)
    return all_ids, high_ids


def _load_arena_rewards(path: Path) -> pd.DataFrame:
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


def _load_reference_pairs(path: Path, max_pairs: int, max_theta_gap: float | None) -> list[ModelPair]:
    ref = pd.read_csv(path)
    needed = {"model_name", "theta"}
    missing = needed.difference(ref.columns)
    if missing:
        raise ValueError(f"reference model ranking missing columns: {sorted(missing)}")
    ref = ref.sort_values("theta", ascending=False).reset_index(drop=True)
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
                    "source": "discriminative_power_static",
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


def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _to_rank_rm_reward_df(reward_df: pd.DataFrame) -> pd.DataFrame:
    out = reward_df.copy()
    out["source"] = "discriminative_power"
    out["benchmark"] = "Arena"
    out["question_id"] = out["item_id"].astype(str)
    out["reward_raw"] = out["reward"].astype(float)
    return out[["source", "benchmark", "model_name", "question_id", "reward_raw"]]


def _fit_theta_map(
    reward_df: pd.DataFrame,
    static_df: pd.DataFrame,
    item_ids: set[str],
    fit_fn: Callable[..., Any],
    args: argparse.Namespace,
) -> tuple[pd.Series, int]:
    sub = reward_df[reward_df["item_id"].isin(item_ids)].copy()
    sampled_static = (
        static_df[static_df["question_id"].isin(item_ids)].copy()
        if not static_df.empty
        else pd.DataFrame()
    )
    if sub.empty and sampled_static.empty:
        return pd.Series(dtype=float), 0
    rank_rm_reward = _to_rank_rm_reward_df(sub) if not sub.empty else pd.DataFrame()
    model_params, _, _ = fit_fn(
        static_df=sampled_static if not sampled_static.empty else None,
        pairwise_df=None,
        reward_df=rank_rm_reward if not rank_rm_reward.empty else None,
        arena_mode="regression",
        num_epochs=args.rank_rm_num_epochs,
        lr=args.rank_rm_lr,
        lambda_static=args.rank_rm_lambda_static if not sampled_static.empty else 0.0,
        lambda_arena=args.rank_rm_lambda_arena if not rank_rm_reward.empty else 0.0,
        lambda_bb=0.0,
        reg_lambda=args.rank_rm_reg_lambda,
        verbose=not args.rank_rm_quiet,
    )
    theta_map = model_params.set_index("model_name")["theta"]
    return theta_map, int(len(item_ids))


def _bootstrap_theta_gap_ci(
    reward_df: pd.DataFrame,
    static_df: pd.DataFrame,
    item_ids: list[str],
    pair: ModelPair,
    fit_fn: Callable[..., Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not item_ids:
        return float("nan"), float("nan")
    gaps: list[float] = []
    n = len(item_ids)
    for _ in range(max(1, args.bootstrap_samples)):
        sampled = [item_ids[i] for i in rng.integers(0, n, size=n)]
        theta_map, _ = _fit_theta_map(reward_df, static_df, set(sampled), fit_fn, args)
        if pair.model_a in theta_map.index and pair.model_b in theta_map.index:
            gaps.append(float(theta_map[pair.model_a] - theta_map[pair.model_b]))
    if not gaps:
        return float("nan"), float("nan")
    arr = np.array(gaps, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _configure_torch_runtime()
    fit_fn = _load_rank_rm_fit_fn()
    rng = np.random.default_rng(args.seed)
    all_ids, high_ids = _load_question_ranking(Path(args.question_ranking), args.percentile)
    reward_df = _load_arena_rewards(Path(args.arena_jsonl))
    static_df = _load_static_jsonl(Path(args.static_jsonl)) if args.static_jsonl else pd.DataFrame()
    pairs = _load_reference_pairs(
        Path(args.reference_model_ranking),
        max_pairs=args.max_adjacent_pairs,
        max_theta_gap=args.max_theta_gap,
    )
    if not pairs:
        raise RuntimeError("No eligible adjacent model pairs found under current constraints.")

    observed_ids = set(reward_df["item_id"].unique().tolist())
    if not static_df.empty:
        observed_ids.update(static_df["question_id"].astype(str).unique().tolist())
    all_pool = [qid for qid in all_ids if qid in observed_ids]
    high_pool = [qid for qid in high_ids if qid in observed_ids]
    high_pool_set = set(high_pool)

    group_rows: list[dict] = []
    compare_rows: list[dict] = []

    for pair in pairs:
        # Pair-specific eligibility requires both models to appear on prompt.
        pair_rows = reward_df[reward_df["model_name"].isin([pair.model_a, pair.model_b])]
        pair_item_counts = pair_rows.groupby("item_id")["model_name"].nunique()
        eligible_ids = set(pair_item_counts[pair_item_counts >= 2].index.tolist())

        high_candidates = [qid for qid in high_pool if qid in eligible_ids]
        random_candidates = [qid for qid in all_pool if qid in eligible_ids and qid not in high_pool_set]

        n_eval = min(len(high_candidates), len(random_candidates))
        if n_eval < 2:
            continue

        high_sample = set(rng.choice(np.array(high_candidates, dtype=object), size=n_eval, replace=False).tolist())
        random_sample = set(rng.choice(np.array(random_candidates, dtype=object), size=n_eval, replace=False).tolist())

        high_theta_map, high_n = _fit_theta_map(reward_df, static_df, high_sample, fit_fn, args)
        rand_theta_map, rand_n = _fit_theta_map(reward_df, static_df, random_sample, fit_fn, args)
        if (
            pair.model_a not in high_theta_map.index
            or pair.model_b not in high_theta_map.index
            or pair.model_a not in rand_theta_map.index
            or pair.model_b not in rand_theta_map.index
            or high_n < 2
            or rand_n < 2
        ):
            continue

        high_gap = float(high_theta_map[pair.model_a] - high_theta_map[pair.model_b])
        rand_gap = float(rand_theta_map[pair.model_a] - rand_theta_map[pair.model_b])
        high_ci_l, high_ci_u = _bootstrap_theta_gap_ci(
            reward_df, static_df, sorted(high_sample), pair, fit_fn, args, rng
        )
        rand_ci_l, rand_ci_u = _bootstrap_theta_gap_ci(
            reward_df, static_df, sorted(random_sample), pair, fit_fn, args, rng
        )
        gap_diff = abs(high_gap) - abs(rand_gap)
        se = math.sqrt(
            max(
                1e-12,
                ((high_ci_u - high_ci_l) / (2 * 1.96)) ** 2 + ((rand_ci_u - rand_ci_l) / (2 * 1.96)) ** 2,
            )
        )
        z_stat = gap_diff / se
        p_val = 2.0 * _normal_sf(abs(z_stat))

        group_rows.extend(
            [
                {
                    "model_a": pair.model_a,
                    "model_b": pair.model_b,
                    "rank_a": pair.rank_a,
                    "rank_b": pair.rank_b,
                    "theta_gap": pair.theta_gap,
                    "group": "high_differential",
                    "n_prompts": high_n,
                    "theta_model_a": float(high_theta_map[pair.model_a]),
                    "theta_model_b": float(high_theta_map[pair.model_b]),
                    "theta_gap_model_a_minus_b": high_gap,
                    "ci95_theta_gap_low": high_ci_l,
                    "ci95_theta_gap_high": high_ci_u,
                },
                {
                    "model_a": pair.model_a,
                    "model_b": pair.model_b,
                    "rank_a": pair.rank_a,
                    "rank_b": pair.rank_b,
                    "theta_gap": pair.theta_gap,
                    "group": "random",
                    "n_prompts": rand_n,
                    "theta_model_a": float(rand_theta_map[pair.model_a]),
                    "theta_model_b": float(rand_theta_map[pair.model_b]),
                    "theta_gap_model_a_minus_b": rand_gap,
                    "ci95_theta_gap_low": rand_ci_l,
                    "ci95_theta_gap_high": rand_ci_u,
                },
            ]
        )
        compare_rows.append(
            {
                "model_a": pair.model_a,
                "model_b": pair.model_b,
                "rank_a": pair.rank_a,
                "rank_b": pair.rank_b,
                "theta_gap": pair.theta_gap,
                "n_prompts_per_group": n_eval,
                "theta_gap_high": high_gap,
                "theta_gap_random": rand_gap,
                "abs_theta_gap_high": abs(high_gap),
                "abs_theta_gap_random": abs(rand_gap),
                "gap_gain_high_minus_random": abs(high_gap) - abs(rand_gap),
                "z_stat_gap_high_vs_random": z_stat,
                "p_value_gap_high_vs_random": p_val,
                "high_better_separation": bool(abs(high_gap) > abs(rand_gap)),
            }
        )

    if not compare_rows:
        raise RuntimeError("No model pairs had enough paired prompts in both high and random groups.")

    group_df = pd.DataFrame(group_rows).sort_values(["rank_a", "group"]).reset_index(drop=True)
    compare_df = pd.DataFrame(compare_rows).sort_values(["rank_a"]).reset_index(drop=True)
    summary = {
        "question_ranking": str(Path(args.question_ranking)),
        "arena_jsonl": str(Path(args.arena_jsonl)),
        "static_jsonl": str(Path(args.static_jsonl)) if args.static_jsonl else None,
        "reference_model_ranking": str(Path(args.reference_model_ranking)),
        "percentile": args.percentile,
        "max_adjacent_pairs": args.max_adjacent_pairs,
        "max_theta_gap": args.max_theta_gap,
        "seed": args.seed,
        "ranking_method": (
            "rank_rm.fit_joint_reward_distilled_irt(arena_mode=regression, static+arena)"
            if not static_df.empty
            else "rank_rm.fit_joint_reward_distilled_irt(arena_mode=regression, arena-only)"
        ),
        "rank_rm_hparams": {
            "num_epochs": args.rank_rm_num_epochs,
            "lr": args.rank_rm_lr,
            "lambda_static": args.rank_rm_lambda_static if not static_df.empty else 0.0,
            "lambda_arena": args.rank_rm_lambda_arena,
            "reg_lambda": args.rank_rm_reg_lambda,
            "quiet": args.rank_rm_quiet,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "pool_sizes": {
            "all_pool_observed": len(all_pool),
            "high_pool_observed": len(high_pool),
        },
        "pairs_requested": len(pairs),
        "pairs_evaluated": int(len(compare_df)),
        "n_pairs_high_better": int(compare_df["high_better_separation"].sum()),
        "mean_gap_gain_high_minus_random": float(compare_df["gap_gain_high_minus_random"].mean()),
    }
    return group_df, compare_df, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_df, compare_df, summary = run_experiment(args)
    per_group_path = out_dir / "pair_group_winrate_stats.csv"
    comparison_path = out_dir / "pair_high_vs_random_comparison.csv"
    summary_path = out_dir / "run_summary.json"

    group_df.to_csv(per_group_path, index=False)
    compare_df.to_csv(comparison_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved close-model separation outputs:")
    print(f"  per-group stats: {per_group_path}")
    print(f"  pair comparison: {comparison_path}")
    print(f"  summary:         {summary_path}")
    print("\nPair comparison preview:")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
