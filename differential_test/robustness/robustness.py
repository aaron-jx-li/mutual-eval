#!/usr/bin/env python3
"""
Robustness test: are item-level properties invariant across model subsets?

Uses ``rank_v1.fit_irt_v1`` with ``arena_mode='pairwise+regression'`` (i.e. the
``both-pr`` mode when static data is supplied, ``arena-pr`` otherwise).

Supports two evaluation modes:
1) random_subset:
   - sample random model subsets for each fraction
   - compare subset item rankings vs full-model reference
2) disjoint_partition:
   - split models into disjoint groups (e.g., 3 groups of 5)
   - compare each group vs reference and group-vs-group alignments
   - if ``partition_count * models_per_partition`` exceeds the model pool size,
     groups are sampled with overlap (each group internally without replacement)
     so the experiment still runs; the realised inter-group overlap is recorded
     per repeat.

Headline metric for the invariance claim:
   Spearman rank correlation of difficulty_b and discrimination_exp_k between
   item parameter sets (subset-vs-reference or group-vs-group).  Top-k overlap
   columns are supplementary; with --topk-fraction set, the per-row k scales
   with the common-item count so the metric is comparable across rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
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

from rank_rm import (  # noqa: E402  -- shared data loaders / pairwise helpers
    build_soft_pairwise_targets,
    load_arena_reward_jsonl,
    load_static_jsonl,
    resolve_pairwise_thresholds,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Item-property invariance robustness experiment.")
    parser.add_argument(
        "--question-ranking",
        required=True,
        help=(
            "Reference (full-model) item parameters.  Accepts either a single CSV "
            "with columns {question_id, difficulty_b, discrimination_exp_k} or a "
            "directory containing rank_v1's static_question_params.csv and/or "
            "arena_question_params.csv (which will be concatenated)."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--arena-jsonl", default=None, help="Arena reward JSONL path.")
    input_group.add_argument("--static-jsonl", default=None, help="Static eval JSONL path.")
    parser.add_argument(
        "--reference-model-ranking",
        default=None,
        help="Optional model_ranking.csv. If provided, subset sampling is restricted to these models.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["random_subset", "disjoint_partition"],
        default="random_subset",
        help="Evaluation design: random subset fractions or disjoint model partitions.",
    )
    parser.add_argument(
        "--subset-fractions",
        default="0.5,0.7,0.9",
        help="Comma-separated model subset fractions in (0,1], e.g. 0.4,0.6,0.8",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of random subsets / partition draws per condition.")
    parser.add_argument("--partition-count", type=int, default=3, help="Number of partition groups.")
    parser.add_argument("--models-per-partition", type=int, default=5, help="Models per partition group.")
    parser.add_argument(
        "--allow-partition-overlap",
        action="store_true",
        help=(
            "If set, allow partition groups to share models when "
            "partition_count * models_per_partition exceeds the model pool size. "
            "Each group is still sampled without replacement internally; only the "
            "across-group disjointness constraint is relaxed.  Realised inter-group "
            "overlap is recorded per repeat."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Absolute cap on top-k for hardest / most-discriminative overlap. Set <=0 to disable the cap.",
    )
    parser.add_argument(
        "--topk-fraction",
        type=float,
        default=0.1,
        help=(
            "Top-k as a fraction of common items per row, making top-k overlap comparable "
            "across rows with differing n_items_common.  When set together with --topk, "
            "the effective k is min(topk_cap, ceil(fraction * n_items_common)).  Set to 0 "
            "to use only the absolute --topk cap (legacy behavior; not comparable across rows)."
        ),
    )
    parser.add_argument("--min-models", type=int, default=4, help="Minimum sampled models per random subset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--rank-rm-num-epochs", type=int, default=1500)
    parser.add_argument("--rank-rm-lr", type=float, default=0.02)
    parser.add_argument("--rank-rm-lambda-arena", type=float, default=1.0)
    parser.add_argument("--rank-rm-lambda-static", type=float, default=0.395)
    parser.add_argument(
        "--rank-rm-lambda-reg",
        type=float,
        default=1.0,
        help="rank_v1 regression-loss weight (pairwise+regression mode).",
    )
    parser.add_argument("--rank-rm-reg-lambda", type=float, default=1e-4)
    parser.add_argument(
        "--rank-rm-both-bad-threshold",
        type=float,
        default=-0.5,
        help="z-score threshold for both-bad pairs when building pairwise targets.",
    )
    parser.add_argument("--rank-rm-quiet", action="store_true")
    return parser.parse_args()


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _item_id_series(question_id_series: pd.Series) -> pd.Series:
    return question_id_series.astype(str).str.split("::", n=1).str[-1]


def _source_item_key(source_series: pd.Series, item_id_series: pd.Series) -> pd.Series:
    # Keep arena/static item spaces disjoint so comparisons never mix them.
    return source_series.astype(str) + "::" + item_id_series.astype(str)


def _load_reference_question_params(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if path.is_dir():
        for fname in ("static_question_params.csv", "arena_question_params.csv"):
            f = path / fname
            if f.exists():
                src = "static" if fname.startswith("static_") else "arena"
                frame = pd.read_csv(f)
                frame["set_source"] = src
                frames.append(frame)
        if not frames:
            raise ValueError(
                f"No question_params CSVs found in directory {path}. "
                f"Expected static_question_params.csv and/or arena_question_params.csv."
            )
    else:
        frame = pd.read_csv(path)
        if "set_source" not in frame.columns:
            frame["set_source"] = "unknown"
        frames.append(frame)
    qdf = pd.concat(frames, ignore_index=True)
    needed = {"question_id", "difficulty_b", "discrimination_exp_k"}
    missing = needed.difference(qdf.columns)
    if missing:
        raise ValueError(f"Reference question params missing columns: {sorted(missing)}")
    out = qdf.copy()
    out["item_id"] = _item_id_series(out["question_id"])
    out["source_item_id"] = _source_item_key(out["set_source"], out["item_id"])
    out = out.dropna(subset=["difficulty_b", "discrimination_exp_k"])
    out = out.drop_duplicates(subset=["source_item_id"], keep="last")
    return out[["set_source", "item_id", "source_item_id", "difficulty_b", "discrimination_exp_k"]].reset_index(drop=True)


def _load_model_pool(
    reference_model_ranking: Path | None,
    arena_df: pd.DataFrame,
    static_df: pd.DataFrame,
) -> list[str]:
    if reference_model_ranking is not None:
        mdf = pd.read_csv(reference_model_ranking)
        if "model_name" not in mdf.columns:
            raise ValueError("reference model ranking missing 'model_name' column.")
        return mdf["model_name"].astype(str).drop_duplicates().tolist()
    pool: set[str] = set()
    if not arena_df.empty:
        pool.update(arena_df["model_name"].astype(str).unique().tolist())
    if not static_df.empty:
        pool.update(static_df["model_name"].astype(str).unique().tolist())
    return sorted(pool)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _resolve_topk(n_common: int, topk_fraction: float, topk_cap: int) -> int:
    """Return the effective k for top-k overlap on a given common-item count.

    With topk_fraction > 0, k scales with n_common (so top-k is comparable across
    rows).  topk_cap (>0) is an additional upper bound; <=0 means uncapped.
    """
    if n_common <= 0:
        return 0
    cap = topk_cap if topk_cap and topk_cap > 0 else n_common
    if topk_fraction and topk_fraction > 0:
        k = max(1, int(math.ceil(topk_fraction * n_common)))
        k = min(k, cap)
    else:
        k = min(cap, n_common)
    return min(k, n_common)


def _topk_overlap(values_a: pd.Series, values_b: pd.Series, k: int) -> float:
    if values_a.empty or values_b.empty or k <= 0:
        return float("nan")
    top_a = set(values_a.sort_values(ascending=False).head(k).index.tolist())
    top_b = set(values_b.sort_values(ascending=False).head(k).index.tolist())
    return float(len(top_a & top_b) / k)


def _compute_alignment_metrics(
    item_df_a: pd.DataFrame,
    item_df_b: pd.DataFrame,
    *,
    topk_fraction: float,
    topk_cap: int,
) -> dict[str, float | int]:
    key_col_a = "source_item_id" if "source_item_id" in item_df_a.columns else "item_id"
    key_col_b = "source_item_id" if "source_item_id" in item_df_b.columns else "item_id"
    diff_a = item_df_a.set_index(key_col_a)["difficulty_b"]
    disc_a = item_df_a.set_index(key_col_a)["discrimination_exp_k"]
    diff_b = item_df_b.set_index(key_col_b)["difficulty_b"]
    disc_b = item_df_b.set_index(key_col_b)["discrimination_exp_k"]
    common_items = (
        diff_a.index.intersection(disc_a.index)
        .intersection(diff_b.index)
        .intersection(disc_b.index)
    )
    n_common = int(len(common_items))
    if n_common < 2:
        return {
            "n_items_common": n_common,
            "spearman_difficulty": float("nan"),
            "spearman_discrimination": float("nan"),
            "topk_used": 0,
            "topk_fraction_of_common": float("nan"),
            "topk_hardest_overlap": float("nan"),
            "topk_discriminative_overlap": float("nan"),
        }
    diff_a_c = diff_a.loc[common_items]
    disc_a_c = disc_a.loc[common_items]
    diff_b_c = diff_b.loc[common_items]
    disc_b_c = disc_b.loc[common_items]
    k_used = _resolve_topk(n_common, topk_fraction, topk_cap)
    return {
        "n_items_common": n_common,
        "spearman_difficulty": _spearman_corr(diff_a_c.to_numpy(dtype=float), diff_b_c.to_numpy(dtype=float)),
        "spearman_discrimination": _spearman_corr(disc_a_c.to_numpy(dtype=float), disc_b_c.to_numpy(dtype=float)),
        "topk_used": int(k_used),
        "topk_fraction_of_common": float(k_used / n_common) if n_common else float("nan"),
        "topk_hardest_overlap": _topk_overlap(diff_a_c, diff_b_c, k_used),
        "topk_discriminative_overlap": _topk_overlap(disc_a_c, disc_b_c, k_used),
    }


# Output column ordering: keep Spearman columns first because they are the
# headline metric for the invariance claim; top-k columns are supplementary.
_TRIAL_METRIC_COLS: tuple[str, ...] = (
    "spearman_difficulty",
    "spearman_discrimination",
    "n_items_common",
    "topk_used",
    "topk_fraction_of_common",
    "topk_hardest_overlap",
    "topk_discriminative_overlap",
)


# ---------------------------------------------------------------------------
# rank_v1 fitting on a model subset
# ---------------------------------------------------------------------------

def _fit_subset_question_params(
    arena_df: pd.DataFrame,           # full reward_df with reward_z column
    static_df: pd.DataFrame,
    subset_models: set[str],
    fit_fn: Callable[..., Any],
    args: argparse.Namespace,
    both_bad_threshold: float,
    tie_delta: float,
) -> pd.DataFrame:
    sub_reward = arena_df[arena_df["model_name"].isin(subset_models)].copy()
    sub_static = (
        static_df[static_df["model_name"].isin(subset_models)].copy()
        if not static_df.empty else pd.DataFrame()
    )
    if sub_reward.empty and sub_static.empty:
        return pd.DataFrame(columns=["set_source", "item_id", "source_item_id", "difficulty_b", "discrimination_exp_k"])

    sub_pairwise = pd.DataFrame()
    if not sub_reward.empty:
        sub_pairwise = build_soft_pairwise_targets(
            sub_reward,
            both_bad_threshold=both_bad_threshold,
            both_bad_use_zscore=True,
            tie_delta=tie_delta,
        )

    use_static = not sub_static.empty
    use_arena = not sub_reward.empty

    _model_params, static_qp, arena_qp, _meta = fit_fn(
        static_df=sub_static if use_static else None,
        pairwise_df=sub_pairwise if not sub_pairwise.empty else None,
        reward_df=sub_reward if use_arena else None,
        arena_mode="pairwise+regression",
        num_epochs=args.rank_rm_num_epochs,
        lr=args.rank_rm_lr,
        lambda_static=args.rank_rm_lambda_static if use_static else 0.0,
        lambda_arena=args.rank_rm_lambda_arena if use_arena else 0.0,
        lambda_reg=args.rank_rm_lambda_reg if use_arena else 0.0,
        reg_lambda=args.rank_rm_reg_lambda,
        verbose=not args.rank_rm_quiet,
    )

    frames: list[pd.DataFrame] = []
    for src, qp in (("static", static_qp), ("arena", arena_qp)):
        if qp is None or qp.empty:
            continue
        if "difficulty_b" not in qp.columns or "discrimination_exp_k" not in qp.columns:
            continue
        frame = qp[["question_id", "difficulty_b", "discrimination_exp_k"]].copy()
        frame["set_source"] = src
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["set_source", "item_id", "source_item_id", "difficulty_b", "discrimination_exp_k"])
    out = pd.concat(frames, ignore_index=True)
    out["item_id"] = _item_id_series(out["question_id"])
    out["source_item_id"] = _source_item_key(out["set_source"], out["item_id"])
    out = out.dropna(subset=["difficulty_b", "discrimination_exp_k"])
    out = out.drop_duplicates(subset=["source_item_id"], keep="last")
    return out[["set_source", "item_id", "source_item_id", "difficulty_b", "discrimination_exp_k"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Group sampling
# ---------------------------------------------------------------------------

def _sample_partition_groups(
    rng: np.random.Generator,
    model_pool: list[str],
    partition_count: int,
    models_per_partition: int,
    allow_overlap: bool,
) -> tuple[list[set[str]], dict[str, Any]]:
    pool_arr = np.array(model_pool, dtype=object)
    pool_size = len(pool_arr)
    needed = partition_count * models_per_partition

    if pool_size >= needed:
        chosen = rng.choice(pool_arr, size=needed, replace=False).tolist()
        groups = [
            set(chosen[i * models_per_partition : (i + 1) * models_per_partition])
            for i in range(partition_count)
        ]
        overlap_used = False
    else:
        if not allow_overlap:
            raise RuntimeError(
                f"Model pool has {pool_size} models, but disjoint partition mode needs at least "
                f"{needed} ({partition_count} x {models_per_partition}). "
                f"Pass --allow-partition-overlap to permit overlapping groups."
            )
        per_group = min(models_per_partition, pool_size)
        groups = [
            set(rng.choice(pool_arr, size=per_group, replace=False).tolist())
            for _ in range(partition_count)
        ]
        overlap_used = True

    overlaps: list[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            inter = len(groups[i] & groups[j])
            denom = min(len(groups[i]), len(groups[j])) or 1
            overlaps.append(inter / denom)
    stats: dict[str, Any] = {
        "overlap_used": bool(overlap_used),
        "mean_pairwise_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "max_pairwise_overlap": float(max(overlaps)) if overlaps else 0.0,
    }
    return groups, stats


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _run_random_subset_mode(
    args: argparse.Namespace,
    rng: np.random.Generator,
    fit_fn: Callable[..., Any],
    ref_q: pd.DataFrame,
    arena_df: pd.DataFrame,
    static_df: pd.DataFrame,
    model_pool: list[str],
    both_bad_threshold: float,
    tie_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fractions = [float(s.strip()) for s in args.subset_fractions.split(",") if s.strip()]
    if not fractions:
        raise ValueError("No valid --subset-fractions provided.")
    for frac in fractions:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"Subset fraction must be in (0,1], got {frac}")

    total_fits = len(fractions) * args.repeats
    fit_no = 0
    rows: list[dict[str, object]] = []
    for frac in fractions:
        n_models = max(args.min_models, int(np.floor(len(model_pool) * frac)))
        n_models = min(n_models, len(model_pool))
        for rep in range(args.repeats):
            fit_no += 1
            print(
                f"[random_subset] fit {fit_no}/{total_fits} | fraction={frac:.2f} ({n_models} models) | repeat={rep + 1}/{args.repeats}",
                flush=True,
            )
            sampled_models = set(
                rng.choice(np.array(model_pool, dtype=object), size=n_models, replace=False).tolist()
            )
            sub_q = _fit_subset_question_params(
                arena_df, static_df, sampled_models, fit_fn, args,
                both_bad_threshold=both_bad_threshold, tie_delta=tie_delta,
            )
            if sub_q.empty:
                print(f"  -> skipped (no items)", flush=True)
                continue
            metrics = _compute_alignment_metrics(
                ref_q, sub_q, topk_fraction=args.topk_fraction, topk_cap=args.topk,
            )
            if metrics["n_items_common"] < 2:
                print(f"  -> skipped (n_items_common={metrics['n_items_common']})", flush=True)
                continue
            print(
                f"  -> spearman_difficulty={metrics['spearman_difficulty']:.3f}  "
                f"spearman_discrimination={metrics['spearman_discrimination']:.3f}  "
                f"n_common={metrics['n_items_common']}",
                flush=True,
            )
            rows.append(
                {
                    "evaluation_mode": "random_subset",
                    "subset_fraction": frac,
                    "effective_fraction": float(n_models / len(model_pool)) if model_pool else float("nan"),
                    "repeat": rep,
                    "group_id": "subset",
                    "n_models": n_models,
                    **metrics,
                }
            )

    if not rows:
        raise RuntimeError("No successful subset fits produced comparable item rankings.")
    ref_df = pd.DataFrame(rows).sort_values(["subset_fraction", "repeat"]).reset_index(drop=True)
    pair_df = pd.DataFrame(columns=["evaluation_mode", "repeat", "group_a", "group_b"])
    return ref_df, pair_df


def _run_disjoint_partition_mode(
    args: argparse.Namespace,
    rng: np.random.Generator,
    fit_fn: Callable[..., Any],
    ref_q: pd.DataFrame,
    arena_df: pd.DataFrame,
    static_df: pd.DataFrame,
    model_pool: list[str],
    both_bad_threshold: float,
    tie_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.partition_count < 2:
        raise ValueError("--partition-count must be >= 2 for disjoint partition mode.")
    if args.models_per_partition < 2:
        raise ValueError("--models-per-partition must be >= 2.")

    total_fits = args.repeats * args.partition_count
    fit_no = 0
    ref_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for rep in range(args.repeats):
        print(
            f"[disjoint_partition] repeat {rep + 1}/{args.repeats} | "
            f"{args.partition_count} groups × {args.models_per_partition} models",
            flush=True,
        )
        groups, overlap_stats = _sample_partition_groups(
            rng, model_pool, args.partition_count, args.models_per_partition,
            allow_overlap=args.allow_partition_overlap,
        )
        if overlap_stats["overlap_used"]:
            print(
                f"  overlap: mean={overlap_stats['mean_pairwise_overlap']:.2f}  "
                f"max={overlap_stats['max_pairwise_overlap']:.2f}",
                flush=True,
            )
        group_item_params: list[pd.DataFrame] = []
        group_indices: list[int] = []
        for gidx, group_models in enumerate(groups):
            fit_no += 1
            print(
                f"  fitting group {gidx + 1}/{args.partition_count} "
                f"(fit {fit_no}/{total_fits}) | {sorted(group_models)}",
                flush=True,
            )
            sub_q = _fit_subset_question_params(
                arena_df, static_df, group_models, fit_fn, args,
                both_bad_threshold=both_bad_threshold, tie_delta=tie_delta,
            )
            if sub_q.empty:
                print(f"    -> skipped (no items)", flush=True)
                continue
            group_item_params.append(sub_q)
            group_indices.append(gidx)
            metrics_ref = _compute_alignment_metrics(
                ref_q, sub_q, topk_fraction=args.topk_fraction, topk_cap=args.topk,
            )
            if metrics_ref["n_items_common"] < 2:
                print(f"    -> skipped (n_items_common={metrics_ref['n_items_common']})", flush=True)
                continue
            print(
                f"    -> vs ref: spearman_difficulty={metrics_ref['spearman_difficulty']:.3f}  "
                f"spearman_discrimination={metrics_ref['spearman_discrimination']:.3f}  "
                f"n_common={metrics_ref['n_items_common']}",
                flush=True,
            )
            ref_rows.append(
                {
                    "evaluation_mode": "disjoint_partition",
                    "repeat": rep,
                    "group_id": f"group_{gidx + 1}",
                    "n_models": len(group_models),
                    "subset_fraction": float(len(group_models) / len(model_pool)) if model_pool else float("nan"),
                    "effective_fraction": float(len(group_models) / len(model_pool)) if model_pool else float("nan"),
                    "overlap_used": overlap_stats["overlap_used"],
                    "mean_pairwise_overlap": overlap_stats["mean_pairwise_overlap"],
                    "max_pairwise_overlap": overlap_stats["max_pairwise_overlap"],
                    **metrics_ref,
                }
            )

        for i in range(len(group_item_params)):
            for j in range(i + 1, len(group_item_params)):
                gi, gj = group_indices[i], group_indices[j]
                metrics_pair = _compute_alignment_metrics(
                    group_item_params[i], group_item_params[j],
                    topk_fraction=args.topk_fraction, topk_cap=args.topk,
                )
                if metrics_pair["n_items_common"] < 2:
                    continue
                print(
                    f"  group_{gi + 1} vs group_{gj + 1}: "
                    f"spearman_difficulty={metrics_pair['spearman_difficulty']:.3f}  "
                    f"spearman_discrimination={metrics_pair['spearman_discrimination']:.3f}  "
                    f"n_common={metrics_pair['n_items_common']}",
                    flush=True,
                )
                shared = len(groups[gi] & groups[gj])
                denom = min(len(groups[gi]), len(groups[gj])) or 1
                pair_rows.append(
                    {
                        "evaluation_mode": "disjoint_partition",
                        "repeat": rep,
                        "group_a": f"group_{gi + 1}",
                        "group_b": f"group_{gj + 1}",
                        "n_models_per_group": args.models_per_partition,
                        "shared_models": int(shared),
                        "pair_overlap_fraction": float(shared / denom),
                        "overlap_used": overlap_stats["overlap_used"],
                        **metrics_pair,
                    }
                )

    if not ref_rows:
        raise RuntimeError("No successful disjoint-group fits produced comparable item rankings.")
    ref_df = pd.DataFrame(ref_rows).sort_values(["repeat", "group_id"]).reset_index(drop=True)
    pair_df = (
        pd.DataFrame(pair_rows).sort_values(["repeat", "group_a", "group_b"]).reset_index(drop=True)
        if pair_rows else pd.DataFrame()
    )
    return ref_df, pair_df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_experiment(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _configure_torch_runtime()
    fit_fn = _load_rank_v1_fit_fn()
    rng = np.random.default_rng(args.seed)

    if bool(args.arena_jsonl) == bool(args.static_jsonl):
        raise ValueError("Provide exactly one of --arena-jsonl or --static-jsonl (they cannot co-exist).")

    print("Loading reference item parameters ...", flush=True)
    ref_q = _load_reference_question_params(Path(args.question_ranking))
    print(f"  {len(ref_q)} reference items loaded", flush=True)

    if args.arena_jsonl:
        print("Loading arena rewards ...", flush=True)
        arena_df = load_arena_reward_jsonl([args.arena_jsonl])
        if arena_df.empty:
            raise RuntimeError(f"No usable arena reward rows loaded from {args.arena_jsonl}.")
        print(
            f"  {len(arena_df)} rows | {arena_df['model_name'].nunique()} models | "
            f"{arena_df['question_id'].nunique()} questions",
            flush=True,
        )
    else:
        arena_df = pd.DataFrame()
        print("Skipping arena rewards (no --arena-jsonl provided).", flush=True)

    if args.static_jsonl:
        print("Loading static eval data ...", flush=True)
        static_df = load_static_jsonl([args.static_jsonl])
        print(
            f"  {len(static_df)} rows | {static_df['model_name'].nunique()} models | "
            f"{static_df['question_id'].nunique()} questions",
            flush=True,
        )
    else:
        static_df = pd.DataFrame()

    if arena_df.empty:
        both_bad_threshold = float("nan")
        tie_delta = float("nan")
        print("Skipping pairwise threshold resolution (no arena data loaded).", flush=True)
    else:
        both_bad_threshold, tie_delta = resolve_pairwise_thresholds(
            arena_df,
            bb_ratio=None,
            tie_ratio=None,
            both_bad_threshold=args.rank_rm_both_bad_threshold,
            both_bad_use_zscore=True,
        )
        print(
            f"Pairwise thresholds: both_bad={both_bad_threshold:.4f}  tie_delta={tie_delta:.4f}",
            flush=True,
        )

    model_pool = _load_model_pool(
        Path(args.reference_model_ranking) if args.reference_model_ranking else None,
        arena_df,
        static_df,
    )
    if len(model_pool) < args.min_models:
        raise RuntimeError(f"Model pool too small ({len(model_pool)}) for --min-models={args.min_models}")
    print(
        f"Model pool: {len(model_pool)} models | "
        f"mode={args.evaluation_mode} | repeats={args.repeats}",
        flush=True,
    )

    if args.evaluation_mode == "disjoint_partition":
        trial_df, pair_df = _run_disjoint_partition_mode(
            args, rng, fit_fn, ref_q, arena_df, static_df, model_pool,
            both_bad_threshold=both_bad_threshold, tie_delta=tie_delta,
        )
    else:
        trial_df, pair_df = _run_random_subset_mode(
            args, rng, fit_fn, ref_q, arena_df, static_df, model_pool,
            both_bad_threshold=both_bad_threshold, tie_delta=tie_delta,
        )

    summary_ref = (
        trial_df.groupby(["evaluation_mode", "subset_fraction", "n_models"], as_index=False)
        .agg(
            n_rows=("repeat", "count"),
            mean_n_items_common=("n_items_common", "mean"),
            mean_spearman_difficulty=("spearman_difficulty", "mean"),
            std_spearman_difficulty=("spearman_difficulty", "std"),
            mean_spearman_discrimination=("spearman_discrimination", "mean"),
            std_spearman_discrimination=("spearman_discrimination", "std"),
            mean_topk_hardest_overlap=("topk_hardest_overlap", "mean"),
            mean_topk_discriminative_overlap=("topk_discriminative_overlap", "mean"),
        )
        .sort_values(["evaluation_mode", "subset_fraction"])
        .reset_index(drop=True)
    )
    summary_ref["comparison_type"] = "group_vs_reference"

    if pair_df.empty:
        summary_pair = pd.DataFrame(columns=summary_ref.columns)
    else:
        summary_pair = (
            pair_df.groupby(["evaluation_mode"], as_index=False)
            .agg(
                n_rows=("repeat", "count"),
                mean_n_items_common=("n_items_common", "mean"),
                mean_spearman_difficulty=("spearman_difficulty", "mean"),
                std_spearman_difficulty=("spearman_difficulty", "std"),
                mean_spearman_discrimination=("spearman_discrimination", "mean"),
                std_spearman_discrimination=("spearman_discrimination", "std"),
                mean_topk_hardest_overlap=("topk_hardest_overlap", "mean"),
                mean_topk_discriminative_overlap=("topk_discriminative_overlap", "mean"),
            )
            .reset_index(drop=True)
        )
        summary_pair["subset_fraction"] = float(args.models_per_partition / max(1, len(model_pool)))
        summary_pair["n_models"] = args.models_per_partition
        summary_pair["comparison_type"] = "group_vs_group"
        summary_pair = summary_pair[summary_ref.columns]
    summary_df = pd.concat([summary_ref, summary_pair], ignore_index=True)

    overlap_summary: dict[str, Any] = {}
    if args.evaluation_mode == "disjoint_partition" and "overlap_used" in trial_df.columns:
        overlap_summary = {
            "any_overlap_used": bool(trial_df["overlap_used"].any()),
            "mean_pairwise_overlap": float(trial_df["mean_pairwise_overlap"].mean()),
            "max_pairwise_overlap": float(trial_df["max_pairwise_overlap"].max()),
            "needed_models": int(args.partition_count * args.models_per_partition),
            "model_pool_size": int(len(model_pool)),
        }

    run_summary: dict[str, Any] = {
        "headline_metric": (
            "Spearman rank correlation of difficulty_b and discrimination_exp_k between item "
            "parameter sets (subset-vs-reference and, in disjoint_partition mode, group-vs-group). "
            "Top-k overlap columns are supplementary; with --topk-fraction set, k scales with "
            "n_items_common per row, otherwise top-k is a raw absolute count and is not strictly "
            "comparable across rows with differing n_items_common."
        ),
        "question_ranking": str(Path(args.question_ranking)),
        "arena_jsonl": str(Path(args.arena_jsonl)) if args.arena_jsonl else None,
        "static_jsonl": str(Path(args.static_jsonl)) if args.static_jsonl else None,
        "reference_model_ranking": str(Path(args.reference_model_ranking)) if args.reference_model_ranking else None,
        "evaluation_mode": args.evaluation_mode,
        "subset_fractions": (
            [float(s.strip()) for s in args.subset_fractions.split(",") if s.strip()]
            if args.evaluation_mode == "random_subset"
            else []
        ),
        "partition_count": args.partition_count if args.evaluation_mode == "disjoint_partition" else None,
        "models_per_partition": args.models_per_partition if args.evaluation_mode == "disjoint_partition" else None,
        "allow_partition_overlap": bool(args.allow_partition_overlap),
        "partition_overlap_summary": overlap_summary,
        "repeats": args.repeats,
        "topk": args.topk,
        "topk_fraction": args.topk_fraction,
        "min_models": args.min_models,
        "seed": args.seed,
        "model_pool_size": len(model_pool),
        "reference_n_items": int(len(ref_q)),
        "ranking_method": "rank_v1.fit_irt_v1(arena_mode=pairwise+regression; mode=both-pr when static provided, arena-pr otherwise)",
        "rank_v1_hparams": {
            "num_epochs": args.rank_rm_num_epochs,
            "lr": args.rank_rm_lr,
            "lambda_static": args.rank_rm_lambda_static if args.static_jsonl else 0.0,
            "lambda_arena": args.rank_rm_lambda_arena,
            "lambda_reg": args.rank_rm_lambda_reg,
            "reg_lambda": args.rank_rm_reg_lambda,
            "both_bad_threshold": args.rank_rm_both_bad_threshold,
            "resolved_both_bad_threshold": float(both_bad_threshold),
            "resolved_tie_delta": float(tie_delta),
            "quiet": args.rank_rm_quiet,
        },
    }
    return trial_df, pair_df, summary_df, run_summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_df, pair_df, summary_df, run_summary = run_experiment(args)
    trial_path = out_dir / "subset_item_property_correlations.csv"
    pair_path = out_dir / "subset_item_property_pairwise_correlations.csv"
    summary_path = out_dir / "subset_item_property_summary.csv"
    config_path = out_dir / "run_summary.json"

    trial_df.to_csv(trial_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Saved item-property invariance outputs:")
    print(f"  trial metrics: {trial_path}")
    print(f"  pair metrics:  {pair_path}")
    print(f"  summary:       {summary_path}")
    print(f"  run summary:   {config_path}")
    if run_summary.get("partition_overlap_summary"):
        print(f"\nPartition overlap summary: {run_summary['partition_overlap_summary']}")
    print("\nSummary preview (Spearman columns are the headline metric):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
