#!/usr/bin/env python3
"""Online replay for estimating new arena item parameters.

This is the item-side analogue of ``cold_start_efficiency.py``.  It treats each
question as a newly added benchmark item, freezes full-data model abilities, and
reveals the existing model responses for that question one at a time.  After each
revealed response it re-estimates only the item's difficulty ``b_q`` and
log-sharpness ``k_q`` under the reward-IRT regression model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranking import fit_reward_irt  # noqa: E402
from rm_analysis.cold_start.data import (  # noqa: E402
    DEFAULT_ARENA_JSONLS,
    filter_models,
    load_reward_responses,
    restrict_questions_per_source,
)


@dataclass(frozen=True)
class ItemGrids:
    b_flat: np.ndarray
    k_flat: np.ndarray
    a_flat: np.ndarray
    prior: np.ndarray


def ability_spread_model_order(model_names: list[str], theta_map: dict[str, float]) -> list[str]:
    """Order models high, low, second-high, second-low, ... by reference ability."""
    ordered = sorted(model_names, key=lambda name: (-theta_map[name], name))
    out: list[str] = []
    lo = 0
    hi = len(ordered) - 1
    while lo <= hi:
        out.append(ordered[lo])
        if lo != hi:
            out.append(ordered[hi])
        lo += 1
        hi -= 1
    return out


def make_item_grids(
    question_params: pd.DataFrame,
    *,
    b_pad: float,
    k_pad: float,
    b_grid_size: int,
    k_grid_size: int,
    reg_lambda: float,
) -> ItemGrids:
    b_values = question_params["difficulty_b"].to_numpy(dtype=float)
    k_values = question_params["k_raw"].to_numpy(dtype=float)
    b_grid = np.linspace(float(b_values.min()) - b_pad, float(b_values.max()) + b_pad, b_grid_size)
    k_grid = np.linspace(float(k_values.min()) - k_pad, float(k_values.max()) + k_pad, k_grid_size)
    b_mesh, k_mesh = np.meshgrid(b_grid, k_grid, indexing="ij")
    b_flat = b_mesh.ravel()
    k_flat = k_mesh.ravel()
    return ItemGrids(
        b_flat=b_flat,
        k_flat=k_flat,
        a_flat=np.exp(k_flat),
        prior=reg_lambda * (b_flat**2 + k_flat**2),
    )


def load_or_fit_reference(
    reference_dir: Path,
    reward_df: pd.DataFrame,
    *,
    fit_if_missing: bool,
    reference_epochs: int,
    reference_lr: float,
    irt_reg_lambda: float,
    quiet: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = reference_dir / "reference_irt_arena.csv"
    question_path = reference_dir / "reference_questions_irt_arena.csv"
    if model_path.exists() and question_path.exists():
        return pd.read_csv(model_path), pd.read_csv(question_path)
    if not fit_if_missing:
        raise SystemExit(
            f"Missing reference files in {reference_dir}. "
            "Pass --fit-reference-if-missing to fit them."
        )

    reference_dir.mkdir(parents=True, exist_ok=True)
    model_params, question_params = fit_reward_irt(
        reward_df[["model_name", "question_id", "reward"]],
        num_epochs=reference_epochs,
        lr=reference_lr,
        reg_lambda=irt_reg_lambda,
        verbose=not quiet,
    )
    model_params.to_csv(model_path, index=False)
    question_params.to_csv(question_path, index=False)
    return model_params, question_params


def _ordered_question_records(
    group: pd.DataFrame,
    theta_map: dict[str, float],
    *,
    order_strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    usable = group[group["model_name"].isin(theta_map)].copy()
    if usable.empty:
        return []

    if order_strategy == "ability_spread":
        model_order = ability_spread_model_order(usable["model_name"].tolist(), theta_map)
    elif order_strategy == "random":
        rng = np.random.default_rng(seed)
        model_order = usable["model_name"].tolist()
        rng.shuffle(model_order)
    else:
        raise ValueError(f"Unknown order strategy: {order_strategy}")

    by_model = usable.set_index("model_name")
    return [
        {
            "model_name": str(model_name),
            "theta": float(theta_map[str(model_name)]),
            "reward": float(by_model.loc[model_name, "reward"]),
        }
        for model_name in model_order
    ]


def difficulty_rank(b_value: float, other_b_values: np.ndarray) -> int:
    return int(1 + np.sum(other_b_values > b_value))


def value_at_cutoff(ns: np.ndarray, values: np.ndarray, cutoff: int) -> float:
    idxs = np.flatnonzero(ns <= cutoff)
    idx = int(idxs[-1]) if len(idxs) else 0
    return float(values[idx])


def replay_one_question(
    question_id: str,
    group: pd.DataFrame,
    *,
    theta_map: dict[str, float],
    question_ref: dict[str, Any],
    other_b_values: np.ndarray,
    grids: ItemGrids,
    order_strategy: str,
    seed: int,
    record_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = _ordered_question_records(
        group,
        theta_map,
        order_strategy=order_strategy,
        seed=seed,
    )
    source = str(group.iloc[0]["source"]) if not group.empty else ""
    if not ordered:
        empty_summary = {
            "question_id": question_id,
            "source": source,
            "seed": seed,
            "n_obs": 0,
            "difficulty_b_ref": float(question_ref["difficulty_b"]),
            "k_ref": float(question_ref["k_raw"]),
            "a_ref": float(question_ref["discrimination_exp_k"]),
            "difficulty_rank_ref": float("nan"),
            "final_b_abs_error": float("nan"),
            "final_k_abs_error": float("nan"),
            "final_a_abs_error": float("nan"),
        }
        return pd.DataFrame(), empty_summary

    theta = np.array([row["theta"] for row in ordered], dtype=float)
    reward = np.array([row["reward"] for row in ordered], dtype=float)
    pred = grids.a_flat[None, :] * (theta[:, None] - grids.b_flat[None, :])
    losses = (pred - reward[:, None]) ** 2
    cumulative = np.cumsum(losses, axis=0)
    objective = cumulative + grids.prior[None, :]
    argmins = np.argmin(objective, axis=1)

    b_hat = grids.b_flat[argmins]
    k_hat = grids.k_flat[argmins]
    a_hat = grids.a_flat[argmins]
    ns = np.arange(1, len(ordered) + 1)

    b_ref = float(question_ref["difficulty_b"])
    k_ref = float(question_ref["k_raw"])
    a_ref = float(question_ref["discrimination_exp_k"])
    rank_ref = difficulty_rank(b_ref, other_b_values)
    rank_hat = np.array([difficulty_rank(float(b), other_b_values) for b in b_hat], dtype=int)

    total_loss = losses.sum(axis=0)
    remaining_loss = total_loss[None, :] - cumulative
    heldout_counts = len(ordered) - ns
    heldout_mse = np.full(len(ns), np.nan, dtype=float)
    nonzero = heldout_counts > 0
    heldout_mse[nonzero] = (
        remaining_loss[np.arange(len(ns))[nonzero], argmins[nonzero]] / heldout_counts[nonzero]
    )

    b_abs_error = np.abs(b_hat - b_ref)
    k_abs_error = np.abs(k_hat - k_ref)
    a_abs_error = np.abs(a_hat - a_ref)
    rank_error = np.abs(rank_hat - rank_ref)
    record_mask = (ns == 1) | (ns == len(ns)) | (ns % max(1, record_every) == 0)

    rows: list[dict[str, Any]] = []
    for idx in np.flatnonzero(record_mask):
        obs = ordered[int(idx)]
        rows.append(
            {
                "question_id": question_id,
                "source": source,
                "seed": seed,
                "n": int(ns[idx]),
                "model_name": obs["model_name"],
                "model_theta": float(obs["theta"]),
                "reward": float(obs["reward"]),
                "difficulty_b_hat": float(b_hat[idx]),
                "difficulty_b_ref": b_ref,
                "k_hat": float(k_hat[idx]),
                "k_ref": k_ref,
                "a_hat": float(a_hat[idx]),
                "a_ref": a_ref,
                "difficulty_rank_hat": int(rank_hat[idx]),
                "difficulty_rank_ref": int(rank_ref),
                "difficulty_rank_error": int(rank_error[idx]),
                "b_abs_error": float(b_abs_error[idx]),
                "k_abs_error": float(k_abs_error[idx]),
                "a_abs_error": float(a_abs_error[idx]),
                "heldout_mse": float(heldout_mse[idx]) if np.isfinite(heldout_mse[idx]) else np.nan,
                "n_total_obs": int(len(ordered)),
            }
        )

    summary = {
        "question_id": question_id,
        "source": source,
        "seed": seed,
        "n_obs": int(len(ordered)),
        "difficulty_b_ref": b_ref,
        "k_ref": k_ref,
        "a_ref": a_ref,
        "difficulty_rank_ref": int(rank_ref),
        "b_abs_error_at_3": value_at_cutoff(ns, b_abs_error, 3),
        "b_abs_error_at_5": value_at_cutoff(ns, b_abs_error, 5),
        "b_abs_error_at_10": value_at_cutoff(ns, b_abs_error, 10),
        "k_abs_error_at_3": value_at_cutoff(ns, k_abs_error, 3),
        "k_abs_error_at_5": value_at_cutoff(ns, k_abs_error, 5),
        "k_abs_error_at_10": value_at_cutoff(ns, k_abs_error, 10),
        "a_abs_error_at_3": value_at_cutoff(ns, a_abs_error, 3),
        "a_abs_error_at_5": value_at_cutoff(ns, a_abs_error, 5),
        "a_abs_error_at_10": value_at_cutoff(ns, a_abs_error, 10),
        "rank_error_at_3": value_at_cutoff(ns, rank_error, 3),
        "rank_error_at_5": value_at_cutoff(ns, rank_error, 5),
        "rank_error_at_10": value_at_cutoff(ns, rank_error, 10),
        "final_b_hat": float(b_hat[-1]),
        "final_k_hat": float(k_hat[-1]),
        "final_a_hat": float(a_hat[-1]),
        "final_b_abs_error": float(b_abs_error[-1]),
        "final_k_abs_error": float(k_abs_error[-1]),
        "final_a_abs_error": float(a_abs_error[-1]),
        "final_rank_error": int(rank_error[-1]),
    }
    return pd.DataFrame(rows), summary


def summarise_item_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "b_abs_error_at_3",
        "b_abs_error_at_5",
        "b_abs_error_at_10",
        "k_abs_error_at_3",
        "k_abs_error_at_5",
        "k_abs_error_at_10",
        "a_abs_error_at_3",
        "a_abs_error_at_5",
        "a_abs_error_at_10",
        "rank_error_at_3",
        "rank_error_at_5",
        "rank_error_at_10",
        "final_b_abs_error",
        "final_k_abs_error",
        "final_a_abs_error",
        "final_rank_error",
    ]
    rows: list[dict[str, Any]] = []
    for source, group in [("all", run_df), *list(run_df.groupby("source", sort=True))]:
        row: dict[str, Any] = {
            "source": source,
            "n_questions": int(group["question_id"].nunique()),
            "n_runs": int(len(group)),
            "mean_n_obs": float(group["n_obs"].mean()),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"median_{metric}"] = float(group[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run question-item online replay.")
    parser.add_argument("--arena-jsonl", nargs="+", type=Path, default=DEFAULT_ARENA_JSONLS)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REPO_ROOT / "results" / "cold_start" / "v1_arena_adaptive",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "cold_start" / "v1_question_items",
    )
    parser.add_argument("--fit-reference-if-missing", action="store_true")
    parser.add_argument("--reference-epochs", type=int, default=1000)
    parser.add_argument("--reference-lr", type=float, default=0.02)
    parser.add_argument("--irt-reg-lambda", type=float, default=1e-4)
    parser.add_argument("--replay-reg-lambda", type=float, default=1e-2)
    parser.add_argument("--reward-normalization", choices=("none", "global", "per_source"), default="per_source")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-questions-per-source", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--order-strategy", choices=("ability_spread", "random"), default="ability_spread")
    parser.add_argument("--b-pad", type=float, default=0.5)
    parser.add_argument("--k-pad", type=float, default=0.5)
    parser.add_argument("--b-grid-size", type=int, default=121)
    parser.add_argument("--k-grid-size", type=int, default=101)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_seeds < 1:
        raise SystemExit("--n-seeds must be at least 1.")
    if args.record_every < 1:
        raise SystemExit("--record-every must be at least 1.")
    if args.b_grid_size < 11 or args.k_grid_size < 11:
        raise SystemExit("--b-grid-size and --k-grid-size must be at least 11.")


def write_metadata(
    args: argparse.Namespace,
    reward_stats: dict[str, dict[str, float]],
    reward_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
) -> None:
    metadata = {
        "arena_jsonl": [str(path) for path in args.arena_jsonl],
        "reference_dir": str(args.reference_dir),
        "order_strategy": args.order_strategy,
        "n_seeds": args.n_seeds,
        "seed_offset": args.seed_offset,
        "record_every": args.record_every,
        "reward_normalization": args.reward_normalization,
        "max_models": args.max_models,
        "models": args.models,
        "max_questions": args.max_questions,
        "max_questions_per_source": args.max_questions_per_source,
        "replay_reg_lambda": args.replay_reg_lambda,
        "b_grid_size": args.b_grid_size,
        "k_grid_size": args.k_grid_size,
        "usable_rows": int(len(reward_df)),
        "models_loaded": int(model_params["model_name"].nunique()),
        "questions_loaded": int(question_params["question_id"].nunique()),
        "reward_stats": reward_stats,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = args.output_dir / "trajectories.csv"
    run_summary_path = args.output_dir / "item_summary_by_run.csv"
    summary_path = args.output_dir / "summary.csv"
    if trajectories_path.exists():
        trajectories_path.unlink()

    reward_df, reward_stats = load_reward_responses(args.arena_jsonl, normalize=args.reward_normalization)
    reward_df = restrict_questions_per_source(
        reward_df,
        max_questions_per_source=args.max_questions_per_source,
    )
    reward_df = filter_models(reward_df, models=args.models, max_models=args.max_models)
    model_params, question_params = load_or_fit_reference(
        args.reference_dir,
        reward_df,
        fit_if_missing=args.fit_reference_if_missing,
        reference_epochs=args.reference_epochs,
        reference_lr=args.reference_lr,
        irt_reg_lambda=args.irt_reg_lambda,
        quiet=args.quiet,
    )

    theta_map = {
        str(row["model_name"]): float(row["theta"])
        for row in model_params.to_dict(orient="records")
    }
    question_ref = {
        str(row["question_id"]): row
        for row in question_params.to_dict(orient="records")
    }
    reward_df = reward_df[
        reward_df["model_name"].isin(theta_map.keys())
        & reward_df["question_id"].isin(question_ref.keys())
    ].copy()
    question_ids = sorted(reward_df["question_id"].unique())
    if args.max_questions is not None:
        question_ids = question_ids[: args.max_questions]
        reward_df = reward_df[reward_df["question_id"].isin(question_ids)].copy()

    print(
        f"Loaded item replay data: {len(reward_df)} rows, "
        f"{reward_df['model_name'].nunique()} models, {len(question_ids)} questions",
        flush=True,
    )

    grids = make_item_grids(
        question_params,
        b_pad=args.b_pad,
        k_pad=args.k_pad,
        b_grid_size=args.b_grid_size,
        k_grid_size=args.k_grid_size,
        reg_lambda=args.replay_reg_lambda,
    )
    write_metadata(args, reward_stats, reward_df, model_params, question_params)

    run_rows: list[dict[str, Any]] = []
    wrote_header = False
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
    b_map = question_params.set_index("question_id")["difficulty_b"].astype(float)

    for question_id, group in reward_df.groupby("question_id", sort=True):
        other_b_values = b_map.drop(index=question_id, errors="ignore").to_numpy(dtype=float)
        for seed in seeds:
            traj_df, run_summary = replay_one_question(
                str(question_id),
                group,
                theta_map=theta_map,
                question_ref=question_ref[str(question_id)],
                other_b_values=other_b_values,
                grids=grids,
                order_strategy=args.order_strategy,
                seed=seed,
                record_every=args.record_every,
            )
            run_rows.append(run_summary)
            if not traj_df.empty:
                traj_df.to_csv(
                    trajectories_path,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(run_summary_path, index=False)
    summary_df = summarise_item_runs(run_df)
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Question item online-update summary ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nOutputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
