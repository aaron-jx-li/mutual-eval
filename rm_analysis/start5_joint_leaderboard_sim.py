#!/usr/bin/env python3
"""Seed-5 online leaderboard replay for joint MutualEval versus 2PL+BT.

This is a thin experiment runner over ``online_leaderboard_sim.py``.  It starts
from an initial leaderboard of five known models, then places the remaining
models in release order.  Before each new model arrives, calibration parameters
are fit only on the currently known models.  The new model is then replayed at
fixed fractions of its available static+arena observations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rm_analysis.online_leaderboard_sim import (  # noqa: E402
    DEFAULT_ARENA_JSONLS,
    DEFAULT_STATIC_JSONLS,
    aggregate_results,
    build_observations,
    fit_method,
    load_arena_jsonls,
    load_static_jsonls,
    rank_map,
    replay_model,
    restrict_questions_per_source,
)


JOINT_METHODS = ("dualeval_joint", "2pl_bt_joint")
METHOD_LABELS = {
    "dualeval_joint": "MutualEval (joint)",
    "2pl_bt_joint": "2PL+BT (joint)",
}
METHOD_COLORS = {
    "dualeval_joint": "#C53030",
    "2pl_bt_joint": "#2B6CB0",
}
COMPARE_SPECS = [
    ("2pl_bt_joint", "random", "2PL+BT random", "#2B6CB0"),
    ("dualeval_joint", "random", "MutualEval random", "#C53030"),
    ("dualeval_joint", "fisher", "MutualEval fisher", "#2F855A"),
    ("dualeval_joint", "sharpness", "MutualEval sharpness", "#805AD5"),
]
DEFAULT_FRACTIONS = [round(i / 10, 1) for i in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-jsonl", nargs="+", type=Path, default=DEFAULT_STATIC_JSONLS)
    parser.add_argument("--arena-jsonl", nargs="+", type=Path, default=DEFAULT_ARENA_JSONLS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "cold_start" / "start5_joint_strategy_fraction",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "cold_start" / "start5_joint_strategy_fraction",
    )
    parser.add_argument("--methods", nargs="+", choices=JOINT_METHODS, default=list(JOINT_METHODS))
    parser.add_argument(
        "--dualeval-strategies",
        nargs="+",
        choices=("random", "fisher", "sharpness"),
        default=["random", "fisher", "sharpness"],
    )
    parser.add_argument(
        "--bt-strategies",
        nargs="+",
        choices=("random", "fisher", "sharpness"),
        default=["random"],
    )
    parser.add_argument("--reference-method", choices=JOINT_METHODS, default="dualeval_joint")
    parser.add_argument("--initial-model-count", type=int, default=5)
    parser.add_argument("--arrival-order", choices=("release", "alphabetical", "random"), default="release")
    parser.add_argument(
        "--release-order-json",
        type=Path,
        default=REPO_ROOT / "data" / "model_release_order.json",
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-static-questions-per-source", type=int, default=None)
    parser.add_argument("--max-arena-questions-per-source", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum revealed observations per arriving model. Use 0 for all available observations.",
    )
    parser.add_argument("--record-fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--reference-epochs", type=int, default=300)
    parser.add_argument("--calibration-epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lambda-static", type=float, default=1.0)
    parser.add_argument("--lambda-arena", type=float, default=1.0)
    parser.add_argument("--lambda-bb", type=float, default=0.2)
    parser.add_argument("--reg-lambda", type=float, default=1e-2)
    parser.add_argument("--bb-ratio", type=float, default=0.15)
    parser.add_argument("--tie-ratio", type=float, default=0.15)
    parser.add_argument("--theta-pad", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=801)
    parser.add_argument("--replay-reg-lambda", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.initial_model_count < 2:
        raise SystemExit("--initial-model-count must be at least 2.")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative; use 0 for all observations.")
    if args.grid_size < 11:
        raise SystemExit("--grid-size must be at least 11.")
    if any(frac <= 0.0 or frac > 1.0 for frac in args.record_fractions):
        raise SystemExit("--record-fractions values must be in (0, 1].")
    if args.reference_method not in args.methods:
        raise SystemExit("--reference-method must be included in --methods.")
    if "dualeval_joint" in args.methods and not args.dualeval_strategies:
        raise SystemExit("--dualeval-strategies must include at least one strategy.")
    if "2pl_bt_joint" in args.methods and not args.bt_strategies:
        raise SystemExit("--bt-strategies must include at least one strategy.")
    args.strategies = sorted(
        set(
            (args.dualeval_strategies if "dualeval_joint" in args.methods else [])
            + (args.bt_strategies if "2pl_bt_joint" in args.methods else [])
        )
    )
    args.record_every = 1


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def ordered_models(models: list[str], args: argparse.Namespace) -> list[str]:
    if args.arrival_order == "alphabetical":
        ordered = sorted(models)
    elif args.arrival_order == "random":
        ordered = sorted(models)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(ordered)
    else:
        release_path = resolve_path(args.release_order_json)
        release_records: list[dict[str, Any]] = []
        if release_path.exists():
            data = json.loads(release_path.read_text(encoding="utf-8"))
            release_records = list(data.get("models", []))
        order_key: dict[str, tuple[str, int, str]] = {}
        for row in release_records:
            name = str(row.get("model_name"))
            order_key[name] = (
                str(row.get("public_release_date") or "9999-12-31"),
                int(row.get("manual_order") or 10**9),
                name,
            )
        ordered = sorted(models, key=lambda name: order_key.get(name, ("9999-12-31", 10**9, name)))
    if args.max_models is not None:
        ordered = ordered[: args.max_models]
    return ordered


def subset_models(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keep = set(models)
    return df[df["model_name"].astype(str).isin(keep)].copy().reset_index(drop=True)


def write_metadata(
    args: argparse.Namespace,
    *,
    static_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    model_order: list[str],
    initial_models: list[str],
    arrivals: list[str],
    reference_ranks: dict[str, dict[str, int]],
) -> None:
    metadata = {
        "protocol": "release_order_seed5_joint_online_leaderboard",
        "methods": args.methods,
        "strategies_by_method": {
            "dualeval_joint": args.dualeval_strategies if "dualeval_joint" in args.methods else [],
            "2pl_bt_joint": args.bt_strategies if "2pl_bt_joint" in args.methods else [],
        },
        "reference_method": args.reference_method,
        "initial_model_count": args.initial_model_count,
        "arrival_order": args.arrival_order,
        "model_order": model_order,
        "initial_models": initial_models,
        "arrival_models": arrivals,
        "record_fractions": sorted(set(float(frac) for frac in args.record_fractions)),
        "max_steps": args.max_steps,
        "reference_epochs": args.reference_epochs,
        "calibration_epochs": args.calibration_epochs,
        "lambda_static": args.lambda_static,
        "lambda_arena": args.lambda_arena,
        "lambda_bb": args.lambda_bb,
        "reg_lambda": args.reg_lambda,
        "bb_ratio": args.bb_ratio,
        "tie_ratio": args.tie_ratio,
        "n_static_rows": int(len(static_df)),
        "n_static_questions": int(static_df["question_id"].nunique()) if not static_df.empty else 0,
        "n_arena_rows": int(len(reward_df)),
        "n_arena_questions": int(reward_df["question_id"].nunique()) if not reward_df.empty else 0,
        "n_models": int(len(model_order)),
        "reference_ranks": reference_ranks,
        "leakage_guard": (
            "For each arriving model, calibration fits use only models already "
            "on the leaderboard. The arriving model is excluded from reward "
            "normalization, pair-threshold estimation, item calibration, and "
            "known-model ability fitting until its replay is complete."
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def add_schedule_columns(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    initial_models: list[str],
    known_before: list[str],
    arrival_models: list[str],
    arrival_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule = {
        "initial_model_count": len(initial_models),
        "initial_models": "|".join(initial_models),
        "known_models_before": "|".join(known_before),
        "arrival_order": "|".join(arrival_models),
        "arrival_index": arrival_index,
        "n_known_before": len(known_before),
        "n_ranked_after": len(known_before) + 1,
    }
    for row in rows:
        row.update(schedule)
    summary.update(schedule)
    return rows, summary


def summarize_by_fraction(trajectories: pd.DataFrame) -> pd.DataFrame:
    if trajectories.empty:
        return pd.DataFrame()
    return (
        trajectories.groupby(["method", "strategy", "target_fraction"], as_index=False)
        .agg(
            mean_n=("n", "mean"),
            mean_reveal_fraction=("reveal_fraction", "mean"),
            mean_rank_error=("rank_error", "mean"),
            median_rank_error=("rank_error", "median"),
            q25=("rank_error", lambda s: float(np.percentile(s, 25))),
            q75=("rank_error", lambda s: float(np.percentile(s, 75))),
            mean_spearman=("spearman", "mean"),
            mean_kendall=("kendall", "mean"),
            n_arrivals=("m_new", "nunique"),
        )
        .sort_values(["method", "strategy", "target_fraction"])
    )


def plot_fraction_comparison(summary: pd.DataFrame, save_dir: Path) -> None:
    if summary.empty:
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for method, strategy, label, color in COMPARE_SPECS:
        part = summary[
            (summary["method"] == method) & (summary["strategy"] == strategy)
        ].sort_values("target_fraction")
        if part.empty:
            continue
        x = part["target_fraction"].to_numpy(dtype=float) * 100.0
        y = part["mean_rank_error"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.1,
            color=color,
            label=label,
        )
        ax.fill_between(
            x,
            part["q25"].to_numpy(dtype=float),
            part["q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.set_title("Seed-5 Joint Cold-Start Leaderboard Placement")
    ax.set_xlabel("Held-out model data used (%)")
    ax.set_ylabel("Mean absolute rank error")
    ax.set_xticks(sorted(summary["target_fraction"].unique() * 100.0))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(save_dir / f"joint_rank_error_by_fraction.{ext}", dpi=180 if ext == "png" else None)
    plt.close(fig)
    summary.to_csv(save_dir / "joint_rank_error_by_fraction.csv", index=False)


def strategies_for_method(method: str, args: argparse.Namespace) -> list[str]:
    if method == "dualeval_joint":
        return list(args.dualeval_strategies)
    if method == "2pl_bt_joint":
        return list(args.bt_strategies)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = resolve_path(args.output_dir)
    args.figures_dir = resolve_path(args.figures_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        args.output_dir / "trajectories.csv",
        args.output_dir / "model_summaries.csv",
        args.output_dir / "trajectory_summary.csv",
        args.output_dir / "trajectory_fraction_summary.csv",
        args.output_dir / "final_summary.csv",
        args.output_dir / "run_metadata.json",
    ]
    if any(path.exists() for path in output_files) and not args.overwrite:
        raise SystemExit(f"Output files already exist in {args.output_dir}; pass --overwrite to replace.")
    for path in output_files:
        if path.exists():
            path.unlink()

    static_df = load_static_jsonls(args.static_jsonl)
    reward_df = load_arena_jsonls(args.arena_jsonl)
    static_df = restrict_questions_per_source(
        static_df,
        max_questions_per_source=args.max_static_questions_per_source,
    )
    reward_df = restrict_questions_per_source(
        reward_df,
        max_questions_per_source=args.max_arena_questions_per_source,
    )

    available_models = sorted(set(static_df["model_name"].astype(str)) | set(reward_df["model_name"].astype(str)))
    if args.models:
        available_models = [model for model in available_models if model in set(args.models)]
    model_order = ordered_models(available_models, args)
    if len(model_order) <= args.initial_model_count:
        raise SystemExit("Need more models than --initial-model-count to run arrivals.")
    static_df = subset_models(static_df, model_order)
    reward_df = subset_models(reward_df, model_order)
    initial_models = model_order[: args.initial_model_count]
    arrivals = model_order[args.initial_model_count :]

    print(
        f"Loaded {len(static_df)} static rows / {static_df['question_id'].nunique()} static questions, "
        f"{len(reward_df)} arena rows / {reward_df['question_id'].nunique()} arena questions, "
        f"{len(model_order)} models.",
        flush=True,
    )
    print(f"Initial models ({len(initial_models)}): {', '.join(initial_models)}", flush=True)
    print(f"Arrival models ({len(arrivals)}): {', '.join(arrivals)}", flush=True)

    reference_ranks: dict[str, dict[str, int]] = {}
    for method in args.methods:
        print(f"Fitting full-data reference: {method}", flush=True)
        reference = fit_method(
            method,
            static_df,
            reward_df,
            args=args,
            num_epochs=args.reference_epochs,
        )
        reference.model_params.to_csv(args.output_dir / f"reference_{method}_models.csv", index=False)
        reference.question_params.to_csv(args.output_dir / f"reference_{method}_questions.csv", index=False)
        reference_ranks[method] = rank_map(reference.model_params)

    write_metadata(
        args,
        static_df=static_df,
        reward_df=reward_df,
        model_order=model_order,
        initial_models=initial_models,
        arrivals=arrivals,
        reference_ranks=reference_ranks,
    )

    eval_reference_rank = reference_ranks[args.reference_method]
    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    total_runs = len(args.methods) * len(arrivals)
    run_idx = 0

    for method in args.methods:
        known_models = list(initial_models)
        method_reference_rank = reference_ranks[method]
        for arrival_index, model_name in enumerate(arrivals, start=1):
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] {method}: calibrating on "
                f"{len(known_models)} known models before {model_name}",
                flush=True,
            )
            static_train = subset_models(static_df, known_models)
            reward_train = subset_models(reward_df, known_models)
            calibration = fit_method(
                method,
                static_train,
                reward_train,
                args=args,
                num_epochs=args.calibration_epochs,
            )
            observations = build_observations(model_name, static_df, reward_df, calibration)
            for strategy in strategies_for_method(method, args):
                rows, summary = replay_model(
                    model_name=model_name,
                    observations=observations,
                    calibration=calibration,
                    reference_rank=eval_reference_rank,
                    method_reference_rank=method_reference_rank,
                    strategy=strategy,
                    args=args,
                )
                rows, summary = add_schedule_columns(
                    rows,
                    summary,
                    initial_models=initial_models,
                    known_before=known_models,
                    arrival_models=arrivals,
                    arrival_index=arrival_index,
                )
                all_rows.extend(rows)
                all_summaries.append(summary)
            pd.DataFrame(all_rows).to_csv(args.output_dir / "trajectories.csv", index=False)
            pd.DataFrame(all_summaries).to_csv(args.output_dir / "model_summaries.csv", index=False)
            known_models.append(model_name)

    trajectory_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(all_summaries)
    aggregate_results(trajectory_df, summary_df, args.output_dir)
    fraction_summary = summarize_by_fraction(trajectory_df)
    fraction_summary.to_csv(args.output_dir / "trajectory_fraction_summary.csv", index=False)
    plot_fraction_comparison(fraction_summary, args.figures_dir)

    print(f"Outputs saved to {args.output_dir}", flush=True)
    print(f"Figures saved to {args.figures_dir}", flush=True)
    if not fraction_summary.empty:
        print(
            fraction_summary[
                ["method", "strategy", "target_fraction", "mean_n", "mean_rank_error", "mean_spearman"]
            ].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
