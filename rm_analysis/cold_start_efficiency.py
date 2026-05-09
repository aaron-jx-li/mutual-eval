#!/usr/bin/env python3
"""Arena-only cold-start replay: reward-IRT versus Bradley-Terry.

This is the clean v1 comparison: both methods consume only ``data/hf`` arena
reward JSONLs.  The replay treats each held-out model as newly released, reveals
its existing responses one at a time, and asks how quickly its rank stabilizes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranking import fit_reward_irt  # noqa: E402
from rm_analysis.cold_start.bt import (  # noqa: E402
    build_arena_reference_pairs,
    build_arena_response_observations,
    fit_bt,
)
from rm_analysis.cold_start.data import (  # noqa: E402
    DEFAULT_ARENA_JSONLS,
    filter_models,
    load_reward_responses,
    restrict_questions_per_source,
)
from rm_analysis.cold_start.math_utils import summarise_runs  # noqa: E402
from rm_analysis.cold_start.replay import (  # noqa: E402
    ARENA_METHODS,
    ReferenceFit,
    prepare_replay_problem,
    replay_prepared_problem,
)


def fit_reference(
    method: str,
    reward_df: pd.DataFrame,
    *,
    reference_epochs: int,
    reference_lr: float,
    bt_epochs: int,
    bt_lr: float,
    bt_reg_lambda: float,
    irt_reg_lambda: float,
    quiet: bool,
) -> ReferenceFit:
    verbose = not quiet
    if method == "irt_arena":
        print("Fitting full-data reward-IRT reference ...", flush=True)
        model_params, question_params = fit_reward_irt(
            reward_df[["model_name", "question_id", "reward"]],
            num_epochs=reference_epochs,
            lr=reference_lr,
            reg_lambda=irt_reg_lambda,
            verbose=verbose,
        )
        return ReferenceFit(method=method, model_params=model_params, question_params=question_params)

    if method == "bt_arena":
        print("Fitting full-data arena BT reference ...", flush=True)
        model_params = fit_bt(
            build_arena_reference_pairs(reward_df),
            num_epochs=bt_epochs,
            lr=bt_lr,
            reg_lambda=bt_reg_lambda,
        )
        return ReferenceFit(method=method, model_params=model_params)

    raise ValueError(f"Arena cold-start supports only {ARENA_METHODS}; got {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run arena-only v1 cold-start replay for reward-IRT vs BT.",
    )
    parser.add_argument("--arena-jsonl", nargs="+", type=Path, default=DEFAULT_ARENA_JSONLS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "cold_start" / "v1_arena_adaptive",
    )
    parser.add_argument("--methods", nargs="+", choices=ARENA_METHODS, default=ARENA_METHODS)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-questions-per-source", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--k-stability", type=int, default=20)
    parser.add_argument("--reference-epochs", type=int, default=1000)
    parser.add_argument("--reference-lr", type=float, default=0.02)
    parser.add_argument("--bt-epochs", type=int, default=1000)
    parser.add_argument("--bt-lr", type=float, default=0.05)
    parser.add_argument("--bt-reg-lambda", type=float, default=1e-4)
    parser.add_argument("--irt-reg-lambda", type=float, default=1e-4)
    parser.add_argument("--replay-reg-lambda", type=float, default=1e-2)
    parser.add_argument(
        "--reward-normalization",
        choices=("none", "global", "per_source"),
        default="per_source",
    )
    parser.add_argument("--theta-pad", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=801)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_seeds < 1:
        raise SystemExit("--n-seeds must be at least 1.")
    if args.grid_size < 11:
        raise SystemExit("--grid-size must be at least 11.")
    if args.record_every < 1:
        raise SystemExit("--record-every must be at least 1.")
    if "irt_arena" not in args.methods:
        raise SystemExit("Adaptive replay order requires method 'irt_arena' to estimate a_q.")
    if args.methods[0] != "irt_arena":
        raise SystemExit("Adaptive replay requires 'irt_arena' to be the first method.")


def write_metadata(
    args: argparse.Namespace,
    reward_stats: dict[str, dict[str, float]],
    *,
    reward_df: pd.DataFrame,
) -> None:
    metadata = {
        "arena_jsonl": [str(path) for path in args.arena_jsonl],
        "methods": args.methods,
        "n_seeds": args.n_seeds,
        "seed_offset": args.seed_offset,
        "max_models": args.max_models,
        "models": args.models,
        "max_questions_per_source": args.max_questions_per_source,
        "max_steps": args.max_steps,
        "record_every": args.record_every,
        "replay_order": "descending_irt_discrimination",
        "k_stability": args.k_stability,
        "reference_epochs": args.reference_epochs,
        "bt_epochs": args.bt_epochs,
        "reward_normalization": args.reward_normalization,
        "usable_rows": int(len(reward_df)),
        "models_loaded": int(reward_df["model_name"].nunique()),
        "questions_loaded": int(reward_df["question_id"].nunique()),
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
    run_summary_path = args.output_dir / "n_star_by_run.csv"
    summary_path = args.output_dir / "summary.csv"
    if trajectories_path.exists():
        trajectories_path.unlink()

    reward_df, reward_stats = load_reward_responses(
        args.arena_jsonl,
        normalize=args.reward_normalization,
    )
    reward_df = restrict_questions_per_source(
        reward_df,
        max_questions_per_source=args.max_questions_per_source,
    )
    reward_df = filter_models(reward_df, models=args.models, max_models=args.max_models)
    print(
        f"Loaded arena rewards: {len(reward_df)} rows, "
        f"{reward_df['model_name'].nunique()} models, "
        f"{reward_df['question_id'].nunique()} questions",
        flush=True,
    )
    write_metadata(args, reward_stats, reward_df=reward_df)

    bt_obs_df = build_arena_response_observations(reward_df)
    adaptive_a_by_question: dict[str, float] | None = None
    run_rows: list[dict] = []
    wrote_header = False
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    for method in args.methods:
        ref = fit_reference(
            method,
            reward_df,
            reference_epochs=args.reference_epochs,
            reference_lr=args.reference_lr,
            bt_epochs=args.bt_epochs,
            bt_lr=args.bt_lr,
            bt_reg_lambda=args.bt_reg_lambda,
            irt_reg_lambda=args.irt_reg_lambda,
            quiet=args.quiet,
        )
        ref.model_params.to_csv(args.output_dir / f"reference_{method}.csv", index=False)
        if ref.question_params is not None:
            ref.question_params.to_csv(
                args.output_dir / f"reference_questions_{method}.csv",
                index=False,
            )
            if method == "irt_arena":
                adaptive_a_by_question = (
                    ref.question_params.set_index("question_id")["discrimination_exp_k"]
                    .astype(float)
                    .to_dict()
                )

        for model_name in ref.model_params["model_name"].tolist():
            if adaptive_a_by_question is None:
                raise RuntimeError(
                    "Adaptive replay requires the irt_arena reference to run before bt_arena."
                )
            problem = prepare_replay_problem(
                str(model_name),
                ref,
                reward_df,
                bt_obs_df=bt_obs_df if method == "bt_arena" else None,
                adaptive_a_by_question=adaptive_a_by_question,
                theta_pad=args.theta_pad,
                grid_size=args.grid_size,
            )
            for seed in seeds:
                traj_df, run_summary = replay_prepared_problem(
                    problem,
                    seed=seed,
                    k_stability=args.k_stability,
                    replay_reg_lambda=args.replay_reg_lambda,
                    max_steps=args.max_steps,
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
            print(f"  {method}: replayed {model_name} across {len(seeds)} seeds", flush=True)

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(run_summary_path, index=False)
    summary_df = summarise_runs(run_df)
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Arena cold-start summary ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nOutputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
