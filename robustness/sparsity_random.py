"""
Exp 1: Random question subsampling.

Randomly keeps a fraction of distinct questions and re-fits IRT, then compares
the resulting model ranking against the hardcoded expected rankings from
robustness/README.md.

Supported modes:
  static      — subsample from a static JSONL (or CSV), compare vs. static reference
  arena       — subsample from data/pairwise_results_900.csv, compare vs. Pairwise IRT ranking
  both        — subsample from both simultaneously, run fit_joint_irt, compare vs.
                2PL+Pairwise IRT (Overall) ranking
  reward      — subsample from an arena reward JSONL, compare vs. reward reference
  both-reward — subsample from static JSONL + arena reward JSONL simultaneously,
                run fit_joint_reward_irt, compare vs. joint reference

Usage:
    python robustness/sparsity_random.py \\
        --mode static \\
        --static-jsonl data/new/static_math_v0.jsonl \\
        --fractions 0.05 0.1 0.2 0.3 0.5 0.7 \\
        --seeds 0 1 2 3 4

    python robustness/sparsity_random.py --mode reward \\
        --arena-jsonl data/new/arena_math_v0.jsonl --fractions 0.25 0.5 0.75

    python robustness/sparsity_random.py --mode both-reward \\
        --static-jsonl data/new/static_math_v0.jsonl \\
        --arena-jsonl data/new/arena_math_v0.jsonl --fractions 0.1 0.25 0.5

    # Old CSV-based modes still work:
    python robustness/sparsity_random.py --mode arena --arena-csv data/pairwise_results_900.csv
    python robustness/sparsity_random.py --mode both  --static-csv data/static_10_models.csv \\
        --arena-csv data/pairwise_results_900.csv --fractions 0.1 0.25 0.5
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime

from robustness.common_cli import base_parser
from robustness.data_utils import (
    load_static, load_static_jsonl, load_arena, load_arena_reward,
    subsample_questions,
)
from robustness.metrics import compute_all_metrics
from robustness.reference_rankings import REFERENCE_RANKINGS, JOINT_REFERENCE_RANKINGS
from ranking import fit_static_irt, fit_arena_irt, fit_joint_irt, fit_reward_irt, fit_joint_reward_irt


def _file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _dataset_label(mode: str, static_csv: str | None, static_jsonl: str, arena_jsonl: str) -> str:
    """Extract human-readable dataset label (math/coding/generic/all) for folder naming."""
    if mode in ("arena", "both") or (mode == "static" and static_csv):
        return ""  # old CSV modes — no domain label
    stem = _file_stem(static_jsonl if mode in ("static", "both-reward") else arena_jsonl)
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _subsample_arena_questions(arena_df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Sample a fraction of distinct question_ids from the arena DataFrame."""
    kept = (
        pd.Series(arena_df["question_id"].unique())
        .sample(frac=frac, random_state=seed)
    )
    return arena_df[arena_df["question_id"].isin(set(kept))].reset_index(drop=True)


def save_fraction_results(
    frac: float,
    mode: str,
    dataset: str,
    seed_rows: list[dict],
    last_sparse_static: pd.DataFrame | None,
    last_sparse_arena: pd.DataFrame | None,
    last_sparse_reward: pd.DataFrame | None,
    last_model_params: pd.DataFrame,
    out_dir: str,
    timestamp: str,
) -> str:
    """
    Write per-fraction output to a timestamped folder.

    Folder: {out_dir}/random/{mode}_{dataset}_{timestamp}/f{frac:.2f}/
    Files:
      sampled_questions.csv   — question_ids sampled (last seed)
      ranking.csv             — model ranking from last seed
      metrics.csv             — per-seed metrics + aggregate row
    """
    prefix = f"{mode}_{dataset}" if dataset else mode
    folder = os.path.join(
        out_dir, "random",
        f"{prefix}_{timestamp}",
        f"f{frac:.2f}",
    )
    os.makedirs(folder, exist_ok=True)

    # 1. sampled_questions.csv (use last seed's sample as representative)
    if last_sparse_static is not None:
        cols = ["question_id"] + (["level"] if "level" in last_sparse_static.columns else [])
        q_df = last_sparse_static[cols].drop_duplicates("question_id").reset_index(drop=True)
    elif last_sparse_reward is not None:
        q_df = pd.DataFrame({"question_id": last_sparse_reward["question_id"].unique()})
    elif last_sparse_arena is not None:
        q_df = pd.DataFrame({"question_id": last_sparse_arena["question_id"].unique()})
    else:
        q_df = pd.DataFrame()
    q_df.to_csv(os.path.join(folder, "sampled_questions.csv"), index=False)

    # 2. ranking.csv (last seed's IRT output)
    rank_df = last_model_params.copy().reset_index(drop=True)
    rank_df.insert(0, "rank", range(1, len(rank_df) + 1))
    rank_df.to_csv(os.path.join(folder, "ranking.csv"), index=False)

    # 3. metrics.csv — per-seed rows + aggregate
    metrics_df = pd.DataFrame(seed_rows)
    numeric_cols = ["spearman_rho", "kendall_tau", "top3_acc", "top5_acc", "exact_matches"]
    mean_row = {"seed": "mean", **{c: metrics_df[c].mean() for c in numeric_cols}}
    std_row  = {"seed": "std",  **{c: metrics_df[c].std()  for c in numeric_cols}}
    summary = pd.concat([metrics_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary.to_csv(os.path.join(folder, "metrics.csv"), index=False)

    print(f"  Saved: {folder}/")
    return folder


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run(
    mode: str = "static",
    fractions: list[float] | None = None,
    seeds: list[int] | None = None,
    static_csv: str | None = None,
    arena_csv: str | None = None,
    static_jsonl: str = "data/new/static_math_v0.jsonl",
    arena_jsonl: str = "data/new/arena_math_v0.jsonl",
    out_dir: str = "robustness/results",
    num_epochs: int = 2000,
    quiet: bool = True,
) -> list[dict]:
    """
    Run Exp 1 for one mode across all requested fractions and seeds.

    Returns a flat list of result dicts (one per fraction×seed).
    """
    if fractions is None:
        fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- Load data ---
    static_df = arena_df = reward_df = None

    if mode in ("static", "both", "both-reward"):
        if static_csv:
            static_df = load_static(static_csv)
        else:
            static_df = load_static_jsonl(static_jsonl)

    if mode in ("arena", "both"):
        arena_df = load_arena(arena_csv)

    if mode in ("reward", "both-reward"):
        reward_df = load_arena_reward(arena_jsonl)

    # --- Dataset label for folder naming ---
    dataset = _dataset_label(mode, static_csv, static_jsonl, arena_jsonl)

    # --- Select reference ranking dict ---
    if mode == "both-reward":
        s_key = _file_stem(static_jsonl)
        a_key = _file_stem(arena_jsonl)
        ref_dict = JOINT_REFERENCE_RANKINGS[(s_key, a_key)]
    elif mode == "arena":
        ref_dict = REFERENCE_RANKINGS["arena"]
    elif mode == "both":
        ref_dict = REFERENCE_RANKINGS["both"]
    elif mode == "static":
        ref_dict = REFERENCE_RANKINGS["static"] if static_csv else REFERENCE_RANKINGS[_file_stem(static_jsonl)]
    else:  # reward
        ref_dict = REFERENCE_RANKINGS[_file_stem(arena_jsonl)]

    all_rows: list[dict] = []

    for frac in fractions:
        print(f"\n--- mode={mode}  fraction={frac:.2f} ---")
        seed_rows: list[dict] = []
        last_sparse_static = None
        last_sparse_arena  = None
        last_sparse_reward = None
        last_model_params  = None

        for seed in seeds:
            print(f"  seed={seed}", end="  ", flush=True)

            sparse_static = None
            sparse_arena  = None
            sparse_reward = None

            if mode in ("static", "both", "both-reward"):
                sparse_static = subsample_questions(static_df, frac, seed)

            if mode in ("arena", "both"):
                sparse_arena = _subsample_arena_questions(arena_df, frac, seed)

            if mode in ("reward", "both-reward"):
                sparse_reward = subsample_questions(reward_df, frac, seed)

            # --- fit IRT ---
            fit_kw = dict(num_epochs=num_epochs, verbose=not quiet)
            if mode == "static":
                mp, _ = fit_static_irt(sparse_static, lr=0.05, **fit_kw)
            elif mode == "arena":
                mp, _ = fit_arena_irt(sparse_arena, lr=0.05, **fit_kw)
            elif mode == "both":
                mp, _ = fit_joint_irt(sparse_static, sparse_arena, lr=0.05, **fit_kw)
            elif mode == "reward":
                mp, _ = fit_reward_irt(sparse_reward, lr=0.05, **fit_kw)
            else:  # both-reward
                mp, _ = fit_joint_reward_irt(sparse_static, sparse_reward, lr=0.02, **fit_kw)

            # --- compute metrics ---
            metrics = compute_all_metrics(mp, ref_dict)
            print(f"ρ={metrics['spearman_rho']:.3f}  τ={metrics['kendall_tau']:.3f}  "
                  f"top3={metrics['top3_acc']:.2f}  top5={metrics['top5_acc']:.2f}  "
                  f"exact={metrics['exact_matches']}")

            row = {"mode": mode, "fraction": frac, "seed": seed, **metrics}
            seed_rows.append({"seed": seed, **metrics})
            all_rows.append(row)

            last_sparse_static = sparse_static
            last_sparse_arena  = sparse_arena
            last_sparse_reward = sparse_reward
            last_model_params  = mp

        # --- save per-fraction results ---
        save_fraction_results(
            frac=frac,
            mode=mode,
            dataset=dataset,
            seed_rows=seed_rows,
            last_sparse_static=last_sparse_static,
            last_sparse_arena=last_sparse_arena,
            last_sparse_reward=last_sparse_reward,
            last_model_params=last_model_params,
            out_dir=out_dir,
            timestamp=timestamp,
        )

    # Print aggregate summary
    print("\n=== Summary (mean across seeds) ===")
    summary_df = (
        pd.DataFrame(all_rows)
        .groupby("fraction")[["spearman_rho", "kendall_tau", "top3_acc", "top5_acc", "exact_matches"]]
        .mean()
        .round(3)
    )
    print(summary_df.to_string())

    return all_rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = base_parser("Exp 1: Random question subsampling")
    p.add_argument("--mode", choices=["static", "arena", "both", "reward", "both-reward"], default="static", help="Which dataset(s) to sample from")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0], help="Fractions of questions to retain")
    args = p.parse_args()

    run(
        mode=args.mode,
        fractions=args.fractions,
        seeds=args.seeds,
        static_csv=args.static_csv,
        arena_csv=args.arena_csv,
        static_jsonl=args.static_jsonl,
        arena_jsonl=args.arena_jsonl,
        out_dir=args.out_dir,
        num_epochs=args.num_epochs,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
