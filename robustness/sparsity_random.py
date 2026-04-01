"""
Exp 1: Random question subsampling.

Randomly keeps a fraction of distinct questions and re-fits IRT, then compares
the resulting model ranking against the hardcoded expected rankings from
robustness/README.md.

Supported modes:
  static  — subsample from data/static_10_models.csv, compare vs. 2PL ranking
  arena   — subsample from data/pairwise_results_900.csv, compare vs. Pairwise IRT ranking
  both    — subsample from both simultaneously, run fit_joint_irt, compare vs.
             2PL+Pairwise IRT (Overall) ranking

Usage:
    python robustness/sparsity_random.py \\
        --mode static \\
        --fractions 0.05 0.1 0.2 0.3 0.5 0.7 \\
        --seeds 0 1 2 3 4

    python robustness/sparsity_random.py --mode arena --fractions 0.25 0.5 0.75
    python robustness/sparsity_random.py --mode both  --fractions 0.1 0.25 0.5
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr, kendalltau

from robustness.common_cli import base_parser
from robustness.data_utils import load_static, load_arena, subsample_questions
from ranking import fit_static_irt, fit_arena_irt, fit_joint_irt


# ---------------------------------------------------------------------------
# Hardcoded reference rankings from robustness/README.md Expected Results
# Keys are lowercase model names matching the actual CSV values.
# ---------------------------------------------------------------------------

# 2PL column (static dataset)
_STATIC_REF = {
    "grok-3-mini-beta":           1,
    "gpt-4.1-mini-2025-04-14":    2,
    "deepseek-v3-0324":           3,
    "mistral-medium-2505":        4,
    "claude-3-5-haiku-20241022":  5,
    "llama-3.3-70b-it":           6,
    "gpt-4o-mini":                7,
    "gemini-2.0-flash":           8,
    "claude-3-7-sonnet-20250219": 9,
    "gemma-3-27b-it":             10,
}

# Pairwise IRT (Static) column — same ordering as 2PL
_ARENA_REF = {
    "deepseek-v3-0324":           1,
    "gpt-4.1-mini-2025-04-14":    2,
    "mistral-medium-2505":        3,
    "claude-3-7-sonnet-20250219": 4,
    "grok-3-mini-beta":           5,
    "gpt-4o-mini":                6,
    "gemma-3-27b-it":             7,
    "claude-3-5-haiku-20241022":  8,
    "llama-3.3-70b-it":           9,
    "gemini-2.0-flash":           10,
}

# 2PL+Pairwise IRT (Overall) column
_BOTH_REF = {
    "deepseek-v3-0324":           1,
    "gpt-4.1-mini-2025-04-14":    2,
    "mistral-medium-2505":        3,
    "grok-3-mini-beta":           4,
    "claude-3-5-haiku-20241022":  5,
    "gpt-4o-mini":                6,
    "llama-3.3-70b-it":           7,
    "gemini-2.0-flash":           8,
    "claude-3-7-sonnet-20250219": 9,
    "gemma-3-27b-it":             10,
}

REFERENCE_RANKINGS: dict[str, dict[str, int]] = {
    "static": _STATIC_REF,
    "arena":  _ARENA_REF,
    "both":   _BOTH_REF,
}


def _model_key(name: str) -> str:
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(model_params: pd.DataFrame, mode: str) -> dict:
    """
    Compare IRT-generated model ranking against the hardcoded reference for `mode`.

    Parameters
    ----------
    model_params : pd.DataFrame
        Output of fit_*_irt — must have columns [model_name, theta], already
        sorted by theta descending (rank 1 = highest theta).
    mode : str
        One of "static", "arena", "both".

    Returns
    -------
    dict with keys: spearman_rho, kendall_tau, top3_acc, top5_acc, exact_matches
    """
    ref = REFERENCE_RANKINGS[mode]

    # Build aligned lists of (pred_rank, ref_rank) for models present in both
    mp = model_params.copy().reset_index(drop=True)
    mp["pred_rank"] = range(1, len(mp) + 1)
    mp["model_key"] = mp["model_name"].map(_model_key)

    aligned = []
    for _, row in mp.iterrows():
        key = row["model_key"]
        if key in ref:
            aligned.append((int(row["pred_rank"]), ref[key]))

    if len(aligned) < 2:
        return dict(spearman_rho=float("nan"), kendall_tau=float("nan"),
                    top3_acc=float("nan"), top5_acc=float("nan"),
                    exact_matches=0)

    pred_ranks = [a[0] for a in aligned]
    ref_ranks  = [a[1] for a in aligned]

    rho, _  = spearmanr(pred_ranks, ref_ranks)
    tau, _  = kendalltau(pred_ranks, ref_ranks)

    # Top-k accuracy: fraction of top-k predicted that appear in top-k reference
    ref_top3 = {m for m, r in ref.items() if r <= 3}
    ref_top5 = {m for m, r in ref.items() if r <= 5}
    pred_top3 = set(mp[mp["pred_rank"] <= 3]["model_key"].tolist())
    pred_top5 = set(mp[mp["pred_rank"] <= 5]["model_key"].tolist())

    top3_acc = len(pred_top3 & ref_top3) / 3.0
    top5_acc = len(pred_top5 & ref_top5) / 5.0

    exact_matches = sum(p == r for p, r in aligned)

    return dict(
        spearman_rho=float(rho),
        kendall_tau=float(tau),
        top3_acc=float(top3_acc),
        top5_acc=float(top5_acc),
        exact_matches=int(exact_matches),
    )


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
    seed_rows: list[dict],          # one dict per seed
    last_sparse_static: pd.DataFrame | None,
    last_sparse_arena: pd.DataFrame | None,
    last_model_params: pd.DataFrame,
    out_dir: str,
    timestamp: str,
) -> str:
    """
    Write per-fraction output to a timestamped folder.

    Folder: {out_dir}/random/{mode}_f{frac:.2f}_{timestamp}/
    Files:
      sampled_questions.csv   — question_ids sampled (last seed)
      ranking.csv             — model ranking from last seed
      metrics.csv             — per-seed metrics + aggregate row
    """
    folder = os.path.join(
        out_dir, "random",
        f"{mode}_f{frac:.2f}_{timestamp}",
    )
    os.makedirs(folder, exist_ok=True)

    # 1. sampled_questions.csv (use last seed's sample as representative)
    if last_sparse_static is not None:
        cols = ["question_id"] + (["level"] if "level" in last_sparse_static.columns else [])
        q_df = last_sparse_static[cols].drop_duplicates("question_id").reset_index(drop=True)
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
    static_csv: str = "data/static_10_models.csv",
    arena_csv: str = "data/pairwise_results_900.csv",
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

    static_df = load_static(static_csv) if mode in ("static", "both") else None
    arena_df  = load_arena(arena_csv)   if mode in ("arena",  "both") else None

    all_rows: list[dict] = []

    for frac in fractions:
        print(f"\n--- mode={mode}  fraction={frac:.2f} ---")
        seed_rows: list[dict] = []
        last_sparse_static = None
        last_sparse_arena  = None
        last_model_params  = None

        for seed in seeds:
            print(f"  seed={seed}", end="  ", flush=True)

            # --- sample questions ---
            sparse_static = None
            sparse_arena  = None

            if mode in ("static", "both"):
                sparse_static = subsample_questions(static_df, frac, seed)

            if mode in ("arena", "both"):
                sparse_arena = _subsample_arena_questions(arena_df, frac, seed)

            # --- fit IRT ---
            fit_kw = dict(num_epochs=num_epochs, verbose=not quiet)
            if mode == "static":
                mp, _ = fit_static_irt(sparse_static, **fit_kw)
            elif mode == "arena":
                mp, _ = fit_arena_irt(sparse_arena, **fit_kw)
            else:  # both
                mp, _ = fit_joint_irt(sparse_static, sparse_arena, **fit_kw)

            # --- compute metrics ---
            metrics = compute_all_metrics(mp, mode)
            print(f"ρ={metrics['spearman_rho']:.3f}  τ={metrics['kendall_tau']:.3f}  "
                  f"top3={metrics['top3_acc']:.2f}  top5={metrics['top5_acc']:.2f}  "
                  f"exact={metrics['exact_matches']}")

            row = {"mode": mode, "fraction": frac, "seed": seed, **metrics}
            seed_rows.append({"seed": seed, **metrics})
            all_rows.append(row)

            last_sparse_static = sparse_static
            last_sparse_arena  = sparse_arena
            last_model_params  = mp

        # --- save per-fraction results ---
        save_fraction_results(
            frac=frac,
            mode=mode,
            seed_rows=seed_rows,
            last_sparse_static=last_sparse_static,
            last_sparse_arena=last_sparse_arena,
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
    p.add_argument("--mode", choices=["static", "arena", "both"], default="static", help="Which dataset(s) to sample from")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0], help="Fractions of questions to retain")
    args = p.parse_args()

    run(
        mode=args.mode,
        fractions=args.fractions,
        seeds=args.seeds,
        static_csv=args.static_csv,
        arena_csv=args.arena_csv,
        out_dir=args.out_dir,
        num_epochs=args.num_epochs,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
