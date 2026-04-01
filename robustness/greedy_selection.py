"""
robustness/greedy_selection.py

(--strategy random):  Forward greedy with random candidate subsampling.
(--strategy diverse):  Forward greedy with level-diversity-based batch selection.

Both experiments incrementally build a question set whose IRT-derived model ranking
converges to the full-dataset reference ranking, measured by Kendall's tau.

Usage
-----
# Random candidate subsampling
python robustness/greedy_selection.py --strategy random \\
    --static-csv data/static_10_models.csv \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2

# Level-diversity batch greedy
python robustness/greedy_selection.py --strategy diverse \\
    --static-csv data/static_10_models.csv \\
    --hard-cap 150 --n-batches 10 --seeds 0 1 2
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau as scipy_kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ranking import fit_static_irt
from robustness.data_utils import load_static
from robustness.metrics import save_results
from robustness.common_cli import base_parser


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _kendall_tau(model_params: pd.DataFrame, ref_ranking: dict) -> float:
    """
    Kendall's tau between the fitted model ranking and a reference ranking dict.

    model_params : DataFrame [model_name, theta], sorted by theta desc (rank 1 = best).
    ref_ranking  : {model_name_lower: int rank (1 = best)}.
    Returns tau in [-1, 1], or 0.0 if fewer than 2 common models.
    """
    names = [m.lower() for m in model_params["model_name"]]
    fitted_rank = {m: i + 1 for i, m in enumerate(names)}
    common = [m for m in names if m in ref_ranking]
    if len(common) < 2:
        return 0.0
    pred = [fitted_rank[m] for m in common]
    ref  = [ref_ranking[m] for m in common]
    tau, _ = scipy_kendalltau(pred, ref)
    return float(tau) if not np.isnan(tau) else 0.0


def _fit_tau(df: pd.DataFrame, ref_ranking: dict, num_epochs: int) -> float:
    """
    Fit IRT on df and return Kendall's tau vs. ref_ranking.
    Always runs silently (verbose=False). Returns -2.0 on failure.
    """
    try:
        mp, _ = fit_static_irt(df, num_epochs=num_epochs, verbose=False)
        return _kendall_tau(mp, ref_ranking)
    except Exception:
        return -2.0


# ---------------------------------------------------------------------------
# Level-group helpers
# ---------------------------------------------------------------------------

def _norm_level(v) -> object:
    """Normalise a level value: NaN / empty string -> None (GSM-8k), else keep as-is."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if str(v).strip() == "":
        return None
    return v


def _build_level_groups(static_df: pd.DataFrame) -> dict[object, list]:
    """
    Return {level_key: [question_id, ...]} where level_key is None for GSM-8k
    and an int/str (1–5) for Hendrycks MATH levels.
    """
    q_level = static_df[["question_id", "level"]].drop_duplicates("question_id")
    groups: dict = {}
    for _, row in q_level.iterrows():
        key = _norm_level(row["level"])
        groups.setdefault(key, []).append(row["question_id"])
    return groups


def _build_qid_level_map(static_df: pd.DataFrame) -> dict:
    """Return {question_id: normalised_level} for printable labels."""
    q_level = static_df[["question_id", "level"]].drop_duplicates("question_id")
    return {row["question_id"]: _norm_level(row["level"]) for _, row in q_level.iterrows()}


def _level_label(v) -> str:
    """Human-readable level label."""
    return "gsm" if v is None else str(v)


# ---------------------------------------------------------------------------
# Strategy: random
# ---------------------------------------------------------------------------

def run_greedy_random(
    static_df: pd.DataFrame,
    ref_ranking: dict,
    *,
    hard_cap: int,
    candidates_per_step: int,
    tau_threshold: float,
    consecutive_needed: int,
    seed: int,
    num_epochs: int,
    quiet: bool,
) -> list[dict]:
    """
    Forward greedy with random candidate subsampling.

    At each step:
      1. Sample `candidates_per_step` questions at random from the remaining pool.
      2. Evaluate each by fitting IRT on (current set ∪ {candidate}).
      3. Add the candidate that maximises Kendall's tau.

    Stops when tau >= tau_threshold for `consecutive_needed` consecutive steps
    or when `hard_cap` questions have been added.
    """
    rng = np.random.default_rng(seed)
    all_qids = static_df["question_id"].unique()
    qid_level = _build_qid_level_map(static_df)

    start_qid = rng.choice(all_qids)
    greedy_set: set = {start_qid}
    remaining: set = set(all_qids) - greedy_set

    rows: list[dict] = []
    consecutive_above = 0
    step = 0

    while len(greedy_set) < hard_cap and remaining:
        step += 1
        m = min(candidates_per_step, len(remaining))
        candidates = rng.choice(list(remaining), size=m, replace=False)

        best_q, best_tau = None, -2.0
        for q in candidates:
            trial_df = static_df[static_df["question_id"].isin(greedy_set | {q})]
            tau = _fit_tau(trial_df, ref_ranking, num_epochs)
            if tau > best_tau:
                best_tau, best_q = tau, q

        greedy_set.add(best_q)
        remaining.discard(best_q)

        lvl = _level_label(qid_level.get(best_q))
        rows.append({
            "step":       step,
            "question_id": best_q,
            "level":      lvl,
            "tau":        best_tau,
            "n_questions": len(greedy_set),
            "seed":       seed,
            "strategy":   "random",
        })

        if not quiet:
            print(
                f"  [random] step {step:3d} | added {best_q} (level={lvl}) "
                f"| τ={best_tau:.4f} | n={len(greedy_set)}"
            )

        if best_tau >= tau_threshold:
            consecutive_above += 1
            if consecutive_above >= consecutive_needed:
                if not quiet:
                    print(
                        f"  Converged: τ ≥ {tau_threshold} for "
                        f"{consecutive_needed} consecutive steps."
                    )
                break
        else:
            consecutive_above = 0

    return rows


# ---------------------------------------------------------------------------
# Strategy: diverse
# ---------------------------------------------------------------------------

def run_greedy_diverse(
    static_df: pd.DataFrame,
    ref_ranking: dict,
    *,
    hard_cap: int,
    n_batches: int,
    tau_threshold: float,
    consecutive_needed: int,
    seed: int,
    num_epochs: int,
    quiet: bool,
) -> list[dict]:
    """
    Forward greedy with level-diversity-based batch selection.

    At each step:
      1. Build `n_batches` diverse candidate batches by sampling 1 question
         from each non-empty level group (GSM-8k + Levels 1–5) not yet in the set.
      2. Evaluate each batch by fitting IRT on (current set ∪ batch).
      3. Add all questions from the best-scoring batch.

    Enforces difficulty diversity by construction; each step adds up to 6 questions.
    Stops when tau >= tau_threshold for `consecutive_needed` consecutive steps
    or when `hard_cap` questions have been added.
    """
    rng = np.random.default_rng(seed)
    all_qids = static_df["question_id"].unique()
    level_groups = _build_level_groups(static_df)

    start_qid = rng.choice(all_qids)
    greedy_set: set = {start_qid}

    rows: list[dict] = []
    consecutive_above = 0
    step = 0

    while len(greedy_set) < hard_cap:
        step += 1

        # Build n_batches diverse candidate batches
        batches: list[list] = []
        for _ in range(n_batches):
            batch = []
            for qids in level_groups.values():
                available = [q for q in qids if q not in greedy_set]
                if available:
                    batch.append(rng.choice(available))
            if batch:
                batches.append(batch)

        if not batches:
            break  # all questions exhausted

        best_batch, best_tau = None, -2.0
        for batch in batches:
            trial_df = static_df[static_df["question_id"].isin(greedy_set | set(batch))]
            tau = _fit_tau(trial_df, ref_ranking, num_epochs)
            if tau > best_tau:
                best_tau, best_batch = tau, batch

        greedy_set.update(best_batch)

        # Summarise level composition of the added batch
        level_comp: dict[str, int] = {}
        for q in best_batch:
            lbl = _level_label(
                next((k for k, v in level_groups.items() if q in v), None)
            )
            level_comp[lbl] = level_comp.get(lbl, 0) + 1

        rows.append({
            "step":             step,
            "questions_added":  len(best_batch),
            "n_questions":      len(greedy_set),
            "tau":              best_tau,
            "level_composition": str(level_comp),
            "seed":             seed,
            "strategy":         "diverse",
        })

        if not quiet:
            print(
                f"  [diverse] step {step:3d} | added {len(best_batch)} questions "
                f"({level_comp}) | τ={best_tau:.4f} | n={len(greedy_set)}"
            )

        if best_tau >= tau_threshold:
            consecutive_above += 1
            if consecutive_above >= consecutive_needed:
                if not quiet:
                    print(
                        f"  Converged: τ ≥ {tau_threshold} for "
                        f"{consecutive_needed} consecutive steps."
                    )
                break
        else:
            consecutive_above = 0

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = base_parser("Greedy question selection")
    parser.add_argument("--strategy", choices=["random", "diverse"], default="random", help="random: random candidate subsampling; diverse: level-diversity batch greedy")
    parser.add_argument("--hard-cap", type=int, default=150, help="Maximum total questions in the selected set")
    parser.add_argument("--candidates", type=int, default=50, help="[random] Candidate questions sampled per step")
    parser.add_argument("--n-batches", type=int, default=10, help="[diverse] Diverse batches evaluated per step")
    parser.add_argument("--tau-threshold", type=float, default=0.95, help="Kendall tau convergence threshold")
    parser.add_argument("--consecutive", type=int, default=3, help="Consecutive steps above tau-threshold to declare convergence")
    # Lower epoch default for speed; user can override with --num-epochs
    parser.set_defaults(num_epochs=500, seeds=[0, 1, 2])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    static_df = load_static(args.static_csv)

    # ------------------------------------------------------------------
    # Fit reference IRT once on the full dataset
    # ------------------------------------------------------------------
    if not args.quiet:
        print("Fitting reference IRT on full dataset ...")
    ref_mp, _ = fit_static_irt(static_df, num_epochs=args.num_epochs, verbose=not args.quiet)
    ref_ranking: dict = {
        m.lower(): i + 1 for i, m in enumerate(ref_mp["model_name"])
    }
    if not args.quiet:
        print("Reference ranking:")
        for name, rank in sorted(ref_ranking.items(), key=lambda x: x[1]):
            print(f"  {rank:2d}. {name}")
        print()

    # ------------------------------------------------------------------
    # Run greedy for each seed
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    for seed in args.seeds:
        if not args.quiet:
            print(f"=== Seed {seed} | strategy={args.strategy} ===")

        if args.strategy == "random":
            rows = run_greedy_random(
                static_df,
                ref_ranking,
                hard_cap=args.hard_cap,
                candidates_per_step=args.candidates,
                tau_threshold=args.tau_threshold,
                consecutive_needed=args.consecutive,
                seed=seed,
                num_epochs=args.num_epochs,
                quiet=args.quiet,
            )
        else:
            rows = run_greedy_diverse(
                static_df,
                ref_ranking,
                hard_cap=args.hard_cap,
                n_batches=args.n_batches,
                tau_threshold=args.tau_threshold,
                consecutive_needed=args.consecutive,
                seed=seed,
                num_epochs=args.num_epochs,
                quiet=args.quiet,
            )

        all_rows.extend(rows)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    filename = "greedy_random.csv" if args.strategy == "random" else "greedy_diverse.csv"
    save_results(all_rows, args.out_dir, filename)
    print(f"\nResults saved to {os.path.join(args.out_dir, filename)}")


if __name__ == "__main__":
    main()
