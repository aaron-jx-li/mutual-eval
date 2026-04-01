"""
Exp 1: Random sparsification of static and arena data.

Removes random fractions of rows, questions, or pairwise comparisons and
measures how well IRT recovers the reference ranking.

Usage:
    python robustness/sparsity_random.py \
        --mode static --sparsity-type rows \
        --fractions 0.1 0.25 0.5 0.75 0.9 \
        --seeds 0 1 2 3 4

    python robustness/sparsity_random.py \
        --mode arena --sparsity-type pairs \
        --fractions 0.25 0.5 0.75 \
        --seeds 0 1 2
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustness.common_cli import base_parser
from robustness.data_utils import (
    load_static, load_arena,
    subsample_rows, subsample_questions, subsample_pairs,
)
from robustness.metrics import fit_and_compare, save_results
from ranking import fit_static_irt, fit_arena_irt


def run(
    mode: str = "static",
    sparsity_type: str = "rows",
    fractions: list[float] | None = None,
    seeds: list[int] | None = None,
    static_csv: str = "data/static_10_models.csv",
    arena_csv: str = "data/pairwise_results_900.csv",
    out_dir: str = "robustness/results",
    num_epochs: int = 2000,
    quiet: bool = True,
) -> list[dict]:
    if fractions is None:
        fractions = [0.1, 0.25, 0.5, 0.75, 0.9]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    static_df = load_static(static_csv) if mode in ("static", "both") else None
    arena_df = load_arena(arena_csv) if mode in ("arena", "both") else None

    # Fit reference model on full data
    print("Fitting reference IRT on full data...")
    if mode == "static":
        ref_mp, ref_qp = fit_static_irt(static_df, num_epochs=num_epochs, verbose=not quiet)
    elif mode == "arena":
        ref_mp, ref_qp = fit_arena_irt(arena_df, num_epochs=num_epochs, verbose=not quiet)
    else:  # both — use static as reference
        ref_mp, ref_qp = fit_static_irt(static_df, num_epochs=num_epochs, verbose=not quiet)

    rows = []
    total = len(fractions) * len(seeds)
    done = 0

    for frac in fractions:
        for seed in seeds:
            print(f"  [{done+1}/{total}] mode={mode} type={sparsity_type} frac={frac} seed={seed}")

            if mode == "static" or mode == "both":
                if sparsity_type == "rows":
                    sparse = subsample_rows(static_df, frac, seed)
                elif sparsity_type == "questions":
                    sparse = subsample_questions(static_df, frac, seed)
                else:
                    raise ValueError(f"sparsity_type '{sparsity_type}' not valid for static mode; use rows or questions")

                result = fit_and_compare(
                    sparse, ref_mp, ref_qp,
                    mode="static",
                    num_epochs=num_epochs,
                    verbose=not quiet,
                )
            elif mode == "arena":
                if sparsity_type == "pairs":
                    sparse = subsample_pairs(arena_df, frac, seed)
                elif sparsity_type == "questions":
                    # Subsample distinct question_ids in arena df
                    import pandas as pd
                    kept_qs = pd.Series(arena_df["question_id"].unique()).sample(
                        frac=frac, random_state=seed
                    )
                    sparse = arena_df[arena_df["question_id"].isin(kept_qs)].reset_index(drop=True)
                else:
                    raise ValueError(f"sparsity_type '{sparsity_type}' not valid for arena mode; use pairs or questions")

                result = fit_and_compare(
                    sparse, ref_mp, ref_qp,
                    mode="arena",
                    num_epochs=num_epochs,
                    verbose=not quiet,
                )

            rows.append({
                "mode": mode,
                "sparsity_type": sparsity_type,
                "fraction": frac,
                "seed": seed,
                "model_rho": result["model_rho"],
                "question_rho": result["question_rho"],
            })
            done += 1

    fname = f"sparsity_random_{mode}_{sparsity_type}.csv"
    save_results(rows, out_dir, fname)
    return rows


def main() -> None:
    p = base_parser("Exp 1: Random sparsification")
    p.add_argument("--mode", choices=["static", "arena", "both"], default="static")
    p.add_argument("--sparsity-type", choices=["rows", "questions", "pairs"], default="rows")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9], help="Fractions of data to keep")
    args = p.parse_args()

    run(
        mode=args.mode,
        sparsity_type=args.sparsity_type,
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
