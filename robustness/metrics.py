"""Thin wrappers around ranking.py metric functions for robustness experiments."""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ranking import (
    fit_static_irt,
    fit_arena_irt,
    fit_joint_irt,
    compute_rank_correlations,
)


def fit_and_compare(
    sparse_df: pd.DataFrame,
    ref_mp: pd.DataFrame,
    ref_qp: pd.DataFrame,
    mode: str = "static",
    arena_df: pd.DataFrame | None = None,
    **kwargs,
) -> dict:
    """
    Fit IRT on sparse_df (and optionally arena_df for joint mode),
    then compute Spearman rho vs. reference params.

    Returns dict with keys: model_rho, question_rho, plus any extra kwargs passed in.
    """
    verbose = kwargs.pop("verbose", False)
    extra = {k: v for k, v in kwargs.items() if k not in (
        "num_epochs", "lr", "reg_lambda", "lambda_static", "lambda_arena",
        "lambda_tie", "lambda_bb",
    )}
    fit_kwargs = {k: v for k, v in kwargs.items() if k not in extra}

    if mode == "static":
        mp, qp = fit_static_irt(sparse_df, verbose=verbose, **fit_kwargs)
    elif mode == "arena":
        mp, qp = fit_arena_irt(sparse_df, verbose=verbose, **fit_kwargs)
    elif mode == "joint":
        assert arena_df is not None, "arena_df required for joint mode"
        mp, qp = fit_joint_irt(sparse_df, arena_df, verbose=verbose, **fit_kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    corr = compute_rank_correlations(mp, qp, ref_mp, ref_qp)
    return {
        "model_rho": corr["model_spearman_rho"],
        "question_rho": corr["question_spearman_rho"],
        **extra,
    }


def save_results(rows: list[dict], out_dir: str, filename: str) -> pd.DataFrame:
    """Save results to CSV and print a tabular summary."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"\nResults saved to {path}")
    print(df.to_string(index=False))
    return df
