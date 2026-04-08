"""Thin wrappers around ranking.py metric functions for robustness experiments."""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy.stats import spearmanr, kendalltau as _kendalltau

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


def _model_key(name: str) -> str:
    return name.strip().lower()


def compute_all_metrics(model_params: pd.DataFrame, ref_dict: dict[str, int]) -> dict:
    """
    Compare IRT-generated model ranking against a reference dict.

    Parameters
    ----------
    model_params : pd.DataFrame
        Output of fit_*_irt — columns [model_name, theta], sorted by theta descending.
    ref_dict : dict[str, int]
        Lowercase model name → reference rank (1 = best).

    Returns
    -------
    dict with keys: spearman_rho, kendall_tau, top3_acc, top5_acc, exact_matches
    """
    mp = model_params.copy().reset_index(drop=True)
    mp["pred_rank"] = range(1, len(mp) + 1)
    mp["model_key"] = mp["model_name"].map(_model_key)

    aligned = [
        (int(row["pred_rank"]), ref_dict[row["model_key"]])
        for _, row in mp.iterrows()
        if row["model_key"] in ref_dict
    ]

    if len(aligned) < 2:
        return dict(spearman_rho=float("nan"), kendall_tau=float("nan"),
                    top3_acc=float("nan"), top5_acc=float("nan"), exact_matches=0)

    pred_ranks = [a[0] for a in aligned]
    ref_ranks  = [a[1] for a in aligned]
    rho, _ = spearmanr(pred_ranks, ref_ranks)
    tau, _ = _kendalltau(pred_ranks, ref_ranks)

    ref_top3  = {m for m, r in ref_dict.items() if r <= 3}
    ref_top5  = {m for m, r in ref_dict.items() if r <= 5}
    pred_top3 = set(mp[mp["pred_rank"] <= 3]["model_key"].tolist())
    pred_top5 = set(mp[mp["pred_rank"] <= 5]["model_key"].tolist())

    return dict(
        spearman_rho=float(rho),
        kendall_tau=float(tau),
        top3_acc=float(len(pred_top3 & ref_top3) / 3.0),
        top5_acc=float(len(pred_top5 & ref_top5) / 5.0),
        exact_matches=int(sum(p == r for p, r in aligned)),
    )


def save_results(rows: list[dict], out_dir: str, filename: str) -> pd.DataFrame:
    """Save results to CSV and print a tabular summary."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"\nResults saved to {path}")
    print(df.to_string(index=False))
    return df
