from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid_np(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr, dtype=float)
    positive = arr >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def bce_with_logits_np(logits: np.ndarray, targets: np.ndarray | float) -> np.ndarray:
    return softplus_np(logits) - targets * logits


def rank_from_theta(theta: float, other_thetas: np.ndarray) -> int:
    return int(1 + np.sum(other_thetas > theta))


def compute_n_star(ns: np.ndarray, ranks: np.ndarray, *, k_stability: int) -> float:
    """Smallest n whose rank remains unchanged for the next K query steps."""
    if len(ns) == 0:
        return float("nan")
    for idx, n_value in enumerate(ns):
        target_n = n_value + k_stability
        in_window = (ns >= n_value) & (ns <= target_n)
        if not np.any(ns[in_window] >= target_n):
            continue
        if np.all(ranks[in_window] == ranks[idx]):
            return float(n_value)
    return float("nan")


def rank_error_at(ns: np.ndarray, ranks: np.ndarray, rank_ref: int, cutoff: int) -> float:
    if len(ns) == 0:
        return float("nan")
    idxs = np.flatnonzero(ns <= cutoff)
    idx = int(idxs[-1]) if len(idxs) else 0
    return float(abs(int(ranks[idx]) - rank_ref))


def nan_percentile(series: pd.Series, q: float) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def summarise_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    return (
        run_df.groupby("method", as_index=False)
        .agg(
            mean_n_star=("n_star", "mean"),
            median_n_star=("n_star", "median"),
            q25_n_star=("n_star", lambda s: nan_percentile(s, 25)),
            q75_n_star=("n_star", lambda s: nan_percentile(s, 75)),
            mean_rank_error_at_50=("rank_error_at_50", "mean"),
            mean_rank_error_at_200=("rank_error_at_200", "mean"),
            mean_final_rank_error=("final_rank_error", "mean"),
            n_runs=("seed", "count"),
            n_models=("m_new", "nunique"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
