from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rm_analysis.cold_start.bt import bt_loss_matrix, bt_observations_for_model
from rm_analysis.cold_start.irt import irt_information, irt_loss_matrix, irt_observations_for_model
from rm_analysis.cold_start.math_utils import compute_n_star, rank_error_at, rank_from_theta, sigmoid_np

ARENA_METHODS = ["irt_arena", "bt_arena"]


@dataclass(frozen=True)
class ReferenceFit:
    method: str
    model_params: pd.DataFrame
    question_params: pd.DataFrame | None = None

    @property
    def theta_map(self) -> dict[str, float]:
        return {
            str(row["model_name"]): float(row["theta"])
            for row in self.model_params.to_dict(orient="records")
        }

    @property
    def rank_map(self) -> dict[str, int]:
        ordered = self.model_params.sort_values("theta", ascending=False)["model_name"].tolist()
        return {str(model_name): rank for rank, model_name in enumerate(ordered, start=1)}


@dataclass(frozen=True)
class ReplayProblem:
    method: str
    model_name: str
    observations: list[dict[str, Any]]
    loss_matrix: np.ndarray
    theta_grid: np.ndarray
    theta_ref: float
    rank_ref: int
    other_thetas: np.ndarray
    selection_scores: np.ndarray


def theta_grid_for_reference(ref: ReferenceFit, *, theta_pad: float, grid_size: int) -> np.ndarray:
    vals = ref.model_params["theta"].to_numpy(dtype=float)
    lo = min(float(vals.min()) - theta_pad, -theta_pad)
    hi = max(float(vals.max()) + theta_pad, theta_pad)
    return np.linspace(lo, hi, grid_size)


def prepare_replay_problem(
    model_name: str,
    ref: ReferenceFit,
    reward_df: pd.DataFrame,
    *,
    bt_obs_df: pd.DataFrame | None,
    adaptive_a_by_question: dict[str, float],
    theta_pad: float,
    grid_size: int,
) -> ReplayProblem:
    if ref.method not in ARENA_METHODS:
        raise ValueError(f"Arena cold-start supports only {ARENA_METHODS}; got {ref.method}")

    theta_map = {m: t for m, t in ref.theta_map.items() if m != model_name}
    theta_grid = theta_grid_for_reference(ref, theta_pad=theta_pad, grid_size=grid_size)

    if ref.method == "irt_arena":
        if ref.question_params is None:
            raise RuntimeError("IRT replay requires question parameters.")
        observations = irt_observations_for_model(model_name, reward_df, ref.question_params)
        loss_matrix = irt_loss_matrix(observations, theta_grid)
    else:
        if bt_obs_df is None:
            raise RuntimeError("BT replay requires precomputed response observations.")
        observations = bt_observations_for_model(model_name, bt_obs_df, theta_map)
        loss_matrix = bt_loss_matrix(observations, theta_grid)

    selection_scores = np.array(
        [float(adaptive_a_by_question.get(obs["question_id"], 0.0)) for obs in observations],
        dtype=float,
    )

    return ReplayProblem(
        method=ref.method,
        model_name=model_name,
        observations=observations,
        loss_matrix=loss_matrix,
        theta_grid=theta_grid,
        theta_ref=ref.theta_map[model_name],
        rank_ref=ref.rank_map[model_name],
        other_thetas=np.array(list(theta_map.values()), dtype=float),
        selection_scores=selection_scores,
    )


def ordered_info(
    method: str,
    ordered_observations: list[dict[str, Any]],
    theta_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    info_x = np.full(len(ordered_observations), np.nan, dtype=float)
    info_value = np.full(len(ordered_observations), np.nan, dtype=float)
    theta_prev = np.concatenate([[0.0], theta_hat[:-1]])

    for idx, obs in enumerate(ordered_observations):
        prev = float(theta_prev[idx])
        if method == "irt_arena":
            info_x[idx], info_value[idx] = irt_information(obs, prev)
        elif method == "bt_arena":
            opp_theta = obs["opp_theta"]
            gaps = np.abs(opp_theta - prev)
            p = sigmoid_np(prev - opp_theta)
            info_x[idx] = float(gaps.mean())
            info_value[idx] = float((p * (1.0 - p)).mean())
    return info_x, info_value


def replay_prepared_problem(
    problem: ReplayProblem,
    *,
    seed: int,
    k_stability: int,
    replay_reg_lambda: float,
    max_steps: int | None,
    record_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observations = problem.observations
    loss_matrix = problem.loss_matrix
    theta_grid = problem.theta_grid
    n_total = len(observations)
    if n_total == 0:
        return pd.DataFrame(), {
            "method": problem.method,
            "m_new": problem.model_name,
            "seed": seed,
            "n_obs": 0,
            "rank_ref": problem.rank_ref,
            "theta_ref": problem.theta_ref,
            "n_star": float("nan"),
            "rank_error_at_50": float("nan"),
            "rank_error_at_200": float("nan"),
            "final_rank_hat": float("nan"),
            "final_rank_error": float("nan"),
            "theta_final": float("nan"),
        }

    order = np.array(
        sorted(
            range(n_total),
            key=lambda idx: (-problem.selection_scores[idx], observations[idx]["obs_id"]),
        ),
        dtype=int,
    )
    if max_steps is not None:
        order = order[: min(max_steps, n_total)]

    ordered_losses = loss_matrix[order]
    ordered_observations = [observations[int(idx)] for idx in order]
    ns = np.arange(1, len(order) + 1)
    prior = replay_reg_lambda * theta_grid**2
    cumulative_loss = np.cumsum(ordered_losses, axis=0)
    objective = cumulative_loss + prior[None, :]
    argmins = np.argmin(objective, axis=1)
    theta_hat = theta_grid[argmins]
    ranks = np.array(
        [rank_from_theta(theta, problem.other_thetas) for theta in theta_hat],
        dtype=int,
    )

    total_loss = loss_matrix.sum(axis=0)
    heldout_counts = n_total - ns
    remaining_loss = total_loss[None, :] - cumulative_loss
    heldout_loss = np.full(len(ns), np.nan, dtype=float)
    nonzero = heldout_counts > 0
    heldout_loss[nonzero] = (
        remaining_loss[np.arange(len(ns))[nonzero], argmins[nonzero]] / heldout_counts[nonzero]
    )
    ll_held_out = -heldout_loss
    ll_gain = np.concatenate([[np.nan], np.diff(ll_held_out)])
    info_x, info_value = ordered_info(problem.method, ordered_observations, theta_hat)

    n_star = compute_n_star(ns, ranks, k_stability=k_stability)
    rank_error = np.abs(ranks - problem.rank_ref)
    record_mask = (ns == 1) | (ns == len(ns)) | (ns % max(1, record_every) == 0)

    rows: list[dict[str, Any]] = []
    for idx in np.flatnonzero(record_mask):
        obs = ordered_observations[int(idx)]
        rows.append(
            {
                "method": problem.method,
                "m_new": problem.model_name,
                "seed": seed,
                "n": int(ns[idx]),
                "obs_id": obs["obs_id"],
                "obs_kind": obs["kind"],
                "source": obs["source"],
                "question_id": obs["question_id"],
                "theta_hat": float(theta_hat[idx]),
                "theta_ref": float(problem.theta_ref),
                "rank_hat": int(ranks[idx]),
                "rank_ref": int(problem.rank_ref),
                "rank_error": int(rank_error[idx]),
                "ll_held_out": float(ll_held_out[idx]) if math.isfinite(ll_held_out[idx]) else np.nan,
                "ll_gain": float(ll_gain[idx]) if math.isfinite(ll_gain[idx]) else np.nan,
                "info_x": float(info_x[idx]) if math.isfinite(info_x[idx]) else np.nan,
                "info_value": (
                    float(info_value[idx]) if math.isfinite(info_value[idx]) else np.nan
                ),
                "selection_a": float(problem.selection_scores[int(order[idx])]),
                "n_total_obs": int(n_total),
                "n_star": n_star,
            }
        )

    summary = {
        "method": problem.method,
        "m_new": problem.model_name,
        "seed": seed,
        "n_obs": int(n_total),
        "n_replayed": int(len(ns)),
        "rank_ref": int(problem.rank_ref),
        "theta_ref": float(problem.theta_ref),
        "n_star": n_star,
        "rank_error_at_50": rank_error_at(ns, ranks, problem.rank_ref, 50),
        "rank_error_at_200": rank_error_at(ns, ranks, problem.rank_ref, 200),
        "final_rank_hat": int(ranks[-1]),
        "final_rank_error": int(rank_error[-1]),
        "theta_final": float(theta_hat[-1]),
    }
    return pd.DataFrame(rows), summary
