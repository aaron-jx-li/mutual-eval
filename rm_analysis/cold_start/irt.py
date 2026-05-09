from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def question_maps(question_params: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    qp = question_params.set_index("question_id")
    return qp["difficulty_b"], qp["discrimination_exp_k"]


def irt_observations_for_model(
    model_name: str,
    reward_df: pd.DataFrame,
    question_params: pd.DataFrame,
) -> list[dict[str, Any]]:
    b_map, a_map = question_maps(question_params)
    observations: list[dict[str, Any]] = []
    subset = reward_df[reward_df["model_name"] == model_name]
    for row in subset.to_dict(orient="records"):
        qid = row["question_id"]
        if qid not in b_map.index:
            continue
        observations.append(
            {
                "obs_id": f"arena::{qid}::{model_name}",
                "kind": "arena",
                "source": row["source"],
                "question_id": qid,
                "b": float(b_map.loc[qid]),
                "a": float(a_map.loc[qid]),
                "target": float(row["reward"]),
            }
        )
    return observations


def irt_loss_matrix(
    observations: list[dict[str, Any]],
    theta_grid: np.ndarray,
) -> np.ndarray:
    losses = np.empty((len(observations), len(theta_grid)), dtype=np.float64)
    for idx, obs in enumerate(observations):
        pred = float(obs["a"]) * (theta_grid - float(obs["b"]))
        losses[idx] = (pred - float(obs["target"])) ** 2
    return losses


def irt_information(obs: dict[str, Any], _theta_prev: float) -> tuple[float, float]:
    a = float(obs["a"])
    # Arena reward regression contributes curvature a^2 to theta.
    return a, a * a
