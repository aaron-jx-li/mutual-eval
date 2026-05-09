from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rm_analysis.baseline_efficiency import build_bt_pairs, fit_bt
from rm_analysis.cold_start.math_utils import bce_with_logits_np, sigmoid_np


def build_arena_reference_pairs(reward_df: pd.DataFrame) -> pd.DataFrame:
    """Reuse the repo's soft BT pair construction with normalized rewards."""
    return build_bt_pairs(reward_df.rename(columns={"reward": "reward_z"}))


def build_arena_response_observations(reward_df: pd.DataFrame) -> pd.DataFrame:
    """Directional pair rows keyed by the revealed arena response row."""
    rows: list[dict[str, Any]] = []
    for question_id, group in reward_df.groupby("question_id", sort=False):
        group = group.sort_values("model_name").reset_index(drop=True)
        source = str(group.iloc[0]["source"])
        records = group.to_dict(orient="records")
        for current in records:
            obs_id = f"arena::{question_id}::{current['model_name']}"
            for other in records:
                if current["model_name"] == other["model_name"]:
                    continue
                target = float(sigmoid_np(float(current["reward"]) - float(other["reward"])))
                rows.append(
                    {
                        "obs_id": obs_id,
                        "source": source,
                        "question_id": question_id,
                        "model_name": str(current["model_name"]),
                        "opponent": str(other["model_name"]),
                        "target": target,
                    }
                )
    return pd.DataFrame(rows)


def bt_observations_for_model(
    model_name: str,
    bt_obs_df: pd.DataFrame,
    theta_map: dict[str, float],
) -> list[dict[str, Any]]:
    subset = bt_obs_df[bt_obs_df["model_name"] == model_name]
    observations: list[dict[str, Any]] = []
    for obs_id, group in subset.groupby("obs_id", sort=False):
        group = group[group["opponent"].isin(theta_map)]
        if group.empty:
            continue
        first = group.iloc[0]
        observations.append(
            {
                "obs_id": str(obs_id),
                "kind": "arena",
                "source": str(first["source"]),
                "question_id": str(first["question_id"]),
                "opp_theta": group["opponent"].map(theta_map).to_numpy(dtype=float),
                "target": group["target"].to_numpy(dtype=float),
            }
        )
    return observations


def bt_loss_matrix(
    observations: list[dict[str, Any]],
    theta_grid: np.ndarray,
) -> np.ndarray:
    losses = np.empty((len(observations), len(theta_grid)), dtype=np.float64)
    for idx, obs in enumerate(observations):
        logits = theta_grid[:, None] - obs["opp_theta"][None, :]
        losses[idx] = bce_with_logits_np(logits, obs["target"][None, :]).mean(axis=1)
    return losses
