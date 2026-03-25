"""Data loading and subsampling helpers for robustness experiments."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ranking import load_arena_pairs


def load_static(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_arena(path: str) -> pd.DataFrame:
    return load_arena_pairs(path)


def subsample_rows(static_df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Random fraction of (model, question) rows."""
    return static_df.sample(frac=frac, random_state=seed).reset_index(drop=True)


def subsample_questions(static_df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Random fraction of distinct question_ids; keeps all models for retained questions."""
    rng = pd.Series(static_df["question_id"].unique()).sample(frac=frac, random_state=seed)
    kept = set(rng)
    return static_df[static_df["question_id"].isin(kept)].reset_index(drop=True)


def subsample_pairs(arena_df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Random fraction of pairwise comparison rows."""
    return arena_df.sample(frac=frac, random_state=seed).reset_index(drop=True)


def drop_models(df: pd.DataFrame, models_to_drop: list[str]) -> pd.DataFrame:
    """Remove all rows for specified model(s). Works for both static and arena DataFrames."""
    if "model_name" in df.columns:
        return df[~df["model_name"].isin(models_to_drop)].reset_index(drop=True)
    # Arena format: model_1 / model_2 columns
    mask = df["model_1"].isin(models_to_drop) | df["model_2"].isin(models_to_drop)
    return df[~mask].reset_index(drop=True)
