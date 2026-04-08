"""Data loading and subsampling helpers for robustness experiments."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ranking import (
    load_arena_pairs,
    load_static_jsonl as _rank_load_static_jsonl,
    load_arena_reward_jsonl as _rank_load_arena_reward_jsonl,
)


def load_static(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_static_jsonl(path: str) -> pd.DataFrame:
    """Load static_eval JSONL → columns: model_name, question_id, judge_result."""
    return _rank_load_static_jsonl(path)


def load_arena(path: str) -> pd.DataFrame:
    return load_arena_pairs(path)


def load_arena_reward(path: str) -> pd.DataFrame:
    """Load arena_eval reward JSONL → columns: model_name, question_id, reward."""
    return _rank_load_arena_reward_jsonl(path)


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
