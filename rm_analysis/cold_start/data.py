from __future__ import annotations

from pathlib import Path

import pandas as pd

from rm_analysis.compare_arena_bt_vs_ours import (
    apply_reward_normalization,
    load_arena_reward_jsonl,
    restrict_questions_per_source,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARENA_JSONLS = [
    REPO_ROOT / "data" / "hf" / "v1_arena_math.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_arena_coding.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_arena_generic.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_arena_misc.jsonl",
]


def load_reward_responses(
    paths: list[Path],
    *,
    normalize: str = "per_source",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    df = load_arena_reward_jsonl(paths)
    if df.empty:
        raise SystemExit("No usable reward rows found.")
    return apply_reward_normalization(df, mode=normalize)


def filter_models(
    reward_df: pd.DataFrame,
    *,
    models: list[str] | None,
    max_models: int | None,
) -> pd.DataFrame:
    if models:
        keep = set(models)
    elif max_models is not None:
        keep = set(sorted(reward_df["model_name"].unique())[:max_models])
    else:
        return reward_df

    out = reward_df[reward_df["model_name"].isin(keep)].copy()
    if out.empty:
        raise SystemExit("No rows remain after applying model filters.")
    return out
