#!/usr/bin/env python3
"""
Reward-only comparison for static-only, arena-only, and joint methods.

This script is designed for the math JSONL files in `data/hf/`:

- `static_math_v0.jsonl` contains model responses on static math questions, but
  its cached labels come from an LLM judge. This script ignores those judge
  fields and uses `RewardClient` to score the static responses instead.
- `arena_math_v0.jsonl` already contains reward-model scores.

Protocol
--------
1. Score static responses with `RewardClient` when a cached reward is missing.
2. Hold out whole questions from the static and arena sources separately.
3. Fit three reward-only models:
   - static-only: train on reward-scored static questions
   - arena-only: train on arena reward questions
   - joint: train on the union of both
4. Evaluate on held-out questions using pairwise accuracy:
   for each held-out question, compare every pair of model responses. The
   "gold" winner is whichever response has higher reward. The model prediction
   is whichever model has higher learned ability `theta`.
5. Report a mixed held-out score:

       mixed_accuracy(p) = p * static_accuracy + (1 - p) * arena_accuracy

   where `0 <= p < 1` is the static share of the held-out metric.

Important
---------
This script never reads `correct`, `grading_method`, `judge_reason`, or any
other LLM-judge outputs from the static file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranking import fit_reward_irt  # noqa: E402
from reward_client import RewardClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare static-only, arena-only, and joint reward-only methods on "
            "a held-out mixed math set."
        )
    )
    parser.add_argument(
        "--static-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "hf" / "static_math_v0.jsonl",
        help="Static math JSONL. Judge fields are ignored.",
    )
    parser.add_argument(
        "--arena-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "hf" / "arena_math_v0.jsonl",
        help="Arena math JSONL with reward scores.",
    )
    parser.add_argument(
        "--static-reward-cache",
        type=Path,
        default=REPO_ROOT / "results" / "reward_eval" / "static_math_v0_reward.jsonl",
        help="Cache path for RewardClient-scored static rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "reward_compare" / "math_joint_static_arena",
        help="Directory for summary outputs.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of questions to hold out per source.",
    )
    parser.add_argument(
        "--test-static-ratio",
        type=float,
        default=0.5,
        help=(
            "Held-out mixture weight p with mixed_accuracy = p*static + (1-p)*arena. "
            "Expected range: 0 <= p < 1."
        ),
    )
    parser.add_argument(
        "--pair-tie-eps",
        type=float,
        default=0.0,
        help="Ignore held-out reward pairs with abs(reward_a - reward_b) <= this value.",
    )
    parser.add_argument(
        "--theta-tie-eps",
        type=float,
        default=0.0,
        help="Ignore predicted pairs with abs(theta_a - theta_b) <= this value.",
    )
    parser.add_argument(
        "--normalize-rewards",
        choices=("none", "global", "per_source"),
        default="per_source",
        help="How to normalize train rewards before fitting.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model-label subset. Defaults to the static/arena intersection.",
    )
    parser.add_argument(
        "--max-static-questions",
        type=int,
        default=None,
        help="Optional cap on the number of static questions, for smoke tests.",
    )
    parser.add_argument(
        "--max-arena-questions",
        type=int,
        default=None,
        help="Optional cap on the number of arena questions, for smoke tests.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2000,
        help="Training epochs for reward IRT.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for reward IRT.",
    )
    parser.add_argument(
        "--reward-timeout",
        type=int,
        default=300,
        help="RewardClient timeout for static rescoring.",
    )
    parser.add_argument(
        "--reward-max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent RewardClient requests for static rescoring.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the static reward cache every N newly scored rows.",
    )
    parser.add_argument(
        "--rm-base-url",
        default=None,
        help="Reward model base URL. Falls back to ARENA_RM_BASE_URL or REWARD_MODEL_BASE_URL.",
    )
    parser.add_argument(
        "--rm-token",
        default=None,
        help="Reward model token. Falls back to ARENA_RM_TOKEN or REWARD_MODEL_TOKEN.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress reward-IRT training logs.",
    )
    return parser.parse_args()


def resolve_env_path() -> Path:
    return REPO_ROOT / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def static_question_id(row: dict[str, Any]) -> str:
    return f"{row['dataset']}::{row['sample_index']}"


def arena_question_id(row: dict[str, Any]) -> str:
    return str(row["item_id"])


def static_row_key(row: dict[str, Any]) -> str:
    return f"{static_question_id(row)}::{row['model_label']}"


def build_reward_conversation(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def normalize_reward(raw_reward: Any) -> tuple[float | None, str | None]:
    if isinstance(raw_reward, Exception):
        return None, str(raw_reward)
    if isinstance(raw_reward, (int, float)):
        value = float(raw_reward)
        if math.isfinite(value):
            return value, None
        return None, f"Non-finite reward: {value}"
    try:
        value = float(raw_reward)
        if math.isfinite(value):
            return value, None
    except (TypeError, ValueError):
        pass
    return None, f"Unexpected reward payload: {repr(raw_reward)[:500]}"


def score_static_row(
    row: dict[str, Any],
    reward_client: RewardClient,
    timeout: int,
) -> dict[str, Any]:
    started = time.time()
    reward_raw = reward_client.get_reward(
        build_reward_conversation(row["question"], row.get("response_text") or ""),
        timeout=timeout,
    )
    reward_value, reward_error = normalize_reward(reward_raw)
    status = "ok" if reward_error is None else "reward_error"
    return {
        "dataset": row["dataset"],
        "sample_index": row["sample_index"],
        "question_id": static_question_id(row),
        "model_label": row["model_label"],
        "model_provider": row.get("model_provider"),
        "model_id": row.get("model_id"),
        "question": row["question"],
        "prompt": row.get("prompt", row["question"]),
        "response_text": row.get("response_text"),
        "reward": reward_value,
        "status": status,
        "error": reward_error,
        "reward_latency_s": round(time.time() - started, 2),
        "processed_at": datetime.now().isoformat(),
    }


def ensure_static_rewards(
    static_rows: list[dict[str, Any]],
    *,
    cache_path: Path,
    reward_client: RewardClient | None,
    reward_timeout: int,
    reward_max_concurrency: int,
    save_every: int,
) -> list[dict[str, Any]]:
    cache_rows = load_jsonl(cache_path)
    cache_by_key = {
        f"{row['question_id']}::{row['model_label']}": row
        for row in cache_rows
        if row.get("question_id") and row.get("model_label")
    }

    for row in static_rows:
        if row.get("status") == "ok" and row.get("reward") is not None:
            cache_by_key.setdefault(
                static_row_key(row),
                {
                    "dataset": row["dataset"],
                    "sample_index": row["sample_index"],
                    "question_id": static_question_id(row),
                    "model_label": row["model_label"],
                    "model_provider": row.get("model_provider"),
                    "model_id": row.get("model_id"),
                    "question": row["question"],
                    "prompt": row.get("prompt", row["question"]),
                    "response_text": row.get("response_text"),
                    "reward": float(row["reward"]),
                    "status": "ok",
                    "error": row.get("error"),
                    "reward_latency_s": row.get("reward_latency_s"),
                    "processed_at": row.get("processed_at"),
                },
            )

    missing_rows: list[dict[str, Any]] = []
    for row in static_rows:
        key = static_row_key(row)
        cached = cache_by_key.get(key)
        if cached is None or cached.get("status") != "ok" or cached.get("reward") is None:
            missing_rows.append(row)

    if missing_rows:
        if reward_client is None:
            raise SystemExit(
                "Static rows are missing reward scores. Pass --rm-base-url/--rm-token "
                "or set ARENA_RM_BASE_URL/ARENA_RM_TOKEN so RewardClient can rescore them."
            )
        print(
            f"Scoring {len(missing_rows)} static rows with RewardClient "
            f"(cache: {cache_path}) ..."
        )
        newly_scored = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, reward_max_concurrency)
        ) as executor:
            future_to_row = {
                executor.submit(score_static_row, row, reward_client, reward_timeout): row
                for row in missing_rows
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_row),
                total=len(future_to_row),
                desc="static reward",
            ):
                scored = future.result()
                key = f"{scored['question_id']}::{scored['model_label']}"
                cache_by_key[key] = scored
                newly_scored += 1
                if save_every > 0 and newly_scored % save_every == 0:
                    write_jsonl(
                        cache_path,
                        sorted(
                            cache_by_key.values(),
                            key=lambda row: (row["question_id"], row["model_label"]),
                        ),
                    )
        write_jsonl(
            cache_path,
            sorted(
                cache_by_key.values(),
                key=lambda row: (row["question_id"], row["model_label"]),
            ),
        )

    resolved_rows: list[dict[str, Any]] = []
    for row in static_rows:
        resolved = cache_by_key.get(static_row_key(row))
        if resolved is not None:
            resolved_rows.append(resolved)
    return resolved_rows


def rows_to_reward_df(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok" or row.get("reward") is None:
            continue
        if source == "static":
            question_id = row.get("question_id") or static_question_id(row)
        else:
            question_id = arena_question_id(row)
        records.append(
            {
                "source": source,
                "question_id": question_id,
                "model_name": row["model_label"],
                "reward": float(row["reward"]),
            }
        )
    return pd.DataFrame(records)


def restrict_raw_rows(
    rows: list[dict[str, Any]],
    *,
    question_id_fn,
    max_questions: int | None,
) -> list[dict[str, Any]]:
    if max_questions is None:
        return rows
    selected_question_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        question_id = str(question_id_fn(row))
        if question_id in seen:
            continue
        seen.add(question_id)
        selected_question_ids.append(question_id)
        if len(selected_question_ids) >= max_questions:
            break
    allowed = set(selected_question_ids)
    return [row for row in rows if str(question_id_fn(row)) in allowed]


def restrict_questions(
    df: pd.DataFrame,
    *,
    max_questions: int | None,
) -> pd.DataFrame:
    if max_questions is None:
        return df
    question_ids = sorted(df["question_id"].unique())[:max_questions]
    return df[df["question_id"].isin(question_ids)].copy()


def split_questions(
    df: pd.DataFrame,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    question_ids = sorted(df["question_id"].unique())
    if len(question_ids) < 2 or test_fraction <= 0:
        return df.copy(), df.iloc[0:0].copy()

    rng = random.Random(seed)
    rng.shuffle(question_ids)

    n_test = int(round(len(question_ids) * test_fraction))
    n_test = max(1, min(n_test, len(question_ids) - 1))
    test_ids = set(question_ids[:n_test])

    train_df = df[~df["question_id"].isin(test_ids)].copy()
    test_df = df[df["question_id"].isin(test_ids)].copy()
    return train_df, test_df


def apply_reward_normalization(
    static_train_df: pd.DataFrame,
    arena_train_df: pd.DataFrame,
    *,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    static_norm = static_train_df.copy()
    arena_norm = arena_train_df.copy()
    stats: dict[str, dict[str, float]] = {}

    def _normalize(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if df.empty:
            stats[key] = {"mean": 0.0, "std": 1.0}
            return df.copy()
        mean = float(df["reward"].mean())
        std = float(df["reward"].std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            std = 1.0
        stats[key] = {"mean": mean, "std": std}
        out = df.copy()
        out["reward"] = (out["reward"] - mean) / std
        return out

    if mode == "none":
        if not static_norm.empty:
            stats["static"] = {
                "mean": float(static_norm["reward"].mean()),
                "std": float(static_norm["reward"].std(ddof=0) or 1.0),
            }
        if not arena_norm.empty:
            stats["arena"] = {
                "mean": float(arena_norm["reward"].mean()),
                "std": float(arena_norm["reward"].std(ddof=0) or 1.0),
            }
        return static_norm, arena_norm, stats

    if mode == "global":
        combined = pd.concat([static_norm, arena_norm], ignore_index=True)
        if combined.empty:
            return static_norm, arena_norm, {"global": {"mean": 0.0, "std": 1.0}}
        mean = float(combined["reward"].mean())
        std = float(combined["reward"].std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            std = 1.0
        stats["global"] = {"mean": mean, "std": std}
        for df in (static_norm, arena_norm):
            if not df.empty:
                df["reward"] = (df["reward"] - mean) / std
        return static_norm, arena_norm, stats

    static_norm = _normalize(static_norm, "static")
    arena_norm = _normalize(arena_norm, "arena")
    return static_norm, arena_norm, stats


def fit_theta_map(
    train_df: pd.DataFrame,
    *,
    num_epochs: int,
    lr: float,
    quiet: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    if train_df.empty:
        raise ValueError("Cannot fit a reward-only method on an empty train set.")
    model_params, _question_params = fit_reward_irt(
        train_df[["model_name", "question_id", "reward"]],
        num_epochs=num_epochs,
        lr=lr,
        verbose=not quiet,
    )
    theta_map = {
        row["model_name"]: float(row["theta"])
        for row in model_params.to_dict(orient="records")
    }
    return theta_map, model_params


def evaluate_pairwise_accuracy(
    test_df: pd.DataFrame,
    theta_map: dict[str, float],
    *,
    pair_tie_eps: float,
    theta_tie_eps: float,
) -> dict[str, Any]:
    correct = 0
    total = 0
    skipped_reward_ties = 0
    skipped_theta_ties = 0
    skipped_missing_theta = 0
    question_counter = 0

    for question_id, group in test_df.groupby("question_id"):
        question_counter += 1
        rows = group[["model_name", "reward"]].dropna().to_dict(orient="records")
        for left, right in combinations(rows, 2):
            reward_delta = float(left["reward"]) - float(right["reward"])
            if abs(reward_delta) <= pair_tie_eps:
                skipped_reward_ties += 1
                continue
            theta_left = theta_map.get(str(left["model_name"]))
            theta_right = theta_map.get(str(right["model_name"]))
            if theta_left is None or theta_right is None:
                skipped_missing_theta += 1
                continue
            theta_delta = theta_left - theta_right
            if abs(theta_delta) <= theta_tie_eps:
                skipped_theta_ties += 1
                continue

            pred_left_wins = theta_delta > 0
            gold_left_wins = reward_delta > 0
            correct += int(pred_left_wins == gold_left_wins)
            total += 1

    return {
        "accuracy": (correct / total) if total else None,
        "correct_pairs": correct,
        "total_pairs": total,
        "questions": question_counter,
        "skipped_reward_ties": skipped_reward_ties,
        "skipped_theta_ties": skipped_theta_ties,
        "skipped_missing_theta": skipped_missing_theta,
    }


def summarize_df(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "questions": 0,
            "models": 0,
        }
    return {
        "rows": int(len(df)),
        "questions": int(df["question_id"].nunique()),
        "models": int(df["model_name"].nunique()),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.test_static_ratio < 1.0):
        raise SystemExit("--test-static-ratio must satisfy 0 <= p < 1.")
    if not (0.0 <= args.test_fraction < 1.0):
        raise SystemExit("--test-fraction must satisfy 0 <= value < 1.")
    if args.reward_max_concurrency < 1:
        raise SystemExit("--reward-max-concurrency must be at least 1.")

    load_env_file(resolve_env_path())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rm_base_url = args.rm_base_url or get_env_value(
        "ARENA_RM_BASE_URL",
        "REWARD_MODEL_BASE_URL",
        "LMARENA_BASE_URL",
    )
    rm_token = args.rm_token or get_env_value(
        "ARENA_RM_TOKEN",
        "REWARD_MODEL_TOKEN",
        "LMARENA_TOKEN",
    )

    static_raw_rows = load_jsonl(args.static_jsonl)
    arena_raw_rows = load_jsonl(args.arena_jsonl)
    if not static_raw_rows:
        raise SystemExit(f"No rows found in {args.static_jsonl}.")
    if not arena_raw_rows:
        raise SystemExit(f"No rows found in {args.arena_jsonl}.")

    static_models = {row["model_label"] for row in static_raw_rows if row.get("model_label")}
    arena_models = {row["model_label"] for row in arena_raw_rows if row.get("model_label")}
    common_models = sorted(static_models & arena_models)
    if args.models:
        requested = set(args.models)
        common_models = [model for model in common_models if model in requested]
    if not common_models:
        raise SystemExit("No overlapping model labels remain after filtering.")

    static_raw_rows = [
        row for row in static_raw_rows
        if row.get("model_label") in common_models and row.get("response_text")
    ]
    arena_raw_rows = [
        row for row in arena_raw_rows
        if row.get("model_label") in common_models
        and row.get("status") == "ok"
        and row.get("reward") is not None
    ]

    static_raw_rows = restrict_raw_rows(
        static_raw_rows,
        question_id_fn=static_question_id,
        max_questions=args.max_static_questions,
    )
    arena_raw_rows = restrict_raw_rows(
        arena_raw_rows,
        question_id_fn=arena_question_id,
        max_questions=args.max_arena_questions,
    )

    reward_client = None
    if rm_base_url and rm_token:
        reward_client = RewardClient(base_url=rm_base_url, token=rm_token)

    static_reward_rows = ensure_static_rewards(
        static_raw_rows,
        cache_path=args.static_reward_cache,
        reward_client=reward_client,
        reward_timeout=args.reward_timeout,
        reward_max_concurrency=args.reward_max_concurrency,
        save_every=args.save_every,
    )

    static_df = rows_to_reward_df(static_reward_rows, source="static")
    arena_df = rows_to_reward_df(arena_raw_rows, source="arena")

    if static_df.empty:
        raise SystemExit("Static reward dataframe is empty after filtering.")
    if arena_df.empty:
        raise SystemExit("Arena reward dataframe is empty after filtering.")

    static_train_df, static_test_df = split_questions(
        static_df,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    arena_train_df, arena_test_df = split_questions(
        arena_df,
        test_fraction=args.test_fraction,
        seed=args.seed + 1,
    )

    static_train_norm, arena_train_norm, norm_stats = apply_reward_normalization(
        static_train_df,
        arena_train_df,
        mode=args.normalize_rewards,
    )

    methods: dict[str, dict[str, Any]] = {}

    static_only_theta, static_only_ranking = fit_theta_map(
        static_train_norm,
        num_epochs=args.num_epochs,
        lr=args.lr,
        quiet=args.quiet,
    )
    methods["static_only"] = {
        "train_rows": int(len(static_train_norm)),
        "train_questions": int(static_train_norm["question_id"].nunique()),
        "model_ranking": static_only_ranking.to_dict(orient="records"),
        "static_test": evaluate_pairwise_accuracy(
            static_test_df,
            static_only_theta,
            pair_tie_eps=args.pair_tie_eps,
            theta_tie_eps=args.theta_tie_eps,
        ),
        "arena_test": evaluate_pairwise_accuracy(
            arena_test_df,
            static_only_theta,
            pair_tie_eps=args.pair_tie_eps,
            theta_tie_eps=args.theta_tie_eps,
        ),
    }

    arena_only_theta, arena_only_ranking = fit_theta_map(
        arena_train_norm,
        num_epochs=args.num_epochs,
        lr=args.lr,
        quiet=args.quiet,
    )
    methods["arena_only"] = {
        "train_rows": int(len(arena_train_norm)),
        "train_questions": int(arena_train_norm["question_id"].nunique()),
        "model_ranking": arena_only_ranking.to_dict(orient="records"),
        "static_test": evaluate_pairwise_accuracy(
            static_test_df,
            arena_only_theta,
            pair_tie_eps=args.pair_tie_eps,
            theta_tie_eps=args.theta_tie_eps,
        ),
        "arena_test": evaluate_pairwise_accuracy(
            arena_test_df,
            arena_only_theta,
            pair_tie_eps=args.pair_tie_eps,
            theta_tie_eps=args.theta_tie_eps,
        ),
    }

    joint_train_norm = pd.concat([static_train_norm, arena_train_norm], ignore_index=True)
    joint_theta, joint_ranking = fit_theta_map(
        joint_train_norm,
        num_epochs=args.num_epochs,
        lr=args.lr,
        quiet=args.quiet,
    )
    methods["joint"] = {
        "train_rows": int(len(joint_train_norm)),
        "train_questions": int(joint_train_norm["question_id"].nunique()),
        "model_ranking": joint_ranking.to_dict(orient="records"),
        "static_test": evaluate_pairwise_accuracy(
            static_test_df,
            joint_theta,
            pair_tie_eps=args.pair_tie_eps,
            theta_tie_eps=args.theta_tie_eps,
        ),
        "arena_test": evaluate_pairwise_accuracy(
            arena_test_df,
            joint_theta,
            pair_tie_eps=args.pair_tie_eps,
            theta_tie_eps=args.theta_tie_eps,
        ),
    }

    for method_name, payload in methods.items():
        static_acc = payload["static_test"]["accuracy"]
        arena_acc = payload["arena_test"]["accuracy"]
        if static_acc is None or arena_acc is None:
            mixed = None
        else:
            mixed = (
                args.test_static_ratio * static_acc
                + (1.0 - args.test_static_ratio) * arena_acc
            )
        payload["mixed_test_accuracy"] = mixed

    summary = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "static_jsonl": str(args.static_jsonl),
            "arena_jsonl": str(args.arena_jsonl),
            "static_reward_cache": str(args.static_reward_cache),
            "test_fraction": args.test_fraction,
            "test_static_ratio": args.test_static_ratio,
            "pair_tie_eps": args.pair_tie_eps,
            "theta_tie_eps": args.theta_tie_eps,
            "normalize_rewards": args.normalize_rewards,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "seed": args.seed,
            "models": common_models,
        },
        "normalization": norm_stats,
        "data": {
            "static_all": summarize_df(static_df),
            "arena_all": summarize_df(arena_df),
            "static_train": summarize_df(static_train_df),
            "static_test": summarize_df(static_test_df),
            "arena_train": summarize_df(arena_train_df),
            "arena_test": summarize_df(arena_test_df),
            "static_question_model_counts": dict(
                Counter(static_df.groupby("question_id")["model_name"].nunique())
            ),
            "arena_question_model_counts": dict(
                Counter(arena_df.groupby("question_id")["model_name"].nunique())
            ),
        },
        "methods": methods,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for method_name, payload in methods.items():
        ranking_path = args.output_dir / f"{method_name}_model_ranking.csv"
        pd.DataFrame(payload["model_ranking"]).to_csv(ranking_path, index=False)

    print("\nHeld-out pairwise accuracy")
    print("-" * 90)
    print(
        f"{'method':14} {'static_test':>12} {'arena_test':>12} {'mixed(p)':>12} "
        f"{'train_q':>8} {'train_rows':>10}"
    )
    for method_name, payload in methods.items():
        static_acc = payload["static_test"]["accuracy"]
        arena_acc = payload["arena_test"]["accuracy"]
        mixed_acc = payload["mixed_test_accuracy"]

        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.4f}"

        print(
            f"{method_name:14} {fmt(static_acc):>12} {fmt(arena_acc):>12} {fmt(mixed_acc):>12} "
            f"{payload['train_questions']:8d} {payload['train_rows']:10d}"
        )
    print("-" * 90)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
