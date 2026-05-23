#!/usr/bin/env python3
"""Leak-free frozen-parameter online cold-start replay.

This implements the mentor-feedback protocol for placing a completely held-out
new model:

1. fit calibration parameters on all other models only;
2. freeze item parameters and known-model abilities;
3. reveal the held-out model's item responses one at a time;
4. update only the held-out model ability ``theta_new``; and
5. choose the next item from current ``theta_new`` using expected Fisher
   information, without using the unrevealed held-out outcome.

The headline comparison is joint static+arena MutualEval versus a shared-theta
2PL+BT baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DUALEVAL_PATH = REPO_ROOT / "ranking" / "dualeval.py"
dualeval_spec = importlib.util.spec_from_file_location("dualeval_module", DUALEVAL_PATH)
if dualeval_spec is None or dualeval_spec.loader is None:
    raise ImportError(f"Could not load DualEval module from {DUALEVAL_PATH}")
dualeval = importlib.util.module_from_spec(dualeval_spec)
dualeval_spec.loader.exec_module(dualeval)

DEFAULT_STATIC_JSONLS = [
    REPO_ROOT / "data" / "hf" / "v1_static_math.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_static_coding.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_static_misc.jsonl",
]
DEFAULT_ARENA_JSONLS = [
    REPO_ROOT / "data" / "hf" / "v1_arena_math.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_arena_coding.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_arena_generic.jsonl",
    REPO_ROOT / "data" / "hf" / "v1_arena_misc.jsonl",
]

JOINT_METHODS = ("dualeval_joint", "2pl_bt_joint")
STRATEGIES = ("fisher", "sharpness", "random")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    uses_static: bool
    uses_arena: bool
    is_dualeval_irt: bool
    is_direct_bt: bool


METHOD_SPECS = {
    "dualeval_joint": MethodSpec("dualeval_joint", True, True, True, False),
    "2pl_bt_joint": MethodSpec("2pl_bt_joint", True, True, False, True),
}


@dataclass
class CalibrationFit:
    method: str
    model_params: pd.DataFrame
    question_params: pd.DataFrame
    metadata: dict[str, Any]
    reward_mean: float
    reward_std: float
    both_bad_threshold: float
    tie_delta: float
    static_rows: int
    arena_rows: int
    arena_pairs: int

    @property
    def theta_map(self) -> dict[str, float]:
        return {
            str(row["model_name"]): float(row["theta"])
            for row in self.model_params.to_dict(orient="records")
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-jsonl", nargs="+", type=Path, default=DEFAULT_STATIC_JSONLS)
    parser.add_argument("--arena-jsonl", nargs="+", type=Path, default=DEFAULT_ARENA_JSONLS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "cold_start" / "v2_online_frozen_joint",
    )
    parser.add_argument("--methods", nargs="+", choices=JOINT_METHODS, default=list(JOINT_METHODS))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--reference-method", choices=JOINT_METHODS, default="dualeval_joint")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-static-questions-per-source", type=int, default=None)
    parser.add_argument("--max-arena-questions-per-source", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum revealed items per held-out model. Use 0 to reveal all available items.",
    )
    parser.add_argument("--record-every", type=int, default=25)
    parser.add_argument(
        "--record-fractions",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional checkpoints as fractions of each model's reveal budget, "
            "for example 0.2 0.4 0.6 0.8 1.0. Overrides fixed --record-every checkpoints."
        ),
    )
    parser.add_argument("--reference-epochs", type=int, default=400)
    parser.add_argument("--calibration-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lambda-static", type=float, default=1.0)
    parser.add_argument("--lambda-arena", type=float, default=1.0)
    parser.add_argument("--lambda-bb", type=float, default=0.2)
    parser.add_argument("--reg-lambda", type=float, default=1e-2)
    parser.add_argument("--bb-ratio", type=float, default=0.15)
    parser.add_argument("--tie-ratio", type=float, default=0.15)
    parser.add_argument("--theta-pad", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=801)
    parser.add_argument("--replay-reg-lambda", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative; use 0 to reveal all available items.")
    if args.record_every < 1:
        raise SystemExit("--record-every must be positive.")
    if args.record_fractions is not None:
        if not args.record_fractions:
            raise SystemExit("--record-fractions must include at least one value when provided.")
        bad = [frac for frac in args.record_fractions if not 0.0 < frac <= 1.0]
        if bad:
            raise SystemExit(f"--record-fractions values must be in (0, 1], got {bad}.")
    if args.grid_size < 11:
        raise SystemExit("--grid-size must be at least 11.")
    if not 0.0 <= args.bb_ratio <= 1.0:
        raise SystemExit("--bb-ratio must be in [0, 1].")
    if not 0.0 <= args.tie_ratio <= 1.0:
        raise SystemExit("--tie-ratio must be in [0, 1].")
    if args.reference_method not in args.methods:
        raise SystemExit("--reference-method must be included in --methods.")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def source_tag_for_path(path: Path) -> str:
    return path.stem


def load_static_jsonls(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = resolve_path(raw_path)
        tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("model_label") is None:
                    continue
                status = str(record.get("status", "")).strip().lower()
                is_errored = status != "ok"
                if not is_errored and record.get("correct") is None:
                    continue
                dataset = str(record.get("dataset", "unknown"))
                sample_index = record.get("sample_index")
                rows.append(
                    {
                        "source": tag,
                        "benchmark": dataset,
                        "model_name": str(record["model_label"]),
                        "question_id": f"{tag}::{dataset}_{sample_index}",
                        "judge_result": 0 if is_errored else int(bool(record["correct"])),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["source", "benchmark", "model_name", "question_id", "judge_result"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["model_name", "question_id"], keep="last")
        .reset_index(drop=True)
    )


def load_arena_jsonls(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = resolve_path(raw_path)
        tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") != "ok" or record.get("reward") is None:
                    continue
                rows.append(
                    {
                        "source": tag,
                        "benchmark": "Arena",
                        "model_name": str(record["model_label"]),
                        "question_id": f"{tag}::{record['item_id']}",
                        "reward_raw": float(record["reward"]),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["source", "benchmark", "model_name", "question_id", "reward_raw"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["model_name", "question_id"], keep="last")
        .reset_index(drop=True)
    )


def restrict_questions_per_source(
    df: pd.DataFrame,
    *,
    max_questions_per_source: int | None,
) -> pd.DataFrame:
    if df.empty or max_questions_per_source is None:
        return df
    keep: set[str] = set()
    for _, group in df[["source", "question_id"]].drop_duplicates().groupby("source", sort=True):
        keep.update(group["question_id"].astype(str).sort_values().head(max_questions_per_source))
    return df[df["question_id"].astype(str).isin(keep)].copy().reset_index(drop=True)


def filter_models(
    static_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    *,
    models: list[str] | None,
    max_models: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_models = sorted(set(static_df["model_name"].astype(str)) | set(reward_df["model_name"].astype(str)))
    if models:
        keep = set(models)
    elif max_models is not None:
        keep = set(all_models[:max_models])
    else:
        return static_df, reward_df
    return (
        static_df[static_df["model_name"].astype(str).isin(keep)].copy().reset_index(drop=True),
        reward_df[reward_df["model_name"].astype(str).isin(keep)].copy().reset_index(drop=True),
    )


def standardize_reward(
    reward_df: pd.DataFrame,
    *,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    out = reward_df.copy()
    if out.empty:
        return out.assign(reward_z=pd.Series(dtype=float)), 0.0, 1.0
    mu = float(out["reward_raw"].mean()) if mean is None else float(mean)
    sigma = float(out["reward_raw"].std(ddof=0)) if std is None else float(std)
    if not math.isfinite(sigma) or sigma < 1e-8:
        sigma = 1.0
    out["reward_z"] = (out["reward_raw"].astype(float) - mu) / sigma
    return out, mu, sigma


def build_pairwise(
    reward_z_df: pd.DataFrame,
    *,
    bb_ratio: float,
    tie_ratio: float,
    both_bad_threshold: float | None = None,
    tie_delta: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    if reward_z_df.empty:
        return pd.DataFrame(), -math.inf, 0.0
    if both_bad_threshold is None or tie_delta is None:
        both_bad_threshold, tie_delta = dualeval.resolve_pairwise_thresholds(
            reward_z_df,
            bb_ratio=bb_ratio,
            tie_ratio=tie_ratio,
        )
    pairwise = dualeval.build_soft_pairwise_targets(
        reward_z_df,
        both_bad_threshold=both_bad_threshold,
        tie_delta=tie_delta,
    )
    return pairwise, float(both_bad_threshold), float(tie_delta)


def rank_map(model_params: pd.DataFrame) -> dict[str, int]:
    ordered = model_params.sort_values("theta", ascending=False)["model_name"].astype(str).tolist()
    return {model: idx for idx, model in enumerate(ordered, start=1)}


def online_rank_map(theta_map: dict[str, float]) -> dict[str, int]:
    ordered = sorted(theta_map, key=lambda model: (-theta_map[model], model))
    return {model: idx for idx, model in enumerate(ordered, start=1)}


def ranking_metrics(theta_map: dict[str, float], reference_rank: dict[str, int]) -> dict[str, float]:
    common = sorted(set(theta_map) & set(reference_rank))
    if len(common) < 2:
        return {"spearman": float("nan"), "kendall": float("nan"), "exact_matches": float("nan")}
    online = online_rank_map({model: theta_map[model] for model in common})
    ref_restricted = {
        model: idx
        for idx, model in enumerate(sorted(common, key=lambda item: (reference_rank[item], item)), start=1)
    }
    ref_values = [ref_restricted[model] for model in common]
    online_values = [online[model] for model in common]
    rho, _ = spearmanr(ref_values, online_values)
    tau, _ = kendalltau(ref_values, online_values)
    exact = float(np.mean([online[model] == ref_restricted[model] for model in common]))
    return {"spearman": float(rho), "kendall": float(tau), "exact_matches": exact}


def sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def bce_with_logits_np(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))


def fit_2pl_bt(
    static_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    *,
    num_epochs: int,
    lr: float,
    lambda_static: float,
    lambda_arena: float,
    reg_lambda: float,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    static = static_df.copy()
    non_tie = pairwise_df[~pairwise_df["tie"].astype(bool)].copy()
    if static.empty:
        raise SystemExit("2PL+BT baseline needs static rows.")
    if non_tie.empty:
        raise SystemExit("2PL+BT baseline needs non-tie arena pairs.")

    all_models = pd.Index(
        pd.unique(pd.concat([static["model_name"], non_tie["model_1"], non_tie["model_2"]], ignore_index=True)),
        name="model_name",
    )
    all_questions = pd.Index(pd.unique(static["question_id"]), name="question_id")
    model_to_idx = {model: idx for idx, model in enumerate(all_models)}
    q_to_idx = {question: idx for idx, question in enumerate(all_questions)}

    static["m_idx"] = static["model_name"].map(model_to_idx)
    static["q_idx"] = static["question_id"].map(q_to_idx)
    non_tie["m1_idx"] = non_tie["model_1"].map(model_to_idx)
    non_tie["m2_idx"] = non_tie["model_2"].map(model_to_idx)

    device = dualeval._get_device()
    m_s = torch.tensor(static["m_idx"].values, dtype=torch.long, device=device)
    q_s = torch.tensor(static["q_idx"].values, dtype=torch.long, device=device)
    y_s = torch.tensor(static["judge_result"].values, dtype=torch.float32, device=device)
    m1_t = torch.tensor(non_tie["m1_idx"].values, dtype=torch.long, device=device)
    m2_t = torch.tensor(non_tie["m2_idx"].values, dtype=torch.long, device=device)
    soft_t = torch.tensor(non_tie["target_prob"].values, dtype=torch.float32, device=device)

    theta = nn.Embedding(len(all_models), 1, device=device)
    b = nn.Embedding(len(all_questions), 1, device=device)
    k = nn.Embedding(len(all_questions), 1, device=device)
    nn.init.zeros_(theta.weight)
    nn.init.zeros_(b.weight)
    nn.init.zeros_(k.weight)

    optimizer = optim.Adam([*theta.parameters(), *b.parameters(), *k.parameters()], lr=lr)
    bce_logits = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits_static = torch.exp(k(q_s).squeeze(-1)) * (theta(m_s).squeeze(-1) - b(q_s).squeeze(-1))
        loss_static = bce_logits(logits_static, y_s)
        logits_bt = theta(m1_t).squeeze(-1) - theta(m2_t).squeeze(-1)
        loss_arena = bce_logits(logits_bt, soft_t)
        reg = reg_lambda * (theta.weight.pow(2).mean() + b.weight.pow(2).mean() + k.weight.pow(2).mean())
        loss = lambda_static * loss_static + lambda_arena * loss_arena + reg
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            b.weight.sub_(shift)

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(
                f"  Epoch {epoch:5d} | static={loss_static.item():.4f} "
                f"bt={loss_arena.item():.4f} total={loss.item():.4f}",
                flush=True,
            )

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    b_np = b.weight.detach().cpu().numpy().squeeze(-1)
    k_np = k.weight.detach().cpu().numpy().squeeze(-1)
    model_params = (
        pd.DataFrame({"model_name": all_models, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )
    q_meta = (
        static[["question_id", "source", "benchmark"]]
        .drop_duplicates("question_id")
        .set_index("question_id")
        .to_dict(orient="index")
    )
    question_params = pd.DataFrame(
        [
            {
                "question_id": question_id,
                "source": q_meta.get(str(question_id), {}).get("source", "unknown"),
                "benchmark": q_meta.get(str(question_id), {}).get("benchmark", "unknown"),
                "difficulty_b": float(b_np[idx]),
                "k_raw": float(k_np[idx]),
                "discrimination_exp_k": math.exp(float(k_np[idx])),
            }
            for idx, question_id in enumerate(all_questions)
        ]
    )
    question_params = question_params.sort_values("difficulty_b", ascending=False).reset_index(drop=True)
    return model_params, question_params, {
        "learned_gamma": None,
        "n_models": int(len(all_models)),
        "n_questions": int(len(all_questions)),
        "has_static": True,
        "has_arena": True,
    }


def fit_method(
    method: str,
    static_train: pd.DataFrame,
    reward_train_raw: pd.DataFrame,
    *,
    args: argparse.Namespace,
    num_epochs: int,
) -> CalibrationFit:
    spec = METHOD_SPECS[method]
    static_input = static_train.copy() if spec.uses_static else pd.DataFrame()
    reward_input_raw = reward_train_raw.copy() if spec.uses_arena else pd.DataFrame()
    reward_z, reward_mean, reward_std = standardize_reward(reward_input_raw)
    pairwise, both_bad_threshold, tie_delta = build_pairwise(
        reward_z,
        bb_ratio=args.bb_ratio,
        tie_ratio=args.tie_ratio,
    )

    verbose = not args.quiet
    if method == "dualeval_joint":
        model_params, question_params, metadata = dualeval.fit_irt(
            static_input if spec.uses_static else None,
            pairwise if spec.uses_arena else None,
            num_epochs=num_epochs,
            lr=args.lr,
            lambda_static=args.lambda_static if spec.uses_static else 0.0,
            lambda_arena=args.lambda_arena if spec.uses_arena else 0.0,
            lambda_bb=args.lambda_bb if spec.uses_arena else 0.0,
            reg_lambda=args.reg_lambda,
            verbose=verbose,
        )
    elif method == "2pl_bt_joint":
        model_params, question_params, metadata = fit_2pl_bt(
            static_input,
            pairwise,
            num_epochs=num_epochs,
            lr=args.lr,
            lambda_static=args.lambda_static,
            lambda_arena=args.lambda_arena,
            reg_lambda=args.reg_lambda,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return CalibrationFit(
        method=method,
        model_params=model_params,
        question_params=question_params,
        metadata=metadata,
        reward_mean=reward_mean,
        reward_std=reward_std,
        both_bad_threshold=both_bad_threshold,
        tie_delta=tie_delta,
        static_rows=int(len(static_input)),
        arena_rows=int(len(reward_input_raw)),
        arena_pairs=int(len(pairwise)),
    )


def question_param_maps(question_params: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    if question_params.empty:
        return {}, {}
    qp = question_params.set_index("question_id")
    return (
        {str(k): float(v) for k, v in qp["difficulty_b"].to_dict().items()},
        {str(k): float(v) for k, v in qp["discrimination_exp_k"].to_dict().items()},
    )


def build_observations(
    model_name: str,
    static_df: pd.DataFrame,
    reward_raw_df: pd.DataFrame,
    calibration: CalibrationFit,
) -> list[dict[str, Any]]:
    spec = METHOD_SPECS[calibration.method]
    theta_map = calibration.theta_map
    b_map, a_map = question_param_maps(calibration.question_params)
    observations: list[dict[str, Any]] = []

    if spec.uses_static:
        subset = static_df[static_df["model_name"].astype(str) == model_name].copy()
        subset = subset[subset["question_id"].astype(str).isin(b_map)].copy()
        for row in subset.sort_values(["source", "question_id"]).to_dict(orient="records"):
            qid = str(row["question_id"])
            observations.append(
                {
                    "obs_id": f"static::{qid}::{model_name}",
                    "kind": "static",
                    "source": str(row["source"]),
                    "benchmark": str(row["benchmark"]),
                    "question_id": qid,
                    "target": float(row["judge_result"]),
                    "a": a_map[qid],
                    "b": b_map[qid],
                }
            )

    if spec.uses_arena:
        reward_z_all, _, _ = standardize_reward(
            reward_raw_df,
            mean=calibration.reward_mean,
            std=calibration.reward_std,
        )
        heldout = reward_z_all[reward_z_all["model_name"].astype(str) == model_name].copy()
        known = reward_z_all[reward_z_all["model_name"].astype(str).isin(theta_map)].copy()
        known_by_question = {str(qid): group for qid, group in known.groupby("question_id", sort=False)}
        for row in heldout.sort_values(["source", "question_id"]).to_dict(orient="records"):
            qid = str(row["question_id"])
            if spec.is_dualeval_irt and qid not in b_map:
                continue
            group = known_by_question.get(qid)
            if group is None or group.empty:
                continue
            group = group[group["model_name"].astype(str).isin(theta_map)].sort_values("model_name")
            if group.empty:
                continue
            z_new = float(row["reward_z"])
            z_opp = group["reward_z"].astype(float).to_numpy()
            opp_theta = group["model_name"].map(theta_map).astype(float).to_numpy()
            targets = np.array([1.0 / (1.0 + math.exp(-(z_new - float(z)))) for z in z_opp], dtype=float)
            tie = np.abs(z_new - z_opp) < calibration.tie_delta
            both_bad = np.maximum(z_new, z_opp) < calibration.both_bad_threshold
            obs: dict[str, Any] = {
                "obs_id": f"arena::{qid}::{model_name}",
                "kind": "arena",
                "source": str(row["source"]),
                "benchmark": "Arena",
                "question_id": qid,
                "target": targets,
                "opp_theta": opp_theta,
                "tie": tie.astype(bool),
                "both_bad": both_bad.astype(bool),
            }
            if spec.is_dualeval_irt:
                a = a_map[qid]
                b = b_map[qid]
                p_opp = sigmoid_np(a * (opp_theta - b))
                obs.update({"a": a, "b": b, "p_opp": np.asarray(p_opp, dtype=float)})
            observations.append(obs)

    return sorted(observations, key=lambda item: (item["kind"], item["source"], item["question_id"]))


def theta_grid(calibration: CalibrationFit, *, theta_pad: float, grid_size: int) -> np.ndarray:
    values = calibration.model_params["theta"].astype(float).to_numpy()
    lo = min(float(values.min()) - theta_pad, -theta_pad)
    hi = max(float(values.max()) + theta_pad, theta_pad)
    return np.linspace(lo, hi, grid_size)


def loss_grid_for_observations(
    observed: list[dict[str, Any]],
    theta_values: np.ndarray,
    calibration: CalibrationFit,
    *,
    args: argparse.Namespace,
) -> np.ndarray:
    static_loss_sum = np.zeros_like(theta_values, dtype=float)
    static_count = 0
    arena_loss_sum = np.zeros_like(theta_values, dtype=float)
    arena_pair_count = 0
    bb_loss_sum = np.zeros_like(theta_values, dtype=float)
    bb_pair_count = 0

    for obs in observed:
        (
            static_part,
            static_part_count,
            arena_part,
            arena_part_count,
            bb_part,
            bb_part_count,
        ) = observation_loss_contribution(obs, theta_values, calibration)
        static_loss_sum += static_part
        static_count += static_part_count
        arena_loss_sum += arena_part
        arena_pair_count += arena_part_count
        bb_loss_sum += bb_part
        bb_pair_count += bb_part_count

    return total_loss_from_sums(
        theta_values,
        args=args,
        static_loss_sum=static_loss_sum,
        static_count=static_count,
        arena_loss_sum=arena_loss_sum,
        arena_pair_count=arena_pair_count,
        bb_loss_sum=bb_loss_sum,
        bb_pair_count=bb_pair_count,
    )


def observation_loss_contribution(
    obs: dict[str, Any],
    theta_values: np.ndarray,
    calibration: CalibrationFit,
) -> tuple[np.ndarray, int, np.ndarray, int, np.ndarray, int]:
    spec = METHOD_SPECS[calibration.method]
    zeros = np.zeros_like(theta_values, dtype=float)
    if obs["kind"] == "static":
        logits = float(obs["a"]) * (theta_values - float(obs["b"]))
        target = np.full_like(theta_values, float(obs["target"]), dtype=float)
        return bce_with_logits_np(logits, target), 1, zeros.copy(), 0, zeros.copy(), 0

    target = np.asarray(obs["target"], dtype=float)
    tie = np.asarray(obs["tie"], dtype=bool)
    both_bad = np.asarray(obs["both_bad"], dtype=bool)
    arena_loss = zeros.copy()
    arena_pair_count = 0
    bb_loss = zeros.copy()
    bb_pair_count = 0

    if spec.is_dualeval_irt:
        a = float(obs["a"])
        b = float(obs["b"])
        gamma = float(calibration.metadata.get("learned_gamma") or 1.0)
        p_new = sigmoid_np(a * (theta_values - b))
        p_opp = np.asarray(obs["p_opp"], dtype=float)
        logits = gamma * (p_new[:, None] - p_opp[None, :])
        hard = ~tie & ~both_bad
        if hard.any():
            arena_loss += bce_with_logits_np(logits[:, hard], target[None, hard]).sum(axis=1)
            arena_pair_count += int(hard.sum())
        bb = both_bad & ~tie
        if bb.any():
            bb_loss += int(bb.sum()) * (-np.log(1.0 - p_new + 1e-6))
            bb_pair_count += int(bb.sum())
    elif spec.is_direct_bt:
        non_tie = ~tie
        if non_tie.any():
            opp_theta = np.asarray(obs["opp_theta"], dtype=float)
            logits = theta_values[:, None] - opp_theta[None, :]
            arena_loss += bce_with_logits_np(logits[:, non_tie], target[None, non_tie]).sum(axis=1)
            arena_pair_count += int(non_tie.sum())

    return zeros.copy(), 0, arena_loss, arena_pair_count, bb_loss, bb_pair_count


def total_loss_from_sums(
    theta_values: np.ndarray,
    *,
    args: argparse.Namespace,
    static_loss_sum: np.ndarray,
    static_count: int,
    arena_loss_sum: np.ndarray,
    arena_pair_count: int,
    bb_loss_sum: np.ndarray,
    bb_pair_count: int,
) -> np.ndarray:
    total = args.replay_reg_lambda * theta_values**2
    if static_count:
        total += args.lambda_static * (static_loss_sum / static_count)
    if arena_pair_count:
        total += args.lambda_arena * (arena_loss_sum / arena_pair_count)
    if bb_pair_count:
        total += args.lambda_bb * (bb_loss_sum / bb_pair_count)
    return total


def fisher_score(obs: dict[str, Any], theta_hat: float, calibration: CalibrationFit, args: argparse.Namespace) -> float:
    spec = METHOD_SPECS[calibration.method]
    if obs["kind"] == "static":
        a = float(obs["a"])
        p = float(sigmoid_np(a * (theta_hat - float(obs["b"]))))
        return float(args.lambda_static * a * a * p * (1.0 - p))

    tie = np.asarray(obs["tie"], dtype=bool)
    both_bad = np.asarray(obs["both_bad"], dtype=bool)
    usable = ~tie & ~both_bad if spec.is_dualeval_irt else ~tie
    if not usable.any():
        return 0.0
    if spec.is_dualeval_irt:
        a = float(obs["a"])
        b = float(obs["b"])
        gamma = float(calibration.metadata.get("learned_gamma") or 1.0)
        p_new = float(sigmoid_np(a * (theta_hat - b)))
        p_opp = np.asarray(obs["p_opp"], dtype=float)[usable]
        mu = sigmoid_np(gamma * (p_new - p_opp))
        derivative = gamma * a * p_new * (1.0 - p_new)
        return float(args.lambda_arena * np.mean(mu * (1.0 - mu) * derivative * derivative))

    opp_theta = np.asarray(obs["opp_theta"], dtype=float)[usable]
    mu = sigmoid_np(theta_hat - opp_theta)
    return float(args.lambda_arena * np.mean(mu * (1.0 - mu)))


def sharpness_score(obs: dict[str, Any], theta_hat: float, calibration: CalibrationFit, args: argparse.Namespace) -> float:
    del theta_hat, calibration, args
    if "a" not in obs:
        return 0.0
    a = float(obs["a"])
    return a * a


def choose_next_index(
    remaining: list[int],
    observations: list[dict[str, Any]],
    *,
    strategy: str,
    theta_hat: float,
    calibration: CalibrationFit,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> int:
    if strategy == "random":
        return int(rng.choice(np.array(remaining, dtype=int)))
    scorer = fisher_score if strategy == "fisher" else sharpness_score
    scored = [
        (
            scorer(observations[idx], theta_hat, calibration, args),
            observations[idx]["obs_id"],
            idx,
        )
        for idx in remaining
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return int(scored[0][2])


def rank_error_for_model(
    model_name: str,
    theta_map: dict[str, float],
    reference_rank: dict[str, int],
) -> tuple[int, int, int]:
    common = sorted(set(theta_map) & set(reference_rank))
    online = online_rank_map({model: theta_map[model] for model in common})
    ref_restricted = {
        model: idx
        for idx, model in enumerate(sorted(common, key=lambda item: (reference_rank[item], item)), start=1)
    }
    return (
        int(online[model_name]),
        int(ref_restricted[model_name]),
        int(abs(online[model_name] - ref_restricted[model_name])),
    )


def replay_budget(args: argparse.Namespace, n_available: int) -> int:
    return n_available if args.max_steps == 0 else min(args.max_steps, n_available)


def record_checkpoints(args: argparse.Namespace, *, max_steps: int) -> dict[int, float]:
    if args.record_fractions is not None:
        checkpoints: dict[int, float] = {}
        for fraction in sorted(set(float(frac) for frac in args.record_fractions)):
            n = max(1, min(max_steps, int(math.ceil(fraction * max_steps))))
            checkpoints[n] = fraction
        return checkpoints

    checkpoints = {1: 1.0 / max_steps, max_steps: 1.0}
    for n in range(args.record_every, max_steps + 1, args.record_every):
        checkpoints[n] = n / max_steps
    return checkpoints


def replay_model(
    *,
    model_name: str,
    observations: list[dict[str, Any]],
    calibration: CalibrationFit,
    reference_rank: dict[str, int],
    method_reference_rank: dict[str, int],
    strategy: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not observations:
        summary = {
            "method": calibration.method,
            "strategy": strategy,
            "m_new": model_name,
            "n_available": 0,
            "n_final": 0,
            "theta_final": float("nan"),
            "rank_error_final": float("nan"),
        }
        return [], summary

    grid = theta_grid(calibration, theta_pad=args.theta_pad, grid_size=args.grid_size)
    theta_hat = float(grid[np.argmin(np.abs(grid))])
    remaining = list(range(len(observations)))
    rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in f"{calibration.method}:{strategy}:{model_name}"))
    max_steps = replay_budget(args, len(observations))
    record_ns = record_checkpoints(args, max_steps=max_steps)
    rows: list[dict[str, Any]] = []
    rank_path: list[int] = []
    static_loss_sum = np.zeros_like(grid, dtype=float)
    static_count = 0
    arena_loss_sum = np.zeros_like(grid, dtype=float)
    arena_pair_count = 0
    bb_loss_sum = np.zeros_like(grid, dtype=float)
    bb_pair_count = 0

    for n in range(1, max_steps + 1):
        next_idx = choose_next_index(
            remaining,
            observations,
            strategy=strategy,
            theta_hat=theta_hat,
            calibration=calibration,
            args=args,
            rng=rng,
        )
        remaining.remove(next_idx)
        selected = observations[next_idx]
        (
            static_part,
            static_part_count,
            arena_part,
            arena_part_count,
            bb_part,
            bb_part_count,
        ) = observation_loss_contribution(selected, grid, calibration)
        static_loss_sum += static_part
        static_count += static_part_count
        arena_loss_sum += arena_part
        arena_pair_count += arena_part_count
        bb_loss_sum += bb_part
        bb_pair_count += bb_part_count
        losses = total_loss_from_sums(
            grid,
            args=args,
            static_loss_sum=static_loss_sum,
            static_count=static_count,
            arena_loss_sum=arena_loss_sum,
            arena_pair_count=arena_pair_count,
            bb_loss_sum=bb_loss_sum,
            bb_pair_count=bb_pair_count,
        )
        theta_hat = float(grid[int(np.argmin(losses))])

        theta_map = dict(calibration.theta_map)
        theta_map[model_name] = theta_hat
        rank_hat, rank_ref, rank_error = rank_error_for_model(model_name, theta_map, reference_rank)
        method_rank_hat, method_rank_ref, method_rank_error = rank_error_for_model(
            model_name,
            theta_map,
            method_reference_rank,
        )
        rank_path.append(rank_ref)
        if n in record_ns:
            metrics = ranking_metrics(theta_map, reference_rank)
            rows.append(
                {
                    "method": calibration.method,
                    "strategy": strategy,
                    "m_new": model_name,
                    "n": n,
                    "n_available": len(observations),
                    "reveal_fraction": n / len(observations),
                    "target_fraction": record_ns[n],
                    "obs_id": selected["obs_id"],
                    "obs_kind": selected["kind"],
                    "source": selected["source"],
                    "question_id": selected["question_id"],
                    "theta_hat": theta_hat,
                    "rank_hat": rank_hat,
                    "rank_ref": rank_ref,
                    "rank_error": rank_error,
                    "method_rank_hat": method_rank_hat,
                    "method_rank_ref": method_rank_ref,
                    "method_rank_error": method_rank_error,
                    "spearman": metrics["spearman"],
                    "kendall": metrics["kendall"],
                    "exact_matches": metrics["exact_matches"],
                    "fit_static_rows": calibration.static_rows,
                    "fit_arena_rows": calibration.arena_rows,
                    "fit_arena_pairs": calibration.arena_pairs,
                }
            )

    final_theta_map = dict(calibration.theta_map)
    final_theta_map[model_name] = theta_hat
    rank_hat, rank_ref, rank_error = rank_error_for_model(model_name, final_theta_map, reference_rank)
    method_rank_hat, method_rank_ref, method_rank_error = rank_error_for_model(
        model_name,
        final_theta_map,
        method_reference_rank,
    )
    summary = {
        "method": calibration.method,
        "strategy": strategy,
        "m_new": model_name,
        "n_available": int(len(observations)),
        "n_final": int(max_steps),
        "final_reveal_fraction": max_steps / len(observations),
        "theta_final": float(theta_hat),
        "rank_hat_final": rank_hat,
        "rank_ref": rank_ref,
        "rank_error_final": rank_error,
        "method_rank_hat_final": method_rank_hat,
        "method_rank_ref": method_rank_ref,
        "method_rank_error_final": method_rank_error,
        "fit_static_rows": calibration.static_rows,
        "fit_arena_rows": calibration.arena_rows,
        "fit_arena_pairs": calibration.arena_pairs,
    }
    return rows, summary


def aggregate_results(trajectory_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    if not trajectory_df.empty:
        curve = (
            trajectory_df.groupby(["method", "strategy", "n"], as_index=False)
            .agg(
                mean_rank_error=("rank_error", "mean"),
                median_rank_error=("rank_error", "median"),
                mean_method_rank_error=("method_rank_error", "mean"),
                mean_spearman=("spearman", "mean"),
                mean_kendall=("kendall", "mean"),
                n_models=("m_new", "nunique"),
            )
            .sort_values(["method", "strategy", "n"])
        )
        curve.to_csv(output_dir / "trajectory_summary.csv", index=False)
        if "target_fraction" in trajectory_df.columns:
            fraction_curve = (
                trajectory_df.groupby(["method", "strategy", "target_fraction"], as_index=False)
                .agg(
                    mean_n=("n", "mean"),
                    mean_reveal_fraction=("reveal_fraction", "mean"),
                    mean_rank_error=("rank_error", "mean"),
                    median_rank_error=("rank_error", "median"),
                    mean_method_rank_error=("method_rank_error", "mean"),
                    mean_spearman=("spearman", "mean"),
                    mean_kendall=("kendall", "mean"),
                    n_models=("m_new", "nunique"),
                )
                .sort_values(["method", "strategy", "target_fraction"])
            )
            fraction_curve.to_csv(output_dir / "trajectory_fraction_summary.csv", index=False)
    if not summary_df.empty:
        final = (
            summary_df.groupby(["method", "strategy"], as_index=False)
            .agg(
                mean_final_rank_error=("rank_error_final", "mean"),
                median_final_rank_error=("rank_error_final", "median"),
                mean_method_final_rank_error=("method_rank_error_final", "mean"),
                median_method_final_rank_error=("method_rank_error_final", "median"),
                mean_n_available=("n_available", "mean"),
                mean_n_final=("n_final", "mean"),
                n_models=("m_new", "nunique"),
            )
            .sort_values(["method", "strategy"])
        )
        final.to_csv(output_dir / "final_summary.csv", index=False)


def write_metadata(
    args: argparse.Namespace,
    static_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    reference_ranks: dict[str, dict[str, int]],
) -> None:
    metadata = {
        "protocol": "leave_one_model_out_frozen_item_online_theta_update",
        "static_jsonl": [str(path) for path in args.static_jsonl],
        "arena_jsonl": [str(path) for path in args.arena_jsonl],
        "methods": args.methods,
        "strategies": args.strategies,
        "reference_method": args.reference_method,
        "max_steps": args.max_steps,
        "record_every": args.record_every,
        "record_fractions": args.record_fractions,
        "reference_epochs": args.reference_epochs,
        "calibration_epochs": args.calibration_epochs,
        "lambda_static": args.lambda_static,
        "lambda_arena": args.lambda_arena,
        "lambda_bb": args.lambda_bb,
        "reg_lambda": args.reg_lambda,
        "bb_ratio": args.bb_ratio,
        "tie_ratio": args.tie_ratio,
        "n_static_rows": int(len(static_df)),
        "n_static_questions": int(static_df["question_id"].nunique()) if not static_df.empty else 0,
        "n_arena_rows": int(len(reward_df)),
        "n_arena_questions": int(reward_df["question_id"].nunique()) if not reward_df.empty else 0,
        "n_models": int(len(set(static_df["model_name"].astype(str)) | set(reward_df["model_name"].astype(str)))),
        "reference_ranks": reference_ranks,
        "leakage_guard": (
            "Each held-out model is removed before reward normalization, pair-threshold "
            "estimation, calibration fitting, item selection, and theta updates. "
            "Full-data references are used only for evaluation."
        ),
        "fisher_static": "lambda_static * a_q^2 * p_new,q * (1 - p_new,q)",
        "fisher_arena_theta_new": (
            "lambda_arena * mean_j mu_new,j,q(1-mu_new,j,q) * "
            "(gamma * a_q * p_new,q * (1-p_new,q))^2"
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        args.output_dir / "trajectories.csv",
        args.output_dir / "model_summaries.csv",
        args.output_dir / "trajectory_summary.csv",
        args.output_dir / "trajectory_fraction_summary.csv",
        args.output_dir / "final_summary.csv",
        args.output_dir / "run_metadata.json",
    ]
    if any(path.exists() for path in output_files) and not args.overwrite:
        raise SystemExit(f"Output files already exist in {args.output_dir}; pass --overwrite to replace.")
    for path in output_files:
        if path.exists():
            path.unlink()

    static_df = load_static_jsonls(args.static_jsonl)
    reward_df = load_arena_jsonls(args.arena_jsonl)
    static_df = restrict_questions_per_source(
        static_df,
        max_questions_per_source=args.max_static_questions_per_source,
    )
    reward_df = restrict_questions_per_source(
        reward_df,
        max_questions_per_source=args.max_arena_questions_per_source,
    )
    static_df, reward_df = filter_models(static_df, reward_df, models=args.models, max_models=args.max_models)
    models = sorted(set(static_df["model_name"].astype(str)) | set(reward_df["model_name"].astype(str)))
    if len(models) < 3:
        raise SystemExit("Need at least three models for leave-one-model-out replay.")

    print(
        f"Loaded {len(static_df)} static rows / {static_df['question_id'].nunique()} static questions, "
        f"{len(reward_df)} arena rows / {reward_df['question_id'].nunique()} arena questions, "
        f"{len(models)} models.",
        flush=True,
    )

    reference_fits: dict[str, CalibrationFit] = {}
    reference_ranks: dict[str, dict[str, int]] = {}
    for method in args.methods:
        print(f"Fitting full-data reference: {method}", flush=True)
        reference = fit_method(
            method,
            static_df,
            reward_df,
            args=args,
            num_epochs=args.reference_epochs,
        )
        reference_fits[method] = reference
        reference.model_params.to_csv(args.output_dir / f"reference_{method}_models.csv", index=False)
        reference.question_params.to_csv(args.output_dir / f"reference_{method}_questions.csv", index=False)
        reference_ranks[method] = rank_map(reference.model_params)

    eval_reference_rank = reference_ranks[args.reference_method]
    write_metadata(args, static_df, reward_df, reference_ranks)

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    total_calibrations = len(args.methods) * len(models)
    calibration_idx = 0
    for method in args.methods:
        method_ref_rank = reference_ranks[method]
        for model_name in models:
            calibration_idx += 1
            print(f"[{calibration_idx}/{total_calibrations}] calibrating {method} without {model_name}", flush=True)
            static_train = static_df[static_df["model_name"].astype(str) != model_name].copy()
            reward_train = reward_df[reward_df["model_name"].astype(str) != model_name].copy()
            calibration = fit_method(
                method,
                static_train,
                reward_train,
                args=args,
                num_epochs=args.calibration_epochs,
            )
            observations = build_observations(model_name, static_df, reward_df, calibration)
            for strategy in args.strategies:
                rows, summary = replay_model(
                    model_name=model_name,
                    observations=observations,
                    calibration=calibration,
                    reference_rank=eval_reference_rank,
                    method_reference_rank=method_ref_rank,
                    strategy=strategy,
                    args=args,
                )
                all_rows.extend(rows)
                all_summaries.append(summary)
            pd.DataFrame(all_rows).to_csv(args.output_dir / "trajectories.csv", index=False)
            pd.DataFrame(all_summaries).to_csv(args.output_dir / "model_summaries.csv", index=False)

    trajectory_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(all_summaries)
    aggregate_results(trajectory_df, summary_df, args.output_dir)
    print(f"Outputs saved to {args.output_dir}", flush=True)
    final_path = args.output_dir / "final_summary.csv"
    if final_path.exists():
        print(pd.read_csv(final_path).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
