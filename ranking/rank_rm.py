#!/usr/bin/env python3
"""
IRT ranking for static benchmarks plus arena reward signals.

Supported modes mirror the JSONL-based reward modes in ``ranking.py``:

1. ``static``: static 2PL-IRT only
       P(y_{i,q}=1) = sigmoid(a_q * (theta_i - b_q))

2. ``arena`` / ``both``: reward-distilled soft-pairwise arena IRT
   - Convert per-response reward scores into globally standardized z-scores.
   - For each question and model pair, define a soft preference target:
       p*_{ijq} = sigmoid(z_{i,q} - z_{j,q})
   - Fit the nested pairwise likelihood:
       pi_{i,q} = sigmoid(theta_i - b_q)
       P_hat(i > j | q) = sigmoid(gamma * a_q * (pi_{i,q} - pi_{j,q}))
   - Optionally apply both-bad anchoring when both reward scores are low.

3. ``reward`` / ``both-reward``: direct reward-regression IRT
       r_{i,q} ~= a_q * (theta_i - b_q)

Example:
    python ranking/rank_rm.py --config ranking/config_ranking_rm.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODE_TO_ARENA_MODE = {
    "static": "soft_pairwise",
    "arena": "soft_pairwise",
    "both": "soft_pairwise",
    "reward": "regression",
    "both-reward": "regression",
}


def mode_uses_static(mode: str) -> bool:
    return mode in {"static", "both", "both-reward"}


def mode_uses_arena(mode: str) -> bool:
    return mode in {"arena", "both", "reward", "both-reward"}


def infer_mode(
    *,
    configured_mode: str | None,
    configured_arena_mode: str | None,
    has_static_input: bool,
    has_arena_input: bool,
) -> str | None:
    if configured_mode:
        return configured_mode
    if configured_arena_mode == "regression":
        if has_static_input and has_arena_input:
            return "both-reward"
        if has_arena_input:
            return "reward"
    elif configured_arena_mode == "soft_pairwise":
        if has_static_input and has_arena_input:
            return "both"
        if has_arena_input:
            return "arena"
        if has_static_input:
            return "static"
    elif has_static_input and has_arena_input:
        return "both"
    elif has_arena_input:
        return "arena"
    elif has_static_input:
        return "static"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit IRT rankings from static JSONL data and arena reward JSONL data.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file.",
    )
    parser.add_argument(
        "--static-jsonl",
        nargs="*",
        default=None,
        help="One or more static_eval responses.jsonl files.",
    )
    parser.add_argument(
        "--arena-reward-jsonl",
        nargs="*",
        default=None,
        help="One or more arena_eval responses.jsonl files with reward scores.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=sorted(MODE_TO_ARENA_MODE),
        help=(
            "Training mode, aligned with ranking.py. "
            "'static': static 2PL only. "
            "'arena': reward-distilled soft-pairwise arena only. "
            "'both': joint static + reward-distilled arena. "
            "'reward': direct reward-regression IRT only. "
            "'both-reward': joint static + direct reward-regression IRT."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        help="Optional path for the ranking plot.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--lambda-static",
        type=float,
        default=None,
        help="Weight for static 2PL loss.",
    )
    parser.add_argument(
        "--lambda-arena",
        type=float,
        default=None,
        help="Weight for arena soft-pairwise loss.",
    )
    parser.add_argument(
        "--lambda-bb",
        type=float,
        default=None,
        help="Weight for both-bad anchoring loss.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=None,
        help="L2 regularization coefficient.",
    )
    parser.add_argument(
        "--both-bad-threshold",
        type=float,
        default=None,
        help="Threshold tau for identifying both-bad arena pairs.",
    )
    parser.add_argument(
        "--both-bad-use-zscore",
        action="store_true",
        help="Threshold both-bad pairs on standardized reward z-scores.",
    )
    parser.add_argument(
        "--both-bad-use-raw",
        action="store_true",
        help="Threshold both-bad pairs on raw reward scores.",
    )
    parser.add_argument(
        "--save-pairwise-targets",
        action="store_true",
        help="Write the distilled arena pairwise targets to CSV.",
    )
    parser.add_argument(
        "--arena-mode",
        default=None,
        choices=["soft_pairwise", "regression"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress logs.",
    )
    return parser.parse_args()


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def _ensure_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _resolve_paths(paths: list[str] | None) -> list[str] | None:
    if paths is None:
        return None
    return [str(_resolve_path(path)) for path in paths]


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    input_section = config.get("input", {})
    training = config.get("training", {})
    output = config.get("output", {})

    if args.static_jsonl is None:
        args.static_jsonl = _ensure_list(input_section.get("static_jsonl"))
    if args.arena_reward_jsonl is None:
        args.arena_reward_jsonl = _ensure_list(input_section.get("arena_reward_jsonl"))
    if args.output_dir is None:
        args.output_dir = str(_resolve_path(output.get("output_dir", "results/ranking_rm/default")))
    if args.save_plot is None and output.get("save_plot") is not None:
        args.save_plot = str(_resolve_path(str(output.get("save_plot"))))
    if args.num_epochs is None:
        args.num_epochs = int(training.get("num_epochs", 5000))
    if args.lr is None:
        args.lr = float(training.get("lr", 0.02))
    if args.lambda_static is None:
        args.lambda_static = float(training.get("lambda_static", 1.0))
    if args.lambda_arena is None:
        args.lambda_arena = float(training.get("lambda_arena", 1.0))
    if args.lambda_bb is None:
        args.lambda_bb = float(training.get("lambda_bb", 0.3))
    if args.reg_lambda is None:
        args.reg_lambda = float(training.get("reg_lambda", 1e-4))
    if args.both_bad_threshold is None:
        args.both_bad_threshold = float(training.get("both_bad_threshold", -0.5))

    if not args.both_bad_use_zscore and not args.both_bad_use_raw:
        both_bad_mode = str(training.get("both_bad_mode", "zscore")).strip().lower()
        args.both_bad_use_zscore = both_bad_mode != "raw"
        args.both_bad_use_raw = both_bad_mode == "raw"

    if args.mode is None and training.get("mode") is not None:
        args.mode = str(training.get("mode")).strip().lower()
    if args.arena_mode is None and training.get("arena_mode") is not None:
        args.arena_mode = str(training.get("arena_mode")).strip().lower()

    args.mode = infer_mode(
        configured_mode=args.mode,
        configured_arena_mode=args.arena_mode,
        has_static_input=bool(args.static_jsonl),
        has_arena_input=bool(args.arena_reward_jsonl),
    )
    if args.mode is None:
        args.mode = "both"

    expected_arena_mode = MODE_TO_ARENA_MODE.get(args.mode)
    if expected_arena_mode is None:
        valid_modes = ", ".join(sorted(MODE_TO_ARENA_MODE))
        raise SystemExit(f"Invalid mode '{args.mode}'. Expected one of: {valid_modes}.")
    if args.arena_mode is not None and args.arena_mode != expected_arena_mode:
        raise SystemExit(
            f"Conflicting configuration: mode='{args.mode}' implies arena_mode='{expected_arena_mode}', "
            f"but arena_mode='{args.arena_mode}' was also provided."
        )
    args.arena_mode = expected_arena_mode
    if not args.save_pairwise_targets:
        args.save_pairwise_targets = bool(output.get("save_pairwise_targets", False))
    if not args.no_plot:
        args.no_plot = bool(output.get("no_plot", False))
    if not args.quiet:
        args.quiet = bool(output.get("quiet", False))

    args.static_jsonl = _resolve_paths(args.static_jsonl)
    args.arena_reward_jsonl = _resolve_paths(args.arena_reward_jsonl)
    return args


def source_tag_for_path(path: Path) -> str:
    parent_name = path.parent.name.strip()
    if parent_name:
        return parent_name
    return path.stem


def load_static_jsonl(jsonl_paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        path = Path(jsonl_path)
        source_tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("status") != "ok":
                    continue
                if d.get("correct") is None:
                    continue
                dataset = str(d.get("dataset", "unknown"))
                sample_index = d.get("sample_index")
                raw_question_id = f"{dataset}_{sample_index}"
                rows.append(
                    {
                        "source": source_tag,
                        "benchmark": dataset,
                        "model_name": str(d["model_label"]),
                        "question_id": f"{source_tag}::{raw_question_id}",
                        "judge_result": int(bool(d["correct"])),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["model_name", "question_id"], keep="last").reset_index(drop=True)


def load_arena_reward_jsonl(jsonl_paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        path = Path(jsonl_path)
        source_tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("status") != "ok" or d.get("reward") is None:
                    continue
                raw_question_id = str(d["item_id"])
                rows.append(
                    {
                        "source": source_tag,
                        "benchmark": "Arena",
                        "model_name": str(d["model_label"]),
                        "question_id": f"{source_tag}::{raw_question_id}",
                        "reward_raw": float(d["reward"]),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["model_name", "question_id"], keep="last").reset_index(drop=True)
    reward_mean = float(df["reward_raw"].mean())
    reward_std = float(df["reward_raw"].std(ddof=0))
    if not math.isfinite(reward_std) or reward_std < 1e-8:
        reward_std = 1.0
    df["reward_z"] = (df["reward_raw"] - reward_mean) / reward_std
    return df


def build_soft_pairwise_targets(
    reward_df: pd.DataFrame,
    *,
    both_bad_threshold: float,
    both_bad_use_zscore: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_col = "reward_z" if both_bad_use_zscore else "reward_raw"
    for question_id, group in reward_df.groupby("question_id", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values("model_name").reset_index(drop=True)
        for idx1, idx2 in combinations(range(len(group)), 2):
            r1 = group.iloc[idx1]
            r2 = group.iloc[idx2]
            z1 = float(r1["reward_z"])
            z2 = float(r2["reward_z"])
            soft_pref = 1.0 / (1.0 + math.exp(-(z1 - z2)))
            score1 = float(r1[score_col])
            score2 = float(r2[score_col])
            rows.append(
                {
                    "source": r1["source"],
                    "benchmark": "Arena",
                    "question_id": question_id,
                    "model_1": r1["model_name"],
                    "model_2": r2["model_name"],
                    "reward_raw_1": float(r1["reward_raw"]),
                    "reward_raw_2": float(r2["reward_raw"]),
                    "reward_z_1": z1,
                    "reward_z_2": z2,
                    "target_prob": soft_pref,
                    "both_bad": bool(score1 < both_bad_threshold and score2 < both_bad_threshold),
                }
            )
    return pd.DataFrame(rows)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_joint_reward_distilled_irt(
    static_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    reward_df: pd.DataFrame | None = None,
    *,
    arena_mode: str = "soft_pairwise",
    num_epochs: int,
    lr: float,
    lambda_static: float,
    lambda_arena: float,
    lambda_bb: float,
    reg_lambda: float,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit IRT model jointly from static benchmark and arena reward data.

    arena_mode:
      'soft_pairwise' — distil rewards into soft pairwise targets (BCE loss).
      'regression'    — regress a_q*(theta_i - b_q) directly onto raw rewards (MSE loss).
    """
    static = static_df.copy() if static_df is not None else pd.DataFrame()
    pairwise = pairwise_df.copy() if pairwise_df is not None and arena_mode == "soft_pairwise" else pd.DataFrame()
    reward = reward_df.copy() if reward_df is not None and arena_mode == "regression" else pd.DataFrame()

    has_static = not static.empty
    has_arena = not pairwise.empty if arena_mode == "soft_pairwise" else not reward.empty
    if not has_static and not has_arena:
        raise SystemExit("Need at least one non-empty data source.")

    if has_static:
        model_series = [static["model_name"]]
        question_series = [static["question_id"]]
    else:
        model_series = []
        question_series = []
    if has_arena:
        if arena_mode == "soft_pairwise":
            model_series.extend([pairwise["model_1"], pairwise["model_2"]])
            question_series.append(pairwise["question_id"])
        else:
            model_series.append(reward["model_name"])
            question_series.append(reward["question_id"])

    all_models = pd.Index(pd.unique(pd.concat(model_series, ignore_index=True)), name="model_name")
    all_questions = pd.Index(pd.unique(pd.concat(question_series, ignore_index=True)), name="question_id")
    model_to_idx = {m: i for i, m in enumerate(all_models)}
    q_to_idx = {q: i for i, q in enumerate(all_questions)}

    question_meta: dict[str, dict[str, str]] = {}
    if has_static:
        for _, row in static[["question_id", "source", "benchmark"]].drop_duplicates().iterrows():
            question_meta[str(row["question_id"])] = {
                "source": str(row["source"]),
                "benchmark": str(row["benchmark"]),
            }
        static["m_idx"] = static["model_name"].map(model_to_idx)
        static["q_idx"] = static["question_id"].map(q_to_idx)
    if has_arena:
        if arena_mode == "soft_pairwise":
            for _, row in pairwise[["question_id", "source", "benchmark"]].drop_duplicates().iterrows():
                question_meta[str(row["question_id"])] = {
                    "source": str(row["source"]),
                    "benchmark": str(row["benchmark"]),
                }
            pairwise["m1_idx"] = pairwise["model_1"].map(model_to_idx)
            pairwise["m2_idx"] = pairwise["model_2"].map(model_to_idx)
            pairwise["q_idx"] = pairwise["question_id"].map(q_to_idx)
        else:
            for _, row in reward[["question_id", "source", "benchmark"]].drop_duplicates().iterrows():
                question_meta[str(row["question_id"])] = {
                    "source": str(row["source"]),
                    "benchmark": str(row["benchmark"]),
                }
            reward["m_idx"] = reward["model_name"].map(model_to_idx)
            reward["q_idx"] = reward["question_id"].map(q_to_idx)

    device = _get_device()

    m_s = q_s = y_s = None
    if has_static:
        m_s = torch.tensor(static["m_idx"].values, dtype=torch.long, device=device)
        q_s = torch.tensor(static["q_idx"].values, dtype=torch.long, device=device)
        y_s = torch.tensor(static["judge_result"].values, dtype=torch.float32, device=device)

    # Soft pairwise tensors
    m1_t = m2_t = q_t = soft_t = None
    m1_bb = m2_bb = q_bb = None
    if has_arena and arena_mode == "soft_pairwise":
        m1_t = torch.tensor(pairwise["m1_idx"].values, dtype=torch.long, device=device)
        m2_t = torch.tensor(pairwise["m2_idx"].values, dtype=torch.long, device=device)
        q_t = torch.tensor(pairwise["q_idx"].values, dtype=torch.long, device=device)
        soft_t = torch.tensor(pairwise["target_prob"].values, dtype=torch.float32, device=device)
        both_bad = pairwise[pairwise["both_bad"]].copy()
        if not both_bad.empty:
            m1_bb = torch.tensor(both_bad["m1_idx"].values, dtype=torch.long, device=device)
            m2_bb = torch.tensor(both_bad["m2_idx"].values, dtype=torch.long, device=device)
            q_bb = torch.tensor(both_bad["q_idx"].values, dtype=torch.long, device=device)

    # Regression tensors
    m_r = q_r = r_t = None
    if has_arena and arena_mode == "regression":
        m_r = torch.tensor(reward["m_idx"].values, dtype=torch.long, device=device)
        q_r = torch.tensor(reward["q_idx"].values, dtype=torch.long, device=device)
        r_t = torch.tensor(reward["reward_raw"].values, dtype=torch.float32, device=device)

    n_models = len(all_models)
    n_questions = len(all_questions)

    theta = nn.Embedding(n_models, 1, device=device)
    b = nn.Embedding(n_questions, 1, device=device)
    k = nn.Embedding(n_questions, 1, device=device)
    nn.init.zeros_(theta.weight)
    nn.init.zeros_(b.weight)
    nn.init.zeros_(k.weight)

    # log_gamma is only used in soft_pairwise mode
    log_gamma = nn.Parameter(torch.tensor(0.0, device=device))
    opt_params = [*theta.parameters(), *b.parameters(), *k.parameters()]
    if arena_mode == "soft_pairwise":
        opt_params.append(log_gamma)

    optimizer = optim.Adam(opt_params, lr=lr)
    bce_logits = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        loss_static = torch.tensor(0.0, device=device)
        if has_static and m_s is not None and len(m_s):
            theta_s = theta(m_s).squeeze(-1)
            b_s = b(q_s).squeeze(-1)
            k_s = k(q_s).squeeze(-1)
            logits_s = torch.exp(k_s) * (theta_s - b_s)
            loss_static = bce_logits(logits_s, y_s)

        loss_arena = torch.tensor(0.0, device=device)
        if has_arena and arena_mode == "soft_pairwise" and m1_t is not None and len(m1_t):
            theta_1 = theta(m1_t).squeeze(-1)
            theta_2 = theta(m2_t).squeeze(-1)
            b_q = b(q_t).squeeze(-1)
            k_q = k(q_t).squeeze(-1)
            a_q = torch.exp(k_q)
            gamma = torch.exp(log_gamma)
            pi_1 = torch.sigmoid(theta_1 - b_q)
            pi_2 = torch.sigmoid(theta_2 - b_q)
            arena_logits = gamma * a_q * (pi_1 - pi_2)
            loss_arena = bce_logits(arena_logits, soft_t)
        elif has_arena and arena_mode == "regression" and m_r is not None and len(m_r):
            theta_r = theta(m_r).squeeze(-1)
            b_r = b(q_r).squeeze(-1)
            k_r = k(q_r).squeeze(-1)
            pred = torch.exp(k_r) * (theta_r - b_r)
            loss_arena = mse(pred, r_t)

        loss_bb = torch.tensor(0.0, device=device)
        if has_arena and arena_mode == "soft_pairwise" and m1_bb is not None and len(m1_bb):
            theta_1_bb = theta(m1_bb).squeeze(-1)
            theta_2_bb = theta(m2_bb).squeeze(-1)
            b_q_bb = b(q_bb).squeeze(-1)
            pi_1_bb = torch.sigmoid(theta_1_bb - b_q_bb)
            pi_2_bb = torch.sigmoid(theta_2_bb - b_q_bb)
            loss_bb = -(
                torch.log(1.0 - pi_1_bb + 1e-6).mean()
                + torch.log(1.0 - pi_2_bb + 1e-6).mean()
            )

        reg = reg_lambda * (
            theta.weight.pow(2).mean()
            + b.weight.pow(2).mean()
            + k.weight.pow(2).mean()
        )
        if arena_mode == "soft_pairwise":
            reg = reg + reg_lambda * log_gamma.pow(2)
        total = lambda_static * loss_static + lambda_arena * loss_arena + lambda_bb * loss_bb + reg
        total.backward()
        optimizer.step()

        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            b.weight.sub_(shift)

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            if arena_mode == "soft_pairwise":
                print(
                    f"  Epoch {epoch:5d} | "
                    f"static={loss_static.item():.4f}  "
                    f"arena={loss_arena.item():.4f}  "
                    f"bb={loss_bb.item():.4f}  "
                    f"gamma={torch.exp(log_gamma).item():.4f}  "
                    f"total={total.item():.4f}"
                )
            else:
                print(
                    f"  Epoch {epoch:5d} | "
                    f"static={loss_static.item():.4f}  "
                    f"reward_mse={loss_arena.item():.4f}  "
                    f"total={total.item():.4f}"
                )

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    b_np = b.weight.detach().cpu().numpy().squeeze(-1)
    k_np = k.weight.detach().cpu().numpy().squeeze(-1)
    gamma_np = float(torch.exp(log_gamma).detach().cpu().item()) if arena_mode == "soft_pairwise" else None

    model_params = (
        pd.DataFrame({"model_name": all_models, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )

    question_rows: list[dict[str, Any]] = []
    for i, question_id in enumerate(all_questions):
        meta = question_meta.get(str(question_id), {})
        question_rows.append(
            {
                "question_id": question_id,
                "source": meta.get("source", "unknown"),
                "benchmark": meta.get("benchmark", "unknown"),
                "difficulty_b": b_np[i],
                "k_raw": k_np[i],
                "discrimination_exp_k": math.exp(k_np[i]),
            }
        )
    question_params = (
        pd.DataFrame(question_rows)
        .sort_values("difficulty_b", ascending=False)
        .reset_index(drop=True)
    )

    metadata = {
        "arena_mode": arena_mode,
        "learned_gamma": gamma_np,
        "n_models": int(n_models),
        "n_questions": int(n_questions),
        "has_static": bool(has_static),
        "has_arena": bool(has_arena),
    }
    return model_params, question_params, metadata


def compute_metrics(
    static_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    arena_mode: str = "soft_pairwise",
    learned_gamma: float | None = None,
    reward_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    theta_map = model_params.set_index("model_name")["theta"]
    question_map = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
    metrics: dict[str, Any] = {}

    if static_df is not None and not static_df.empty:
        st = static_df.copy()
        st["theta"] = st["model_name"].map(theta_map)
        st["difficulty_b"] = st["question_id"].map(question_map["difficulty_b"])
        st["a_q"] = st["question_id"].map(question_map["discrimination_exp_k"])
        st = st.dropna(subset=["theta", "difficulty_b", "a_q"])
        logits = st["a_q"] * (st["theta"] - st["difficulty_b"])
        st["pred_correct"] = (logits >= 0).astype(int)
        metrics["static_accuracy"] = float((st["pred_correct"] == st["judge_result"].astype(int)).mean())

    if arena_mode == "soft_pairwise" and pairwise_df is not None and not pairwise_df.empty:
        ar = pairwise_df.copy()
        ar["theta_1"] = ar["model_1"].map(theta_map)
        ar["theta_2"] = ar["model_2"].map(theta_map)
        ar["difficulty_b"] = ar["question_id"].map(question_map["difficulty_b"])
        ar["a_q"] = ar["question_id"].map(question_map["discrimination_exp_k"])
        ar = ar.dropna(subset=["theta_1", "theta_2", "difficulty_b", "a_q"])
        pi1 = 1.0 / (1.0 + np.exp(-(ar["theta_1"] - ar["difficulty_b"])))
        pi2 = 1.0 / (1.0 + np.exp(-(ar["theta_2"] - ar["difficulty_b"])))
        gamma = learned_gamma if learned_gamma is not None else 1.0
        logits = gamma * ar["a_q"].to_numpy() * (pi1 - pi2)
        pred_prob = 1.0 / (1.0 + np.exp(-logits))
        target = ar["target_prob"].to_numpy()
        eps = 1e-6
        logloss = -np.mean(target * np.log(pred_prob + eps) + (1.0 - target) * np.log(1.0 - pred_prob + eps))
        metrics["arena_soft_logloss"] = float(logloss)
        metrics["arena_both_bad_pairs"] = int(ar["both_bad"].sum())
        tie_mask = np.isclose(target, 0.5)
        hard_eval_mask = (~ar["both_bad"].to_numpy(dtype=bool)) & (~tie_mask)
        metrics["arena_tie_pairs"] = int(tie_mask.sum())
        metrics["arena_hard_eval_pairs"] = int(hard_eval_mask.sum())
        if hard_eval_mask.any():
            metrics["arena_hard_pair_accuracy"] = float(
                ((pred_prob[hard_eval_mask] >= 0.5) == (target[hard_eval_mask] >= 0.5)).mean()
            )
        else:
            metrics["arena_hard_pair_accuracy"] = float("nan")

    elif arena_mode == "regression" and reward_df is not None and not reward_df.empty:
        rw = reward_df.copy()
        rw["theta"] = rw["model_name"].map(theta_map)
        rw["difficulty_b"] = rw["question_id"].map(question_map["difficulty_b"])
        rw["a_q"] = rw["question_id"].map(question_map["discrimination_exp_k"])
        rw = rw.dropna(subset=["theta", "difficulty_b", "a_q"])
        pred = rw["a_q"] * (rw["theta"] - rw["difficulty_b"])
        residuals = pred - rw["reward_raw"]
        metrics["arena_reward_mse"] = float((residuals ** 2).mean())
        metrics["arena_reward_mae"] = float(residuals.abs().mean())

    return metrics


def plot_difficulty_and_ability(
    model_ranking: pd.DataFrame,
    question_ranking: pd.DataFrame,
    *,
    save_path: str | None,
    title: str,
) -> None:
    if question_ranking.empty or model_ranking.empty or not save_path:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    qr = question_ranking.copy()
    benchmarks = sorted(qr["benchmark"].fillna("unknown").unique())
    colors = plt.cm.tab10.colors[: max(1, len(benchmarks))]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bins = np.linspace(qr["difficulty_b"].min(), qr["difficulty_b"].max(), 41)
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)
    bottoms = np.zeros(len(centers))

    for idx, benchmark in enumerate(benchmarks):
        counts, _ = np.histogram(qr.loc[qr["benchmark"] == benchmark, "difficulty_b"], bins=bins)
        ax.bar(
            centers,
            counts,
            width=widths,
            bottom=bottoms,
            color=colors[idx % len(colors)],
            alpha=0.85,
            align="center",
            edgecolor="white",
            linewidth=0.3,
            label=benchmark,
        )
        bottoms += counts

    y_top = max(1.0, bottoms.max() * 1.5)
    ax.set_ylim(0, y_top)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.35)

    model_colors = plt.cm.Dark2.colors
    benchmark_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=6, label=benchmark)
        for idx, benchmark in enumerate(benchmarks)
    ]
    model_handles: list[Line2D] = []
    for idx, (_, row) in enumerate(model_ranking.iterrows()):
        color = model_colors[idx % len(model_colors)]
        ax.axvline(row["theta"], 0, 0.85, color=color, linestyle="--", linewidth=1.3, alpha=0.9)
        ax.text(
            row["theta"],
            y_top * (0.88 + 0.03 * (idx % 2)),
            f"{idx + 1}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )
        model_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                marker="o",
                markersize=5,
                label=f"{idx + 1}. {row['model_name']}",
            )
        )

    ax.set_xlabel(r"Latent scale (ability $\theta$ / difficulty $b$)")
    ax.set_ylabel("Number of questions")
    ax.set_title(title)

    # Place both legends side-by-side below the axes.
    # ncol=2 for models keeps the block compact when there are many models.
    # bbox_inches='tight' at save time ensures they are not clipped.
    n_model_cols = 2 if len(model_handles) > 8 else 1
    benchmark_legend = ax.legend(
        handles=benchmark_handles,
        title="Question Sources",
        frameon=True,
        framealpha=0.9,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.14),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(benchmark_legend)
    ax.legend(
        handles=model_handles,
        title="Model Rankings",
        frameon=True,
        framealpha=0.9,
        loc="upper left",
        bbox_to_anchor=(0.42, -0.14),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        ncol=n_model_cols,
        fontsize=8,
        title_fontsize=9,
    )

    plt.tight_layout()
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_obj, dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args = apply_config_defaults(args)
    verbose = not args.quiet

    if args.num_epochs < 1:
        raise SystemExit("--num-epochs must be at least 1.")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive.")
    if args.reg_lambda < 0:
        raise SystemExit("--reg-lambda must be non-negative.")
    if args.both_bad_use_zscore and args.both_bad_use_raw:
        raise SystemExit("Choose only one of --both-bad-use-zscore or --both-bad-use-raw.")

    use_static = mode_uses_static(args.mode)
    use_arena = mode_uses_arena(args.mode)

    static_df = pd.DataFrame()
    if use_static:
        if not args.static_jsonl:
            raise SystemExit(f"Mode '{args.mode}' requires --static-jsonl.")
        print(f"Loading static JSONL files: {len(args.static_jsonl)}")
        static_df = load_static_jsonl(args.static_jsonl)
        print(
            f"  {len(static_df)} rows, "
            f"{static_df['model_name'].nunique() if not static_df.empty else 0} models, "
            f"{static_df['question_id'].nunique() if not static_df.empty else 0} questions"
        )
    elif args.static_jsonl:
        print(f"Note: ignoring static_jsonl because mode='{args.mode}' does not use static data.")

    reward_df = pd.DataFrame()
    pairwise_df = pd.DataFrame()
    if use_arena:
        if not args.arena_reward_jsonl:
            raise SystemExit(f"Mode '{args.mode}' requires --arena-reward-jsonl.")
        print(f"Loading arena reward JSONL files: {len(args.arena_reward_jsonl)}")
        reward_df = load_arena_reward_jsonl(args.arena_reward_jsonl)
        if reward_df.empty:
            print("  0 usable reward rows found.")
        else:
            print(
                f"  {len(reward_df)} rows, "
                f"{reward_df['model_name'].nunique()} models, "
                f"{reward_df['question_id'].nunique()} questions, "
                f"reward mean={reward_df['reward_raw'].mean():.3f}, "
                f"reward std={reward_df['reward_raw'].std(ddof=0):.3f}"
            )
            if args.arena_mode == "soft_pairwise":
                pairwise_df = build_soft_pairwise_targets(
                    reward_df,
                    both_bad_threshold=args.both_bad_threshold,
                    both_bad_use_zscore=args.both_bad_use_zscore,
                )
                print(
                    f"  distilled into {len(pairwise_df)} soft pairwise rows; "
                    f"both_bad={int(pairwise_df['both_bad'].sum()) if not pairwise_df.empty else 0}"
                )
            else:
                print(f"  arena_mode=regression: using raw rewards directly")
    elif args.arena_reward_jsonl:
        print(f"Note: ignoring arena_reward_jsonl because mode='{args.mode}' does not use arena reward data.")

    has_arena_data = (not pairwise_df.empty) if args.arena_mode == "soft_pairwise" else (not reward_df.empty)
    if static_df.empty and not has_arena_data:
        raise SystemExit("No usable static or arena-reward data loaded.")

    print(f"\nFitting IRT model (mode={args.mode}, arena_mode={args.arena_mode}) ...")
    model_params, question_params, fit_meta = fit_joint_reward_distilled_irt(
        static_df if not static_df.empty else None,
        pairwise_df if not pairwise_df.empty else None,
        reward_df if not reward_df.empty else None,
        arena_mode=args.arena_mode,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=args.lambda_static,
        lambda_arena=args.lambda_arena,
        lambda_bb=args.lambda_bb,
        reg_lambda=args.reg_lambda,
        verbose=verbose,
    )

    metrics = compute_metrics(
        static_df if not static_df.empty else None,
        pairwise_df if not pairwise_df.empty else None,
        model_params,
        question_params,
        arena_mode=args.arena_mode,
        learned_gamma=fit_meta["learned_gamma"],
        reward_df=reward_df if not reward_df.empty else None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_params.to_csv(output_dir / "model_ranking.csv", index=False)
    question_params.to_csv(output_dir / "question_ranking.csv", index=False)
    if args.save_pairwise_targets and not pairwise_df.empty:
        pairwise_df.to_csv(output_dir / "arena_soft_pairwise_targets.csv", index=False)

    run_summary = {
        "config": args.config,
        "mode": args.mode,
        "arena_mode": args.arena_mode,
        "static_jsonl": args.static_jsonl,
        "arena_reward_jsonl": args.arena_reward_jsonl,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "lambda_static": args.lambda_static,
        "lambda_arena": args.lambda_arena,
        "lambda_bb": args.lambda_bb,
        "reg_lambda": args.reg_lambda,
        "both_bad_threshold": args.both_bad_threshold,
        "both_bad_mode": "zscore" if args.both_bad_use_zscore else "raw",
        "save_plot": args.save_plot,
        "metrics": metrics,
        **fit_meta,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\nModel ranking (highest ability first):")
    print(model_params.to_string(index=False))
    print(f"\nQuestion difficulty (top 20 hardest of {len(question_params)}):")
    print(question_params.head(20).to_string(index=False))
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    if fit_meta["learned_gamma"] is not None:
        print(f"  learned_gamma: {fit_meta['learned_gamma']}")

    if not args.no_plot and args.save_plot:
        if args.mode == "static":
            title = "Static IRT Ranking"
        elif args.arena_mode == "soft_pairwise":
            title = "Reward-Distilled IRT Ranking"
        else:
            title = "Reward Regression IRT Ranking"
        plot_difficulty_and_ability(
            model_params,
            question_params,
            save_path=args.save_plot,
            title=title,
        )
        print(f"\nSaved plot to {args.save_plot}")

    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
