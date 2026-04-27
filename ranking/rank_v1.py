#!/usr/bin/env python3
"""
IRT ranking v1 — direct pairwise formulation.

Key differences from rank_rm.py:
  1. Arena pairwise loss uses the direct formulation (no nested sigmoid, no γ):
         P(i > j | q) = sigmoid(a_q * (theta_i - theta_j))
     b_q cancels out of pairwise comparisons and is unidentified from pairwise
     data alone.  In pure pairwise modes, arena items carry a_q only.
  2. Static items retain the full 2PL parameterisation (θ, b_q, a_q).
  3. No both-bad anchoring loss — tie pairs (|z_i - z_j| < tie_delta) are
     simply excluded from the arena loss.
  4. z-regression mode: regress a_q*(θ_i - b_q) onto z_{i,q} (identifies b_q).
  5. pairwise+regression mode: combined loss that identifies both a_q and b_q
     for arena items without static data:
         L = lambda_arena * BCE(σ(a_q*(θ_i-θ_j)), σ(z_i-z_j))   [relative]
           + lambda_reg   * MSE(a_q*(θ_i-b_q),    z_{i,q})       [absolute]

Supported modes:
  static            — static 2PL IRT only
  arena             — direct pairwise arena IRT only (a_q, no b_q for arena)
  both              — joint static 2PL + direct pairwise IRT
  arena-pr          — pairwise+regression arena IRT (a_q and b_q for arena)
  both-pr           — joint static 2PL + pairwise+regression arena IRT
  z-regression      — z-score reward regression IRT only
  both-z-regression — joint static 2PL + z-score reward regression IRT

Example:
    python ranking/rank_v1.py --config ranking/config_ranking_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
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

# Re-use data-loading and config helpers from rank_rm without modification.
from rank_rm import (
    _ensure_list,
    _expand_env,
    _resolve_path,
    _resolve_paths,
    build_soft_pairwise_targets,
    load_arena_reward_jsonl,
    load_static_jsonl,
    load_yaml_config,
    resolve_pairwise_thresholds,
    source_tag_for_path,
)

# v1 extends the mode set from rank_rm to add pairwise+regression modes.
MODE_TO_ARENA_MODE: dict[str, str] = {
    "static":            "soft_pairwise",
    "arena":             "soft_pairwise",
    "both":              "soft_pairwise",
    "arena-pr":          "pairwise+regression",
    "both-pr":           "pairwise+regression",
    "z-regression":      "z-regression",
    "both-z-regression": "z-regression",
}


def mode_uses_static(mode: str) -> bool:
    return mode in {"static", "both", "both-pr", "both-z-regression"}


def mode_uses_arena(mode: str) -> bool:
    return mode in {"arena", "both", "arena-pr", "both-pr", "z-regression", "both-z-regression"}


def infer_mode(
    *,
    configured_mode: str | None,
    configured_arena_mode: str | None,
    has_static_input: bool,
    has_arena_input: bool,
) -> str | None:
    if configured_mode:
        return configured_mode
    if configured_arena_mode == "pairwise+regression":
        if has_static_input and has_arena_input:
            return "both-pr"
        if has_arena_input:
            return "arena-pr"
    elif configured_arena_mode == "z-regression":
        if has_static_input and has_arena_input:
            return "both-z-regression"
        if has_arena_input:
            return "z-regression"
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit IRT rankings (v1 direct pairwise) from static and arena reward data.",
    )
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--static-jsonl", nargs="*", default=None)
    parser.add_argument("--arena-reward-jsonl", nargs="*", default=None)
    parser.add_argument(
        "--mode", default=None,
        choices=sorted(MODE_TO_ARENA_MODE),
        help=(
            "'static': static 2PL only. "
            "'arena': direct pairwise arena only (a_q, no b_q for arena). "
            "'both': joint static + direct pairwise. "
            "'arena-pr': pairwise+regression arena (identifies b_q for arena). "
            "'both-pr': joint static + pairwise+regression. "
            "'z-regression': z-score reward regression only. "
            "'both-z-regression': joint static + z-score reward regression."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-plot", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-static", type=float, default=None)
    parser.add_argument("--lambda-arena", type=float, default=None,
                        help="Weight for arena pairwise loss (pairwise+regression mode).")
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="Weight for arena regression loss (pairwise+regression mode).")
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--bb-ratio", type=float, default=None,
                        help="Target fraction of both-bad pairs. Overrides --both-bad-threshold.")
    parser.add_argument("--tie-ratio", type=float, default=None,
                        help="Target fraction of tie pairs.")
    parser.add_argument("--both-bad-threshold", type=float, default=None,
                        help="Fallback z-score threshold for both-bad pairs when --bb-ratio not set.")
    parser.add_argument("--both-bad-use-zscore", action="store_true")
    parser.add_argument("--both-bad-use-raw", action="store_true")
    parser.add_argument("--save-pairwise-targets", action="store_true")
    parser.add_argument("--arena-mode", default=None,
                        choices=["soft_pairwise", "pairwise+regression", "z-regression"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


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
        args.output_dir = str(_resolve_path(output.get("output_dir", "results/ranking_v1/default")))
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
    if args.lambda_reg is None:
        args.lambda_reg = float(training.get("lambda_reg", 1.0))
    if args.reg_lambda is None:
        args.reg_lambda = float(training.get("reg_lambda", 1e-4))
    if args.bb_ratio is None and training.get("bb_ratio") is not None:
        args.bb_ratio = float(training["bb_ratio"])
    if args.tie_ratio is None and training.get("tie_ratio") is not None:
        args.tie_ratio = float(training["tie_ratio"])
    if args.both_bad_threshold is None:
        args.both_bad_threshold = float(training.get("both_bad_threshold", -0.5))

    if not args.both_bad_use_zscore and not args.both_bad_use_raw:
        both_bad_mode = str(training.get("both_bad_mode", "zscore")).strip().lower()
        args.both_bad_use_zscore = both_bad_mode != "raw"
        args.both_bad_use_raw = both_bad_mode == "raw"

    if args.mode is None and training.get("mode") is not None:
        args.mode = str(training["mode"]).strip().lower()
    if args.arena_mode is None and training.get("arena_mode") is not None:
        args.arena_mode = str(training["arena_mode"]).strip().lower()

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
            f"Conflicting configuration: mode='{args.mode}' implies "
            f"arena_mode='{expected_arena_mode}', but arena_mode='{args.arena_mode}' was provided."
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


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_irt_v1(
    static_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    reward_df: pd.DataFrame | None = None,
    *,
    arena_mode: str = "soft_pairwise",
    num_epochs: int,
    lr: float,
    lambda_static: float,
    lambda_arena: float,
    lambda_reg: float = 1.0,
    reg_lambda: float,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit IRT model using the direct pairwise formulation for arena data.

    arena_mode:
      'soft_pairwise'       — direct pairwise BCE only; arena items have a_q, no b_q.
      'pairwise+regression' — combined BCE + MSE; identifies both a_q and b_q for arena.
      'z-regression'        — MSE regression onto z-scores; arena items have a_q and b_q.

    Returns:
        model_params           — theta per model
        static_question_params — b_q and a_q per static question
        arena_question_params  — a_q (and b_q when identified) per arena question
        metadata
    """
    static = static_df.copy() if static_df is not None else pd.DataFrame()
    use_pairwise = arena_mode in ("soft_pairwise", "pairwise+regression")
    use_regression = arena_mode in ("z-regression", "pairwise+regression")
    pairwise = pairwise_df.copy() if pairwise_df is not None and use_pairwise else pd.DataFrame()
    reward = reward_df.copy() if reward_df is not None and use_regression else pd.DataFrame()

    has_static = not static.empty
    has_pairwise = not pairwise.empty
    has_reward = not reward.empty
    has_arena = has_pairwise or has_reward
    if not has_static and not has_arena:
        raise SystemExit("Need at least one non-empty data source.")

    # --- Build model index (shared across all sources) ---
    model_series = []
    if has_static:
        model_series.append(static["model_name"])
    if has_pairwise:
        model_series.extend([pairwise["model_1"], pairwise["model_2"]])
    if has_reward:
        model_series.append(reward["model_name"])
    all_models = pd.Index(
        pd.unique(pd.concat(model_series, ignore_index=True)), name="model_name"
    )
    model_to_idx = {m: i for i, m in enumerate(all_models)}

    # --- Build separate question indices for static and arena ---
    static_questions: pd.Index = pd.Index([], name="question_id")
    arena_questions: pd.Index = pd.Index([], name="question_id")
    static_q_meta: dict[str, dict[str, str]] = {}
    arena_q_meta: dict[str, dict[str, str]] = {}

    if has_static:
        static_questions = pd.Index(pd.unique(static["question_id"]), name="question_id")
        for _, row in static[["question_id", "source", "benchmark"]].drop_duplicates().iterrows():
            static_q_meta[str(row["question_id"])] = {
                "source": str(row["source"]), "benchmark": str(row["benchmark"]),
            }
        static["m_idx"] = static["model_name"].map(model_to_idx)
        static["qs_idx"] = static["question_id"].map({q: i for i, q in enumerate(static_questions)})

    if has_arena:
        # Arena question index: union of pairwise and reward question IDs
        arena_qids = pd.unique(pd.concat(
            ([pairwise["question_id"]] if has_pairwise else []) +
            ([reward["question_id"]] if has_reward else []),
            ignore_index=True,
        ))
        arena_questions = pd.Index(arena_qids, name="question_id")
        arena_q_to_idx = {q: i for i, q in enumerate(arena_questions)}
        src_df = pairwise if has_pairwise else reward
        for _, row in src_df[["question_id", "source", "benchmark"]].drop_duplicates().iterrows():
            arena_q_meta[str(row["question_id"])] = {
                "source": str(row["source"]), "benchmark": str(row["benchmark"]),
            }
        if has_pairwise:
            pairwise["m1_idx"] = pairwise["model_1"].map(model_to_idx)
            pairwise["m2_idx"] = pairwise["model_2"].map(model_to_idx)
            pairwise["qa_idx"] = pairwise["question_id"].map(arena_q_to_idx)
        if has_reward:
            reward["m_idx"] = reward["model_name"].map(model_to_idx)
            reward["qa_idx"] = reward["question_id"].map(arena_q_to_idx)

    device = _get_device()
    n_models = len(all_models)
    n_static_q = len(static_questions)
    n_arena_q = len(arena_questions)

    # --- Parameters ---
    # Shared ability
    theta = nn.Embedding(n_models, 1, device=device)
    nn.init.zeros_(theta.weight)

    # Static-only: difficulty b_s and log-discrimination k_s
    b_s = nn.Embedding(max(n_static_q, 1), 1, device=device)
    k_s = nn.Embedding(max(n_static_q, 1), 1, device=device)
    nn.init.zeros_(b_s.weight)
    nn.init.zeros_(k_s.weight)

    # Arena: log-discrimination k_a always; b_a when regression signal is present
    arena_has_difficulty = arena_mode in ("z-regression", "pairwise+regression")
    k_a = nn.Embedding(max(n_arena_q, 1), 1, device=device)
    b_a = nn.Embedding(max(n_arena_q, 1), 1, device=device)
    nn.init.zeros_(k_a.weight)
    nn.init.zeros_(b_a.weight)

    opt_params = [*theta.parameters(), *b_s.parameters(), *k_s.parameters(), *k_a.parameters()]
    if arena_has_difficulty:
        opt_params += [*b_a.parameters()]

    optimizer = optim.Adam(opt_params, lr=lr)
    bce_logits = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # --- Static tensors ---
    m_s_t = qs_t = y_s_t = None
    if has_static:
        m_s_t = torch.tensor(static["m_idx"].values, dtype=torch.long, device=device)
        qs_t = torch.tensor(static["qs_idx"].values, dtype=torch.long, device=device)
        y_s_t = torch.tensor(static["judge_result"].values, dtype=torch.float32, device=device)

    # --- Arena tensors: soft pairwise (non-tie pairs only) ---
    m1_t = m2_t = qa_t = soft_t = None
    if has_pairwise:
        hard = pairwise[~pairwise["tie"]].copy()
        m1_t = torch.tensor(hard["m1_idx"].values, dtype=torch.long, device=device)
        m2_t = torch.tensor(hard["m2_idx"].values, dtype=torch.long, device=device)
        qa_t = torch.tensor(hard["qa_idx"].values, dtype=torch.long, device=device)
        soft_t = torch.tensor(hard["target_prob"].values, dtype=torch.float32, device=device)

    # --- Arena tensors: regression (z-regression or pairwise+regression) ---
    m_r_t = qa_r_t = z_t = None
    if has_reward:
        m_r_t = torch.tensor(reward["m_idx"].values, dtype=torch.long, device=device)
        qa_r_t = torch.tensor(reward["qa_idx"].values, dtype=torch.long, device=device)
        z_t = torch.tensor(reward["reward_z"].values, dtype=torch.float32, device=device)

    # --- Training loop ---
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        loss_static = torch.tensor(0.0, device=device)
        if has_static and m_s_t is not None and len(m_s_t):
            a_s = torch.exp(k_s(qs_t).squeeze(-1))
            logits_s = a_s * (theta(m_s_t).squeeze(-1) - b_s(qs_t).squeeze(-1))
            loss_static = bce_logits(logits_s, y_s_t)

        # --- Pairwise loss (soft_pairwise and pairwise+regression modes) ---
        loss_pairwise = torch.tensor(0.0, device=device)
        if m1_t is not None and len(m1_t):
            a_a = torch.exp(k_a(qa_t).squeeze(-1))
            logits_a = a_a * (theta(m1_t).squeeze(-1) - theta(m2_t).squeeze(-1))
            loss_pairwise = bce_logits(logits_a, soft_t)

        # --- Regression loss (z-regression and pairwise+regression modes) ---
        loss_regression = torch.tensor(0.0, device=device)
        if m_r_t is not None and len(m_r_t):
            a_a_r = torch.exp(k_a(qa_r_t).squeeze(-1))
            pred = a_a_r * (theta(m_r_t).squeeze(-1) - b_a(qa_r_t).squeeze(-1))
            loss_regression = mse(pred, z_t)

        # Variance-targeting prior on θ: penalise deviation from unit spread rather
        # than shrinking toward zero.  After zero-mean centering E[θ]=0, so
        # theta.pow(2).mean() == Var(θ), and (Var(θ)-1)² is minimised at Var=1.
        # L2 is kept on log-discriminations (k_a) — those should stay near 0.
        reg = (
            reg_lambda * (theta.weight.pow(2).mean() - 1.0).pow(2)
            + reg_lambda * k_a.weight.pow(2).mean()
        )
        if has_static:
            reg = reg + reg_lambda * (b_s.weight.pow(2).mean() + k_s.weight.pow(2).mean())
        if arena_has_difficulty:
            reg = reg + reg_lambda * b_a.weight.pow(2).mean()

        total = (
            lambda_static * loss_static
            + lambda_arena * loss_pairwise
            + lambda_reg * loss_regression
            + reg
        )
        total.backward()
        optimizer.step()

        # Zero-mean theta; shift difficulty embeddings to stay on the same scale
        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            if has_static:
                b_s.weight.sub_(shift)
            if arena_has_difficulty:
                b_a.weight.sub_(shift)

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(
                f"  Epoch {epoch:5d} | "
                f"static={loss_static.item():.4f}  "
                f"pairwise={loss_pairwise.item():.4f}  "
                f"regression={loss_regression.item():.4f}  "
                f"total={total.item():.4f}"
            )

    # --- Extract parameters ---
    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    model_params = (
        pd.DataFrame({"model_name": all_models, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )

    # Static question params: b and a both meaningful
    static_q_rows: list[dict[str, Any]] = []
    if has_static:
        b_s_np = b_s.weight.detach().cpu().numpy().squeeze(-1)
        k_s_np = k_s.weight.detach().cpu().numpy().squeeze(-1)
        for i, qid in enumerate(static_questions):
            meta = static_q_meta.get(str(qid), {})
            static_q_rows.append({
                "question_id": qid,
                "source": meta.get("source", "unknown"),
                "benchmark": meta.get("benchmark", "unknown"),
                "difficulty_b": b_s_np[i],
                "k_raw": k_s_np[i],
                "discrimination_exp_k": math.exp(k_s_np[i]),
            })
    static_question_params = (
        pd.DataFrame(static_q_rows)
        .sort_values("difficulty_b", ascending=False)
        .reset_index(drop=True)
        if static_q_rows else pd.DataFrame()
    )

    # Arena question params: a_q always; b_q when regression signal was present
    arena_q_rows: list[dict[str, Any]] = []
    if has_arena:
        k_a_np = k_a.weight.detach().cpu().numpy().squeeze(-1)
        b_a_np = b_a.weight.detach().cpu().numpy().squeeze(-1) if arena_has_difficulty else None
        for i, qid in enumerate(arena_questions):
            meta = arena_q_meta.get(str(qid), {})
            row: dict[str, Any] = {
                "question_id": qid,
                "source": meta.get("source", "unknown"),
                "benchmark": meta.get("benchmark", "unknown"),
                "k_raw": k_a_np[i],
                "discrimination_exp_k": math.exp(k_a_np[i]),
            }
            if b_a_np is not None:
                row["difficulty_b"] = b_a_np[i]
            arena_q_rows.append(row)
    arena_question_params = (
        pd.DataFrame(arena_q_rows)
        .sort_values("discrimination_exp_k", ascending=False)
        .reset_index(drop=True)
        if arena_q_rows else pd.DataFrame()
    )

    metadata: dict[str, Any] = {
        "arena_mode": arena_mode,
        "arena_has_difficulty": bool(arena_has_difficulty),
        "n_models": int(n_models),
        "n_static_questions": int(n_static_q),
        "n_arena_questions": int(n_arena_q),
        "has_static": bool(has_static),
        "has_arena": bool(has_arena),
    }
    return model_params, static_question_params, arena_question_params, metadata


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    static_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    model_params: pd.DataFrame,
    static_question_params: pd.DataFrame,
    arena_question_params: pd.DataFrame,
    *,
    arena_mode: str = "soft_pairwise",
    reward_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    theta_map = model_params.set_index("model_name")["theta"]
    metrics: dict[str, Any] = {}

    # Static accuracy
    if static_df is not None and not static_df.empty and not static_question_params.empty:
        sq_map = static_question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
        st = static_df.copy()
        st["theta"] = st["model_name"].map(theta_map)
        st["difficulty_b"] = st["question_id"].map(sq_map["difficulty_b"])
        st["a_q"] = st["question_id"].map(sq_map["discrimination_exp_k"])
        st = st.dropna(subset=["theta", "difficulty_b", "a_q"])
        logits = st["a_q"] * (st["theta"] - st["difficulty_b"])
        metrics["static_accuracy"] = float((logits >= 0).astype(int).eq(st["judge_result"].astype(int)).mean())

    # Arena soft pairwise metrics (soft_pairwise and pairwise+regression modes)
    if arena_mode in ("soft_pairwise", "pairwise+regression") \
            and pairwise_df is not None and not pairwise_df.empty \
            and not arena_question_params.empty:
        aq_map = arena_question_params.set_index("question_id")["discrimination_exp_k"]
        ar = pairwise_df.copy()
        ar["theta_1"] = ar["model_1"].map(theta_map)
        ar["theta_2"] = ar["model_2"].map(theta_map)
        ar["a_q"] = ar["question_id"].map(aq_map)
        ar = ar.dropna(subset=["theta_1", "theta_2", "a_q"])

        logits = ar["a_q"].to_numpy() * (ar["theta_1"].to_numpy() - ar["theta_2"].to_numpy())
        pred_prob = 1.0 / (1.0 + np.exp(-logits))
        target = ar["target_prob"].to_numpy()

        eps = 1e-6
        logloss = -np.mean(
            target * np.log(pred_prob + eps) + (1.0 - target) * np.log(1.0 - pred_prob + eps)
        )
        metrics["arena_soft_logloss"] = float(logloss)
        metrics["arena_both_bad_pairs"] = int(ar["both_bad"].sum())

        tie_mask = ar["tie"].to_numpy(dtype=bool) if "tie" in ar.columns else np.isclose(target, 0.5)
        hard_eval_mask = (~ar["both_bad"].to_numpy(dtype=bool)) & (~tie_mask)
        metrics["arena_tie_pairs"] = int(tie_mask.sum())
        metrics["arena_hard_eval_pairs"] = int(hard_eval_mask.sum())
        if hard_eval_mask.any():
            metrics["arena_hard_pair_accuracy"] = float(
                ((pred_prob[hard_eval_mask] >= 0.5) == (target[hard_eval_mask] >= 0.5)).mean()
            )
        else:
            metrics["arena_hard_pair_accuracy"] = float("nan")

    # Regression metrics (z-regression and pairwise+regression modes)
    if arena_mode in ("z-regression", "pairwise+regression") \
            and reward_df is not None and not reward_df.empty \
            and not arena_question_params.empty \
            and "difficulty_b" in arena_question_params.columns:
        aq_map = arena_question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
        rw = reward_df.copy()
        rw["theta"] = rw["model_name"].map(theta_map)
        rw["difficulty_b"] = rw["question_id"].map(aq_map["difficulty_b"])
        rw["a_q"] = rw["question_id"].map(aq_map["discrimination_exp_k"])
        rw = rw.dropna(subset=["theta", "difficulty_b", "a_q"])
        pred = rw["a_q"] * (rw["theta"] - rw["difficulty_b"])
        residuals = pred - rw["reward_z"]
        metrics["arena_z_reward_mse"] = float((residuals ** 2).mean())
        metrics["arena_z_reward_mae"] = float(residuals.abs().mean())

    return metrics


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_difficulty_and_ability(
    model_params: pd.DataFrame,
    static_question_params: pd.DataFrame,
    *,
    arena_question_params: pd.DataFrame | None = None,
    save_path: str | None,
    title: str,
) -> None:
    """Plot model abilities and item difficulties on the shared latent scale.

    Static items always have b_q.  Arena items are included only when
    difficulty_b is present (i.e. pairwise+regression or z-regression mode).
    """
    # Build the combined question frame: static + arena items that have b_q.
    frames = []
    if not static_question_params.empty:
        frames.append(static_question_params[["benchmark", "difficulty_b"]])
    if arena_question_params is not None and not arena_question_params.empty \
            and "difficulty_b" in arena_question_params.columns:
        frames.append(arena_question_params[["benchmark", "difficulty_b"]])
    if not frames or model_params.empty or not save_path:
        return
    qr = pd.concat(frames, ignore_index=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    benchmarks = sorted(qr["benchmark"].fillna("unknown").unique())
    colors = plt.cm.tab10.colors[: max(1, len(benchmarks))]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bins = np.linspace(qr["difficulty_b"].min(), qr["difficulty_b"].max(), 41)
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)
    bottoms = np.zeros(len(centers))

    for idx, benchmark in enumerate(benchmarks):
        counts, _ = np.histogram(
            qr.loc[qr["benchmark"] == benchmark, "difficulty_b"], bins=bins
        )
        ax.bar(
            centers, counts, width=widths, bottom=bottoms,
            color=colors[idx % len(colors)], alpha=0.85, align="center",
            edgecolor="white", linewidth=0.3, label=benchmark,
        )
        bottoms += counts

    y_top = max(1.0, bottoms.max() * 1.5)
    ax.set_ylim(0, y_top)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.35)

    model_colors = plt.cm.Dark2.colors
    benchmark_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=6, label=b)
        for idx, b in enumerate(benchmarks)
    ]
    model_handles: list[Line2D] = []
    for idx, (_, row) in enumerate(model_params.iterrows()):
        color = model_colors[idx % len(model_colors)]
        ax.axvline(row["theta"], 0, 0.85, color=color, linestyle="--", linewidth=1.3, alpha=0.9)
        ax.text(
            row["theta"], y_top * (0.88 + 0.03 * (idx % 2)),
            f"{idx + 1}", ha="center", va="bottom", fontsize=8, color=color,
        )
        model_handles.append(Line2D(
            [0], [0], color=color, linestyle="--", marker="o", markersize=5,
            label=f"{idx + 1}. {row['model_name']}",
        ))

    ax.set_xlabel(r"Latent scale (ability $\theta$ / difficulty $b$)")
    ax.set_ylabel("Number of questions")
    ax.set_title(title)

    n_model_cols = 2 if len(model_handles) > 8 else 1
    benchmark_legend = ax.legend(
        handles=benchmark_handles, title="Question Sources", frameon=True,
        framealpha=0.9, loc="upper left", bbox_to_anchor=(0.0, -0.14),
        bbox_transform=ax.transAxes, borderaxespad=0.0, fontsize=8, title_fontsize=9,
    )
    ax.add_artist(benchmark_legend)
    ax.legend(
        handles=model_handles, title="Model Rankings", frameon=True, framealpha=0.9,
        loc="upper left", bbox_to_anchor=(0.42, -0.14), bbox_transform=ax.transAxes,
        borderaxespad=0.0, ncol=n_model_cols, fontsize=8, title_fontsize=9,
    )

    plt.tight_layout()
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_obj, dpi=500, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            if args.arena_mode in ("soft_pairwise", "pairwise+regression"):
                both_bad_threshold, tie_delta = resolve_pairwise_thresholds(
                    reward_df,
                    bb_ratio=args.bb_ratio,
                    tie_ratio=args.tie_ratio,
                    both_bad_threshold=args.both_bad_threshold,
                    both_bad_use_zscore=args.both_bad_use_zscore,
                )
                if args.bb_ratio is not None:
                    print(f"  bb_ratio={args.bb_ratio} → both_bad_threshold={both_bad_threshold:.4f}")
                if args.tie_ratio is not None:
                    print(f"  tie_ratio={args.tie_ratio} → tie_delta={tie_delta:.4f}")
                pairwise_df = build_soft_pairwise_targets(
                    reward_df,
                    both_bad_threshold=both_bad_threshold,
                    both_bad_use_zscore=args.both_bad_use_zscore,
                    tie_delta=tie_delta,
                )
                print(
                    f"  distilled into {len(pairwise_df)} soft pairwise rows; "
                    f"both_bad={int(pairwise_df['both_bad'].sum())}  "
                    f"tie={int(pairwise_df['tie'].sum())}"
                )
                if args.arena_mode == "pairwise+regression":
                    print(f"  pairwise+regression: reward_df also used for regression loss")
            else:
                print("  arena_mode=z-regression: using globally z-scored rewards")
    elif args.arena_reward_jsonl:
        print(f"Note: ignoring arena_reward_jsonl because mode='{args.mode}' does not use arena data.")

    has_arena_data = not pairwise_df.empty or not reward_df.empty
    if static_df.empty and not has_arena_data:
        raise SystemExit("No usable static or arena-reward data loaded.")

    print(f"\nFitting IRT v1 model (mode={args.mode}, arena_mode={args.arena_mode}) ...")
    model_params, static_qp, arena_qp, fit_meta = fit_irt_v1(
        static_df if not static_df.empty else None,
        pairwise_df if not pairwise_df.empty else None,
        reward_df if not reward_df.empty else None,
        arena_mode=args.arena_mode,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=args.lambda_static,
        lambda_arena=args.lambda_arena,
        lambda_reg=args.lambda_reg,
        reg_lambda=args.reg_lambda,
        verbose=verbose,
    )

    metrics = compute_metrics(
        static_df if not static_df.empty else None,
        pairwise_df if not pairwise_df.empty else None,
        model_params,
        static_qp,
        arena_qp,
        arena_mode=args.arena_mode,
        reward_df=reward_df if not reward_df.empty else None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_params.to_csv(output_dir / "model_ranking.csv", index=False)
    if not static_qp.empty:
        static_qp.to_csv(output_dir / "static_question_params.csv", index=False)
    if not arena_qp.empty:
        arena_qp.to_csv(output_dir / "arena_question_params.csv", index=False)
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
        "lambda_reg": args.lambda_reg,
        "reg_lambda": args.reg_lambda,
        "both_bad_threshold": args.both_bad_threshold,
        "both_bad_mode": "zscore" if args.both_bad_use_zscore else "raw",
        "bb_ratio": args.bb_ratio,
        "tie_ratio": args.tie_ratio,
        "save_plot": args.save_plot,
        "metrics": metrics,
        **fit_meta,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    print("\nModel ranking (highest ability first):")
    print(model_params.to_string(index=False))
    if not static_qp.empty:
        print(f"\nStatic question difficulty (top 20 hardest of {len(static_qp)}):")
        print(static_qp.head(20).to_string(index=False))
    if not arena_qp.empty:
        print(f"\nArena question discrimination (top 20 most discriminating of {len(arena_qp)}):")
        print(arena_qp.head(20).to_string(index=False))
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if not args.no_plot and args.save_plot:
        if args.mode == "static":
            title = "Static IRT Ranking (v1)"
        elif args.arena_mode == "soft_pairwise":
            title = "Direct Pairwise IRT Ranking (v1)"
        elif args.arena_mode == "pairwise+regression":
            title = "Pairwise+Regression IRT Ranking (v1)"
        else:
            title = "Z-Score Reward Regression IRT Ranking (v1)"
        plot_difficulty_and_ability(
            model_params,
            static_qp,
            arena_question_params=arena_qp if not arena_qp.empty else None,
            save_path=args.save_plot,
            title=title,
        )
        print(f"\nSaved plot to {args.save_plot}")

    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
