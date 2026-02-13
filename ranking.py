#!/usr/bin/env python3
"""
Unified IRT-based ranking of LLMs using static benchmarks, arena pairwise
preferences, or both jointly.

Usage examples:
    # Static-only 2PL IRT
    python ranking.py --mode static --static-csv ./data/static_10_models.csv

    # Arena-only pairwise IRT
    python ranking.py --mode arena --arena-csv ./data/pairwise_results_900.csv

    # Joint model (static + arena)
    python ranking.py --mode both \\
        --static-csv ./data/static_10_models.csv \\
        --arena-csv ./data/pairwise_results_900.csv

    # Customise training and evaluation
    python ranking.py --mode both \\
        --static-csv ./data/static_10_models.csv \\
        --arena-csv ./data/pairwise_results_900.csv \\
        --num-epochs 5000 --lr 0.05 --lambda-bb 0.3 \\
        --evaluate --save-plot ./figures/ranking.pdf
"""

from __future__ import annotations

import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


# =====================================================================
# Data loading helpers
# =====================================================================

def load_arena_pairs(csv_path: str) -> pd.DataFrame:
    """
    Load an Arena-style CSV where each column ``<model_a>_vs_<model_b>``
    holds values in {0, 1, 2, 3}:
        0 = model_a wins, 1 = model_b wins, 2 = tie, 3 = both_bad.

    Returns a long-format DataFrame with columns:
        question_id, model_1, model_2, label
    """
    df_wide = pd.read_csv(csv_path)
    pair_cols = [c for c in df_wide.columns if "_vs_" in c]

    rows: list[dict] = []
    for _, row in df_wide.iterrows():
        qid = row["id"]
        for col in pair_cols:
            val = row[col]
            if pd.isna(val):
                continue
            model_a, model_b = col.split("_vs_", 1)
            rows.append(
                {
                    "question_id": qid,
                    "model_1": model_a,
                    "model_2": model_b,
                    "label": int(val),
                }
            )

    arena_pairs = pd.DataFrame(rows)
    # Normalise common model-name variants
    arena_pairs[["model_1", "model_2"]] = arena_pairs[["model_1", "model_2"]].replace(
        {
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
            "deepseek-chat": "deepseek-v3-0324",
            "llama-3.3-70b-instruct": "llama-3.3-70b-it",
            "gemini-2.0-flash-001": "gemini-2.0-flash",
        }
    )
    return arena_pairs


def build_pairwise_from_static(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-(model, question) correctness labels into pairwise
    comparisons with outcomes: m1_win, m2_win, tie, both_bad.
    """
    rows: list[dict] = []
    for qid, group in static_df.groupby("question_id"):
        for (_, r1), (_, r2) in combinations(group.iterrows(), 2):
            y1, y2 = r1["judge_result"], r2["judge_result"]
            if y1 == 1 and y2 == 0:
                outcome = "m1_win"
            elif y1 == 0 and y2 == 1:
                outcome = "m2_win"
            elif y1 == 1 and y2 == 1:
                outcome = "tie"
            else:
                outcome = "both_bad"
            rows.append(
                {
                    "question_id": qid,
                    "model_1": r1["model_name"],
                    "model_2": r2["model_name"],
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


# =====================================================================
# Core IRT fitting functions
# =====================================================================

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_static_irt(
    static_df: pd.DataFrame,
    *,
    num_epochs: int = 2000,
    lr: float = 0.05,
    reg_lambda: float = 1e-4,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    2-parameter logistic IRT on static benchmark correctness data.

    Model:  logit P(correct_{i,q}) = a_q * (theta_i - b_q),  a_q = exp(k_q).

    Returns (model_params, question_params) DataFrames.
    """
    df = static_df.copy()
    df["model_cat"] = df["model_name"].astype("category")
    df["question_cat"] = df["question_id"].astype("category")
    df["model_idx"] = df["model_cat"].cat.codes
    df["question_idx"] = df["question_cat"].cat.codes

    n_models = df["model_idx"].nunique()
    n_questions = df["question_idx"].nunique()

    device = _get_device()
    m_idx = torch.tensor(df["model_idx"].values, dtype=torch.long, device=device)
    q_idx = torch.tensor(df["question_idx"].values, dtype=torch.long, device=device)
    y = torch.tensor(df["judge_result"].values, dtype=torch.float32, device=device)

    theta = nn.Embedding(n_models, 1, device=device)
    b = nn.Embedding(n_questions, 1, device=device)
    k = nn.Embedding(n_questions, 1, device=device)
    nn.init.zeros_(theta.weight)
    nn.init.zeros_(b.weight)
    nn.init.zeros_(k.weight)

    optimizer = optim.Adam(
        list(theta.parameters()) + list(b.parameters()) + list(k.parameters()),
        lr=lr,
    )
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        theta_i = theta(m_idx).squeeze(-1)
        b_q = b(q_idx).squeeze(-1)
        k_q = k(q_idx).squeeze(-1)
        a_q = torch.exp(k_q)

        logits = a_q * (theta_i - b_q)
        loss = bce(logits, y) + reg_lambda * (
            theta.weight.pow(2).mean()
            + b.weight.pow(2).mean()
            + k.weight.pow(2).mean()
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            b.weight.sub_(shift)

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(f"  Epoch {epoch:5d} | loss = {loss.item():.4f}")

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    b_np = b.weight.detach().cpu().numpy().squeeze(-1)
    k_np = k.weight.detach().cpu().numpy().squeeze(-1)

    model_params = (
        pd.DataFrame(
            {"model_name": df["model_cat"].cat.categories, "theta": theta_np}
        )
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )
    question_params = (
        pd.DataFrame(
            {
                "question_id": df["question_cat"].cat.categories,
                "difficulty_b": b_np,
                "k_raw": k_np,
                "discrimination_exp_k": np.exp(k_np),
            }
        )
        .sort_values("difficulty_b", ascending=False)
        .reset_index(drop=True)
    )
    return model_params, question_params


def fit_arena_irt(
    arena_df: pd.DataFrame,
    *,
    num_epochs: int = 2000,
    lr: float = 0.05,
    lambda_tie: float = 0.0,
    lambda_bb: float = 1.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise IRT on arena-style pairwise logs.

    Model:
        pi_{i,q} = sigmoid(theta_i - b_q)
        P(i > j | q) = sigmoid(exp(k_q) * (pi_{i,q} - pi_{j,q}))
    """
    df = arena_df.copy()
    label_map = {0: "m1_win", 1: "m2_win", 2: "tie", 3: "both_bad"}
    df["outcome"] = df["label"].map(label_map)

    df_win = df[df["outcome"] == "m1_win"].copy()
    df_loss = df[df["outcome"] == "m2_win"].copy()
    df_tie = df[df["outcome"] == "tie"].copy()
    df_bb = df[df["outcome"] == "both_bad"].copy()

    df_win["label_bin"] = 1.0
    df_loss["label_bin"] = 0.0
    df_dec = pd.concat([df_win, df_loss], axis=0)

    model_ids = pd.Index(
        pd.unique(pd.concat([df["model_1"], df["model_2"]])), name="model_name"
    )
    question_ids = pd.Index(pd.unique(df["question_id"]), name="question_id")
    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    q_to_idx = {q: i for i, q in enumerate(question_ids)}

    def _map(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["m1_idx"] = sub["model_1"].map(model_to_idx)
        sub["m2_idx"] = sub["model_2"].map(model_to_idx)
        sub["q_idx"] = sub["question_id"].map(q_to_idx)
        return sub

    df_dec = _map(df_dec)
    df_tie = _map(df_tie) if len(df_tie) else df_tie
    df_bb = _map(df_bb) if len(df_bb) else df_bb

    device = _get_device()
    m1_dec = torch.tensor(df_dec["m1_idx"].values).long().to(device)
    m2_dec = torch.tensor(df_dec["m2_idx"].values).long().to(device)
    q_dec = torch.tensor(df_dec["q_idx"].values).long().to(device)
    y_dec = torch.tensor(df_dec["label_bin"].values).float().to(device)

    m1_tie = m2_tie = q_tie = None
    if len(df_tie):
        m1_tie = torch.tensor(df_tie["m1_idx"].values).long().to(device)
        m2_tie = torch.tensor(df_tie["m2_idx"].values).long().to(device)
        q_tie = torch.tensor(df_tie["q_idx"].values).long().to(device)

    m1_bb = m2_bb = q_bb = None
    if len(df_bb):
        m1_bb = torch.tensor(df_bb["m1_idx"].values).long().to(device)
        m2_bb = torch.tensor(df_bb["m2_idx"].values).long().to(device)
        q_bb = torch.tensor(df_bb["q_idx"].values).long().to(device)

    n_models = len(model_ids)
    n_questions = len(question_ids)

    theta = nn.Embedding(n_models, 1, device=device)
    b = nn.Embedding(n_questions, 1, device=device)
    k = nn.Embedding(n_questions, 1, device=device)
    nn.init.zeros_(theta.weight)
    nn.init.zeros_(b.weight)
    nn.init.zeros_(k.weight)

    optimizer = optim.Adam(
        list(theta.parameters()) + list(b.parameters()) + list(k.parameters()),
        lr=lr,
    )
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Decisive
        t1 = theta(m1_dec).squeeze(-1)
        t2 = theta(m2_dec).squeeze(-1)
        bq = b(q_dec).squeeze(-1)
        kq = k(q_dec).squeeze(-1)
        aq = torch.exp(kq)
        z = aq * (torch.sigmoid(t1 - bq) - torch.sigmoid(t2 - bq))
        loss_dec = bce(z, y_dec)

        # Ties
        loss_tie = torch.tensor(0.0, device=device)
        if lambda_tie > 0.0 and m1_tie is not None:
            t1t = theta(m1_tie).squeeze(-1)
            t2t = theta(m2_tie).squeeze(-1)
            bqt = b(q_tie).squeeze(-1)
            kqt = k(q_tie).squeeze(-1)
            zt = torch.exp(kqt) * (torch.sigmoid(t1t - bqt) - torch.sigmoid(t2t - bqt))
            loss_tie = bce(zt, torch.full_like(zt, 0.5))

        # Both-bad
        loss_bb = torch.tensor(0.0, device=device)
        if lambda_bb > 0.0 and m1_bb is not None:
            t1b = theta(m1_bb).squeeze(-1)
            t2b = theta(m2_bb).squeeze(-1)
            bqb = b(q_bb).squeeze(-1)
            pi1 = torch.sigmoid(t1b - bqb)
            pi2 = torch.sigmoid(t2b - bqb)
            loss_bb = -(torch.log(1 - pi1 + 1e-6).mean() + torch.log(1 - pi2 + 1e-6).mean())

        reg = 1e-4 * (
            theta.weight.pow(2).mean()
            + b.weight.pow(2).mean()
            + k.weight.pow(2).mean()
        )

        total = loss_dec + lambda_tie * loss_tie + lambda_bb * loss_bb + reg
        total.backward()
        optimizer.step()

        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            b.weight.sub_(shift)

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(
                f"  Epoch {epoch:5d} | "
                f"dec={loss_dec.item():.4f}  "
                f"bb={loss_bb.item():.4f}  "
                f"total={total.item():.4f}"
            )

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    b_np = b.weight.detach().cpu().numpy().squeeze(-1)
    k_np = k.weight.detach().cpu().numpy().squeeze(-1)

    model_params = (
        pd.DataFrame({"model_name": model_ids, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )
    question_params = (
        pd.DataFrame(
            {
                "question_id": question_ids,
                "difficulty_b": b_np,
                "k_raw": k_np,
                "discrimination_exp_k": np.exp(k_np),
            }
        )
        .sort_values("difficulty_b", ascending=False)
        .reset_index(drop=True)
    )
    return model_params, question_params


def fit_joint_irt(
    static_df: pd.DataFrame,
    arena_pairs_df: pd.DataFrame,
    *,
    num_epochs: int = 2000,
    lr: float = 0.02,
    lambda_static: float = 1.0,
    lambda_arena: float = 1.0,
    lambda_tie: float = 0.0,
    lambda_bb: float = 1.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Joint 2PL-IRT (static) + pairwise-IRT (arena) model.

    Static:   P(correct_{i,q}) = sigmoid(a_q * (theta_i - b_q))
    Arena:    P(i > j | q)     = sigmoid(a_q * (pi_{i,q} - pi_{j,q}))
              with pi_{i,q}    = sigmoid(theta_i - b_q),  a_q = exp(k_q)
    """
    static = static_df.copy()
    arena = arena_pairs_df.copy()

    label_map = {0: "m1_win", 1: "m2_win", 2: "tie", 3: "both_bad"}
    arena["outcome"] = arena["label"].map(label_map)

    df_win = arena[arena["outcome"] == "m1_win"].copy()
    df_loss = arena[arena["outcome"] == "m2_win"].copy()
    df_tie = arena[arena["outcome"] == "tie"].copy()
    df_bb = arena[arena["outcome"] == "both_bad"].copy()

    df_win["label_bin"] = 1.0
    df_loss["label_bin"] = 0.0
    df_dec = pd.concat([df_win, df_loss], axis=0)

    # Unified model/question indices across both data sources
    all_models = pd.Index(
        pd.unique(
            pd.concat([static["model_name"], arena["model_1"], arena["model_2"]], ignore_index=True)
        ),
        name="model_name",
    )
    all_questions = pd.Index(
        pd.unique(
            pd.concat([static["question_id"], arena["question_id"]], ignore_index=True)
        ),
        name="question_id",
    )

    model_to_idx = {m: i for i, m in enumerate(all_models)}
    q_to_idx = {q: i for i, q in enumerate(all_questions)}

    static = static[
        static["model_name"].isin(all_models) & static["question_id"].isin(all_questions)
    ].copy()
    static["m_idx"] = static["model_name"].map(model_to_idx)
    static["q_idx"] = static["question_id"].map(q_to_idx)

    def _map(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["m1_idx"] = sub["model_1"].map(model_to_idx)
        sub["m2_idx"] = sub["model_2"].map(model_to_idx)
        sub["q_idx"] = sub["question_id"].map(q_to_idx)
        return sub

    df_dec = _map(df_dec)
    df_tie = _map(df_tie) if len(df_tie) else df_tie
    df_bb = _map(df_bb) if len(df_bb) else df_bb

    device = _get_device()

    # Static tensors
    m_s = torch.tensor(static["m_idx"].values).long().to(device)
    q_s = torch.tensor(static["q_idx"].values).long().to(device)
    y_s = torch.tensor(static["judge_result"].values).float().to(device)

    # Arena decisive tensors
    m1_dec_t = torch.tensor(df_dec["m1_idx"].values).long().to(device)
    m2_dec_t = torch.tensor(df_dec["m2_idx"].values).long().to(device)
    q_dec_t = torch.tensor(df_dec["q_idx"].values).long().to(device)
    y_dec_t = torch.tensor(df_dec["label_bin"].values).float().to(device)

    # Tie tensors
    m1_tie = m2_tie = q_tie_t = None
    if len(df_tie):
        m1_tie = torch.tensor(df_tie["m1_idx"].values).long().to(device)
        m2_tie = torch.tensor(df_tie["m2_idx"].values).long().to(device)
        q_tie_t = torch.tensor(df_tie["q_idx"].values).long().to(device)

    # Both-bad tensors
    m1_bb = m2_bb = q_bb_t = None
    if len(df_bb):
        m1_bb = torch.tensor(df_bb["m1_idx"].values).long().to(device)
        m2_bb = torch.tensor(df_bb["m2_idx"].values).long().to(device)
        q_bb_t = torch.tensor(df_bb["q_idx"].values).long().to(device)

    n_models = len(all_models)
    n_questions = len(all_questions)

    theta = nn.Embedding(n_models, 1, device=device)
    b = nn.Embedding(n_questions, 1, device=device)
    k = nn.Embedding(n_questions, 1, device=device)
    nn.init.zeros_(theta.weight)
    nn.init.zeros_(b.weight)
    nn.init.zeros_(k.weight)

    optimizer = optim.Adam(
        list(theta.parameters()) + list(b.parameters()) + list(k.parameters()),
        lr=lr,
    )
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # -- Static 2PL --
        ts = theta(m_s).squeeze(-1)
        bs = b(q_s).squeeze(-1)
        ks = k(q_s).squeeze(-1)
        logits_s = torch.exp(ks) * (ts - bs)
        loss_static = bce(logits_s, y_s)

        # -- Arena decisive --
        t1 = theta(m1_dec_t).squeeze(-1)
        t2 = theta(m2_dec_t).squeeze(-1)
        bq = b(q_dec_t).squeeze(-1)
        kq = k(q_dec_t).squeeze(-1)
        aq = torch.exp(kq)
        pi1 = torch.sigmoid(t1 - bq)
        pi2 = torch.sigmoid(t2 - bq)
        loss_dec = bce(aq * (pi1 - pi2), y_dec_t)

        # -- Ties --
        loss_tie = torch.tensor(0.0, device=device)
        if lambda_tie > 0.0 and m1_tie is not None and len(m1_tie):
            t1t = theta(m1_tie).squeeze(-1)
            t2t = theta(m2_tie).squeeze(-1)
            bqt = b(q_tie_t).squeeze(-1)
            kqt = k(q_tie_t).squeeze(-1)
            zt = torch.exp(kqt) * (
                torch.sigmoid(t1t - bqt) - torch.sigmoid(t2t - bqt)
            )
            loss_tie = bce(zt, torch.full_like(zt, 0.5))

        # -- Both-bad --
        loss_bb = torch.tensor(0.0, device=device)
        if lambda_bb > 0.0 and m1_bb is not None and len(m1_bb):
            t1b = theta(m1_bb).squeeze(-1)
            t2b = theta(m2_bb).squeeze(-1)
            bqb = b(q_bb_t).squeeze(-1)
            p1 = torch.sigmoid(t1b - bqb)
            p2 = torch.sigmoid(t2b - bqb)
            loss_bb = -(torch.log(1 - p1 + 1e-6).mean() + torch.log(1 - p2 + 1e-6).mean())

        # -- Regularisation --
        reg = 1e-4 * (
            theta.weight.pow(2).mean()
            + b.weight.pow(2).mean()
            + k.weight.pow(2).mean()
        )

        total = (
            lambda_static * loss_static
            + lambda_arena * (loss_dec + lambda_tie * loss_tie + lambda_bb * loss_bb)
            + reg
        )
        total.backward()
        optimizer.step()

        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            b.weight.sub_(shift)

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(
                f"  Epoch {epoch:5d} | "
                f"static={loss_static.item():.4f}  "
                f"dec={loss_dec.item():.4f}  "
                f"tie={loss_tie.item():.4f}  "
                f"bb={loss_bb.item():.4f}  "
                f"total={total.item():.4f}"
            )

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    b_np = b.weight.detach().cpu().numpy().squeeze(-1)
    k_np = k.weight.detach().cpu().numpy().squeeze(-1)

    model_params = (
        pd.DataFrame({"model_name": all_models, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )
    question_params = (
        pd.DataFrame(
            {
                "question_id": all_questions,
                "difficulty_b": b_np,
                "k_raw": k_np,
                "discrimination_exp_k": np.exp(k_np),
            }
        )
        .sort_values("difficulty_b", ascending=False)
        .reset_index(drop=True)
    )
    return model_params, question_params


# =====================================================================
# Evaluation helpers
# =====================================================================

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_static_agreement(
    static_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
) -> pd.DataFrame:
    """Compare IRT predictions (theta_i >= b_q) with actual correctness."""
    theta_map = model_params.set_index("model_name")["theta"]
    b_map = question_params.set_index("question_id")["difficulty_b"]

    df = static_df.copy()
    df = df[df["model_name"].isin(theta_map.index) & df["question_id"].isin(b_map.index)].copy()
    df["theta"] = df["model_name"].map(theta_map)
    df["b"] = df["question_id"].map(b_map)
    df["pred_correct"] = (df["theta"] >= df["b"]).astype(int)
    df["judge_result"] = df["judge_result"].astype(int)

    summary = (
        df.groupby("model_name")
        .apply(
            lambda g: pd.Series(
                {
                    "n_questions": len(g),
                    "agreement_pct": 100.0 * (g["pred_correct"] == g["judge_result"]).mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    total = summary["n_questions"].sum()
    weighted = (summary["agreement_pct"] * summary["n_questions"]).sum() / total
    summary = pd.concat(
        [summary, pd.DataFrame({"model_name": ["__mean__"], "n_questions": [total], "agreement_pct": [weighted]})],
        ignore_index=True,
    )
    return summary


def compute_rank_correlations(
    mp1: pd.DataFrame,
    qp1: pd.DataFrame,
    mp2: pd.DataFrame,
    qp2: pd.DataFrame,
) -> dict:
    """Spearman rank correlations between two fitted models."""
    df_m = mp1[["model_name", "theta"]].merge(mp2[["model_name", "theta"]], on="model_name", suffixes=("_1", "_2"))
    rho_m, _ = spearmanr(df_m["theta_1"], df_m["theta_2"])

    df_q = qp1[["question_id", "difficulty_b"]].merge(
        qp2[["question_id", "difficulty_b"]], on="question_id", suffixes=("_1", "_2")
    )
    rho_q = np.nan
    if len(df_q) > 1:
        rho_q, _ = spearmanr(df_q["difficulty_b_1"], df_q["difficulty_b_2"])

    return {"model_spearman_rho": rho_m, "question_spearman_rho": rho_q}


def evaluate_joint_model(
    static_df: pd.DataFrame | None,
    arena_pairs_df: pd.DataFrame | None,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    tie_margin: float = 0.05,
    both_bad_thresh: float = 0.2,
    plot_heatmap: bool = False,
    save_dir: str | None = None,
) -> dict:
    """Compute accuracy and confusion matrices for the fitted model."""
    theta_map = model_params.set_index("model_name")["theta"]
    b_map = question_params.set_index("question_id")["difficulty_b"]
    results: dict = {}

    # -- Static evaluation --
    if static_df is not None:
        st = static_df.copy()
        st["theta"] = st["model_name"].map(theta_map)
        st["b"] = st["question_id"].map(b_map)
        st = st.dropna(subset=["theta", "b", "judge_result"])
        st["pred"] = (st["theta"] >= st["b"]).astype(int)
        results["static_accuracy"] = float((st["pred"] == st["judge_result"].astype(int)).mean())

        if plot_heatmap:
            cm = pd.crosstab(st["judge_result"].astype(int), st["pred"], rownames=["true"], colnames=["pred"])
            cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=["incorrect", "correct"], yticklabels=["incorrect", "correct"])
            plt.title("Static confusion (row-normalised)")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/confusion_static.pdf", dpi=500, bbox_inches="tight")
            plt.show()

    # -- Arena evaluation --
    if arena_pairs_df is not None:
        ar = arena_pairs_df.copy()
        ar["theta_1"] = ar["model_1"].map(theta_map)
        ar["theta_2"] = ar["model_2"].map(theta_map)
        ar["b"] = ar["question_id"].map(b_map)
        ar = ar.dropna(subset=["theta_1", "theta_2", "b", "label"])
        ar["label"] = ar["label"].astype(int)

        pi1 = _sigmoid_np(ar["theta_1"].to_numpy() - ar["b"].to_numpy())
        pi2 = _sigmoid_np(ar["theta_2"].to_numpy() - ar["b"].to_numpy())
        true_l = ar["label"].to_numpy()
        pred_l = np.zeros_like(true_l)

        bb_mask = (pi1 < both_bad_thresh) & (pi2 < both_bad_thresh)
        pred_l[bb_mask] = 3
        tie_mask = (~bb_mask) & (np.abs(pi1 - pi2) < tie_margin)
        pred_l[tie_mask] = 2
        rest = ~(bb_mask | tie_mask)
        pred_l[rest & (pi1 > pi2)] = 0
        pred_l[rest & (pi2 >= pi1)] = 1

        non_tie = true_l != 2
        results["arena_accuracy"] = float((pred_l[non_tie] == true_l[non_tie]).mean()) if non_tie.any() else np.nan

        if plot_heatmap:
            labels_3 = [0, 1, 3]
            cm4 = pd.crosstab(pd.Series(true_l, name="true"), pd.Series(pred_l, name="pred"),
                              rownames=["true"], colnames=["pred"], dropna=False)
            cm4 = cm4.reindex(index=[0, 1, 2, 3], columns=[0, 1, 2, 3], fill_value=0)
            cm3 = cm4.loc[labels_3, labels_3]
            cm3_norm = cm3.div(cm3.sum(axis=1).replace(0, np.nan), axis=0)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm3_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=["m1_win", "m2_win", "both_bad"],
                        yticklabels=["m1_win", "m2_win", "both_bad"])
            plt.title("Arena confusion (row-normalised, no ties)")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/confusion_arena.pdf", dpi=500, bbox_inches="tight")
            plt.show()

    return results


# =====================================================================
# Plotting
# =====================================================================

def plot_difficulty_and_ability(
    model_ranking: pd.DataFrame,
    question_ranking: pd.DataFrame,
    *,
    bins: int = 40,
    save_path: str | None = None,
    title: str | None = None,
) -> None:
    """
    Stacked histogram of question difficulties (by benchmark) with model
    abilities overlaid as dashed vertical lines.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    qr = question_ranking.copy()

    if "benchmark" not in qr.columns:
        qid_str = qr["question_id"].astype(str)
        has_us = qid_str.str.contains("_")
        bench_static = qid_str.str.rsplit("_", n=1).str[0]
        qr["benchmark"] = np.where(has_us, bench_static, "Arena")

    benchmarks = sorted(qr["benchmark"].unique())
    bench_colors = plt.cm.tab10.colors[: len(benchmarks)]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    b_vals = qr["difficulty_b"].values
    if len(b_vals) == 0:
        raise ValueError("question_ranking is empty.")

    bin_edges = np.linspace(b_vals.min(), b_vals.max(), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    hist_data = np.array([
        np.histogram(qr.loc[qr["benchmark"] == bmk, "difficulty_b"].values, bins=bin_edges)[0]
        for bmk in benchmarks
    ])

    bottoms = np.zeros_like(hist_data[0])
    bench_handles = []
    for i, bmk in enumerate(benchmarks):
        ax.bar(bin_centers, hist_data[i], width=bin_widths, bottom=bottoms,
               color=bench_colors[i], alpha=0.85, align="center", edgecolor="white", linewidth=0.3)
        bottoms += hist_data[i]
        bench_handles.append(Line2D([0], [0], color=bench_colors[i], lw=6, label=bmk))

    y_max = bottoms.max() if len(bottoms) else 1.0
    ax.set_ylim(0, max(1.0, y_max * 1.7))
    ax.axvline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.4, zorder=0)

    mr = model_ranking.sort_values("theta", ascending=False).reset_index(drop=True)

    def _suffix(i: int) -> str:
        return "st" if i == 0 else "nd" if i == 1 else "rd" if i == 2 else "th"

    model_colors = plt.cm.Dark2.colors[: len(mr)]
    model_handles = []
    total_ylim = max(1.0, y_max * 1.7)
    vline_top = y_max * 1.05 / total_ylim
    marker_y = y_max * 1.08
    base_text_y = y_max * 1.18
    text_offset = y_max * 0.04

    for i, row in mr.iterrows():
        theta_val, name = row["theta"], row["model_name"]
        color = model_colors[i % len(model_colors)]
        rank_label = f"{i + 1}{_suffix(i)}"

        ax.axvline(theta_val, 0, vline_top, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
        ax.scatter([theta_val], [marker_y], color=color, s=55, zorder=5)
        ax.text(theta_val, base_text_y + (i % 2) * text_offset, rank_label,
                ha="center", va="bottom", fontsize=8, color=color, fontweight="medium")
        model_handles.append(
            Line2D([0], [0], color=color, linestyle="--", marker="o", markersize=5, label=f"{rank_label}: {name}")
        )

    ax.set_xlabel(r"Latent scale (ability $\theta$ / difficulty $b$)", fontsize=11)
    ax.set_ylabel("Number of questions", fontsize=11)
    ax.set_title(title or "Question Difficulty Distribution & Model Abilities", fontsize=13, pad=10)
    ax.tick_params(axis="both", labelsize=9)

    if len(benchmarks) > 1 or benchmarks[0] not in ("All questions", "Arena"):
        legend1 = ax.legend(handles=bench_handles, title="Benchmarks", frameon=False,
                            bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        ax.add_artist(legend1)

    ax.legend(handles=model_handles, title="Models", frameon=False,
              bbox_to_anchor=(1.02, 0.4), loc="upper left", borderaxespad=0.0, fontsize=8, title_fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        plt.savefig(save_path, dpi=800, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank LLMs via IRT on static benchmarks, arena pairwise data, or both.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", required=True, choices=["static", "arena", "both"],
                   help="Which data sources to fit on.")
    p.add_argument("--static-csv", default=None, help="Path to static benchmark CSV.")
    p.add_argument("--arena-csv", default=None, help="Path to arena pairwise CSV.")

    # Training hyper-parameters
    p.add_argument("--num-epochs", type=int, default=2000, help="Training epochs (default: 2000).")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05).")
    p.add_argument("--lambda-static", type=float, default=1.0, help="Weight for static loss (both mode).")
    p.add_argument("--lambda-arena", type=float, default=1.0, help="Weight for arena loss (both mode).")
    p.add_argument("--lambda-tie", type=float, default=0.0, help="Weight for tie loss (default: 0.0).")
    p.add_argument("--lambda-bb", type=float, default=1.0, help="Weight for both-bad loss (default: 1.0).")

    # Output / evaluation
    p.add_argument("--save-plot", default=None, help="Path to save the ranking plot (e.g. ./figures/ranking.pdf).")
    p.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    p.add_argument("--evaluate", action="store_true", help="Run evaluation after fitting.")
    p.add_argument("--quiet", action="store_true", help="Disable verbose training output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    static_df = None
    arena_df = None

    # ── Validate and load data ───────────────────────────────────────
    if args.mode in ("static", "both"):
        if not args.static_csv:
            raise SystemExit("Error: --static-csv is required for mode '{}'.".format(args.mode))
        print(f"Loading static data from {args.static_csv} ...")
        static_df = pd.read_csv(args.static_csv)
        print(f"  {len(static_df)} rows, {static_df['model_name'].nunique()} models, "
              f"{static_df['question_id'].nunique()} questions")

    if args.mode in ("arena", "both"):
        if not args.arena_csv:
            raise SystemExit("Error: --arena-csv is required for mode '{}'.".format(args.mode))
        print(f"Loading arena data from {args.arena_csv} ...")
        arena_df = load_arena_pairs(args.arena_csv)
        n_arena_models = pd.unique(pd.concat([arena_df["model_1"], arena_df["model_2"]])).shape[0]
        print(f"  {len(arena_df)} pairs, {n_arena_models} models, "
              f"{arena_df['question_id'].nunique()} questions")

    # ── Fit ──────────────────────────────────────────────────────────
    print(f"\nFitting IRT model (mode={args.mode}) ...")
    if args.mode == "static":
        assert static_df is not None
        model_params, question_params = fit_static_irt(
            static_df, num_epochs=args.num_epochs, lr=args.lr, verbose=verbose,
        )
    elif args.mode == "arena":
        assert arena_df is not None
        model_params, question_params = fit_arena_irt(
            arena_df, num_epochs=args.num_epochs, lr=args.lr,
            lambda_tie=args.lambda_tie, lambda_bb=args.lambda_bb, verbose=verbose,
        )
    else:  # both
        assert static_df is not None and arena_df is not None
        model_params, question_params = fit_joint_irt(
            static_df, arena_df,
            num_epochs=args.num_epochs, lr=args.lr,
            lambda_static=args.lambda_static, lambda_arena=args.lambda_arena,
            lambda_tie=args.lambda_tie, lambda_bb=args.lambda_bb, verbose=verbose,
        )

    # ── Print rankings ───────────────────────────────────────────────
    print("\nModel ranking (highest ability first):")
    print(model_params.to_string(index=False))

    print(f"\nQuestion difficulty (top 20 hardest of {len(question_params)}):")
    print(question_params.head(20).to_string(index=False))

    # ── Evaluate ─────────────────────────────────────────────────────
    if args.evaluate:
        print("\nEvaluation:")
        metrics = evaluate_joint_model(static_df, arena_df, model_params, question_params)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        if static_df is not None:
            print("\nStatic agreement by model:")
            agreement = compute_static_agreement(static_df, model_params, question_params)
            print(agreement.to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_difficulty_and_ability(
            model_params, question_params, save_path=args.save_plot,
        )


if __name__ == "__main__":
    main()
