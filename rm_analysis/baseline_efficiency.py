#!/usr/bin/env python3
"""
Baseline BT efficiency experiment: randomly sample a fraction of questions,
fit Bradley-Terry, and measure ranking stability vs the full-data BT ranking.

This serves as a comparison baseline for the proposed greedy IRT-based
data-efficient evaluation method (see method.md, section on greedy selection).

Usage:
  python rm_analysis/baseline_efficiency.py \
      --reward-jsonl results/arena_eval/coding_v0/responses.jsonl \
      --output-dir results/baseline_efficiency/coding_v0

  python rm_analysis/baseline_efficiency.py \
      --reward-jsonl results/arena_eval/math_v0/responses.jsonl \
                     results/arena_eval/coding_v0/responses.jsonl \
      --output-dir results/baseline_efficiency/joint_v0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FRACTIONS = [0.033, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reward_responses(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Reward JSONL not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") != "ok" or record.get("reward") is None:
                    continue
                rows.append(
                    {
                        "question_id": str(record["item_id"]),
                        "model_name": str(record["model_label"]),
                        "reward_raw": float(record["reward"]),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable reward rows found.")

    df = df.drop_duplicates(subset=["question_id", "model_name"], keep="last").reset_index(drop=True)

    reward_mean = float(df["reward_raw"].mean())
    reward_std = float(df["reward_raw"].std(ddof=0))
    if not math.isfinite(reward_std) or reward_std < 1e-8:
        reward_std = 1.0
    df["reward_z"] = (df["reward_raw"] - reward_mean) / reward_std
    return df


def build_bt_pairs(reward_df: pd.DataFrame) -> pd.DataFrame:
    """Convert reward responses into soft BT pairwise targets (one row per model pair per question)."""
    from itertools import combinations

    rows: list[dict[str, Any]] = []
    for question_id, group in reward_df.groupby("question_id", sort=False):
        group = group.sort_values("model_name").reset_index(drop=True)
        if len(group) < 2:
            continue
        for i, j in combinations(range(len(group)), 2):
            r1, r2 = group.iloc[i], group.iloc[j]
            gap = float(r1["reward_z"] - r2["reward_z"])
            target = 1.0 / (1.0 + math.exp(-gap))
            rows.append(
                {
                    "question_id": str(question_id),
                    "model_1": str(r1["model_name"]),
                    "model_2": str(r2["model_name"]),
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BT model fitting
# ---------------------------------------------------------------------------

def fit_bt(
    pairs: pd.DataFrame,
    *,
    num_epochs: int = 500,
    lr: float = 0.05,
    reg_lambda: float = 1e-4,
) -> pd.DataFrame:
    """Fit Bradley-Terry by SGD; returns DataFrame with columns [model_name, theta]."""
    if pairs.empty:
        return pd.DataFrame(columns=["model_name", "theta"])

    model_ids = pd.Index(
        pd.unique(pd.concat([pairs["model_1"], pairs["model_2"]], ignore_index=True)),
        name="model_name",
    )
    model_to_idx = {m: i for i, m in enumerate(model_ids)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.tensor(pairs["model_1"].map(model_to_idx).values, dtype=torch.long, device=device)
    m2 = torch.tensor(pairs["model_2"].map(model_to_idx).values, dtype=torch.long, device=device)
    targets = torch.tensor(pairs["target"].values, dtype=torch.float32, device=device)

    theta = nn.Embedding(len(model_ids), 1, device=device)
    nn.init.zeros_(theta.weight)
    optimizer = optim.Adam(theta.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(num_epochs):
        optimizer.zero_grad()
        logits = theta(m1).squeeze(-1) - theta(m2).squeeze(-1)
        loss = bce(logits, targets) + reg_lambda * theta.weight.pow(2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.weight.sub_(theta.weight.mean())

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    return (
        pd.DataFrame({"model_name": list(model_ids), "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def rank_series(model_params: pd.DataFrame) -> pd.Series:
    """Return a Series mapping model_name -> rank (1 = best)."""
    ordered = model_params.sort_values("theta", ascending=False)["model_name"].tolist()
    return pd.Series({m: r for r, m in enumerate(ordered, start=1)}, name="rank")


def ranking_metrics(
    ref_params: pd.DataFrame,
    sample_params: pd.DataFrame,
    *,
    top_k: int,
) -> dict[str, float]:
    """Compute Spearman ρ, Kendall τ, exact rank matches, and top-k retention."""
    common = set(ref_params["model_name"]) & set(sample_params["model_name"])
    if len(common) < 2:
        return dict(spearman=float("nan"), kendall=float("nan"),
                    exact_matches=float("nan"), top_k_retention=float("nan"),
                    n_models=len(common))

    ref_rank = rank_series(ref_params[ref_params["model_name"].isin(common)])
    sam_rank = rank_series(sample_params[sample_params["model_name"].isin(common)])
    models = sorted(common)

    r_ref = [ref_rank[m] for m in models]
    r_sam = [sam_rank[m] for m in models]

    rho, _ = spearmanr(r_ref, r_sam)
    tau, _ = kendalltau(r_ref, r_sam)

    exact = float(sum(a == b for a, b in zip(r_ref, r_sam)) / len(models))

    ref_top = {m for m, r in ref_rank.items() if r <= top_k}
    sam_top = {m for m, r in sam_rank.items() if r <= top_k}
    top_k_ret = float(len(ref_top & sam_top) / len(ref_top)) if ref_top else float("nan")

    return dict(
        spearman=float(rho),
        kendall=float(tau),
        exact_matches=exact,
        top_k_retention=top_k_ret,
        n_models=len(common),
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    reward_df: pd.DataFrame,
    *,
    fractions: list[float],
    seeds: list[int],
    top_k: int,
    bt_epochs: int,
    bt_lr: float,
    bt_reg: float,
) -> pd.DataFrame:
    all_question_ids = sorted(reward_df["question_id"].unique())
    n_total = len(all_question_ids)
    print(f"Total questions: {n_total}, models: {reward_df['model_name'].nunique()}", flush=True)

    # Full-data BT reference ranking
    print("Fitting full-data BT reference ...", flush=True)
    full_pairs = build_bt_pairs(reward_df)
    ref_params = fit_bt(full_pairs, num_epochs=bt_epochs, lr=bt_lr, reg_lambda=bt_reg)
    print(f"Reference ranking: {ref_params['model_name'].tolist()}", flush=True)

    rows: list[dict[str, Any]] = []
    for frac in fractions:
        n_sample = max(1, int(round(frac * n_total)))
        n_sample = min(n_sample, n_total)
        actual_frac = n_sample / n_total

        for seed in seeds:
            rng = np.random.default_rng(seed)
            sampled_ids = rng.choice(all_question_ids, size=n_sample, replace=False).tolist()
            subset_df = reward_df[reward_df["question_id"].isin(sampled_ids)]
            pairs_sub = build_bt_pairs(subset_df)

            if pairs_sub.empty:
                continue

            sample_params = fit_bt(pairs_sub, num_epochs=bt_epochs, lr=bt_lr, reg_lambda=bt_reg)
            metrics = ranking_metrics(ref_params, sample_params, top_k=top_k)

            rows.append(
                {
                    "target_fraction": frac,
                    "actual_fraction": actual_frac,
                    "n_questions": n_sample,
                    "n_total_questions": n_total,
                    "seed": seed,
                    **metrics,
                }
            )
            print(
                f"  frac={actual_frac:.2f} ({n_sample}/{n_total}) seed={seed}"
                f"  ρ={metrics['spearman']:.3f}  τ={metrics['kendall']:.3f}"
                f"  exact={metrics['exact_matches']:.3f}  top{top_k}={metrics['top_k_retention']:.3f}",
                flush=True,
            )

    return pd.DataFrame(rows)


def summarise(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["target_fraction", "n_questions", "n_total_questions"], as_index=False)
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            kendall_mean=("kendall", "mean"),
            kendall_std=("kendall", "std"),
            exact_matches_mean=("exact_matches", "mean"),
            exact_matches_std=("exact_matches", "std"),
            top_k_retention_mean=("top_k_retention", "mean"),
            top_k_retention_std=("top_k_retention", "std"),
            n_seeds=("seed", "count"),
        )
        .sort_values("n_questions")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(summary_df: pd.DataFrame, *, top_k: int, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = summary_df["actual_fraction"] if "actual_fraction" in summary_df.columns else summary_df["target_fraction"]

    for ax, (col, ylabel, title) in zip(
        axes,
        [
            ("spearman", "Spearman ρ", "Spearman Rank Correlation"),
            ("kendall", "Kendall τ", "Kendall Rank Correlation"),
            (f"top_k_retention", f"Top-{top_k} retention", f"Top-{top_k} Retention"),
        ],
    ):
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col not in summary_df.columns:
            mean_col = col
            std_col = None

        ax.plot(x, summary_df[mean_col], marker="o", label="BT (random sample)")
        if std_col and std_col in summary_df.columns:
            ax.fill_between(
                x,
                summary_df[mean_col] - summary_df[std_col],
                summary_df[mean_col] + summary_df[std_col],
                alpha=0.2,
            )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Fraction of questions used")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline BT efficiency: random-fraction sampling vs full-data ranking.",
    )
    parser.add_argument(
        "--reward-jsonl",
        nargs="+",
        type=Path,
        required=True,
        help="One or more arena_eval responses.jsonl files with reward scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "baseline_efficiency" / "default",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Question fractions to evaluate (e.g. 0.05 0.10 0.25 0.50 1.0).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k cutoff for retention metric.",
    )
    parser.add_argument(
        "--bt-epochs",
        type=int,
        default=500,
        help="BT training epochs.",
    )
    parser.add_argument(
        "--bt-lr",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--bt-reg",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reward_df = load_reward_responses(args.reward_jsonl)

    results_df = run_experiment(
        reward_df,
        fractions=args.fractions,
        seeds=args.seeds,
        top_k=args.top_k,
        bt_epochs=args.bt_epochs,
        bt_lr=args.bt_lr,
        bt_reg=args.bt_reg,
    )

    results_df.to_csv(args.output_dir / "results_raw.csv", index=False)
    summary_df = summarise(results_df)
    summary_df.to_csv(args.output_dir / "results_summary.csv", index=False)

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    if not args.no_plots:
        plot_results(
            summary_df,
            top_k=args.top_k,
            save_path=args.output_dir / "plots" / "bt_efficiency.png",
        )

    print(f"\nOutputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
