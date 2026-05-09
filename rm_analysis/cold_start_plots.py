#!/usr/bin/env python3
"""Generate arena-only cold-start plots from ``cold_start_efficiency.py`` outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "cold_start" / "v1_arena_adaptive"
DEFAULT_QUESTION_RESULTS_DIR = REPO_ROOT / "results" / "cold_start" / "v1_question_items"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "cold_start"
METHOD_LABELS = {
    "irt_arena": "IRT arena",
    "bt_arena": "BT arena",
}
METHOD_ORDER = ["irt_arena", "bt_arena"]


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def _model_order_by_primary_rank(run_df: pd.DataFrame, *, primary_method: str = "irt_arena") -> list[str]:
    primary = run_df[run_df["method"] == primary_method]
    if primary.empty:
        primary = run_df
    return (
        primary.groupby("m_new", as_index=False)
        .agg(rank_ref=("rank_ref", "median"))
        .sort_values(["rank_ref", "m_new"])["m_new"]
        .tolist()
    )


def plot_convergence(trajectories: pd.DataFrame, save_path: Path) -> None:
    df = trajectories.dropna(subset=["rank_error"]).copy()
    if df.empty:
        raise ValueError("No trajectory rows with rank_error.")
    grouped = (
        df.groupby(["method", "n"], as_index=False)
        .agg(
            mean_rank_error=("rank_error", "mean"),
            q25=("rank_error", lambda s: float(np.percentile(s, 25))),
            q75=("rank_error", lambda s: float(np.percentile(s, 75))),
        )
        .sort_values(["method", "n"])
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for idx, method in enumerate([m for m in METHOD_ORDER if m in set(grouped["method"])]):
        sub = grouped[grouped["method"] == method]
        color = cmap(idx)
        ax.plot(sub["n"], sub["mean_rank_error"], label=_method_label(method), color=color, linewidth=1.8)
        ax.fill_between(sub["n"], sub["q25"], sub["q75"], color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("Revealed responses")
    ax.set_ylabel("Mean absolute rank error")
    ax.set_title("Adaptive Arena Cold-Start Rank Convergence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, save_path)


def plot_n_star_per_model(run_df: pd.DataFrame, save_path: Path) -> None:
    needed = run_df[run_df["method"].isin(METHOD_ORDER)].dropna(subset=["n_star"]).copy()
    if needed.empty:
        raise ValueError("No finite n_star rows for irt_arena/bt_arena.")

    med = (
        needed.groupby(["m_new", "method"], as_index=False)
        .agg(
            median_n_star=("n_star", "median"),
            q25=("n_star", lambda s: float(np.percentile(s, 25))),
            q75=("n_star", lambda s: float(np.percentile(s, 75))),
            rank_ref=("rank_ref", "median"),
        )
        .sort_values(["rank_ref", "m_new"])
    )
    models = _model_order_by_primary_rank(needed)
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.7), 5.2))
    offsets = {"irt_arena": -width / 2, "bt_arena": width / 2}
    colors = {"irt_arena": "#1f77b4", "bt_arena": "#ff7f0e"}
    for method in METHOD_ORDER:
        sub = med[med["method"] == method].set_index("m_new").reindex(models)
        y = sub["median_n_star"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["q25"].to_numpy(dtype=float),
                sub["q75"].to_numpy(dtype=float) - y,
            ]
        )
        ax.bar(
            x + offsets[method],
            y,
            width=width,
            label=_method_label(method),
            color=colors[method],
            alpha=0.88,
        )
        ax.errorbar(
            x + offsets[method],
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Median n*")
    ax.set_title("Adaptive Arena Queries To Rank Stability By Held-Out Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, save_path)


def plot_info_per_question(trajectories: pd.DataFrame, save_path: Path, *, max_points: int) -> None:
    df = trajectories[
        trajectories["method"].isin(METHOD_ORDER)
        & trajectories["info_x"].notna()
        & trajectories["ll_gain"].notna()
    ].copy()
    if df.empty:
        raise ValueError("No information/ll_gain rows for info scatter.")

    frames: list[pd.DataFrame] = []
    for method, group in df.groupby("method", sort=True):
        if len(group) > max_points:
            group = group.sample(max_points, random_state=0)
        frames.append(group)
    df = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, method, xlabel in [
        (axes[0], "irt_arena", "IRT item discrimination"),
        (axes[1], "bt_arena", "BT mean opponent gap"),
    ]:
        sub = df[df["method"] == method]
        if sub.empty:
            ax.set_visible(False)
            continue
        color = "#1f77b4" if method == "irt_arena" else "#ff7f0e"
        ax.scatter(sub["info_x"], sub["ll_gain"], s=9, alpha=0.25, color=color, edgecolors="none")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Held-out log-likelihood gain")
        ax.set_title(_method_label(method))
        ax.grid(alpha=0.2)

    fig.suptitle("Adaptive Arena Information Per Revealed Response", y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


def plot_lift_by_ability(run_df: pd.DataFrame, save_path: Path) -> None:
    paired = run_df[run_df["method"].isin(METHOD_ORDER)].dropna(subset=["n_star"]).copy()
    if paired.empty:
        raise ValueError("No finite n_star rows for lift plot.")
    med = (
        paired.groupby(["m_new", "method"], as_index=False)
        .agg(median_n_star=("n_star", "median"), rank_ref=("rank_ref", "median"), theta_ref=("theta_ref", "median"))
    )
    wide = med.pivot(index="m_new", columns="method", values="median_n_star")
    meta = med.groupby("m_new", as_index=True).agg(rank_ref=("rank_ref", "median"), theta_ref=("theta_ref", "median"))
    plot_df = wide.join(meta).dropna(subset=METHOD_ORDER)
    if plot_df.empty:
        raise ValueError("No paired arena IRT/BT n_star rows for lift plot.")
    plot_df["query_saving"] = plot_df["bt_arena"] - plot_df["irt_arena"]
    plot_df = plot_df.sort_values("rank_ref")

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        plot_df["rank_ref"],
        plot_df["query_saving"],
        c=plot_df["theta_ref"],
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.5,
    )
    for model_name, row in plot_df.iterrows():
        ax.annotate(str(model_name), (row["rank_ref"], row["query_saving"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Reference rank (1 = best)")
    ax.set_ylabel("Median query saving: BT arena n* - IRT arena n*")
    ax.set_title("Adaptive Arena Cold-Start Lift By Model Ability")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Reference theta")
    _save(fig, save_path)


def plot_summary_table(summary_df: pd.DataFrame, save_path: Path) -> None:
    cols = [
        "method",
        "mean_n_star",
        "median_n_star",
        "mean_rank_error_at_50",
        "mean_rank_error_at_200",
        "n_runs",
    ]
    table_df = summary_df[[c for c in cols if c in summary_df.columns]].copy()
    for col in table_df.columns:
        if col != "method":
            table_df[col] = table_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}" if isinstance(x, float) else str(x))
    table_df["method"] = table_df["method"].map(_method_label)

    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.45 * len(table_df) + 1.4)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=[c.replace("_", " ") for c in table_df.columns],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    ax.set_title("Adaptive Arena Cold-Start Summary", pad=14)
    _save(fig, save_path)


def _line_mean(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    label: str,
    color: str,
) -> None:
    grouped = (
        df.groupby(x, as_index=False)
        .agg(
            mean=(y, "mean"),
        )
        .sort_values(x)
    )
    ax.plot(grouped[x], grouped["mean"], color=color, linewidth=2.1, label=label)


def plot_model_online_coldstart(trajectories: pd.DataFrame, run_df: pd.DataFrame, save_path: Path) -> None:
    del run_df
    df = trajectories[
        trajectories["method"].isin(METHOD_ORDER) & trajectories["rank_error"].notna()
    ].copy()
    if df.empty:
        raise ValueError("No model cold-start trajectory rows to plot.")

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    colors = {"irt_arena": "#1f77b4", "bt_arena": "#ff7f0e"}
    labels = {"irt_arena": "DualEval", "bt_arena": "BT"}
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if not sub.empty:
            _line_mean(
                ax,
                sub,
                x="n",
                y="rank_error",
                label=labels[method],
                color=colors[method],
            )

    ax.set_xlim(1, min(250, int(df["n"].max())))
    ax.set_xlabel("Revealed responses for held-out model")
    ax.set_ylabel("Mean absolute rank error")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)


def plot_question_item_online_update(trajectories: pd.DataFrame, save_path: Path) -> None:
    needed = trajectories.dropna(subset=["b_abs_error", "k_abs_error"]).copy()
    if needed.empty:
        raise ValueError("No question-item online trajectory rows to plot.")

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    _line_mean(
        ax,
        needed,
        x="n",
        y="b_abs_error",
        label="Difficulty",
        color="#2ca02c",
    )
    _line_mean(
        ax,
        needed,
        x="n",
        y="k_abs_error",
        label="Log-sharpness",
        color="#9467bd",
    )
    ax.set_xlabel("Revealed model responses for held-out item")
    ax.set_ylabel("Mean absolute parameter error")
    ax.set_xlim(1, min(15, int(needed["n"].max())))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cold-start replay outputs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--question-results-dir", type=Path, default=DEFAULT_QUESTION_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--max-info-points", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories_path = args.results_dir / "trajectories.csv"
    run_path = args.results_dir / "n_star_by_run.csv"
    summary_path = args.results_dir / "summary.csv"
    if not trajectories_path.exists():
        raise SystemExit(f"Missing trajectories CSV: {trajectories_path}")
    if not run_path.exists():
        raise SystemExit(f"Missing run summary CSV: {run_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing summary CSV: {summary_path}")

    trajectories = pd.read_csv(trajectories_path)
    run_df = pd.read_csv(run_path)
    summary_df = pd.read_csv(summary_path)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(trajectories, args.figures_dir / "convergence_irt_vs_bt.pdf")
    plot_n_star_per_model(run_df, args.figures_dir / "n_star_per_model.pdf")
    plot_info_per_question(
        trajectories,
        args.figures_dir / "info_per_question.pdf",
        max_points=args.max_info_points,
    )
    plot_lift_by_ability(run_df, args.figures_dir / "lift_by_ability.pdf")
    plot_summary_table(summary_df, args.figures_dir / "summary_table.pdf")
    plot_model_online_coldstart(
        trajectories,
        run_df,
        args.figures_dir / "model_online_coldstart.pdf",
    )

    question_trajectories_path = args.question_results_dir / "trajectories.csv"
    if question_trajectories_path.exists():
        question_trajectories = pd.read_csv(question_trajectories_path)
        plot_question_item_online_update(
            question_trajectories,
            args.figures_dir / "question_item_online_update.pdf",
        )
    else:
        print(f"Skipping question-item plot; missing {question_trajectories_path}", flush=True)


if __name__ == "__main__":
    main()
