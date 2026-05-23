#!/usr/bin/env python3
"""Debug plots for the v2 online cold-start replay.

The main comparison is ``2pl_bt_joint + random`` versus ``dualeval_joint``.
The script also includes MutualEval's fisher and sharpness reveal policies so it
is clear whether the gap is from the model class or from item ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "cold_start" / "v2_online_frozen_joint"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "cold_start" / "v2_online_debug"

COMPARE_SPECS = [
    ("2pl_bt_joint", "random", "2PL+BT random", "#2B6CB0"),
    ("dualeval_joint", "random", "MutualEval random", "#C53030"),
    ("dualeval_joint", "fisher", "MutualEval fisher", "#2F855A"),
    ("dualeval_joint", "sharpness", "MutualEval sharpness", "#805AD5"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories_path = results_dir / "trajectories.csv"
    summaries_path = results_dir / "model_summaries.csv"
    if not trajectories_path.exists():
        raise SystemExit(f"Missing trajectories: {trajectories_path}")
    if not summaries_path.exists():
        raise SystemExit(f"Missing model summaries: {summaries_path}")
    trajectories = pd.read_csv(trajectories_path)
    summaries = pd.read_csv(summaries_path)
    return trajectories, summaries


def filter_compare(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for method, strategy, label, color in COMPARE_SPECS:
        part = df[(df["method"] == method) & (df["strategy"] == strategy)].copy()
        part["label"] = label
        part["color"] = color
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        raise SystemExit("No rows matched the configured comparison specs.")
    return out


def budget_column(trajectories: pd.DataFrame) -> str:
    return "target_fraction" if "target_fraction" in trajectories.columns else "n"


def budget_label(column: str) -> str:
    return "Reveal fraction of available items" if column == "target_fraction" else "Revealed items for held-out model"


def budget_stem(column: str, base: str) -> str:
    return base.replace("_by_n", "_by_fraction") if column == "target_fraction" else base


def summarize_by_budget(trajectories: pd.DataFrame) -> pd.DataFrame:
    column = budget_column(trajectories)
    grouped = (
        trajectories.groupby(["method", "strategy", "label", "color", column], as_index=False)
        .agg(
            mean_n=("n", "mean"),
            mean_reveal_fraction=("reveal_fraction", "mean") if "reveal_fraction" in trajectories.columns else ("n", "mean"),
            mean_rank_error=("rank_error", "mean"),
            median_rank_error=("rank_error", "median"),
            q25=("rank_error", lambda s: float(np.percentile(s, 25))),
            q75=("rank_error", lambda s: float(np.percentile(s, 75))),
            mean_spearman=("spearman", "mean"),
            n_models=("m_new", "nunique"),
        )
        .sort_values(["method", "strategy", column])
    )
    grouped["budget_column"] = column
    return grouped


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=180 if fmt == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_rank_error_by_budget(summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    column = str(summary["budget_column"].iloc[0]) if "budget_column" in summary.columns else "n"
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for _, _, label, color in COMPARE_SPECS:
        part = summary[summary["label"] == label].sort_values(column)
        if part.empty:
            continue
        ax.plot(part[column], part["mean_rank_error"], marker="o", linewidth=2.1, label=label, color=color)
        ax.fill_between(part[column], part["q25"], part["q75"], color=color, alpha=0.10, linewidth=0)
    ax.set_title("Held-Out Model Rank Error During Online Placement")
    ax.set_xlabel(budget_label(column))
    ax.set_ylabel("Mean absolute rank error")
    ax.set_xticks(sorted(summary[column].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    return save_figure(fig, output_dir, budget_stem(column, "rank_error_by_n"), formats)


def paired_delta_frame(trajectories: pd.DataFrame) -> pd.DataFrame:
    column = budget_column(trajectories)
    key_cols = ["m_new", column]
    pivot = trajectories.pivot_table(
        index=key_cols,
        columns=["method", "strategy"],
        values="rank_error",
        aggfunc="first",
    )
    bt = pivot[("2pl_bt_joint", "random")]
    rows: list[dict[str, Any]] = []
    for method, strategy, label, _ in COMPARE_SPECS:
        if method == "2pl_bt_joint" and strategy == "random":
            continue
        if (method, strategy) not in pivot.columns:
            continue
        values = pivot[(method, strategy)] - bt
        for (model_name, budget_value), delta in values.dropna().items():
            rows.append(
                {
                    "m_new": model_name,
                    column: budget_value,
                    "comparison": f"{label} - 2PL+BT random",
                    "delta_rank_error": float(delta),
                    "budget_column": column,
                }
            )
    return pd.DataFrame(rows)


def plot_delta_by_budget(delta_df: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    column = str(delta_df["budget_column"].iloc[0]) if "budget_column" in delta_df.columns else "n"
    grouped = (
        delta_df.groupby(["comparison", column], as_index=False)
        .agg(
            mean_delta=("delta_rank_error", "mean"),
            q25=("delta_rank_error", lambda s: float(np.percentile(s, 25))),
            q75=("delta_rank_error", lambda s: float(np.percentile(s, 75))),
        )
    )
    colors = {
        "MutualEval random - 2PL+BT random": "#C53030",
        "MutualEval fisher - 2PL+BT random": "#2F855A",
        "MutualEval sharpness - 2PL+BT random": "#805AD5",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    for comparison, part in grouped.groupby("comparison", sort=False):
        part = part.sort_values(column)
        color = colors.get(comparison, "#4A5568")
        ax.plot(part[column], part["mean_delta"], marker="o", linewidth=2.0, label=comparison, color=color)
        ax.fill_between(part[column], part["q25"], part["q75"], color=color, alpha=0.10, linewidth=0)
    ax.set_title("Delta vs 2PL+BT Random")
    ax.set_xlabel(budget_label(column))
    ax.set_ylabel("Mean rank-error delta")
    ax.text(
        0.01,
        0.97,
        "Positive means 2PL+BT random has lower rank error",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8)
    return save_figure(fig, output_dir, budget_stem(column, "delta_rank_error_by_n"), formats)


def final_delta_frame(summaries: pd.DataFrame) -> pd.DataFrame:
    compare = filter_compare(summaries)
    pivot = compare.pivot_table(
        index="m_new",
        columns=["method", "strategy"],
        values=["rank_error_final", "rank_hat_final", "rank_ref"],
        aggfunc="first",
    )
    bt_error = pivot[("rank_error_final", "2pl_bt_joint", "random")]
    rows: list[dict[str, Any]] = []
    for method, strategy, label, _ in COMPARE_SPECS:
        if ("rank_error_final", method, strategy) not in pivot.columns:
            continue
        error = pivot[("rank_error_final", method, strategy)]
        rank_hat = pivot[("rank_hat_final", method, strategy)]
        rank_ref = pivot[("rank_ref", method, strategy)]
        for model_name in error.dropna().index:
            rows.append(
                {
                    "m_new": model_name,
                    "method": method,
                    "strategy": strategy,
                    "label": label,
                    "rank_error_final": float(error.loc[model_name]),
                    "delta_vs_bt_random": float(error.loc[model_name] - bt_error.loc[model_name]),
                    "rank_hat_final": int(rank_hat.loc[model_name]),
                    "rank_ref": int(rank_ref.loc[model_name]),
                }
            )
    return pd.DataFrame(rows)


def plot_per_model_final_delta(final_delta: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    plot_df = final_delta[final_delta["label"].isin(["MutualEval random", "MutualEval fisher"])].copy()
    order = (
        plot_df[plot_df["label"] == "MutualEval random"]
        .sort_values(["delta_vs_bt_random", "m_new"], ascending=[False, True])["m_new"]
        .tolist()
    )
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, max(5.2, 0.38 * len(order) + 1.2)))
    offsets = {"MutualEval random": -0.17, "MutualEval fisher": 0.17}
    colors = {"MutualEval random": "#C53030", "MutualEval fisher": "#2F855A"}
    for label in ["MutualEval random", "MutualEval fisher"]:
        part = plot_df[plot_df["label"] == label].set_index("m_new").loc[order]
        ax.barh(y + offsets[label], part["delta_vs_bt_random"], height=0.30, color=colors[label], label=label)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Final rank-error delta vs 2PL+BT random")
    ax.set_title("Which Held-Out Models Drive the Gap at the Final Budget?")
    ax.text(
        0.01,
        0.98,
        "Positive means 2PL+BT random is better",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=True)
    return save_figure(fig, output_dir, "per_model_final_delta", formats)


def plot_final_rank_scatter(final_delta: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    labels = [
        label
        for label in ["2PL+BT random", "MutualEval random", "MutualEval fisher"]
        if label in set(final_delta["label"])
    ]
    if not labels:
        return []
    colors = {"2PL+BT random": "#2B6CB0", "MutualEval random": "#C53030", "MutualEval fisher": "#2F855A"}
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 4.1), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    max_rank = int(final_delta["rank_ref"].max())
    for ax, label in zip(axes, labels):
        part = final_delta[final_delta["label"] == label]
        ax.scatter(part["rank_ref"], part["rank_hat_final"], s=42, color=colors[label], alpha=0.85)
        for _, row in part.iterrows():
            if abs(int(row["rank_hat_final"]) - int(row["rank_ref"])) >= 3:
                ax.text(row["rank_ref"] + 0.08, row["rank_hat_final"] + 0.08, row["m_new"], fontsize=6)
        ax.plot([1, max_rank], [1, max_rank], color="black", linestyle=":", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Reference rank")
        ax.grid(True, alpha=0.20)
    axes[0].set_ylabel("Predicted final rank")
    axes[0].invert_yaxis()
    axes[0].invert_xaxis()
    fig.suptitle("Final New-Model Placement at the Final Budget", y=1.02)
    return save_figure(fig, output_dir, "final_rank_scatter", formats)


def write_debug_tables(
    output_dir: Path,
    by_budget: pd.DataFrame,
    delta_df: pd.DataFrame,
    final_delta: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    column = str(by_budget["budget_column"].iloc[0]) if "budget_column" in by_budget.columns else "n"
    by_budget.to_csv(output_dir / f"{budget_stem(column, 'rank_error_by_n')}.csv", index=False)
    delta_df.to_csv(output_dir / f"{budget_stem(column, 'delta_rank_error_by_n')}_long.csv", index=False)
    final_delta.to_csv(output_dir / "final_per_model_comparison.csv", index=False)


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        rows.append(cells)
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_markdown_summary(output_dir: Path, by_budget: pd.DataFrame, final_delta: pd.DataFrame) -> None:
    column = str(by_budget["budget_column"].iloc[0]) if "budget_column" in by_budget.columns else "n"
    exact_label = "exact_at_full_budget" if column == "target_fraction" else "exact_at_200"
    final_rows = []
    for _, _, label, _ in COMPARE_SPECS:
        part = final_delta[final_delta["label"] == label]
        if part.empty:
            continue
        final_rows.append(
            {
                "label": label,
                "mean_final_rank_error": part["rank_error_final"].mean(),
                "median_final_rank_error": part["rank_error_final"].median(),
                exact_label: int((part["rank_error_final"] == 0).sum()),
            }
        )
    final_table = pd.DataFrame(final_rows)

    delta_rows = []
    for label in ["MutualEval random", "MutualEval fisher", "MutualEval sharpness"]:
        part = final_delta[final_delta["label"] == label]
        if part.empty:
            continue
        delta_rows.append(
            {
                "comparison": f"{label} - 2PL+BT random",
                "mean_delta": part["delta_vs_bt_random"].mean(),
                "dualeval_better_models": int((part["delta_vs_bt_random"] < 0).sum()),
                "tied_models": int((part["delta_vs_bt_random"] == 0).sum()),
                "bt_better_models": int((part["delta_vs_bt_random"] > 0).sum()),
            }
        )
    delta_table = pd.DataFrame(delta_rows)

    text = [
        "# Online Cold-Start Debug Plots",
        "",
        "Primary comparison: `2pl_bt_joint + random` vs `dualeval_joint`.",
        "",
        "Generated figures:",
        "",
        f"- `{budget_stem(column, 'rank_error_by_n')}`: mean rank error across revealed-item budgets.",
        f"- `{budget_stem(column, 'delta_rank_error_by_n')}`: MutualEval rank-error delta relative to 2PL+BT random.",
        "- `per_model_final_delta`: per-held-out-model final deltas at the final budget.",
        "- `final_rank_scatter`: predicted final rank versus full-data reference rank.",
        "",
        "## Final Metrics",
        "",
        markdown_table(final_table),
        "",
        "## Paired Deltas At Final Budget",
        "",
        "Positive mean delta means 2PL+BT random has lower rank error.",
        "",
        markdown_table(delta_table),
        "",
        f"## Rank Error By {'Fraction' if column == 'target_fraction' else 'n'}",
        "",
        markdown_table(
            by_budget[["label", column, "mean_n", "mean_rank_error", "median_rank_error", "q25", "q75"]]
        ),
        "",
    ]
    (output_dir / "debug_summary.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = resolve_path(args.results_dir)
    output_dir = resolve_path(args.output_dir)
    trajectories, summaries = load_inputs(results_dir)
    compare_traj = filter_compare(trajectories)
    by_budget = summarize_by_budget(compare_traj)
    delta_df = paired_delta_frame(compare_traj)
    final_delta = final_delta_frame(summaries)

    saved: list[Path] = []
    saved.extend(plot_rank_error_by_budget(by_budget, output_dir, args.formats))
    saved.extend(plot_delta_by_budget(delta_df, output_dir, args.formats))
    saved.extend(plot_per_model_final_delta(final_delta, output_dir, args.formats))
    saved.extend(plot_final_rank_scatter(final_delta, output_dir, args.formats))
    write_debug_tables(output_dir, by_budget, delta_df, final_delta)
    write_markdown_summary(output_dir, by_budget, final_delta)

    print(f"Saved debug outputs to {output_dir}", flush=True)
    for path in saved:
        print(path, flush=True)


if __name__ == "__main__":
    main()
