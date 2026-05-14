#!/usr/bin/env python3
"""
Visualize and quantify the relationship between item difficulty and discrimination.

Primary figure requested:
  - x-axis: difficulty bins
  - y-axis: item count in each difficulty bin
  - color: discrimination score groups (quantile buckets)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot difficulty vs discrimination diagnostics.")
    parser.add_argument("--question-ranking", required=True, help="Path to question_ranking.csv.")
    parser.add_argument(
        "--static-question-ranking",
        default=None,
        help=(
            "Optional path to static question params CSV. When provided, rows are merged with "
            "--question-ranking so source-colored scatter can show both arena and static prompts."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save figures and tables.")
    parser.add_argument(
        "--difficulty-col",
        default="difficulty_b",
        help="Difficulty column in question_ranking.csv (default: difficulty_b).",
    )
    parser.add_argument(
        "--discrimination-col",
        default="discrimination_exp_k",
        help="Discrimination column in question_ranking.csv (default: discrimination_exp_k).",
    )
    parser.add_argument(
        "--difficulty-bins",
        type=int,
        default=18,
        help="Number of bins on the difficulty axis for the distribution figure.",
    )
    parser.add_argument(
        "--discrimination-quantiles",
        type=int,
        default=4,
        help="Number of quantile buckets for discrimination coloring.",
    )
    parser.add_argument(
        "--extreme-quantile",
        type=float,
        default=0.75,
        help="Upper quantile for the high threshold in cross-extreme analysis.",
    )
    parser.add_argument(
        "--scatter-y-lower-quantile",
        type=float,
        default=0.01,
        help="Lower quantile for y-axis display range in source scatter (default: 0.01).",
    )
    parser.add_argument(
        "--scatter-y-upper-quantile",
        type=float,
        default=0.99,
        help="Upper quantile for y-axis display range in source scatter (default: 0.99).",
    )
    return parser.parse_args()


def _load_item_table(path: Path, difficulty_col: str, discrimination_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"question_id", difficulty_col, discrimination_col}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"question_ranking.csv missing required columns: {sorted(missing)}")

    source_col = "source" if "source" in df.columns else None
    benchmark_col = "benchmark" if "benchmark" in df.columns else None

    out_cols = ["question_id", difficulty_col, discrimination_col]
    if source_col:
        out_cols.append(source_col)
    if benchmark_col:
        out_cols.append(benchmark_col)

    out = df[out_cols].copy()
    out["item_id"] = out["question_id"].astype(str).str.split("::", n=1).str[-1]
    out["data_split"] = out.apply(_infer_data_split, axis=1)
    out = out.drop_duplicates(subset=["data_split", "item_id"], keep="last")
    out = out.rename(columns={difficulty_col: "difficulty", discrimination_col: "discrimination"})
    out["difficulty"] = pd.to_numeric(out["difficulty"], errors="coerce")
    out["discrimination"] = pd.to_numeric(out["discrimination"], errors="coerce")
    out = out[np.isfinite(out["difficulty"]) & np.isfinite(out["discrimination"])].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No valid rows after cleaning difficulty/discrimination columns.")
    return out


def _load_item_table_multi(
    primary_path: Path,
    static_path: Path | None,
    difficulty_col: str,
    discrimination_col: str,
) -> pd.DataFrame:
    dfs = [_load_item_table(primary_path, difficulty_col, discrimination_col)]
    if static_path is not None:
        dfs.append(_load_item_table(static_path, difficulty_col, discrimination_col))
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["data_split", "item_id"], keep="last").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No valid rows after loading question ranking inputs.")
    return out


def _infer_data_split(row: pd.Series) -> str:
    benchmark = str(row.get("benchmark", "")).strip().lower()
    source = str(row.get("source", "")).strip().lower()
    question_id = str(row.get("question_id", "")).strip().lower()

    arena_tokens = ("arena",)
    static_tokens = ("static", "simpleqa", "mmlu", "hle")

    if any(tok in benchmark for tok in arena_tokens):
        return "arena"
    if any(tok in source for tok in arena_tokens):
        return "arena"
    if any(tok in question_id for tok in arena_tokens):
        return "arena"

    if any(tok in benchmark for tok in static_tokens):
        return "static"
    if any(tok in source for tok in static_tokens):
        return "static"
    if any(tok in question_id for tok in static_tokens):
        return "static"

    # If unknown, keep in static-like bucket so split outputs remain usable.
    return "static"


def _build_distribution_table(
    item_df: pd.DataFrame,
    difficulty_bins: int,
    discrimination_quantiles: int,
) -> tuple[pd.DataFrame, pd.Categorical, pd.Categorical]:
    difficulty_bucket = pd.cut(item_df["difficulty"], bins=difficulty_bins, include_lowest=True)

    # qcut can drop duplicated edges for near-constant discrimination columns.
    discrimination_bucket = pd.qcut(
        item_df["discrimination"],
        q=discrimination_quantiles,
        duplicates="drop",
    )

    table = pd.crosstab(difficulty_bucket, discrimination_bucket).sort_index()
    table.index.name = "difficulty_bin"
    return table, difficulty_bucket, discrimination_bucket


def _plot_difficulty_distribution_with_discrimination(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(table.index))
    bottom = np.zeros(len(table.index), dtype=float)
    cmap = plt.cm.viridis
    colors = [cmap(v) for v in np.linspace(0.15, 0.9, len(table.columns))]

    for color, col in zip(colors, table.columns, strict=False):
        vals = table[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.9, color=color, label=str(col))
        bottom += vals

    centers = [interval.mid for interval in table.index]
    tick_step = max(1, len(centers) // 8)
    tick_idx = np.arange(0, len(centers), tick_step)
    tick_labels = [f"{centers[i]:.2f}" for i in tick_idx]

    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Difficulty (bin center)")
    ax.set_ylabel("Item count")
    ax.set_title("Item distribution by difficulty, colored by discrimination bucket")
    ax.legend(title="Discrimination quantile bucket", fontsize=8, title_fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_hex_relationship(item_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    hb = ax.hexbin(
        item_df["difficulty"].to_numpy(),
        item_df["discrimination"].to_numpy(),
        gridsize=30,
        mincnt=1,
        cmap="magma",
    )
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Discrimination")
    ax.set_title("Difficulty vs discrimination density (hexbin)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Item count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scatter_colored_by_data_split(
    item_df: pd.DataFrame,
    output_path: Path,
    y_lower_q: float,
    y_upper_q: float,
) -> None:
    if not (0.0 <= y_lower_q < y_upper_q <= 1.0):
        raise ValueError("--scatter-y-lower-quantile and --scatter-y-upper-quantile must satisfy 0<=low<high<=1.")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    color_map = {
        "arena": "#e69f00",
        "static": "#56b4e9",
    }
    y_vals = item_df["discrimination"].to_numpy(dtype=float)
    y_low = float(np.quantile(y_vals, y_lower_q))
    y_high = float(np.quantile(y_vals, y_upper_q))

    if np.isclose(y_low, y_high):
        y_pad = max(1.0, abs(y_low) * 0.1)
        y_low -= y_pad
        y_high += y_pad

    for split_name, split_df in item_df.groupby("data_split", sort=True):
        color = color_map.get(split_name, "#7f7f7f")
        ax.scatter(
            split_df["difficulty"].to_numpy(),
            split_df["discrimination"].to_numpy(),
            s=20,
            alpha=0.7,
            c=color,
            edgecolors="none",
            label=f"{split_name} (n={len(split_df)})",
        )

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Differential (discrimination)")
    ax.set_ylim(y_low, y_high)
    n_low = int((y_vals < y_low).sum())
    n_high = int((y_vals > y_high).sum())
    ax.set_title("Difficulty vs differential by prompt source")
    ax.text(
        0.99,
        0.01,
        f"Display range: q[{y_lower_q:.2f}, {y_upper_q:.2f}] | clipped below: {n_low}, above: {n_high}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Prompt source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _compute_summary(item_df: pd.DataFrame, extreme_quantile: float) -> dict:
    if not (0.5 < extreme_quantile < 1.0):
        raise ValueError("--extreme-quantile must be in (0.5, 1.0).")
    low_q = 1.0 - extreme_quantile

    d_high = float(item_df["difficulty"].quantile(extreme_quantile))
    d_low = float(item_df["difficulty"].quantile(low_q))
    a_high = float(item_df["discrimination"].quantile(extreme_quantile))
    a_low = float(item_df["discrimination"].quantile(low_q))

    high_d_low_a = item_df[(item_df["difficulty"] >= d_high) & (item_df["discrimination"] <= a_low)]
    low_d_high_a = item_df[(item_df["difficulty"] <= d_low) & (item_df["discrimination"] >= a_high)]
    high_both = item_df[(item_df["difficulty"] >= d_high) & (item_df["discrimination"] >= a_high)]
    low_both = item_df[(item_df["difficulty"] <= d_low) & (item_df["discrimination"] <= a_low)]

    n = int(len(item_df))
    pearson = float(item_df["difficulty"].corr(item_df["discrimination"], method="pearson"))
    spearman = float(item_df["difficulty"].corr(item_df["discrimination"], method="spearman"))

    return {
        "n_items": n,
        "difficulty_discrimination_correlation": {
            "pearson": pearson,
            "spearman": spearman,
        },
        "extreme_quantile_thresholds": {
            "high_quantile": extreme_quantile,
            "low_quantile": low_q,
            "difficulty_high_threshold": d_high,
            "difficulty_low_threshold": d_low,
            "discrimination_high_threshold": a_high,
            "discrimination_low_threshold": a_low,
        },
        "cross_extreme_counts": {
            "high_difficulty_low_discrimination": int(len(high_d_low_a)),
            "low_difficulty_high_discrimination": int(len(low_d_high_a)),
            "high_both": int(len(high_both)),
            "low_both": int(len(low_both)),
        },
        "cross_extreme_ratios_over_all_items": {
            "high_difficulty_low_discrimination": float(len(high_d_low_a) / n),
            "low_difficulty_high_discrimination": float(len(low_d_high_a) / n),
            "high_both": float(len(high_both) / n),
            "low_both": float(len(low_both) / n),
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    item_df = _load_item_table_multi(
        primary_path=Path(args.question_ranking),
        static_path=Path(args.static_question_ranking) if args.static_question_ranking else None,
        difficulty_col=args.difficulty_col,
        discrimination_col=args.discrimination_col,
    )
    table, _, _ = _build_distribution_table(
        item_df=item_df,
        difficulty_bins=args.difficulty_bins,
        discrimination_quantiles=args.discrimination_quantiles,
    )

    distribution_path = out_dir / "difficulty_distribution_by_discrimination.png"
    hex_path = out_dir / "difficulty_vs_discrimination_hexbin.png"
    source_scatter_path = out_dir / "difficulty_vs_differential_by_source.png"
    table_path = out_dir / "difficulty_discrimination_binned_counts.csv"
    summary_path = out_dir / "difficulty_discrimination_summary.json"

    _plot_difficulty_distribution_with_discrimination(table=table, output_path=distribution_path)
    _plot_hex_relationship(item_df=item_df, output_path=hex_path)
    _plot_scatter_colored_by_data_split(
        item_df=item_df,
        output_path=source_scatter_path,
        y_lower_q=args.scatter_y_lower_quantile,
        y_upper_q=args.scatter_y_upper_quantile,
    )

    flat_table = table.copy()
    flat_table.index = flat_table.index.astype(str)
    flat_table.to_csv(table_path, index=True)

    summary = _compute_summary(item_df=item_df, extreme_quantile=args.extreme_quantile)
    per_split_summary = {
        split_name: _compute_summary(split_df, extreme_quantile=args.extreme_quantile)
        for split_name, split_df in item_df.groupby("data_split", sort=True)
        if not split_df.empty
    }

    summary["breakdown_by_data_split"] = per_split_summary
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved difficulty/discrimination diagnostics:")
    print(f"  distribution plot: {distribution_path}")
    print(f"  hexbin plot:       {hex_path}")
    print(f"  source scatter:    {source_scatter_path}")
    print(f"  binned counts:     {table_path}")
    print(f"  summary json:      {summary_path}")
    print("\nRelationship summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
