"""
Compare reward value distributions across math_v0, coding_v0, and generic_v0 splits,
grouped by model. Saves figures to figures/.
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "results" / "arena_eval"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SPLITS = ["math_v0", "coding_v0", "generic_v0"]
SPLIT_LABELS = {"math_v0": "Math", "coding_v0": "Coding", "generic_v0": "Generic"}


def load_rewards():
    """Returns:
      data:   {split: {model_label: [reward, ...]}}   — only scored rows
      totals: {split: {model_label: int}}              — total rows (including errors)
    """
    data, totals = {}, {}
    for split in SPLITS:
        path = DATA_DIR / split / "responses.jsonl"
        model_rewards = defaultdict(list)
        model_totals = defaultdict(int)
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                model_totals[r["model_label"]] += 1
                if r["status"] != "generation_error" and r["reward"] is not None:
                    model_rewards[r["model_label"]].append(float(r["reward"]))
        data[split] = dict(model_rewards)
        totals[split] = dict(model_totals)
    return data, totals


def plot_violin_by_split(data, totals):
    """One figure: 3 subplots (one per split), violin per model."""
    models = sorted({m for split in data.values() for m in split})
    n_models = len(models)
    model_idx = {m: i for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Reward Distribution by Model and Split", fontsize=14, y=1.01)

    for ax, split in zip(axes, SPLITS):
        split_data = data[split]
        positions = []
        values = []
        for m in models:
            vals = split_data.get(m, [])
            if len(vals) >= 2:  # violinplot requires at least 2 points
                positions.append(model_idx[m])
                values.append(vals)

        if not values:
            ax.set_title(SPLIT_LABELS[split] + " (no data)", fontsize=12)
            continue
        parts = ax.violinplot(values, positions=positions, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)

        ax.set_title(SPLIT_LABELS[split], fontsize=12)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Model")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Reward")
    fig.tight_layout()
    out = FIG_DIR / "rm_dist_violin_by_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_mean_reward_heatmap(data, totals):
    """Heatmap: models x splits, cells = mean reward. Red asterisk if >5% generation errors."""
    models = sorted({m for split in data.values() for m in split})
    matrix = np.full((len(models), len(SPLITS)), np.nan)

    for j, split in enumerate(SPLITS):
        for i, m in enumerate(models):
            vals = data[split].get(m, [])
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(7, max(5, len(models) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    fig.colorbar(im, ax=ax, label="Mean Reward")

    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Mean Reward per Model and Split\n(* = >5% generation errors)", fontsize=11)

    for i, m in enumerate(models):
        for j, split in enumerate(SPLITS):
            if np.isnan(matrix[i, j]):
                continue
            n_scored = len(data[split].get(m, []))
            n_total = totals[split].get(m, n_scored)
            error_rate = (n_total - n_scored) / n_total if n_total else 0
            marker = "*" if error_rate > 0.05 else ""
            ax.text(j, i, f"{matrix[i, j]:.2f}{marker}\nn={n_scored}",
                    ha="center", va="center", fontsize=6)

    fig.tight_layout()
    out = FIG_DIR / "rm_dist_mean_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_rank_by_domain(data, totals):
    """3 horizontal bar charts (one per split) with models ranked by mean reward."""
    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len({m for s in data.values() for m in s}) * 0.35)))
    fig.suptitle("Model Ranking by Mean Reward per Domain", fontsize=14, y=1.01)

    for ax, split in zip(axes, SPLITS):
        split_data = data[split]
        means = {m: np.mean(vals) for m, vals in split_data.items() if vals}
        ranked = sorted(means.items(), key=lambda x: x[1])  # ascending so best is at top
        if not ranked:
            ax.set_title(SPLIT_LABELS[split] + " (no data)", fontsize=12)
            continue
        models_r, scores = zip(*ranked)

        colors = ["#d73027" if s < 0 else "#4575b4" for s in scores]
        bars = ax.barh(range(len(models_r)), scores, color=colors, edgecolor="white", height=0.7)

        ax.set_yticks(range(len(models_r)))
        ax.set_yticklabels(models_r, fontsize=8)
        ax.set_title(SPLIT_LABELS[split], fontsize=12)
        ax.set_xlabel("Mean Reward")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        for i, (bar, score, model) in enumerate(zip(bars, scores, models_r)):
            n_scored = len(data[split].get(model, []))
            n_total = totals[split].get(model, n_scored)
            err_str = f" (*{n_total - n_scored} err)" if n_total - n_scored > 0 else ""
            label = f"{score:.2f}{err_str}"
            ax.text(score + (0.02 if score >= 0 else -0.02), i, label,
                    va="center", ha="left" if score >= 0 else "right", fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / "rm_dist_rank_by_domain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_kde_overlay(data, totals):
    """One figure: 3 subplots, KDE per model overlaid, one subplot per split."""
    from scipy.stats import gaussian_kde

    models = sorted({m for split in data.values() for m in split})
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    color_map = {m: colors[i] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle("Reward KDE by Model and Split", fontsize=14, y=1.01)

    for ax, split in zip(axes, SPLITS):
        split_data = data[split]
        all_vals = [v for vals in split_data.values() for v in vals]
        if not all_vals:
            ax.set_title(SPLIT_LABELS[split] + " (no data)", fontsize=12)
            continue
        x = np.linspace(min(all_vals) - 0.5, max(all_vals) + 0.5, 300)

        for m in models:
            vals = split_data.get(m, [])
            if len(vals) < 5:
                continue
            kde = gaussian_kde(vals, bw_method=0.3)
            ax.plot(x, kde(x), label=m, color=color_map[m], linewidth=1.2)

        ax.set_title(SPLIT_LABELS[split], fontsize=12)
        ax.set_xlabel("Reward")
        ax.set_ylabel("Density")
        ax.grid(linestyle="--", alpha=0.4)

    handles = [plt.Line2D([0], [0], color=color_map[m], linewidth=1.5, label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    out = FIG_DIR / "rm_dist_kde_by_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    data, totals = load_rewards()
    plot_violin_by_split(data, totals)
    plot_mean_reward_heatmap(data, totals)
    plot_rank_by_domain(data, totals)
    plot_kde_overlay(data, totals)
