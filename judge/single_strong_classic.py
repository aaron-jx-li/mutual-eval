#!/usr/bin/env python3
"""
Classic pairwise strong judge for LLM math battles.

Supports two input formats:
1. Legacy JSON: battle data with conversation_a, conversation_b, winner (human label)
2. Arena responses.jsonl: per-question model responses with reward values

For responses.jsonl:
- Groups by item_id, forms pairwise comparisons
- Uses LLM judge to give preference (A/B/C/D)
- Compares with reward-model preference (higher reward = preferred)
- Reports quantitative metrics: agreement rate, Kendall tau, accuracy by margin

Usage (legacy JSON):
    python judge/single_strong_classic.py \
        --input data/arena_140k_math_openai_single.json \
        --output data/arena_140k_math_single_strong_classic.json

Usage (responses.jsonl, compare with reward model):
    python judge/single_strong_classic.py \
        --input results/arena_eval/math_v0/responses.jsonl \
        --output results/arena_eval/math_v0/judge_vs_rm.json \
        --format jsonl

    # Limit pairs for faster runs
    python judge/single_strong_classic.py \
        --input results/arena_eval/math_v0/responses.jsonl \
        --output results/arena_eval/math_v0/judge_vs_rm.json \
        --format jsonl --max-items 50 --max-pairs-per-item 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from openai import OpenAI

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Line2D = None

SYSTEM_INSTRUCTIONS = (
    "You are an impartial mathematics judge. "
    "Compare two responses to the same math question and pick which is better. "
    "Prioritize mathematical correctness over style."
)

CLASSIC_PAIR_PROMPT_TEMPLATE = """\
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
You should choose the assistant that follows the user's instructions and answers the user's question better.
Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail.
Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names of the assistants. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format:
"[[A]]" if assistant A is better, "[[B]]" if assistant B is better, "[[C]]" for a tie,
and "[[D]]" if both responses are mathematically incorrect / both bad.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""


# ── Helpers ───────────────────────────────────────────────────────────

def extract_text(content) -> str:
    """Extract plain text from a content field (string or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def build_pair_prompt(question: str, answer_a: str, answer_b: str) -> str:
    """Build a classic pairwise prompt from question and two answers."""
    return CLASSIC_PAIR_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=answer_a or "(empty response)",
        answer_b=answer_b or "(empty response)",
    )


def build_pair_prompt_legacy(entry: dict) -> str:
    """Build a classic pairwise prompt from a legacy battle entry."""
    conv_a = entry["conversation_a"]
    conv_b = entry["conversation_b"]

    question_parts = []
    turn = 0
    for msg in conv_a:
        if msg["role"] == "user":
            turn += 1
            question_parts.append(f"Turn {turn}: {extract_text(msg['content'])}")
    question_text = "\n\n".join(question_parts)

    response_a_parts = []
    response_b_parts = []
    turn_a = turn_b = 0

    for msg in conv_a:
        if msg["role"] == "user":
            turn_a += 1
        elif msg["role"] == "assistant":
            text = extract_text(msg["content"])
            prefix = f"Turn {turn_a}:\n" if turn_a > 1 or len(conv_a) > 2 else ""
            response_a_parts.append(f"{prefix}{text}")

    for msg in conv_b:
        if msg["role"] == "user":
            turn_b += 1
        elif msg["role"] == "assistant":
            text = extract_text(msg["content"])
            prefix = f"Turn {turn_b}:\n" if turn_b > 1 or len(conv_b) > 2 else ""
            response_b_parts.append(f"{prefix}{text}")

    answer_a = "\n\n".join(response_a_parts)
    answer_b = "\n\n".join(response_b_parts)

    return build_pair_prompt(question_text, answer_a, answer_b)


def call_judge(
    client: OpenAI,
    judge_model: str,
    prompt: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """Call the judge model via the Responses API."""
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=judge_model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=prompt,
                max_output_tokens=4096,
                store=False,
            )
            return resp.output_text.strip()
        except Exception as e:
            print(f"  [Attempt {attempt + 1}/{max_retries}] {judge_model} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    return "Fallback: [[C]]"


def parse_classic_verdict(response_text: str) -> dict:
    """
    Parse classic verdict format:
      [[A]] -> model_a
      [[B]] -> model_b
      [[C]] -> tie
      [[D]] -> both_bad
    """
    text = response_text.strip()
    upper = text.upper()

    if "[[A]]" in upper:
        verdict = "model_a"
    elif "[[B]]" in upper:
        verdict = "model_b"
    elif "[[C]]" in upper:
        verdict = "tie"
    elif "[[D]]" in upper:
        verdict = "both_bad"
    else:
        # Fallback heuristics for non-compliant outputs
        lower = text.lower()
        if ("both" in lower and "bad" in lower) or (
            "both" in lower and ("incorrect" in lower or "wrong" in lower or "errors" in lower)
        ):
            verdict = "both_bad"
        elif "assistant a" in lower and ("better" in lower or "wins" in lower):
            verdict = "model_a"
        elif "assistant b" in lower and ("better" in lower or "wins" in lower):
            verdict = "model_b"
        else:
            verdict = "tie"

    return {
        "verdict": verdict,
        "explanation": text[:2000],
    }


# ── Core evaluation ──────────────────────────────────────────────────

def judge_entry(client: OpenAI, judge_model: str, entry: dict) -> dict:
    """Use a single strong judge to compare both responses (legacy format)."""
    prompt = build_pair_prompt_legacy(entry)
    raw_response = call_judge(client, judge_model, prompt)
    parsed = parse_classic_verdict(raw_response)

    return {
        "id": entry["id"],
        "model_a": entry["model_a"],
        "model_b": entry["model_b"],
        "human_label": entry["winner"],
        "judge_label": parsed["verdict"],
        "judge_model": judge_model,
        "judge_response": {
            "raw_response": raw_response,
            **parsed,
        },
    }


def judge_pair_jsonl(
    client: OpenAI,
    judge_model: str,
    item_id: str,
    rec_a: dict,
    rec_b: dict,
) -> dict:
    """Judge a single pair from responses.jsonl format."""
    question = rec_a.get("question", "")
    answer_a = rec_a.get("response_text") or ""
    answer_b = rec_b.get("response_text") or ""
    reward_a = rec_a.get("reward")
    reward_b = rec_b.get("reward")

    prompt = build_pair_prompt(question, answer_a, answer_b)
    raw_response = call_judge(client, judge_model, prompt)
    parsed = parse_classic_verdict(raw_response)

    pair_id = f"{item_id}::{rec_a['model_label']} vs {rec_b['model_label']}"

    return {
        "id": pair_id,
        "item_id": item_id,
        "model_a": rec_a["model_label"],
        "model_b": rec_b["model_label"],
        "reward_a": reward_a,
        "reward_b": reward_b,
        "judge_label": parsed["verdict"],
        "judge_model": judge_model,
        "judge_response": {
            "raw_response": raw_response,
            **parsed,
        },
    }


# ── Load responses.jsonl ─────────────────────────────────────────────

def load_responses_jsonl(path: str) -> dict[str, list[dict]]:
    """Load responses.jsonl and group by item_id. Only rows with valid reward."""
    items: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("reward") is None:
                continue
            items[d["item_id"]].append(d)
    return dict(items)


def sample_pairs(
    items: dict[str, list[dict]],
    max_items: int | None,
    max_pairs_per_item: int | None,
    max_pairs_total: int | None = None,
    seed: int = 42,
) -> list[tuple[str, dict, dict]]:
    """
    Sample (item_id, rec_a, rec_b) pairs for judging.

    Balanced subsample when max_pairs_total is set:
    - Distributes pairs evenly across items (each item gets ~equal share)
    - Within each item, randomly samples if over quota
    """
    rng = random.Random(seed)
    pairs: list[tuple[str, dict, dict]] = []

    item_ids = sorted(items.keys())
    if max_items:
        item_ids = item_ids[:max_items]

    # Build all pairs per item
    item_pairs: list[tuple[str, list[tuple[dict, dict]]]] = []
    for item_id in item_ids:
        recs = items[item_id]
        if len(recs) < 2:
            continue
        all_pairs = list(combinations(recs, 2))
        if max_pairs_per_item and len(all_pairs) > max_pairs_per_item:
            all_pairs = rng.sample(all_pairs, max_pairs_per_item)
        item_pairs.append((item_id, all_pairs))

    if not item_pairs:
        return pairs

    if max_pairs_total:
        # Balanced: distribute budget across items
        n_items = len(item_pairs)
        quota_per_item = max(1, max_pairs_total // n_items)
        remainder = max_pairs_total - quota_per_item * n_items
        for i, (item_id, all_pairs) in enumerate(item_pairs):
            quota = quota_per_item + (1 if i < remainder else 0)
            take = min(quota, len(all_pairs))
            if take < len(all_pairs):
                selected = rng.sample(all_pairs, take)
            else:
                selected = all_pairs
            for rec_a, rec_b in selected:
                pairs.append((item_id, rec_a, rec_b))
    else:
        for item_id, all_pairs in item_pairs:
            for rec_a, rec_b in all_pairs:
                pairs.append((item_id, rec_a, rec_b))

    return pairs


# ── Evaluation metrics (judge vs reward model) ────────────────────────

def _judge_direction(judge_label: str) -> int | None:
    """Map judge 4-class label to direction: +1 (prefer A), -1 (prefer B), 0 (tie/both_bad), None if invalid."""
    if judge_label == "model_a":
        return 1
    if judge_label == "model_b":
        return -1
    if judge_label in ("tie", "both_bad"):
        return 0
    return None


def compute_judge_vs_rm_metrics(results: list[dict]) -> dict:
    """
    Quantitative metrics comparing LLM judge (4 discrete labels) with reward model (2 scalars).

    Judge: model_a, model_b, tie, both_bad
    RM: reward_a, reward_b (continuous) → direction = sign(reward_a - reward_b), margin = |reward_a - reward_b|

    Metrics (no mapping of RM to discrete labels):
    - dir_agree_judge_decisive: when judge picks model_a or model_b, does sign(Δreward) match?
    - dir_agree_rm_decisive: when |Δreward| > threshold, does judge pick the higher-reward model?
    - spearman: correlation between judge_direction (+1/-1/0) and (reward_a - reward_b)
    - accuracy_by_rm_margin: dir agreement stratified by |reward_a - reward_b|
    - judge_says_tie_rm_margin: when judge says tie/both_bad, mean |Δreward| (is RM also uncertain?)
    """
    valid = [r for r in results if r.get("reward_a") is not None and r.get("reward_b") is not None]
    if not valid:
        return {"error": "No valid pairs with reward values"}

    total = len(valid)

    # 1. Direction agreement when judge is decisive (model_a or model_b)
    judge_decisive = [r for r in valid if r["judge_label"] in ("model_a", "model_b")]
    n_judge_dec = len(judge_decisive)
    if n_judge_dec > 0:
        agree = sum(
            1 for r in judge_decisive
            if (r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
            or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
        )
        dir_agree_judge_decisive = agree / n_judge_dec
    else:
        dir_agree_judge_decisive = None

    # 2. Direction agreement when RM is decisive (|Δreward| > threshold)
    margin_threshold = 0.1
    rm_decisive = [r for r in valid if abs((r["reward_a"] or 0) - (r["reward_b"] or 0)) > margin_threshold]
    n_rm_dec = len(rm_decisive)
    if n_rm_dec > 0:
        agree = sum(
            1 for r in rm_decisive
            if (r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
            or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
        )
        dir_agree_rm_decisive = agree / n_rm_dec
    else:
        dir_agree_rm_decisive = None

    # 3. Spearman correlation: judge_direction vs (reward_a - reward_b)
    judge_scores = []
    rm_diffs = []
    for r in valid:
        jd = _judge_direction(r["judge_label"])
        if jd is not None:
            judge_scores.append(jd)
            rm_diffs.append((r["reward_a"] or 0) - (r["reward_b"] or 0))
    if len(judge_scores) >= 2:
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(judge_scores, rm_diffs)
            spearman = float(rho) if rho is not None else None
        except ImportError:
            spearman = None  # scipy not available
    else:
        spearman = None

    # 4. Accuracy by RM margin (when |Δreward| in bucket)
    margin_buckets = [(0.0, 0.1, "tiny"), (0.1, 0.25, "small"), (0.25, 0.5, "medium"), (0.5, float("inf"), "large")]
    accuracy_by_margin: dict[str, dict] = {}
    for lo, hi, name in margin_buckets:
        subset = [
            r for r in valid
            if lo <= abs((r["reward_a"] or 0) - (r["reward_b"] or 0)) < hi
        ]
        if subset:
            agree = sum(
                1 for r in subset
                if (r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
                or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
            )
            acc = agree / len(subset)
            accuracy_by_margin[name] = {"n": len(subset), "accuracy": acc, "margin_range": f"[{lo}, {hi})"}

    # 5. When judge says tie/both_bad: what's the RM margin? (is RM also uncertain?)
    judge_tie_or_bad = [r for r in valid if r["judge_label"] in ("tie", "both_bad")]
    if judge_tie_or_bad:
        margins = [abs((r["reward_a"] or 0) - (r["reward_b"] or 0)) for r in judge_tie_or_bad]
        mean_margin_when_judge_tie = sum(margins) / len(margins)
    else:
        mean_margin_when_judge_tie = None

    # 6. Judge label distribution (for reference)
    judge_dist = Counter(r["judge_label"] for r in valid)

    return {
        "total_pairs": total,
        "dir_agree_judge_decisive": dir_agree_judge_decisive,
        "n_judge_decisive": n_judge_dec,
        "dir_agree_rm_decisive": dir_agree_rm_decisive,
        "n_rm_decisive": n_rm_dec,
        "rm_margin_threshold": margin_threshold,
        "spearman_rho": spearman,
        "accuracy_by_rm_margin": accuracy_by_margin,
        "mean_rm_margin_when_judge_tie": mean_margin_when_judge_tie,
        "n_judge_tie_or_both_bad": len(judge_tie_or_bad),
        "judge_label_dist": dict(judge_dist),
    }


def print_judge_vs_rm(results: list[dict]) -> None:
    """Print metrics comparing judge with reward model."""
    metrics = compute_judge_vs_rm_metrics(results)
    if "error" in metrics:
        print(metrics["error"])
        return

    print(f"\n{'='*60}")
    print("LLM Judge vs Reward Model — Quantitative Metrics")
    print(f"{'='*60}")
    print("Judge: 4 classes (model_a, model_b, tie, both_bad) | RM: 2 scalars (reward_a, reward_b)")
    print(f"\nTotal pairs: {metrics['total_pairs']}")

    if metrics["dir_agree_judge_decisive"] is not None:
        print(f"\n1. Direction agreement (when judge decisive): {metrics['dir_agree_judge_decisive']:.1%} (n={metrics['n_judge_decisive']})")
        print("   → When judge picks A or B, does sign(reward_a - reward_b) match?")
    if metrics["dir_agree_rm_decisive"] is not None:
        print(f"\n2. Direction agreement (when RM decisive, |Δreward|>{metrics['rm_margin_threshold']}): {metrics['dir_agree_rm_decisive']:.1%} (n={metrics['n_rm_decisive']})")
        print("   → When RM has clear preference, does judge pick the higher-reward model?")
    if metrics.get("spearman_rho") is not None:
        print(f"\n3. Spearman ρ (judge direction vs reward_a - reward_b): {metrics['spearman_rho']:.3f}")

    print("\n4. Accuracy by |reward_a - reward_b| (direction match):")
    for name, m in metrics.get("accuracy_by_rm_margin", {}).items():
        print(f"   {name:>8}: {m['accuracy']:.1%} (n={m['n']}) margin {m['margin_range']}")

    if metrics.get("mean_rm_margin_when_judge_tie") is not None:
        print(f"\n5. When judge says tie/both_bad: mean |Δreward| = {metrics['mean_rm_margin_when_judge_tie']:.3f} (n={metrics['n_judge_tie_or_both_bad']})")
        print("   → Lower = RM also uncertain; higher = RM had clear preference but judge abstained")

    print("\n6. Judge label distribution:")
    for label, count in sorted(metrics.get("judge_label_dist", {}).items()):
        print(f"   {label}: {count}")


def plot_judge_vs_rm_figures(results: list[dict], output_dir: str) -> None:
    """
    Create figures 1, 2, 4 and save as PNG in output_dir.

    Fig 1: Scatter — Judge direction vs reward difference (reward_a - reward_b)
    Fig 2: Bar — Agreement rate vs RM margin bins
    Fig 4: Distribution of Δreward by judge label (histograms/box plots)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping figures.")
        return

    valid = [r for r in results if r.get("reward_a") is not None and r.get("reward_b") is not None]
    if not valid:
        print("No valid pairs with reward values; skipping figures.")
        return

    os.makedirs(output_dir, exist_ok=True)

    label_colors = {"model_a": "#2ecc71", "model_b": "#e74c3c", "tie": "#95a5a6", "both_bad": "#34495e"}

    scatter_x = []
    scatter_y = []
    scatter_c = []
    for r in valid:
        jd = _judge_direction(r["judge_label"])
        if jd is not None:
            scatter_x.append((r["reward_a"] or 0) - (r["reward_b"] or 0))
            scatter_y.append(jd)
            scatter_c.append(label_colors.get(r["judge_label"], "#333"))

    # Add small jitter to y for visibility
    rng = random.Random(42)
    scatter_y_jitter = [y + rng.uniform(-0.08, 0.08) for y in scatter_y]

    # --- Figure 1: Scatter ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(scatter_x, scatter_y_jitter, c=scatter_c, alpha=0.5, s=20, edgecolors="none")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("reward_a − reward_b (RM preference strength)")
    ax1.set_ylabel("Judge direction (+1=A, −1=B, 0=tie/both_bad)")
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(["B preferred", "tie/both_bad", "A preferred"])
    ax1.set_title("Judge direction vs reward difference")
    ax1.set_ylim(-1.5, 1.5)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", label="model_a", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", label="model_b", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", label="tie", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#34495e", label="both_bad", markersize=8),
    ]
    ax1.legend(handles=legend_elements)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "fig1_judge_vs_reward_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved {output_dir}/fig1_judge_vs_reward_scatter.png")

    # --- Figure 2: Agreement rate vs RM margin ---
    margin_buckets = [(0.0, 0.1, "tiny"), (0.1, 0.25, "small"), (0.25, 0.5, "medium"), (0.5, float("inf"), "large")]
    names, accs, ns = [], [], []
    for lo, hi, name in margin_buckets:
        subset = [r for r in valid if lo <= abs((r["reward_a"] or 0) - (r["reward_b"] or 0)) < hi]
        if subset:
            agree = sum(
                1 for r in subset
                if (r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
                or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
            )
            names.append(name)
            accs.append(agree / len(subset))
            ns.append(len(subset))

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x_pos = range(len(names))
    ax2.bar(x_pos, accs, color="steelblue", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{n}\n(n={ns[i]})" for i, n in enumerate(names)])
    ax2.set_ylabel("Direction agreement")
    ax2.set_xlabel("|reward_a − reward_b| margin")
    ax2.set_title("Agreement rate vs RM margin")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    for i, (acc, n) in enumerate(zip(accs, ns)):
        ax2.text(i, acc + 0.02, f"{acc:.1%}", ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig2_agreement_vs_margin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {output_dir}/fig2_agreement_vs_margin.png")

    # --- Figure 4: Distribution of Δreward by judge label ---
    by_label: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        delta = (r["reward_a"] or 0) - (r["reward_b"] or 0)
        by_label[r["judge_label"]].append(delta)

    labels_order = ["model_a", "model_b", "tie", "both_bad"]
    data = [by_label.get(l, []) for l in labels_order if by_label.get(l)]
    plot_labels = [l for l in labels_order if by_label.get(l)]
    plot_colors = [label_colors.get(l, "#333") for l in plot_labels]

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    bp = ax4.boxplot(data, tick_labels=plot_labels, patch_artist=True, showfliers=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(plot_colors[i])
        patch.set_alpha(0.6)
    ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_ylabel("reward_a − reward_b")
    ax4.set_xlabel("Judge label")
    ax4.set_title("Distribution of Δreward by judge label")
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "fig4_delta_by_judge_label.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved {output_dir}/fig4_delta_by_judge_label.png")


def print_evaluation(results: list[dict], has_human_label: bool = False) -> None:
    """Print agreement metrics. For legacy format: judge vs human. For jsonl: judge vs RM."""
    if not results:
        print("No results to evaluate.")
        return

    # Check if we have reward_a/reward_b (jsonl format) or human_label (legacy)
    if results[0].get("reward_a") is not None and results[0].get("reward_b") is not None:
        print_judge_vs_rm(results)
        return

    if not has_human_label:
        print("No human or RM labels to compare against.")
        return

    total = len(results)
    judge_model = results[0].get("judge_model", "unknown")
    exact = sum(1 for r in results if r["judge_label"] == r["human_label"])
    print(f"\n{'='*60}")
    print(f"Classic Pairwise Strong Judge Results — {judge_model} ({total} battles)")
    print(f"{'='*60}")
    print(f"Exact agreement with human labels: {exact}/{total} ({100*exact/total:.1f}%)")

    direction_correct = 0
    direction_total = 0
    for r in results:
        h, j = r["human_label"], r["judge_label"]
        if h in ("model_a", "model_b"):
            direction_total += 1
            if h == j:
                direction_correct += 1
    if direction_total > 0:
        print(
            "Winner direction accuracy (when human picks a winner): "
            f"{direction_correct}/{direction_total} ({100*direction_correct/direction_total:.1f}%)"
        )

    labels = ["model_a", "model_b", "tie", "both_bad"]
    print("\nConfusion matrix (rows=human, cols=judge):")
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    for hl in labels:
        row_counts = []
        for jl in labels:
            count = sum(1 for r in results if r["human_label"] == hl and r["judge_label"] == jl)
            row_counts.append(count)
        row_str = f"{hl:>12}" + "".join(f"{c:>12}" for c in row_counts)
        print(row_str)

    judge_dist = Counter(r["judge_label"] for r in results)
    human_dist = Counter(r["human_label"] for r in results)
    print("\nLabel distribution:")
    print(f"{'Label':>12}  {'Human':>8}  {'Judge':>8}")
    for label in labels:
        print(f"{label:>12}  {human_dist.get(label,0):>8}  {judge_dist.get(label,0):>8}")


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classic pairwise strong judge for LLM math battles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input JSON or JSONL file.")
    p.add_argument("--output", required=True, help="Output JSON file for results.")
    p.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Input format: json (legacy battle) or jsonl (arena responses with rewards).",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-5.4",
        help="Judge model to use (default: gpt-5.2).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already-evaluated entries.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N entries (json format) or first N items (jsonl).",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="[jsonl] Max number of questions (items) to process.",
    )
    p.add_argument(
        "--max-pairs-per-item",
        type=int,
        default=None,
        help="[jsonl] Max pairs per question (samples if exceeded).",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="[jsonl] Total pairs budget. Balanced subsample across items (~equal per item).",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip API calls; just evaluate existing results.",
    )
    p.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory to save figures (default: figures/). Set empty to skip.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="[jsonl] Number of parallel workers for API calls (default: 4).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.format == "jsonl":
        if args.eval_only:
            if not os.path.exists(args.output):
                raise SystemExit(f"Output file {args.output} not found for --eval-only.")
            with open(args.output) as f:
                results = json.load(f)
            print_judge_vs_rm(results)
            if args.figures_dir:
                print("\nGenerating figures...")
                plot_judge_vs_rm_figures(results, args.figures_dir)
            print(f"\nResults in {args.output}")
            return

        items = load_responses_jsonl(args.input)
        print(f"Loaded {len(items)} items from {args.input}")

        pairs = sample_pairs(
            items,
            max_items=args.max_items or args.limit,
            max_pairs_per_item=args.max_pairs_per_item,
            max_pairs_total=args.max_pairs,
        )
        print(f"Sampled {len(pairs)} pairs to judge")

        done_ids: set[str] = set()
        results: list[dict] = []
        if args.resume and os.path.exists(args.output):
            with open(args.output) as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
            print(f"Resuming: {len(done_ids)} pairs already done")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Error: OPENAI_API_KEY environment variable is not set.")
        # Native OpenAI API only (no LiteLLM)
        base_url = "https://api.openai.com/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)

        pending = [(iid, a, b) for iid, a, b in pairs if f"{iid}::{a['model_label']} vs {b['model_label']}" not in done_ids]
        workers = max(1, args.workers)
        print(f"Evaluating {len(pending)} pairs with {args.judge_model} ({workers} workers, {len(done_ids)} already done)\n")

        results_lock = threading.Lock()
        completed = [0]  # mutable for closure

        def process_one(item_id: str, rec_a: dict, rec_b: dict) -> dict:
            return judge_pair_jsonl(client, args.judge_model, item_id, rec_a, rec_b)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_one, item_id, rec_a, rec_b): (item_id, rec_a, rec_b)
                for item_id, rec_a, rec_b in pending
            }
            for future in as_completed(futures):
                item_id, rec_a, rec_b = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  ERROR {item_id[:8]}... {rec_a['model_label']} vs {rec_b['model_label']}: {e}")
                    continue

                with results_lock:
                    results.append(result)
                    completed[0] += 1
                    n = completed[0]
                    total = len(pending)

                j = result["judge_label"]
                ra, rb = result.get("reward_a"), result.get("reward_b")
                delta = (ra - rb) if ra is not None and rb is not None else None
                if delta is not None and j in ("model_a", "model_b"):
                    agree = (j == "model_a" and delta > 0) or (j == "model_b" and delta < 0)
                    match = "✓" if agree else "✗"
                else:
                    match = "—"
                delta_str = f" Δreward={delta:.3f}" if delta is not None else ""
                print(f"[{n}/{total}] {rec_a['model_label']} vs {rec_b['model_label']}  judge: {j}{delta_str}  {match}")

                with results_lock:
                    with open(args.output, "w") as f:
                        json.dump(results, f, indent=2)

        print_judge_vs_rm(results)
        if args.figures_dir:
            print("\nGenerating figures...")
            plot_judge_vs_rm_figures(results, args.figures_dir)
        print(f"\nResults saved to {args.output}")
        return

    # Legacy JSON format
    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {args.input}")

    if args.limit:
        data = data[: args.limit]
        print(f"Limited to first {len(data)} entries")

    done_ids = set()
    results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} entries already done")

    if args.eval_only:
        if not results:
            with open(args.output) as f:
                results = json.load(f)
        print_evaluation(results, has_human_label=True)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is not set.")
    base_url = "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    pending = [d for d in data if d["id"] not in done_ids]
    print(f"Evaluating {len(pending)} entries with {args.judge_model} ({len(done_ids)} already done)\n")

    for i, entry in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] {entry['model_a']} vs {entry['model_b']} (human: {entry['winner']})")

        result = judge_entry(client, args.judge_model, entry)
        results.append(result)

        match = "✓" if result["judge_label"] == result["human_label"] else "✗"
        print(f"  judge: {result['judge_label']}  {match}")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print_evaluation(results, has_human_label=True)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
