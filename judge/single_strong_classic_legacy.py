#!/usr/bin/env python3
"""
Arena math 900: LLM judge + reward model, compare with human labels.

Input: data/arena_math_900.json (question, answer_a, answer_b, human_label)
- Runs LLM judge on each pair
- Runs reward model on (question, answer_a) and (question, answer_b)
- Compares: judge labels, human labels, RM rewards

Usage:
    python judge/single_strong_classic_legacy.py \
        --input data/arena_math_900.json \
        --output results/arena_math_900/judge_human_rm.json

    # Limit entries for testing
    python judge/single_strong_classic_legacy.py \
        --input data/arena_math_900.json \
        --output results/arena_math_900/judge_human_rm.json \
        --limit 100

    # Resume from partial results
    python judge/single_strong_classic_legacy.py \
        --input data/arena_math_900.json \
        --output results/arena_math_900/judge_human_rm.json \
        --resume

    # Eval only (no API calls)
    python judge/single_strong_classic_legacy.py \
        --input data/arena_math_900.json \
        --output results/arena_math_900/judge_human_rm.json \
        --eval-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reward_client import RewardClient

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


def build_pair_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return CLASSIC_PAIR_PROMPT_TEMPLATE.format(
        question=question or "(no question)",
        answer_a=answer_a or "(empty response)",
        answer_b=answer_b or "(empty response)",
    )


def build_reward_conversation(question: str, answer: str) -> list[dict]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def normalize_reward(raw: object) -> tuple[float | None, str | None]:
    if isinstance(raw, Exception):
        return None, str(raw)
    if isinstance(raw, (int, float)):
        v = float(raw)
        return (v, None) if math.isfinite(v) else (None, f"Non-finite: {v}")
    try:
        v = float(raw)
        return (v, None) if math.isfinite(v) else (None, f"Non-finite: {v}")
    except (TypeError, ValueError):
        pass
    return None, f"Unexpected: {repr(raw)[:200]}"


def call_judge(client: OpenAI, judge_model: str, prompt: str, max_retries: int = 3) -> str:
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
            print(f"  [Attempt {attempt+1}/{max_retries}] judge failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5.0 * (attempt + 1))
    return "Fallback: [[C]]"


def parse_verdict(text: str) -> str:
    upper = text.upper()
    if "[[A]]" in upper:
        return "model_a"
    if "[[B]]" in upper:
        return "model_b"
    if "[[C]]" in upper:
        return "tie"
    if "[[D]]" in upper:
        return "both_bad"
    lower = text.lower()
    if ("both" in lower and "bad" in lower) or ("both" in lower and ("incorrect" in lower or "wrong" in lower)):
        return "both_bad"
    if "assistant a" in lower and ("better" in lower or "wins" in lower):
        return "model_a"
    if "assistant b" in lower and ("better" in lower or "wins" in lower):
        return "model_b"
    return "tie"


def _direction(label: str) -> int | None:
    if label == "model_a":
        return 1
    if label == "model_b":
        return -1
    if label in ("tie", "both_bad"):
        return 0
    return None


# ── Process one entry ────────────────────────────────────────────────────

def process_entry(
    entry: dict,
    client: OpenAI,
    reward_client: RewardClient,
    reward_sem: threading.Semaphore,
    judge_model: str,
    rm_timeout: float,
) -> dict:
    """Judge + RM for one arena_math_900 entry."""
    eid = entry["id"]
    question = entry.get("question", "")
    answer_a = entry.get("answer_a", "") or ""
    answer_b = entry.get("answer_b", "") or ""
    human_label = entry.get("human_label", "tie")

    # LLM judge
    prompt = build_pair_prompt(question, answer_a, answer_b)
    raw = call_judge(client, judge_model, prompt)
    judge_label = parse_verdict(raw)

    # Reward model (with semaphore for concurrency)
    reward_a = reward_b = None
    with reward_sem:
        try:
            r_a = reward_client.get_reward(build_reward_conversation(question, answer_a), timeout=rm_timeout)
            reward_a, _ = normalize_reward(r_a)
        except Exception as ex:
            reward_a = None
        try:
            r_b = reward_client.get_reward(build_reward_conversation(question, answer_b), timeout=rm_timeout)
            reward_b, _ = normalize_reward(r_b)
        except Exception as ex:
            reward_b = None

    return {
        "id": eid,
        "model_a": entry.get("model_a"),
        "model_b": entry.get("model_b"),
        "human_label": human_label,
        "judge_label": judge_label,
        "reward_a": reward_a,
        "reward_b": reward_b,
        "judge_model": judge_model,
        "judge_raw": raw[:500],
    }


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r.get("reward_a") is not None and r.get("reward_b") is not None]
    if not valid:
        return {"error": "No valid results with rewards"}

    n = len(valid)

    # Judge vs human
    judge_human_exact = sum(1 for r in valid if r["judge_label"] == r["human_label"]) / n
    human_decisive = [r for r in valid if r["human_label"] in ("model_a", "model_b")]
    judge_human_dir = 0.0
    if human_decisive:
        judge_human_dir = sum(
            1 for r in human_decisive
            if r["judge_label"] == r["human_label"]
        ) / len(human_decisive)

    # Judge vs RM
    judge_dec = [r for r in valid if r["judge_label"] in ("model_a", "model_b")]
    judge_rm_dir = 0.0
    if judge_dec:
        judge_rm_dir = sum(
            1 for r in judge_dec
            if (r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
            or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
        ) / len(judge_dec)

    # Human vs RM
    human_rm_dir = 0.0
    if human_decisive:
        human_rm_dir = sum(
            1 for r in human_decisive
            if (r["human_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
            or (r["human_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
        ) / len(human_decisive)

    # Triple agreement
    all_agree = 0
    for r in valid:
        j, h = r["judge_label"], r["human_label"]
        rm_prefer_a = (r["reward_a"] or 0) > (r["reward_b"] or 0)
        rm_prefer_b = (r["reward_b"] or 0) > (r["reward_a"] or 0)
        if j == h:
            if j == "model_a" and rm_prefer_a:
                all_agree += 1
            elif j == "model_b" and rm_prefer_b:
                all_agree += 1
            elif j in ("tie", "both_bad") and not rm_prefer_a and not rm_prefer_b:
                all_agree += 1  # RM tie
            # else: j==h but RM disagrees
    triple_agree = all_agree / n if n else 0

    return {
        "n": n,
        "judge_human_exact": judge_human_exact,
        "judge_human_dir": judge_human_dir,
        "n_human_decisive": len(human_decisive),
        "judge_rm_dir": judge_rm_dir,
        "n_judge_decisive": len(judge_dec),
        "human_rm_dir": human_rm_dir,
        "triple_agree": triple_agree,
        "judge_dist": dict(Counter(r["judge_label"] for r in valid)),
        "human_dist": dict(Counter(r["human_label"] for r in valid)),
    }


def print_metrics(results: list[dict]) -> None:
    m = compute_metrics(results)
    if "error" in m:
        print(m["error"])
        return

    print(f"\n{'='*60}")
    print("Judge vs Human vs RM — Arena Math 900")
    print(f"{'='*60}")
    print(f"Total: {m['n']}")

    print("\n1. Judge vs Human:")
    print(f"   Exact agreement: {m['judge_human_exact']:.1%}")
    print(f"   Dir agreement (when human decisive): {m['judge_human_dir']:.1%} (n={m['n_human_decisive']})")

    print("\n2. Judge vs RM:")
    print(f"   Dir agreement (when judge decisive): {m['judge_rm_dir']:.1%} (n={m['n_judge_decisive']})")

    print("\n3. Human vs RM:")
    print(f"   Dir agreement (when human decisive): {m['human_rm_dir']:.1%} (n={m['n_human_decisive']})")

    print("\n4. Triple agreement (judge=human & RM direction matches):")
    print(f"   {m['triple_agree']:.1%}")

    print("\n5. Confusion: Judge vs Human")
    labels = ["model_a", "model_b", "tie", "both_bad"]
    print(f"   {'':>10}" + "".join(f"{l:>10}" for l in labels))
    for hl in labels:
        row = [sum(1 for r in results if r["human_label"] == hl and r["judge_label"] == jl) for jl in labels]
        print(f"   {hl:>10}" + "".join(f"{c:>10}" for c in row))

    print("\n6. Label distributions:")
    print(f"   Human: {m['human_dist']}")
    print(f"   Judge: {m['judge_dist']}")


# ── Figures ───────────────────────────────────────────────────────────────

def plot_figures(results: list[dict], output_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping figures.")
        return

    valid = [r for r in results if r.get("reward_a") is not None and r.get("reward_b") is not None]
    if not valid:
        print("No valid results; skipping figures.")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors = {"model_a": "#2ecc71", "model_b": "#e74c3c", "tie": "#95a5a6", "both_bad": "#34495e"}

    # Fig 1: Judge vs Human confusion matrix (heatmap-style bar)
    labels = ["model_a", "model_b", "tie", "both_bad"]
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    mat = [[sum(1 for r in valid if r["human_label"] == hl and r["judge_label"] == jl) for jl in labels] for hl in labels]
    im = ax1.imshow(mat, cmap="Blues")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Judge")
    ax1.set_ylabel("Human")
    ax1.set_title("Judge vs Human confusion matrix")
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, str(mat[i][j]), ha="center", va="center")
    fig1.colorbar(im, ax=ax1, label="count")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "fig1_judge_vs_human_confusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved {output_dir}/fig1_judge_vs_human_confusion.png")

    # Fig 2: Δreward by human label
    by_human: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        by_human[r["human_label"]].append((r["reward_a"] or 0) - (r["reward_b"] or 0))

    order = [l for l in labels if by_human.get(l)]
    data = [by_human[l] for l in order]
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bp = ax2.boxplot(data, tick_labels=order, patch_artist=True)
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(colors.get(order[i], "#333"))
        p.set_alpha(0.6)
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_ylabel("reward_a − reward_b")
    ax2.set_xlabel("Human label")
    ax2.set_title("Δreward by human label")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig2_delta_by_human_label.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {output_dir}/fig2_delta_by_human_label.png")

    # Fig 3: Δreward by judge label
    by_judge: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        by_judge[r["judge_label"]].append((r["reward_a"] or 0) - (r["reward_b"] or 0))

    order = [l for l in labels if by_judge.get(l)]
    data = [by_judge[l] for l in order]
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    bp = ax3.boxplot(data, tick_labels=order, patch_artist=True)
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(colors.get(order[i], "#333"))
        p.set_alpha(0.6)
    ax3.axhline(0, color="gray", linestyle="--")
    ax3.set_ylabel("reward_a − reward_b")
    ax3.set_xlabel("Judge label")
    ax3.set_title("Δreward by judge label")
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "fig3_delta_by_judge_label.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved {output_dir}/fig3_delta_by_judge_label.png")

    # Fig 4: Agreement by RM margin (Judge vs Human, Judge vs RM)
    buckets = [(0.0, 0.1, "tiny"), (0.1, 0.25, "small"), (0.25, 0.5, "medium"), (0.5, float("inf"), "large")]
    names, acc_judge_human, acc_judge_rm, ns = [], [], [], []
    for lo, hi, name in buckets:
        sub = [r for r in valid if lo <= abs((r["reward_a"] or 0) - (r["reward_b"] or 0)) < hi]
        if sub:
            names.append(name)
            ns.append(len(sub))
            jh = sum(1 for r in sub if r["judge_label"] == r["human_label"]) / len(sub)
            jr = sum(
                1 for r in sub
                if (r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
                or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
            ) / len(sub)
            acc_judge_human.append(jh)
            acc_judge_rm.append(jr)

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    w = 0.35
    ax4.bar([i - w/2 for i in x], acc_judge_human, w, label="Judge vs Human", color="steelblue")
    ax4.bar([i + w/2 for i in x], acc_judge_rm, w, label="Judge vs RM", color="coral")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{n}\n(n={ns[i]})" for i, n in enumerate(names)])
    ax4.set_ylabel("Agreement")
    ax4.set_xlabel("|Δreward| margin")
    ax4.set_title("Agreement by RM margin")
    ax4.legend()
    ax4.set_ylim(0, 1.05)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "fig4_agreement_by_margin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved {output_dir}/fig4_agreement_by_margin.png")

    # Fig 5: Triple agreement breakdown
    both_dec = [r for r in valid if r["human_label"] in ("model_a", "model_b") and r["judge_label"] in ("model_a", "model_b")]
    jh_agree = sum(1 for r in both_dec if r["judge_label"] == r["human_label"])
    jr_agree = sum(
        1 for r in both_dec
        if (r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
        or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
    )
    hr_agree = sum(
        1 for r in both_dec
        if (r["human_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
        or (r["human_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0))
    )
    triple = sum(
        1 for r in both_dec
        if r["judge_label"] == r["human_label"]
        and ((r["judge_label"] == "model_a" and (r["reward_a"] or 0) > (r["reward_b"] or 0))
             or (r["judge_label"] == "model_b" and (r["reward_b"] or 0) > (r["reward_a"] or 0)))
    )

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    cats = ["Judge=Human", "Judge=RM", "Human=RM", "All three"]
    vals = [jh_agree/len(both_dec) if both_dec else 0, jr_agree/len(both_dec) if both_dec else 0,
            hr_agree/len(both_dec) if both_dec else 0, triple/len(both_dec) if both_dec else 0]
    ax5.bar(cats, vals, color=["steelblue", "coral", "seagreen", "purple"])
    ax5.set_ylabel("Agreement (when both decisive)")
    ax5.set_title("Pairwise and triple agreement")
    ax5.set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        ax5.text(i, v + 0.02, f"{v:.1%}", ha="center", va="bottom")
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, "fig5_triple_agreement.png"), dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"  Saved {output_dir}/fig5_triple_agreement.png")


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arena math 900: LLM judge + RM, compare with human.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="arena_math_900.json")
    p.add_argument("--output", required=True, help="Output JSON")
    p.add_argument("--judge-model", default="gpt-5.4", help="Judge model")
    p.add_argument("--limit", type=int, default=None, help="Max entries")
    p.add_argument("--resume", action="store_true", help="Resume from output")
    p.add_argument("--eval-only", action="store_true", help="No API calls, eval only")
    p.add_argument("--figures-dir", default="figures/consistency_legacy", help="Figures output dir")
    p.add_argument("--workers", type=int, default=8, help="Parallel judge workers")
    p.add_argument("--rm-max-concurrency", type=int, default=6, help="Max concurrent RM calls")
    p.add_argument("--rm-timeout", type=int, default=300, help="RM request timeout (s)")
    p.add_argument("--rm-base-url", default=None, help="RM base URL (default: ARENA_RM_BASE_URL)")
    p.add_argument("--rm-token", default=None, help="RM token (default: ARENA_RM_TOKEN)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input) as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    print(f"Loaded {len(data)} entries from {args.input}")

    if args.limit:
        data = data[: args.limit]
        print(f"Limited to {len(data)}")

    done_ids = set()
    results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} done")

    if args.eval_only:
        if not results:
            with open(args.output) as f:
                results = json.load(f)
        print_metrics(results)
        if args.figures_dir:
            print("\nGenerating figures...")
            plot_figures(results, args.figures_dir)
        print(f"\nResults in {args.output}")
        return

    # API clients
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY required")
    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    rm_url = args.rm_base_url or os.environ.get("ARENA_RM_BASE_URL")
    rm_token = args.rm_token or os.environ.get("ARENA_RM_TOKEN")
    if not rm_url or not rm_token:
        raise SystemExit("ARENA_RM_BASE_URL and ARENA_RM_TOKEN required (or --rm-base-url, --rm-token)")
    reward_client = RewardClient(base_url=rm_url, token=rm_token)
    reward_sem = threading.Semaphore(min(args.rm_max_concurrency, 8))

    pending = [e for e in data if e["id"] not in done_ids]
    print(f"Processing {len(pending)} entries (judge + RM, {args.workers} workers, RM concurrency≤{args.rm_max_concurrency})\n")

    def process_one(entry: dict) -> dict:
        return process_entry(
            entry, client, reward_client, reward_sem,
            args.judge_model, float(args.rm_timeout),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, e): e for e in pending}
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  ERROR {entry['id']}: {e}")
                continue
            results.append(r)
            j, h = r["judge_label"], r["human_label"]
            match = "✓" if j == h else "✗"
            print(f"[{len(results)}/{len(data)}] {entry['id'][:8]}... judge={j} human={h} {match}")

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print_metrics(results)
    if args.figures_dir:
        print("\nGenerating figures...")
        plot_figures(results, args.figures_dir)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
