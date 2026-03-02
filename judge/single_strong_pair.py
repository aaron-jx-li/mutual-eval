#!/usr/bin/env python3
"""
Pairwise strong judge for LLM math battles.

Uses a strong judge model (default: gpt-5.2) to compare both responses
side-by-side in a single prompt and directly choose from the four labels:
model_a, model_b, tie, both_bad.

Usage:
    python judge/single_strong_pair.py \\
        --input data/arena_140k_math_openai_single.json \\
        --output data/arena_140k_math_single_strong_pair.json

    # Use a different judge model
    python judge/single_strong_pair.py \\
        --input data/arena_140k_math_openai_single.json \\
        --output data/arena_140k_math_single_strong_pair.json \\
        --judge-model gpt-4.1

    # Resume from partial results
    python judge/single_strong_pair.py \\
        --input data/arena_140k_math_openai_single.json \\
        --output data/arena_140k_math_single_strong_pair.json \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter

from openai import OpenAI

SYSTEM_INSTRUCTIONS = (
    "You are an expert mathematics judge. "
    "You compare two responses to the same math question and decide which is better. "
    "Focus ONLY on mathematical correctness, not style or presentation."
)

PAIR_PROMPT_TEMPLATE = """\
Below are two responses (Response A and Response B) to the same math question. \
Compare them and decide which is mathematically better.

## Question

{question}

## Response A

{response_a}

## Response B

{response_b}

## Instructions

Compare the two responses above on mathematical correctness:
- Incorrect calculations or arithmetic
- Flawed logical reasoning or invalid proof steps
- Wrong formulas or theorems applied
- Incorrect final answer
- Missing critical cases or conditions

Choose one of these four verdicts:
- "model_a": Response A is mathematically better (B has errors that A does not)
- "model_b": Response B is mathematically better (A has errors that B does not)
- "tie": Both responses are mathematically correct (or equally good)
- "both_bad": Both responses contain mathematical errors

Respond with ONLY a JSON object (no markdown fencing):
{{"verdict": "model_a" or "model_b" or "tie" or "both_bad", "explanation": "brief justification"}}"""


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


def format_turns(conversation: list[dict], role_filter: str) -> str:
    """Extract all messages of a given role from a conversation."""
    parts = []
    turn = 0
    for msg in conversation:
        role = msg["role"]
        text = extract_text(msg["content"])
        if role == "user":
            turn += 1
        if role == role_filter:
            label = f"Turn {turn}" if turn > 0 else ""
            parts.append(f"**{label}:**\n{text}" if label else text)
    return "\n\n".join(parts)


def build_pair_prompt(entry: dict) -> str:
    """Build a side-by-side comparison prompt from a battle entry."""
    conv_a = entry["conversation_a"]
    conv_b = entry["conversation_b"]

    # Extract user questions (same in both conversations)
    question_parts = []
    turn = 0
    for msg in conv_a:
        if msg["role"] == "user":
            turn += 1
            text = extract_text(msg["content"])
            question_parts.append(f"**Turn {turn}:** {text}")
    question_text = "\n\n".join(question_parts)

    # Extract assistant responses from each conversation
    response_a_parts = []
    response_b_parts = []
    turn_a = turn_b = 0
    for msg in conv_a:
        if msg["role"] == "user":
            turn_a += 1
        elif msg["role"] == "assistant":
            text = extract_text(msg["content"])
            prefix = f"**Turn {turn_a}:**\n" if turn_a > 1 or len(conv_a) > 2 else ""
            response_a_parts.append(f"{prefix}{text}")
    for msg in conv_b:
        if msg["role"] == "user":
            turn_b += 1
        elif msg["role"] == "assistant":
            text = extract_text(msg["content"])
            prefix = f"**Turn {turn_b}:**\n" if turn_b > 1 or len(conv_b) > 2 else ""
            response_b_parts.append(f"{prefix}{text}")

    response_a_text = "\n\n".join(response_a_parts)
    response_b_text = "\n\n".join(response_b_parts)

    return PAIR_PROMPT_TEMPLATE.format(
        question=question_text,
        response_a=response_a_text,
        response_b=response_b_text,
    )


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
    return '{"verdict": "tie", "explanation": "API_ERROR"}'


VALID_VERDICTS = {"model_a", "model_b", "tie", "both_bad"}


def parse_verdict(response_text: str) -> dict:
    """Parse the JSON verdict response, with fallback for malformed output."""
    text = response_text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        verdict = str(parsed.get("verdict", "tie")).lower().strip()
        if verdict not in VALID_VERDICTS:
            verdict = "tie"
        return {
            "verdict": verdict,
            "explanation": str(parsed.get("explanation", "")),
        }
    except json.JSONDecodeError:
        pass

    # Fallback: search for verdict strings in raw text
    lower = text.lower()
    for label in ["model_a", "model_b", "both_bad", "tie"]:
        if f'"verdict": "{label}"' in lower or f'"verdict":"{label}"' in lower:
            return {"verdict": label, "explanation": f"PARSE_FALLBACK: {text[:500]}"}

    # Last resort: keyword heuristics
    if "both" in lower and "bad" in lower:
        return {"verdict": "both_bad", "explanation": f"PARSE_FALLBACK: {text[:500]}"}
    if "response a" in lower and ("better" in lower or "correct" in lower):
        return {"verdict": "model_a", "explanation": f"PARSE_FALLBACK: {text[:500]}"}
    if "response b" in lower and ("better" in lower or "correct" in lower):
        return {"verdict": "model_b", "explanation": f"PARSE_FALLBACK: {text[:500]}"}

    return {"verdict": "tie", "explanation": f"PARSE_FALLBACK: {text[:500]}"}


# ── Core evaluation ──────────────────────────────────────────────────

def judge_entry(client: OpenAI, judge_model: str, entry: dict) -> dict:
    """Use a single strong judge to compare both responses side-by-side."""
    prompt = build_pair_prompt(entry)
    raw_response = call_judge(client, judge_model, prompt)
    parsed = parse_verdict(raw_response)

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


# ── Evaluation metrics ───────────────────────────────────────────────

def print_evaluation(results: list[dict]) -> None:
    """Print agreement metrics between judge labels and human labels."""
    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return

    judge_model = results[0].get("judge_model", "unknown")
    exact = sum(1 for r in results if r["judge_label"] == r["human_label"])
    print(f"\n{'='*60}")
    print(f"Pairwise Strong Judge Results — {judge_model} ({total} battles)")
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
        print(f"Winner direction accuracy (when human picks a winner): "
              f"{direction_correct}/{direction_total} ({100*direction_correct/direction_total:.1f}%)")

    labels = ["model_a", "model_b", "tie", "both_bad"]
    print(f"\nConfusion matrix (rows=human, cols=judge):")
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    for hl in labels:
        row_counts = []
        for jl in labels:
            count = sum(1 for r in results if r["human_label"] == hl and r["judge_label"] == jl)
            row_counts.append(count)
        row_str = f"{hl:>12}" + "".join(f"{c:>12}" for c in row_counts)
        print(row_str)

    print(f"\nPer-label metrics:")
    print(f"{'Label':>12}  {'Precision':>10}  {'Recall':>10}  {'Human N':>8}  {'Pred N':>8}")
    for label in labels:
        pred_n = sum(1 for r in results if r["judge_label"] == label)
        true_n = sum(1 for r in results if r["human_label"] == label)
        tp = sum(1 for r in results if r["judge_label"] == label and r["human_label"] == label)
        prec = tp / pred_n if pred_n > 0 else 0
        rec = tp / true_n if true_n > 0 else 0
        print(f"{label:>12}  {prec:>10.2%}  {rec:>10.2%}  {true_n:>8}  {pred_n:>8}")

    judge_dist = Counter(r["judge_label"] for r in results)
    human_dist = Counter(r["human_label"] for r in results)
    print(f"\nLabel distribution:")
    print(f"{'Label':>12}  {'Human':>8}  {'Judge':>8}")
    for label in labels:
        print(f"{label:>12}  {human_dist.get(label,0):>8}  {judge_dist.get(label,0):>8}")


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairwise strong judge for LLM math battles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input JSON file with battle data.")
    p.add_argument("--output", required=True, help="Output JSON file for results.")
    p.add_argument("--judge-model", default="gpt-5.2",
                   help="Judge model to use (default: gpt-5.2).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing output file, skipping already-evaluated entries.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N entries (for testing).")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip API calls; just evaluate existing results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {args.input}")

    if args.limit:
        data = data[: args.limit]
        print(f"Limited to first {len(data)} entries")

    done_ids: set[str] = set()
    results: list[dict] = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} entries already done")

    if args.eval_only:
        if not results:
            with open(args.output) as f:
                results = json.load(f)
        print_evaluation(results)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is not set.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    pending = [d for d in data if d["id"] not in done_ids]
    print(f"Evaluating {len(pending)} entries with {args.judge_model} "
          f"({len(done_ids)} already done)\n")

    for i, entry in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] {entry['model_a']} vs {entry['model_b']} "
              f"(human: {entry['winner']})")

        result = judge_entry(client, args.judge_model, entry)
        results.append(result)

        match = "✓" if result["judge_label"] == result["human_label"] else "✗"
        print(f"  judge: {result['judge_label']}  {match}")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print_evaluation(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
