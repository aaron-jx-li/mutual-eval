#!/usr/bin/env python3
"""
Mutual pairwise judge for LLM math battles.

Each competing model sees BOTH responses side-by-side and picks a winner.
The two verdicts are then aggregated, weighting concessions (a model
admitting the other's response is better) more heavily than self-serving
claims.

Aggregation rules:
    Both agree on a label                        →  use that label
    One picks a winner, the other says tie        →  use the winner
    They pick opposite winners (both self-serving) →  tie
    Either says both_bad                            →  both_bad

Usage:
    python judge/mutual_pair.py \\
        --input data/arena_140k_math_openai_single.json \\
        --output data/arena_140k_math_mutual_pair.json

    # Resume from partial results
    python judge/mutual_pair.py \\
        --input data/arena_140k_math_openai_single.json \\
        --output data/arena_140k_math_mutual_pair.json \\
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


def extract_questions(conversation: list[dict]) -> str:
    """Extract user questions from a conversation."""
    parts = []
    turn = 0
    for msg in conversation:
        if msg["role"] == "user":
            turn += 1
            text = extract_text(msg["content"])
            parts.append(f"**Turn {turn}:** {text}")
    return "\n\n".join(parts)


def extract_responses(conversation: list[dict]) -> str:
    """Extract assistant responses from a conversation."""
    parts = []
    turn = 0
    multi_turn = sum(1 for m in conversation if m["role"] == "user") > 1
    for msg in conversation:
        if msg["role"] == "user":
            turn += 1
        elif msg["role"] == "assistant":
            text = extract_text(msg["content"])
            prefix = f"**Turn {turn}:**\n" if multi_turn else ""
            parts.append(f"{prefix}{text}")
    return "\n\n".join(parts)


def build_pair_prompt(entry: dict) -> str:
    """Build a side-by-side comparison prompt from a battle entry."""
    question_text = extract_questions(entry["conversation_a"])
    response_a_text = extract_responses(entry["conversation_a"])
    response_b_text = extract_responses(entry["conversation_b"])

    return PAIR_PROMPT_TEMPLATE.format(
        question=question_text,
        response_a=response_a_text,
        response_b=response_b_text,
    )


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """Call a model via the Responses API."""
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=prompt,
                max_output_tokens=4096,
                store=False,
            )
            return resp.output_text.strip()

        except Exception as e:
            print(f"  [Attempt {attempt + 1}/{max_retries}] {model} failed: {e}")
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

    lower = text.lower()
    for label in ["model_a", "model_b", "both_bad", "tie"]:
        if f'"verdict": "{label}"' in lower or f'"verdict":"{label}"' in lower:
            return {"verdict": label, "explanation": f"PARSE_FALLBACK: {text[:500]}"}

    if "both" in lower and "bad" in lower:
        return {"verdict": "both_bad", "explanation": f"PARSE_FALLBACK: {text[:500]}"}
    if "response a" in lower and ("better" in lower or "correct" in lower):
        return {"verdict": "model_a", "explanation": f"PARSE_FALLBACK: {text[:500]}"}
    if "response b" in lower and ("better" in lower or "correct" in lower):
        return {"verdict": "model_b", "explanation": f"PARSE_FALLBACK: {text[:500]}"}

    return {"verdict": "tie", "explanation": f"PARSE_FALLBACK: {text[:500]}"}


# ── Aggregation ──────────────────────────────────────────────────────

AGGREGATE_TABLE = {
    # (verdict_from_A, verdict_from_B) → final_label
    #
    # Agreement
    ("model_a", "model_a"): "model_a",
    ("model_b", "model_b"): "model_b",
    ("tie", "tie"): "tie",
    ("both_bad", "both_bad"): "both_bad",
    # One picks winner, other says tie → trust the winner signal
    ("model_a", "tie"): "model_a",
    ("tie", "model_a"): "model_a",
    ("model_b", "tie"): "model_b",
    ("tie", "model_b"): "model_b",
    # Opposite winners → tie (cancel out)
    ("model_a", "model_b"): "tie",
    ("model_b", "model_a"): "tie",
    # Either says both_bad → both_bad (at least one model sees errors in both)
    ("model_a", "both_bad"): "both_bad",
    ("both_bad", "model_a"): "both_bad",
    ("model_b", "both_bad"): "both_bad",
    ("both_bad", "model_b"): "both_bad",
    ("both_bad", "tie"): "both_bad",
    ("tie", "both_bad"): "both_bad",
}


def aggregate_verdicts(verdict_a: str, verdict_b: str) -> str:
    """Aggregate verdicts from model A and model B into a final label."""
    return AGGREGATE_TABLE.get((verdict_a, verdict_b), "tie")


def classify_verdict(evaluator: str, verdict: str) -> str:
    """Classify a verdict as 'concession', 'self-serving', or 'neutral'."""
    if evaluator == "a":
        if verdict == "model_b":
            return "concession"
        if verdict == "model_a":
            return "self-serving"
    elif evaluator == "b":
        if verdict == "model_a":
            return "concession"
        if verdict == "model_b":
            return "self-serving"
    return "neutral"


# ── Core evaluation ──────────────────────────────────────────────────

def mutual_pair_evaluate_entry(client: OpenAI, entry: dict) -> dict:
    """Both competing models evaluate the pair side-by-side."""
    model_a = entry["model_a"]
    model_b = entry["model_b"]

    prompt = build_pair_prompt(entry)

    raw_a = call_model(client, model_a, prompt)
    raw_b = call_model(client, model_b, prompt)

    parsed_a = parse_verdict(raw_a)
    parsed_b = parse_verdict(raw_b)

    final_label = aggregate_verdicts(parsed_a["verdict"], parsed_b["verdict"])

    return {
        "id": entry["id"],
        "model_a": model_a,
        "model_b": model_b,
        "human_label": entry["winner"],
        "mutual_pair_label": final_label,
        "verdict_from_a": {
            "raw_response": raw_a,
            "type": classify_verdict("a", parsed_a["verdict"]),
            **parsed_a,
        },
        "verdict_from_b": {
            "raw_response": raw_b,
            "type": classify_verdict("b", parsed_b["verdict"]),
            **parsed_b,
        },
    }


# ── Evaluation metrics ───────────────────────────────────────────────

def print_evaluation(results: list[dict]) -> None:
    """Print agreement metrics between mutual-pair labels and human labels."""
    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return

    exact = sum(1 for r in results if r["mutual_pair_label"] == r["human_label"])
    print(f"\n{'='*60}")
    print(f"Mutual Pairwise Results ({total} battles)")
    print(f"{'='*60}")
    print(f"Exact agreement with human labels: {exact}/{total} ({100*exact/total:.1f}%)")

    direction_correct = 0
    direction_total = 0
    for r in results:
        h, m = r["human_label"], r["mutual_pair_label"]
        if h in ("model_a", "model_b"):
            direction_total += 1
            if h == m:
                direction_correct += 1
    if direction_total > 0:
        print(f"Winner direction accuracy (when human picks a winner): "
              f"{direction_correct}/{direction_total} ({100*direction_correct/direction_total:.1f}%)")

    # Concession analysis
    concessions = sum(
        1 for r in results
        if r["verdict_from_a"]["type"] == "concession"
        or r["verdict_from_b"]["type"] == "concession"
    )
    self_serving = sum(
        1 for r in results
        if r["verdict_from_a"]["type"] == "self-serving"
        or r["verdict_from_b"]["type"] == "self-serving"
    )
    print(f"\nConcession analysis:")
    print(f"  Entries with at least one concession: {concessions}/{total}")
    print(f"  Entries with at least one self-serving verdict: {self_serving}/{total}")

    # Agreement between the two evaluators
    agree = sum(1 for r in results
                if r["verdict_from_a"]["verdict"] == r["verdict_from_b"]["verdict"])
    print(f"  Evaluator agreement: {agree}/{total} ({100*agree/total:.1f}%)")

    labels = ["model_a", "model_b", "tie", "both_bad"]
    print(f"\nConfusion matrix (rows=human, cols=mutual_pair):")
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    for hl in labels:
        row_counts = []
        for ml in labels:
            count = sum(1 for r in results if r["human_label"] == hl and r["mutual_pair_label"] == ml)
            row_counts.append(count)
        row_str = f"{hl:>12}" + "".join(f"{c:>12}" for c in row_counts)
        print(row_str)

    print(f"\nPer-label metrics:")
    print(f"{'Label':>12}  {'Precision':>10}  {'Recall':>10}  {'Human N':>8}  {'Pred N':>8}")
    for label in labels:
        pred_n = sum(1 for r in results if r["mutual_pair_label"] == label)
        true_n = sum(1 for r in results if r["human_label"] == label)
        tp = sum(1 for r in results if r["mutual_pair_label"] == label and r["human_label"] == label)
        prec = tp / pred_n if pred_n > 0 else 0
        rec = tp / true_n if true_n > 0 else 0
        print(f"{label:>12}  {prec:>10.2%}  {rec:>10.2%}  {true_n:>8}  {pred_n:>8}")

    mp_dist = Counter(r["mutual_pair_label"] for r in results)
    human_dist = Counter(r["human_label"] for r in results)
    print(f"\nLabel distribution:")
    print(f"{'Label':>12}  {'Human':>8}  {'MutualPair':>10}")
    for label in labels:
        print(f"{label:>12}  {human_dist.get(label,0):>8}  {mp_dist.get(label,0):>10}")


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mutual pairwise judge for LLM math battles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input JSON file with battle data.")
    p.add_argument("--output", required=True, help="Output JSON file for results.")
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
    print(f"Evaluating {len(pending)} entries ({len(done_ids)} already done)\n")

    for i, entry in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] {entry['model_a']} vs {entry['model_b']} "
              f"(human: {entry['winner']})")

        result = mutual_pair_evaluate_entry(client, entry)
        results.append(result)

        va = result["verdict_from_a"]
        vb = result["verdict_from_b"]
        match = "✓" if result["mutual_pair_label"] == result["human_label"] else "✗"
        print(f"  A says: {va['verdict']} ({va['type']})  |  "
              f"B says: {vb['verdict']} ({vb['type']})  |  "
              f"final: {result['mutual_pair_label']}  {match}")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print_evaluation(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
