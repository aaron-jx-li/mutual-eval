#!/usr/bin/env python3
"""
Informal test: estimate the fraction of math-focused prompts in arena-expert-5k.

Uses the same judge prompt and classification logic as sample_arena_math.py,
run against a random sample of English rows from the expert-5k dataset.

Usage:
  # Quick estimate: sample 200 English rows
  python eval_arena/check_expert5k_math_fraction.py

  # Full dataset (expensive: ~3k LLM calls)
  python eval_arena/check_expert5k_math_fraction.py --sample-size 0

  # Custom sample size with a fixed seed
  python eval_arena/check_expert5k_math_fraction.py --sample-size 500 --seed 123
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


JUDGE_SYSTEM = (
    "You are a strict prompt classifier. "
    "Decide whether the user prompt is math-focused."
)

# Identical definition to sample_arena_math.py
JUDGE_USER_TEMPLATE = """\
Classify whether this user prompt is math-focused.

Definition of math-focused:
- The core task requires mathematical reasoning/calculation/proof/derivation,
  symbolic manipulation, quantitative word-problem solving, geometry, algebra,
  number theory, probability/statistics, optimization, or equation solving.
- Include prompts that are primarily about solving/understanding a math problem.
- Exclude general coding prompts unless math reasoning is central.
- Exclude general trivia, writing, translation, legal/medical advice, etc.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_math": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "short explanation"
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate math fraction in arena-expert-5k.")
    p.add_argument("--dataset", default="lmarena-ai/arena-expert-5k")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of English rows to judge. 0 = all English rows.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default="gpt-4.1-mini")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def load_env() -> None:
    here = Path(__file__).resolve().parent
    for env_path in [here / ".env", here.parent / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip().lstrip("export").strip()
            v = v.strip().strip("'\"")
            os.environ.setdefault(k, v)


def extract_user_prompt(row: dict[str, Any]) -> str:
    for conv_key in ("conversation_a", "conversation_b"):
        conv = row.get(conv_key)
        if isinstance(conv, list):
            for msg in conv:
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                    content = msg.get("content", "")
                    text = content if isinstance(content, str) else (
                        "\n".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        ) if isinstance(content, list) else ""
                    )
                    if text.strip():
                        return text.strip()
    return ""


def classify_math(client: OpenAI, model: str, prompt: str, timeout: int) -> dict[str, Any]:
    user = JUDGE_USER_TEMPLATE.format(prompt=prompt[:3000])  # truncate very long prompts
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        return {
            "is_math": bool(parsed.get("is_math", False)),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "reason": str(parsed.get("reason", "")),
        }
    except Exception as e:
        return {"is_math": False, "confidence": 0.0, "reason": f"error: {e}"}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    args = parse_args()
    load_env()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=args.timeout,
    )

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    rows = [dict(r) for r in ds]
    print(f"  Total rows: {len(rows)}")

    # Filter to English
    en_rows = [r for r in rows if str(r.get("language", "")).strip().lower() == "en"]
    print(f"  English rows: {len(en_rows)}")

    # Sample
    rng = random.Random(args.seed)
    rng.shuffle(en_rows)
    sample = en_rows if args.sample_size == 0 else en_rows[: args.sample_size]
    print(f"  Rows to judge: {len(sample)}")
    print()

    results = []
    errors = 0
    for row in tqdm(sample, desc="Classifying"):
        prompt = extract_user_prompt(row)
        if not prompt:
            errors += 1
            continue
        result = classify_math(client, args.judge_model, prompt, args.timeout)
        results.append({
            "id": str(row.get("id", "")),
            "prompt_len": len(prompt),
            **result,
        })

    # Aggregate
    n = len(results)
    n_math = sum(1 for r in results if r["is_math"])
    n_math_hc = sum(1 for r in results if r["is_math"] and r["confidence"] >= args.min_confidence)
    lo, hi = wilson_ci(n_math_hc, n)

    print(f"\n{'='*50}")
    print(f"Judged:          {n} rows  (errors/skipped: {errors})")
    print(f"is_math=True:    {n_math} / {n}  ({100*n_math/n:.1f}%)")
    print(f"is_math=True, confidence >= {args.min_confidence}:  {n_math_hc} / {n}  ({100*n_math_hc/n:.1f}%)")
    print(f"95% Wilson CI:   [{100*lo:.1f}%, {100*hi:.1f}%]")
    print()

    # Extrapolate to full English set
    rate = n_math_hc / n if n else 0
    print(f"Extrapolated to {len(en_rows)} English rows in expert-5k:")
    print(f"  Estimated math questions: {rate * len(en_rows):.0f}  "
          f"[{lo * len(en_rows):.0f}, {hi * len(en_rows):.0f}]")

    # Breakdown: confidence distribution among math rows
    if n_math > 0:
        conf_vals = sorted(r["confidence"] for r in results if r["is_math"])
        print(f"\nConfidence dist (is_math=True, n={n_math}):")
        print(f"  min={conf_vals[0]:.2f}  median={conf_vals[len(conf_vals)//2]:.2f}  max={conf_vals[-1]:.2f}")

    # Show sample math prompts
    math_rows = [r for r in results if r["is_math"] and r["confidence"] >= args.min_confidence]
    if math_rows:
        print(f"\nSample math prompts (up to 5):")
        for r in math_rows[:5]:
            print(f"  [conf={r['confidence']:.2f}, len={r['prompt_len']}] {r['reason']}")


if __name__ == "__main__":
    main()
