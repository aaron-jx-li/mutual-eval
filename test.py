import argparse
import json
import os
import re
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an expert math evaluator. Compare two model answers for the SAME math question. "
    "Focus on mathematical correctness, sound reasoning, and clarity. "
    "Choose which answer is better. "
    "If both answers are equally good, output 'tie'. "
    "If both are clearly wrong or nonsensical, output 'both_bad'. "
    "Use the structured response format and set 'choice' to exactly one value from this set:\n"
    "model_a, model_b, tie, both_bad."
)

SOURCE_PATH = Path("data/arena_140k_math_filtered.json")
OUT_PATH = Path("data/arena_math_gpt5mini_900.json")
MODEL_NAME = "gpt-5-mini"
VALID_LABELS = {"model_a", "model_b", "tie", "both_bad"}
FENCE_RE = re.compile(r"^```[a-z]*\n|\n```$", flags=re.IGNORECASE)
EDGE_PUNCT_RE = re.compile(r"^[`\"'.,:;!?()\[\]{}]+|[`\"'.,:;!?()\[\]{}]+$")


class JudgeChoice(BaseModel):
    choice: Literal["model_a", "model_b", "tie", "both_bad"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=None, help="Only judge first N examples (mini run)")
    return parser.parse_args()


def build_user_prompt(question, ans_a, ans_b):
    return (
        f"Question:\n{question}\n\n"
        f"Answer A (model_a):\n{ans_a}\n\n"
        f"Answer B (model_b):\n{ans_b}\n\n"
        "Which answer is better for correctness and reasoning?"
    )


def get_output_path(n):
    if n is None:
        return OUT_PATH

    base = OUT_PATH
    return base.with_name(f"{base.stem}.mini_n{n}{base.suffix}")


def normalize_token(token):
    return EDGE_PUNCT_RE.sub("", token.strip().lower())


def parse_fallback_choice(raw):
    normalized = FENCE_RE.sub("", raw.strip().lower()).strip()
    if normalize_token(normalized) in VALID_LABELS:
        return normalize_token(normalized)

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for line in reversed(lines):
        token = line.split(maxsplit=1)[0]
        token = normalize_token(token)
        if token in VALID_LABELS:
            return token

    return "unknown"


def load_done_labels(path):
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as f:
        existing = json.load(f)

    done = {}
    for item in existing:
        item_id = item.get("id")
        label = item.get("judge_label")
        if item_id is None or label is None:
            continue
        done[item_id] = label

    return done


def load_source_data(n):
    with SOURCE_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if n is not None:
        return data[:n]
    return data


def save_progress(path, data):
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def extract_judge_label(client, prompt):
    resp = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=JudgeChoice,
        reasoning_effort="low",
        max_completion_tokens=2560,
    )
    message = resp.choices[0].message
    raw = (message.content or "").strip()

    if message.parsed is not None:
        return message.parsed.choice, raw

    return parse_fallback_choice(raw), raw


def main():
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    client = None
    data = load_source_data(args.n)

    out_path = get_output_path(args.n)

    # Resume: load existing progress if available
    done = load_done_labels(out_path)
    if out_path != OUT_PATH:
        # Reuse labels from the full run output while keeping mini-run writes isolated.
        done = {**load_done_labels(OUT_PATH), **done}

    if done:
        judged_in_scope = sum(1 for item in data if item["id"] in done)
        remaining = len(data) - judged_in_scope
        print(f"Resuming: {judged_in_scope} already judged, {remaining} remaining")

    skipped = 0
    for item in tqdm(data, total=len(data), desc="Judging"):
        if item["id"] in done:
            item["judge_label"] = done[item["id"]]
            skipped += 1
            continue

        if client is None:
            if not api_key:
                raise SystemExit("OPENAI_API_KEY is required")
            client = OpenAI(
                api_key=api_key,
                base_url="https://us.api.openai.com/v1",
            )

        prompt = build_user_prompt(item["question"], item["answer_a"], item["answer_b"])
        label, raw = extract_judge_label(client, prompt)

        item["judge_label"] = label

        if args.n is not None:
            print(f"  human={item['human_label']:>10s}  judge={label:>10s}  raw={raw[:80].lower()}")

        # Save after each item so progress is never lost
        save_progress(out_path, data)

    print(f"Done. Skipped {skipped} already-judged. Saved to {out_path}")
    save_progress(out_path, data)


if __name__ == "__main__":
    main()
