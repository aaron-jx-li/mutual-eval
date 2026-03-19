#!/usr/bin/env python3
"""
Filter math-focused prompts from Arena 140k using an LLM judge.

Dataset:
  https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k

Example:
  python eval_arena/sample_arena_math.py \
    --output data/arena_140k_math_openai.jsonl \
    --resume
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


JUDGE_SYSTEM = (
    "You are a strict prompt classifier. "
    "Decide whether the user prompt is math-focused."
)

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
    p = argparse.ArgumentParser(description="LLM-filter Arena 140k for math-focused prompts.")
    p.add_argument(
        "--dataset",
        default="lmarena-ai/arena-human-preference-140k",
        help="Hugging Face dataset path.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="HF dataset config/subset name (default: auto).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train).",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="OpenAI judge model (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--output",
        default="data/arena_140k_math_openai.jsonl",
        help="Output JSONL for rows judged as math-focused.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="Flush buffers every N judged rows (default: 200).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only keep rows with is_math=true and confidence >= this value (default: 0.0).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint/output files.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for quick tests.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="OpenAI client timeout seconds.",
    )
    return p.parse_args()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export ") :].strip()
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)


def resolve_env_path() -> Path:
    here = Path(__file__).resolve()
    local = here.with_name(".env")
    if local.exists():
        return local
    return here.parent.parent / ".env"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t.strip()
    return ""


def extract_user_prompt(row: dict[str, Any]) -> str:
    # Preferred: first user turn from conversation_a
    conv_a = row.get("conversation_a")
    if isinstance(conv_a, list):
        for msg in conv_a:
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                text = extract_text(msg.get("content"))
                if text:
                    return text

    # Fallback: first user turn from conversation_b
    conv_b = row.get("conversation_b")
    if isinstance(conv_b, list):
        for msg in conv_b:
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                text = extract_text(msg.get("content"))
                if text:
                    return text

    # Fallback: if full_conversation has structured user field
    full = row.get("full_conversation")
    if isinstance(full, list):
        for turn in full:
            if isinstance(turn, dict):
                user_blob = turn.get("user")
                if isinstance(user_blob, dict):
                    text = extract_text(user_blob.get("content"))
                    if text:
                        return text

    return ""


def call_judge(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    user = JUDGE_USER_TEMPLATE.format(prompt=prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        lower = raw.lower()
        parsed = {
            "is_math": ('"is_math": true' in lower) or ('"is_math":true' in lower),
            "confidence": 0.0,
            "reason": f"parse_fallback: {raw[:300]}",
        }
    return {
        "is_math": bool(parsed.get("is_math", False)),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reason": str(parsed.get("reason", "")),
        "judge_raw": raw,
    }


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(obj.get("id", "")).strip()
            if rid:
                done.add(rid)
    return done


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)

    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)
    rows = [dict(r) for r in ds]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    output_path = Path(args.output)
    min_conf = float(args.min_confidence)
    if min_conf < 0.0 or min_conf > 1.0:
        raise SystemExit("--min-confidence must be within [0.0, 1.0].")

    done_ids: set[str] = set()
    if args.resume:
        done_ids |= load_done_ids(output_path)

    kept_buffer: list[dict[str, Any]] = []
    judged = 0
    kept = 0

    pbar = tqdm(rows, desc="Judging Arena prompts", unit="row")
    for row in pbar:
        rid = str(row.get("id", "")).strip()
        if rid and rid in done_ids:
            continue

        prompt = extract_user_prompt(row)
        if not prompt:
            if rid:
                done_ids.add(rid)
            judged += 1
            if args.save_every > 0 and judged % args.save_every == 0:
                append_jsonl(output_path, kept_buffer)
                kept_buffer.clear()
            continue

        judge = call_judge(client, args.judge_model, prompt)
        rec = {
            "id": rid,
            "model_a": row.get("model_a"),
            "model_b": row.get("model_b"),
            "winner": row.get("winner"),
            "language": row.get("language"),
            "is_code": row.get("is_code"),
            "is_math": judge["is_math"],
            "confidence": judge["confidence"],
            "reason": judge["reason"],
            "prompt": prompt,
        }
        judged += 1
        if judge["is_math"] and float(judge["confidence"]) >= min_conf:
            kept_buffer.append(rec)
            kept += 1
        if rid:
            done_ids.add(rid)

        pbar.set_postfix_str(f"judged={judged} kept={kept}", refresh=False)

        if args.save_every > 0 and judged % args.save_every == 0:
            append_jsonl(output_path, kept_buffer)
            kept_buffer.clear()

    pbar.close()
    append_jsonl(output_path, kept_buffer)

    print(f"Done. Judged={judged}, kept_math={kept}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
