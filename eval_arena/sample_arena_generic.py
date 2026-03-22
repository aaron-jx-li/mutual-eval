#!/usr/bin/env python3
"""
Sample prompts from Arena Human Preference 140k for generic domain evaluation.

Dataset:
  https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k

No domain filtering is applied. Rows are shuffled with a fixed seed and the
first --max-kept rows with a non-empty extractable prompt are saved.

Examples:
  python eval_arena/sample_arena_generic.py

  python eval_arena/sample_arena_generic.py \\
    --max-kept 1000 \\
    --seed 42 \\
    --output data/arena_140k_generic.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample prompts from Arena Human Preference 140k.",
    )
    parser.add_argument(
        "--dataset",
        default="lmarena-ai/arena-human-preference-140k",
        help="Hugging Face dataset path.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train).",
    )
    parser.add_argument(
        "--output",
        default="data/arena_140k_generic.jsonl",
        help="Output JSONL file for sampled rows.",
    )
    parser.add_argument(
        "--max-kept",
        type=int,
        default=1000,
        help="Number of rows to sample (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--allowed-languages",
        nargs="+",
        default=["en"],
        help="Language allowlist using dataset language codes (default: en).",
    )
    parser.add_argument(
        "--min-prompt-len",
        type=int,
        default=10,
        help="Minimum character length of extracted prompt (default: 10).",
    )
    return parser.parse_args()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export "):].strip()
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


def extract_text_from_serialized_conversation(raw: str) -> str:
    if not raw:
        return ""
    match = re.search(
        r"'role':\s*'user'.*?'text':\s*(?P<text>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",
        raw,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    try:
        value = __import__("ast").literal_eval(match.group("text"))
    except (SyntaxError, ValueError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def extract_user_prompt(row: dict[str, Any]) -> str:
    for conv_key in ("conversation_a", "conversation_b"):
        conv = row.get(conv_key)
        if isinstance(conv, list):
            for msg in conv:
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                    text = extract_text(msg.get("content"))
                    if text:
                        return text
        elif isinstance(conv, str):
            text = extract_text_from_serialized_conversation(conv)
            if text:
                return text

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


def build_output_record(row: dict[str, Any], prompt: str, dataset: str, split: str) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "")).strip(),
        "dataset": dataset,
        "split": split,
        "model_a": row.get("model_a"),
        "model_b": row.get("model_b"),
        "winner": row.get("winner"),
        "language": row.get("language"),
        "prompt": prompt,
        "conversation_a": row.get("conversation_a"),
        "conversation_b": row.get("conversation_b"),
    }


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    rows = [dict(r) for r in ds]
    print(f"Loaded {len(rows):,} rows.")

    allowed = {v.strip().lower() for v in args.allowed_languages}
    rows = [r for r in rows if str(r.get("language") or "").strip().lower() in allowed]
    print(f"After language filter {sorted(allowed)}: {len(rows):,} rows remain.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept: list[dict[str, Any]] = []
    skipped = 0

    with tqdm(rows, desc="Sampling rows", unit="row") as pbar:
        for row in pbar:
            if len(kept) >= args.max_kept:
                break
            prompt = extract_user_prompt(row)
            if len(prompt) < args.min_prompt_len:
                skipped += 1
                continue
            kept.append(build_output_record(row, prompt, args.dataset, args.split))
            pbar.set_postfix_str(f"kept={len(kept)} skipped={skipped}", refresh=False)

    with output_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. kept={len(kept)}, skipped_no_prompt={skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
