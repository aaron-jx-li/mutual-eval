#!/usr/bin/env python3
"""
LLM-filter coding-focused prompts from Arena Expert 5k.

Dataset:
  https://huggingface.co/datasets/lmarena-ai/arena-expert-5k

This script extracts the user prompt from each Arena battle row, asks an LLM
judge whether the prompt belongs to the coding/software domain, and writes the
selected rows to a JSONL sample file.

Unlike the older math sampler, this version supports:
  - parallel judge requests
  - resumable processing via a full judged-record log
  - optional output-size capping for fixed-size samples

Examples:
  python eval_arena/sample_arena_coding.py

  python eval_arena/sample_arena_coding.py \
    --config eval_arena/config_arena_coding.yaml
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import yaml


JUDGE_SYSTEM = (
    "You are a strict domain classifier. "
    "Decide whether the user prompt is primarily about coding or software engineering."
)

JUDGE_USER_TEMPLATE = """\
Classify whether this user prompt is coding-focused.

Definition of coding-focused:
- The core task is about writing, debugging, explaining, reviewing, refactoring,
  testing, optimizing, or running code.
- Include software engineering, scripts, SQL, regex, shell commands, APIs,
  frameworks, web development, ML engineering, devops, build systems, and
  code-generation tasks.
- Include requests that ask for code snippets, implementation plans tied to
  code, or fixing program behavior.
- Exclude pure math, pure writing, general knowledge, product advice, business
  strategy, or other non-programming tasks, even if they mention technology.
- Exclude prompts that only ask about using software as an end user unless the
  answer mainly requires programming or technical implementation.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_coding": true or false,
  "confidence": 0.0 to 1.0,
  "category": "short label",
  "reason": "short explanation"
}}
"""

DONE_STATUSES = {"ok", "skipped_no_prompt", "skipped_language"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-filter Arena Expert 5k for coding-focused prompts.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Sampling settings are read from the 'sampling' section.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Hugging Face dataset path (default: lmarena-ai/arena-expert-5k).",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional HF dataset config/subset name.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (default: train).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenAI-compatible judge model (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file containing selected coding-domain rows.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="JSON checkpoint for resume/state. Defaults to <output stem>_checkpoint.json.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Flush buffers every N completed rows (default: 100).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Only keep rows with is_coding=true and confidence >= this value.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent judge requests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output/checkpoint files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of dataset rows to inspect.",
    )
    parser.add_argument(
        "--max-kept",
        type=int,
        default=None,
        help="Optional cap on number of coding rows to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic row shuffling before judging.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Judge client timeout in seconds.",
    )
    parser.add_argument(
        "--allowed-languages",
        nargs="+",
        default=None,
        help="Optional language allowlist, e.g. --allowed-languages en.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default=None,
        help="Environment variable name holding the judge API key (default: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for the OpenAI-compatible judge endpoint (default: OPENAI_BASE_URL or direct OpenAI).",
    )
    return parser.parse_args()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def resolve_env_path() -> Path:
    here = Path(__file__).resolve()
    local_env = here.with_name(".env")
    if local_env.exists():
        return local_env
    return here.parent.parent / ".env"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    section = config.get("sampling", {})

    if args.dataset is None:
        args.dataset = section.get("dataset", "lmarena-ai/arena-expert-5k")
    if args.subset is None:
        args.subset = section.get("subset")
    if args.split is None:
        args.split = section.get("split", "train")
    if args.judge_model is None:
        args.judge_model = section.get("judge_model", "gpt-4.1-mini")
    if args.output is None:
        args.output = section.get("output", "data/arena_expert_5k_coding.jsonl")
    if args.checkpoint is None:
        args.checkpoint = section.get("checkpoint")
    if args.save_every is None:
        args.save_every = int(section.get("save_every", 100))
    if args.min_confidence is None:
        args.min_confidence = float(section.get("min_confidence", 0.0))
    if args.max_workers is None:
        args.max_workers = int(section.get("max_workers", 8))
    if not args.resume:
        args.resume = bool(section.get("resume", False))
    if args.max_rows is None and section.get("max_rows") is not None:
        args.max_rows = int(section.get("max_rows"))
    if args.max_kept is None and section.get("max_kept") is not None:
        args.max_kept = int(section.get("max_kept"))
    if args.seed is None and section.get("seed") is not None:
        args.seed = int(section.get("seed"))
    if args.timeout is None:
        args.timeout = int(section.get("timeout", 60))
    if args.allowed_languages is None and section.get("allowed_languages") is not None:
        args.allowed_languages = [str(v) for v in section.get("allowed_languages", [])]
    if args.judge_api_key_env is None:
        args.judge_api_key_env = str(section.get("judge_api_key_env", "OPENAI_API_KEY"))
    if args.judge_base_url is None and section.get("judge_base_url") is not None:
        args.judge_base_url = str(section.get("judge_base_url"))
    return args


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
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
    text_literal = match.group("text")
    try:
        value = ast.literal_eval(text_literal)
    except (SyntaxError, ValueError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def extract_user_prompt(row: dict[str, Any]) -> str:
    conv_a = row.get("conversation_a")
    if isinstance(conv_a, list):
        for msg in conv_a:
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                text = extract_text(msg.get("content"))
                if text:
                    return text
    elif isinstance(conv_a, str):
        text = extract_text_from_serialized_conversation(conv_a)
        if text:
            return text

    conv_b = row.get("conversation_b")
    if isinstance(conv_b, list):
        for msg in conv_b:
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                text = extract_text(msg.get("content"))
                if text:
                    return text
    elif isinstance(conv_b, str):
        text = extract_text_from_serialized_conversation(conv_b)
        if text:
            return text

    full_conv = row.get("full_conversation")
    if isinstance(full_conv, list):
        for turn in full_conv:
            if not isinstance(turn, dict):
                continue
            user_blob = turn.get("user")
            if isinstance(user_blob, dict):
                text = extract_text(user_blob.get("content"))
                if text:
                    return text
    elif isinstance(full_conv, str):
        text = extract_text_from_serialized_conversation(full_conv)
        if text:
            return text

    return ""


def build_row_id(row: dict[str, Any], row_index: int) -> str:
    value = str(row.get("id", "")).strip()
    if value:
        return value
    return f"arena_expert_{row_index:06d}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(
    path: Path,
    *,
    done_ids: set[str],
    judged_count: int,
    kept_count: int,
    error_count: int,
    status_counts: dict[str, int],
) -> None:
    payload = {
        "done_ids": sorted(done_ids),
        "judged_rows": judged_count,
        "kept_rows": kept_count,
        "judge_errors": error_count,
        "status_counts": status_counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def call_judge(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    user_prompt = JUDGE_USER_TEMPLATE.format(prompt=prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        lower = raw.lower()
        parsed = {
            "is_coding": ('"is_coding": true' in lower) or ('"is_coding":true' in lower),
            "confidence": 0.0,
            "category": "parse_fallback",
            "reason": raw[:300],
        }
    return {
        "is_coding": bool(parsed.get("is_coding", False)),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "category": str(parsed.get("category", "")),
        "reason": str(parsed.get("reason", "")),
        "judge_raw": raw,
    }


def build_judge_client(args: argparse.Namespace) -> OpenAI:
    api_key_env = str(args.judge_api_key_env or "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"{api_key_env} is required for the judge client.")

    base_url = args.judge_base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)


def judge_row(client: OpenAI, judge_model: str, row: dict[str, Any], min_confidence: float) -> dict[str, Any]:
    allowed_languages = row.get("allowed_languages")
    language = str(row.get("language") or "").strip().lower()
    if allowed_languages:
        allowed_normalized = {str(v).strip().lower() for v in allowed_languages if str(v).strip()}
        if language not in allowed_normalized:
            return {
                "id": row["id"],
                "prompt": row["prompt"],
                "status": "skipped_language",
                "is_coding": False,
                "confidence": 0.0,
                "category": "language_filtered",
                "reason": f"Language {language or 'unknown'} not in allowlist {sorted(allowed_normalized)}.",
                "selected": False,
                "judge_model": judge_model,
                "judge_raw": "",
            }

    prompt = row["prompt"]
    if not prompt:
        return {
            "id": row["id"],
            "prompt": "",
            "status": "skipped_no_prompt",
            "is_coding": False,
            "confidence": 0.0,
            "category": "missing_prompt",
            "reason": "No user prompt could be extracted.",
            "selected": False,
            "judge_model": judge_model,
            "judge_raw": "",
        }

    try:
        judged = call_judge(client, judge_model, prompt)
    except Exception as exc:
        return {
            "id": row["id"],
            "prompt": prompt,
            "status": "judge_error",
            "is_coding": False,
            "confidence": 0.0,
            "category": "judge_error",
            "reason": str(exc),
            "selected": False,
            "judge_model": judge_model,
            "judge_raw": "",
        }

    selected = judged["is_coding"] and float(judged["confidence"]) >= min_confidence
    return {
        "id": row["id"],
        "prompt": prompt,
        "status": "ok",
        "is_coding": judged["is_coding"],
        "confidence": judged["confidence"],
        "category": judged["category"],
        "reason": judged["reason"],
        "selected": selected,
        "judge_model": judge_model,
        "judge_raw": judged["judge_raw"],
    }


def build_selected_record(row: dict[str, Any], judged: dict[str, Any], dataset: str, split: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dataset": dataset,
        "split": split,
        "model_a": row["raw_row"].get("model_a"),
        "model_b": row["raw_row"].get("model_b"),
        "winner": row["raw_row"].get("winner"),
        "evaluation_order": row["raw_row"].get("evaluation_order"),
        "language": row["raw_row"].get("language"),
        "occupational_tags": row["raw_row"].get("occupational_tags"),
        "prompt": row["prompt"],
        "conversation_a": row["raw_row"].get("conversation_a"),
        "conversation_b": row["raw_row"].get("conversation_b"),
        "judge_model": judged["judge_model"],
        "is_coding": judged["is_coding"],
        "confidence": judged["confidence"],
        "category": judged["category"],
        "reason": judged["reason"],
        "judge_raw": judged["judge_raw"],
    }


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)

    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        raise SystemExit("--min-confidence must be within [0.0, 1.0].")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1.")
    if args.save_every < 1:
        raise SystemExit("--save-every must be at least 1.")
    if args.max_kept is not None and args.max_kept < 1:
        raise SystemExit("--max-kept must be at least 1 when provided.")

    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_path.with_name(f"{output_path.stem}_checkpoint.json")

    if not args.resume:
        if output_path.exists():
            output_path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    client = build_judge_client(args)

    if args.subset:
        dataset = load_dataset(args.dataset, args.subset, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    raw_rows = [dict(row) for row in dataset]

    prepared_rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(raw_rows):
        prepared_rows.append(
            {
                "id": build_row_id(raw_row, row_index),
                "row_index": row_index,
                "prompt": extract_user_prompt(raw_row),
                "language": raw_row.get("language"),
                "allowed_languages": args.allowed_languages,
                "raw_row": raw_row,
            }
        )

    if args.seed is not None:
        rng = random.Random(args.seed)
        rng.shuffle(prepared_rows)
    if args.max_rows is not None:
        prepared_rows = prepared_rows[: args.max_rows]

    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    done_ids = {str(v) for v in checkpoint.get("done_ids", [])} if checkpoint else set()
    kept_existing = load_jsonl(output_path) if args.resume else []
    kept_count = len(kept_existing)
    error_count = int(checkpoint.get("judge_errors", 0)) if checkpoint else 0
    judged_count = int(checkpoint.get("judged_rows", len(done_ids))) if checkpoint else len(done_ids)
    status_counts = {
        str(k): int(v)
        for k, v in (checkpoint.get("status_counts", {}) or {}).items()
    }

    if args.max_kept is not None and kept_count >= args.max_kept:
        print(f"Already have {kept_count} kept rows, which meets --max-kept={args.max_kept}.")
        print(f"Output: {output_path}")
        print(f"Checkpoint: {checkpoint_path}")
        return

    pending_rows = [row for row in prepared_rows if row["id"] not in done_ids]

    kept_buffer: list[dict[str, Any]] = []
    processed_since_flush = 0

    def flush() -> None:
        nonlocal processed_since_flush
        append_jsonl(output_path, kept_buffer)
        kept_buffer.clear()
        save_checkpoint(
            checkpoint_path,
            done_ids=done_ids,
            judged_count=judged_count,
            kept_count=kept_count,
            error_count=error_count,
            status_counts=status_counts,
        )
        processed_since_flush = 0

    progress = tqdm(
        total=len(prepared_rows),
        initial=judged_count,
        desc="Judging Arena coding",
        unit="row",
    )

    stop_submitting = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending_iter = iter(pending_rows)
        futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

        def maybe_submit_next() -> bool:
            nonlocal stop_submitting
            if stop_submitting:
                return False
            if args.max_kept is not None and kept_count >= args.max_kept:
                stop_submitting = True
                return False
            try:
                row = next(pending_iter)
            except StopIteration:
                return False
            future = executor.submit(
                judge_row,
                client,
                args.judge_model,
                row,
                float(args.min_confidence),
            )
            futures[future] = row
            return True

        for _ in range(min(args.max_workers, len(pending_rows))):
            if not maybe_submit_next():
                break

        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                row = futures.pop(future)
                judged = future.result()
                processed_since_flush += 1
                status = str(judged["status"])
                status_counts[status] = status_counts.get(status, 0) + 1

                if status in DONE_STATUSES:
                    done_ids.add(row["id"])
                    judged_count += 1
                    progress.update(1)
                elif status == "judge_error":
                    error_count += 1

                if judged["selected"]:
                    if args.max_kept is None or kept_count < args.max_kept:
                        kept_buffer.append(build_selected_record(row, judged, args.dataset, args.split))
                        kept_count += 1
                    else:
                        stop_submitting = True

                progress.set_postfix_str(
                    f"judged={judged_count} kept={kept_count} errors={error_count}",
                    refresh=False,
                )

                if processed_since_flush >= args.save_every:
                    flush()

                maybe_submit_next()

    progress.close()
    flush()

    summary = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "allowed_languages": args.allowed_languages,
        "judge_model": args.judge_model,
        "min_confidence": args.min_confidence,
        "max_workers": args.max_workers,
        "judge_api_key_env": args.judge_api_key_env,
        "judge_base_url": args.judge_base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        "seed": args.seed,
        "max_rows": args.max_rows,
        "max_kept": args.max_kept,
        "judged_rows": judged_count,
        "kept_rows": kept_count,
        "judge_errors": error_count,
        "status_counts": status_counts,
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. judged={judged_count} kept={kept_count} judge_errors={error_count}")
    print(f"Selected output: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
