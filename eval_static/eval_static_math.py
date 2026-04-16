#!/usr/bin/env python3
"""
Static math evaluation pipeline for the current 16-model roster.

Key design choices:
1. Use the model roster from `model_api_smoke_test.py` so the eval stays aligned.
2. Sample static benchmarks rather than running full benchmarks by default.
3. Use a flexible LLM judge for correctness on every example.

Usage examples:
    python eval_static/eval_static_math.py --sample-file results/static_samples/run1/sampled_items.jsonl
    python eval_static/eval_static_math.py --sample-file results/static_samples/run1/sampled_items.jsonl --models gpt-5.4 claude-opus-4-6
    python eval_static/eval_static_math.py --sample-file results/static_samples/run1/sampled_items.jsonl --judge-model gpt-4.1-mini
    python eval_static/eval_static_math.py --config eval_static/config_static.yaml --use-litellm
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anthropic
from datasets import load_dataset
from openai import BadRequestError, OpenAI
from tqdm import tqdm
import yaml

from model_api_smoke_test import (
    MODEL_LOOKUP,
    MODEL_SPECS,
    extract_chat_content,
    get_routed_model_id,
    should_use_litellm_for_model,
    get_env_value,
    load_env_file,
    normalize_base_url,
)


ANSWER_INSTRUCTION = (
    "Solve the math problem carefully. "
    "End your response with a final line of the form 'Final answer: <answer>'."
)

JUDGE_SYSTEM_INSTRUCTIONS = (
    "You are a careful mathematics grader. "
    "Determine whether a model response is mathematically correct. "
    "Mark an answer correct when the final answer is mathematically equivalent to the ground truth, "
    "even if it uses a different but equivalent form. "
    "Treat equivalent forms as correct, including fractions vs decimals (for example 11/2 and 5.5), "
    "unsimplified vs simplified expressions, equivalent algebraic forms, and equivalent interval/set notation "
    "when they represent the same solution. "
    "For multiple-choice questions, accept either the correct option letter or the correct option content. "
    "Do not reward style, verbosity, or formatting. Focus on mathematical correctness."
)

JUDGE_PROMPT_TEMPLATE = """\
Question:
{question}

Ground-truth answer:
{gold_answer}

Model response:
{model_answer}

Decide whether the model response is mathematically correct.

Important grading rules:
- Count mathematically equivalent answers as correct.
- Examples of equivalent answers include 11/2 and 5.5, 0.5 and 1/2, or algebraically equivalent expressions.
- For multiple-choice questions, accept either the correct letter choice or the correct option content.
- Minor notation differences or harmless formatting differences should not make a correct answer wrong.
- If the response contains reasoning plus a final answer, judge based on whether the final mathematical conclusion is correct and supported well enough.

Return ONLY a JSON object:
{{"correct": true or false, "reason": "brief explanation"}}
"""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str
    default_pilot_samples: int
    default_paper_samples: int


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec("gsm8k", "openai/gsm8k", "main", "test", "gsm8k", 30, 50),
    DatasetSpec(
        "mmlu-abstract",
        "brucewlee1/mmlu-abstract-algebra",
        None,
        "test",
        "mmlu",
        10,
        20,
    ),
    DatasetSpec(
        "mmlu-college",
        "brucewlee1/mmlu-college-mathematics",
        None,
        "test",
        "mmlu",
        10,
        20,
    ),
    DatasetSpec("math-algebra", "EleutherAI/hendrycks_math", "algebra", "test", "math", 10, 35),
    DatasetSpec(
        "math-counting",
        "EleutherAI/hendrycks_math",
        "counting_and_probability",
        "test",
        "math",
        10,
        50,
    ),
    DatasetSpec("math-geometry", "EleutherAI/hendrycks_math", "geometry", "test", "math", 10, 50),
    DatasetSpec("math-number", "EleutherAI/hendrycks_math", "number_theory", "test", "math", 10, 50),
    DatasetSpec(
        "math-intermediate",
        "EleutherAI/hendrycks_math",
        "intermediate_algebra",
        "test",
        "math",
        10,
        55,
    ),
    DatasetSpec(
        "math-prealgebra",
        "EleutherAI/hendrycks_math",
        "prealgebra",
        "test",
        "math",
        10,
        15,
    ),
    DatasetSpec(
        "math-precalculus",
        "EleutherAI/hendrycks_math",
        "precalculus",
        "test",
        "math",
        10,
        15,
    ),
    DatasetSpec("aime-2025", "test-time-compute/aime_2025", None, "test", "aime", 5, 30),
    DatasetSpec("aime-2026", "MathArena/aime_2026", None, "train", "aime", 5, 30),
    DatasetSpec("olympiad-math", "math-ai/olympiadbench", None, "test", "olympiad", 10, 80),
]

DATASET_LOOKUP = {spec.name: spec for spec in DATASET_SPECS}


def call_openai_compatible(
    client: OpenAI,
    model_id: str,
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    request_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if not model_id.startswith("gpt-5"):
        request_kwargs["temperature"] = temperature
    if max_tokens is not None:
        if model_id.startswith("gpt-5"):
            request_kwargs["max_completion_tokens"] = max_tokens
        else:
            request_kwargs["max_tokens"] = max_tokens

    try:
        resp = client.chat.completions.create(**request_kwargs)
    except BadRequestError as exc:
        message = str(exc).lower()
        if (
            "max_tokens" in request_kwargs
            and "unsupported" in message
            and "max_tokens" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("max_tokens", None)
            retry_kwargs["max_completion_tokens"] = max_tokens
            resp = client.chat.completions.create(**retry_kwargs)
        elif (
            "max_completion_tokens" in request_kwargs
            and "unsupported" in message
            and "max_completion_tokens" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("max_completion_tokens", None)
            retry_kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**retry_kwargs)
        elif (
            "temperature" in request_kwargs
            and "unsupported" in message
            and "temperature" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**retry_kwargs)
        else:
            raise

    return extract_chat_content(resp.choices[0].message.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current 16-model roster on static math benchmarks.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Evaluation settings are read from the 'evaluation' section.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=[spec.label for spec in MODEL_SPECS],
        help="Models to evaluate. Defaults to all models in model_api_smoke_test.py.",
    )
    parser.add_argument(
        "--sample-file",
        default=None,
        help="Path to a previously generated sampled_items.jsonl file.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenAI judge model used for correctness grading.",
    )
    parser.add_argument(
        "--generation-timeout",
        type=int,
        default=None,
        help="Provider API timeout seconds for generation requests.",
    )
    parser.add_argument(
        "--generation-max-tokens",
        type=int,
        default=None,
        dest="generation_max_tokens",
        help="Maximum tokens for generation (default from config: 32768).",
    )
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=None,
        help="Number of retries for empty or failed generation attempts (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for results. Defaults to results/static_eval/<timestamp>.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Periodically save partial responses and summary every N evaluations (default: 50).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial results in the output directory.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of question items to evaluate in parallel (default: 4). Use 1 for serial execution.",
    )
    parser.add_argument(
        "--use-litellm",
        action="store_true",
        help="Route all model and judge calls through a single OpenAI-compatible LiteLLM endpoint.",
    )
    parser.add_argument(
        "--litellm-models",
        nargs="+",
        default=None,
        choices=[spec.label for spec in MODEL_SPECS],
        help=(
            "Explicit subset of model labels to route through LiteLLM. "
            "When set, this overrides the default all-model behavior."
        ),
    )
    return parser.parse_args()


def build_output_dir(user_output_dir: str | None) -> Path:
    if user_output_dir:
        return Path(user_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "static_eval" / timestamp


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
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _expand_env(raw)


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    section = config.get("evaluation", {})

    if args.models is None:
        args.models = section.get("models", [spec.label for spec in MODEL_SPECS])
    if args.sample_file is None:
        args.sample_file = section.get("sample_file")
        if args.sample_file is None:
            sampling_output_dir = config.get("sampling", {}).get("output_dir")
            if sampling_output_dir:
                args.sample_file = str(Path(sampling_output_dir) / "sampled_items.jsonl")
    if args.judge_model is None:
        args.judge_model = section.get("judge_model", "gpt-4.1-mini")
    if args.generation_timeout is None:
        args.generation_timeout = int(section.get("generation_timeout", 120))
    if args.generation_max_tokens is None:
        args.generation_max_tokens = int(section.get("generation_max_tokens", 32768))
    if args.generation_retries is None:
        args.generation_retries = int(section.get("generation_retries", 2))
    args.model_max_tokens = {
        key: (int(value) if value is not None else None)
        for key, value in section.get("model_max_tokens", {}).items()
    }
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")
    if args.save_every == 50:
        args.save_every = int(section.get("save_every", 50))
    if not args.resume:
        args.resume = bool(section.get("resume", False))
    if args.max_workers == 4:
        args.max_workers = int(section.get("max_workers", 4))
    if not args.use_litellm:
        args.use_litellm = bool(section.get("use_litellm", False))
    if args.litellm_models is None and section.get("litellm_models") is not None:
        args.litellm_models = list(section.get("litellm_models", []))

    if not args.sample_file:
        raise SystemExit("Error: --sample-file is required, either via CLI or config file.")
    return args


def build_sample_plan(selected_datasets: list[str], profile: str, uniform_n: int | None) -> dict[str, int]:
    plan: dict[str, int] = {}
    for name in selected_datasets:
        spec = DATASET_LOOKUP[name]
        if uniform_n is not None:
            plan[name] = uniform_n
        elif profile == "paper":
            plan[name] = spec.default_paper_samples
        else:
            plan[name] = spec.default_pilot_samples
    return plan


def build_clients(selected_models: list[str], generation_timeout: int) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    providers = {MODEL_LOOKUP[name].provider for name in selected_models}

    if "openai" in providers or True:
        key = get_env_value("OPENAI_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("OPENAI_BASE_URL"), "openai")
            clients["openai"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)

    if "anthropic" in providers:
        key = get_env_value("ANTHROPIC_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("ANTHROPIC_BASE_URL"), "anthropic")
            clients["anthropic"] = anthropic.Anthropic(
                api_key=key,
                base_url=base_url,
                timeout=generation_timeout,
            )

    if "google" in providers:
        key = get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("GEMINI_BASE_URL"), "google")
            if base_url:
                clients["google"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)

    if "openrouter" in providers:
        key = get_env_value("OPENROUTER_API_KEY")
        if key:
            clients["openrouter"] = OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                timeout=generation_timeout,
            )

    if "openai" not in clients:
        sys.exit("OPENAI_API_KEY is required for the judge model.")
    return clients


def build_clients_with_mode(
    selected_models: list[str],
    generation_timeout: int,
    *,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> dict[str, Any]:
    clients = build_clients(selected_models, generation_timeout)
    if use_litellm or litellm_models:
        key = get_env_value("LITELLM_API_KEY", "OPENAI_API_KEY")
        base_url = normalize_base_url(
            get_env_value("LITELLM_BASE_URL", "OPENAI_BASE_URL"),
            "openai",
        )
        if not key or not base_url:
            sys.exit(
                "When --use-litellm is enabled, set LITELLM_API_KEY (or OPENAI_API_KEY) "
                "and LITELLM_BASE_URL (or OPENAI_BASE_URL)."
            )
        clients["litellm"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)
    return clients


def _should_use_litellm(spec, *, use_litellm: bool, litellm_models: set[str] | None) -> bool:
    if litellm_models is not None:
        return spec.label in litellm_models and spec.litellm_model_id is not None
    return should_use_litellm_for_model(spec, use_litellm=use_litellm)


def load_raw_rows(spec: DatasetSpec) -> list[dict]:
    if spec.hf_config is None:
        dataset = load_dataset(spec.hf_path)[spec.split]
    else:
        dataset = load_dataset(spec.hf_path, spec.hf_config)[spec.split]

    rows = [dict(row) for row in dataset]
    if spec.kind == "olympiad":
        rows = [
            row
            for row in rows
            if row.get("subject") == "Math"
            and row.get("language") == "English"
            and row.get("modality") == "Text-only"
            and not row.get("is_multiple_answer", False)
        ]
    return rows


def sample_rows(rows: list[dict], *, n: int, seed: int, stratify_field: str | None = None) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    if not stratify_field:
        return rng.sample(rows, n)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(stratify_field, "unknown"))].append(row)

    buckets = list(grouped.values())
    if not buckets:
        return rng.sample(rows, n)

    per_bucket = max(1, n // len(buckets))
    sampled: list[dict] = []
    leftovers: list[dict] = []
    for bucket in buckets:
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        take = min(per_bucket, len(shuffled))
        sampled.extend(shuffled[:take])
        leftovers.extend(shuffled[take:])

    if len(sampled) < n:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: n - len(sampled)])
    elif len(sampled) > n:
        rng.shuffle(sampled)
        sampled = sampled[:n]
    return sampled


def format_mmlu_question(item: dict) -> str:
    letters = ["A", "B", "C", "D", "E", "F"]
    options = item.get("options", [])
    option_lines = [f"({letters[idx]}) {option}" for idx, option in enumerate(options)]
    return f"{item['centerpiece']}\n\nOptions:\n" + "\n".join(option_lines)


def build_eval_prompt(dataset_spec: DatasetSpec, item: dict) -> str:
    return f"{build_raw_question(dataset_spec, item)}\n\n{ANSWER_INSTRUCTION}"


def build_raw_question(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "gsm8k":
        return item["question"]
    if dataset_spec.kind == "mmlu":
        return format_mmlu_question(item)
    if dataset_spec.kind == "aime":
        return item.get("question") or item.get("problem")
    if dataset_spec.kind == "olympiad":
        return item["question"]
    return item["problem"]


def build_gold_answer(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "gsm8k":
        return item["answer"]
    if dataset_spec.kind == "mmlu":
        letter = item["correct_options"][0]
        literal = item["correct_options_literal"][0]
        return f"Option {letter}: {literal}"
    if dataset_spec.kind == "aime":
        return str(item["answer"])
    if dataset_spec.kind == "olympiad":
        final_answer = item.get("final_answer")
        if isinstance(final_answer, list):
            return " ; ".join(str(part) for part in final_answer)
        return str(final_answer)
    return item["solution"]


def get_item_metadata(dataset_spec: DatasetSpec, item: dict) -> dict[str, Any]:
    if dataset_spec.kind == "gsm8k":
        return {"level": None, "subject": "word_problems"}
    if dataset_spec.kind == "mmlu":
        return {"level": None, "subject": dataset_spec.name}
    if dataset_spec.kind == "aime":
        metadata = item.get("metadata") or {}
        problem_type = metadata.get("problem_type")
        if isinstance(problem_type, list) and problem_type:
            subject = problem_type[0]
        else:
            subject = "AIME"
        return {"level": "Competition", "subject": subject}
    if dataset_spec.kind == "olympiad":
        return {
            "level": item.get("difficulty", "Competition"),
            "subject": item.get("subfield") or item.get("subject") or "Olympiad",
        }
    return {"level": item.get("level"), "subject": item.get("type")}


def judge_correctness(
    judge_client: OpenAI,
    judge_model: str,
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    use_litellm: bool = False,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
    )
    if use_litellm:
        request_kwargs: dict[str, Any] = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = judge_client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "response_format" in request_kwargs and "unsupported" in message and "response_format" in message:
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("response_format", None)
                resp = judge_client.chat.completions.create(**retry_kwargs)
            else:
                raise
        raw_text = extract_chat_content(resp.choices[0].message.content).strip()
    else:
        resp = judge_client.responses.create(
            model=judge_model,
            instructions=JUDGE_SYSTEM_INSTRUCTIONS,
            input=prompt,
            max_output_tokens=512,
            store=False,
        )
        raw_text = resp.output_text.strip()
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        correct = bool(parsed.get("correct", False))
        reason = str(parsed.get("reason", ""))
    except json.JSONDecodeError:
        lower = cleaned.lower()
        correct = '"correct": true' in lower or '"correct":true' in lower
        reason = cleaned[:500]
    return {
        "correct": correct,
        "method": "llm_judge",
        "judge_raw": raw_text,
        "judge_reason": reason,
    }


def call_model(
    spec,
    clients: dict[str, Any],
    prompt: str,
    *,
    generation_max_tokens: int | None,
    use_litellm: bool = False,
    litellm_models: set[str] | None = None,
) -> str:
    # Gemini 2.5/3.1 are thinking models that can consume a large output budget
    # before they emit the final answer, so match the arena eval headroom here.
    google_max_tokens = generation_max_tokens
    if spec.provider == "google" and any(prefix in spec.model_id for prefix in ("2.5", "3.1")):
        google_max_tokens = 65536
    if _should_use_litellm(spec, use_litellm=use_litellm, litellm_models=litellm_models):
        return call_openai_compatible(
            clients["litellm"],
            get_routed_model_id(spec, use_litellm=True),
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=generation_max_tokens,
        )
    if spec.provider == "openai":
        return call_openai_compatible(
            clients["openai"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=generation_max_tokens,
        )
    if spec.provider == "google":
        return call_openai_compatible(
            clients["google"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=google_max_tokens,
        )
    if spec.provider == "openrouter":
        return call_openai_compatible(
            clients["openrouter"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=generation_max_tokens,
        )
    if spec.provider == "anthropic":
        resp = clients["anthropic"].messages.create(
            model=spec.model_id,
            max_tokens=generation_max_tokens or 8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
    raise ValueError(f"Unsupported provider: {spec.provider}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def make_eval_key(item_or_record: dict[str, Any], model_label: str | None = None) -> str:
    label = model_label if model_label is not None else str(item_or_record["model_label"])
    return f"{item_or_record['dataset']}::{item_or_record['sample_index']}::{label}"


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "dataset",
                "num_items",
                "num_scored",
                "num_correct",
                "accuracy",
                "judge_graded",
                "generation_errors",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def build_eval_order_lookup(
    sampled_items: list[dict[str, Any]],
    selected_model_specs: list[Any],
) -> dict[str, int]:
    order_lookup: dict[str, int] = {}
    order = 0
    for item in sampled_items:
        for model_spec in selected_model_specs:
            order_lookup[make_eval_key(item, model_spec.label)] = order
            order += 1
    return order_lookup


def order_results(
    rows: list[dict[str, Any]],
    eval_order_lookup: dict[str, int],
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            eval_order_lookup.get(make_eval_key(row), float("inf")),
            row["dataset"],
            row["sample_index"],
            row["model_label"],
        ),
    )


def save_checkpoint(
    *,
    output_dir: Path,
    results: list[dict[str, Any]],
    eval_order_lookup: dict[str, int],
    completed_evals: int,
    total_evals: int,
) -> None:
    ordered = order_results(results, eval_order_lookup)
    write_jsonl(output_dir / "responses.jsonl", ordered)
    summary_rows = summarize_results(ordered)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    checkpoint = {
        "completed_evals": completed_evals,
        "total_evals": total_evals,
        "saved_at": datetime.now().isoformat(),
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["dataset"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_label, dataset_name), group in sorted(grouped.items()):
        all_binary = [1 if row.get("correct") is True else 0 for row in group]
        judge_count = sum(1 for row in group if row["grading_method"] == "llm_judge")
        error_count = sum(1 for row in group if row["status"] != "ok")
        summary_rows.append(
            {
                "model_label": model_label,
                "dataset": dataset_name,
                "num_items": len(group),
                "num_scored": len(group),
                "num_correct": sum(all_binary),
                "accuracy": mean(all_binary) if all_binary else None,
                "judge_graded": judge_count,
                "generation_errors": error_count,
            }
        )
    return summary_rows


def evaluate_item(
    item: dict[str, Any],
    *,
    selected_model_specs: list[Any],
    clients: dict[str, Any],
    judge_model: str,
    completed_keys: set[str],
    generation_max_tokens: int | None,
    generation_retries: int,
    model_max_tokens: dict[str, int | None] | None,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    item_results: list[tuple[str, dict[str, Any]]] = []

    for model_spec in selected_model_specs:
        eval_key = make_eval_key(item, model_spec.label)
        if eval_key in completed_keys:
            continue

        record: dict[str, Any] = {
            "dataset": item["dataset"],
            "dataset_kind": item["dataset_kind"],
            "sample_index": item["sample_index"],
            "model_label": model_spec.label,
            "model_provider": model_spec.provider,
            "model_id": model_spec.model_id,
            "level": item["level"],
            "subject": item["subject"],
            "question": item["question"],
            "prompt": item["prompt"],
            "gold_answer": item["gold_answer"],
            "status": "ok",
            "correct": None,
            "grading_method": "llm_judge",
            "response_text": None,
            "judge_reason": None,
            "judge_raw": None,
            "latency_s": None,
            "generation_attempts": 0,
        }

        started = time.time()
        effective_max_tokens = (model_max_tokens or {}).get(model_spec.label, generation_max_tokens)
        response_text: str | None = None
        last_generation_error: str | None = None
        for attempt in range(generation_retries + 1):
            record["generation_attempts"] = attempt + 1
            try:
                candidate = call_model(
                    model_spec,
                    clients,
                    item["prompt"],
                    generation_max_tokens=effective_max_tokens,
                    use_litellm=use_litellm,
                    litellm_models=litellm_models,
                )
                if candidate.strip():
                    response_text = candidate
                    break
                last_generation_error = "Model returned an empty response after extraction."
            except Exception as exc:
                last_generation_error = str(exc)
            if attempt < generation_retries:
                time.sleep(min(2 ** attempt, 4))

        record["latency_s"] = round(time.time() - started, 2)
        if response_text is None:
            record["status"] = "generation_error"
            record["correct"] = False
            record["grading_method"] = "generation_error"
            record["judge_reason"] = last_generation_error or "Unknown generation failure."
            item_results.append((eval_key, record))
            continue
        record["response_text"] = response_text

        judge_spec = MODEL_LOOKUP.get(judge_model)
        judge_uses_litellm = (
            judge_spec is not None
            and _should_use_litellm(
                judge_spec,
                use_litellm=use_litellm,
                litellm_models=litellm_models,
            )
        )
        effective_judge_model = (
            get_routed_model_id(judge_spec, use_litellm=True)
            if judge_uses_litellm and judge_spec is not None
            else judge_model
        )
        judged = judge_correctness(
            clients["litellm"] if judge_uses_litellm else clients["openai"],
            effective_judge_model,
            question=item["question"],
            gold_answer=item["gold_answer"],
            model_answer=record["response_text"] or "",
            use_litellm=judge_uses_litellm,
        )
        record["correct"] = judged["correct"]
        record["grading_method"] = judged["method"]
        record["judge_reason"] = judged["judge_reason"]
        record["judge_raw"] = judged["judge_raw"]
        item_results.append((eval_key, record))

    return item_results


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nAccuracy summary")
    print("-" * 80)
    for row in summary_rows:
        accuracy = row["accuracy"]
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        print(
            f"{row['model_label']:24} {row['dataset']:18} "
            f"acc={accuracy_text:>8} "
            f"scored={row['num_scored']:4}/{row['num_items']:4} "
            f"judge={row['judge_graded']:4} errors={row['generation_errors']:3}"
        )
    print("-" * 80)


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)

    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_model_specs = [MODEL_LOOKUP[label] for label in args.models]
    litellm_models = set(args.litellm_models) if args.litellm_models else None
    selected_labels = {spec.label for spec in selected_model_specs}
    if litellm_models is not None:
        unselected = sorted(litellm_models - selected_labels)
        if unselected:
            sys.exit(
                "LiteLLM subset includes models not in --models/config: "
                + ", ".join(unselected)
            )
        unsupported_subset = sorted(
            label for label in litellm_models if MODEL_LOOKUP[label].litellm_model_id is None
        )
        if unsupported_subset:
            sys.exit(
                "LiteLLM is not configured for these requested subset models: "
                + ", ".join(unsupported_subset)
            )
    elif args.use_litellm:
        unsupported = [spec.label for spec in selected_model_specs if spec.litellm_model_id is None]
        if unsupported:
            sys.exit(
                "LiteLLM is not configured for these models: "
                + ", ".join(unsupported)
                + ". Remove them from the config/models list or add mappings in MODEL_SPECS."
            )
    clients = build_clients_with_mode(
        args.models,
        args.generation_timeout,
        use_litellm=args.use_litellm,
        litellm_models=litellm_models,
    )
    sample_file = Path(args.sample_file)
    sampled_items = load_jsonl(sample_file)
    dataset_counts: dict[str, int] = defaultdict(int)
    for item in sampled_items:
        dataset_counts[str(item["dataset"])] += 1

    run_config = {
        "models": args.models,
        "sample_file": str(sample_file),
        "datasets": sorted(dataset_counts.keys()),
        "sample_plan": dict(dataset_counts),
        "judge_model": args.judge_model,
        "generation_timeout": args.generation_timeout,
        "generation_max_tokens": args.generation_max_tokens,
        "generation_retries": args.generation_retries,
        "model_max_tokens": args.model_max_tokens,
        "use_litellm": args.use_litellm,
        "litellm_models": sorted(litellm_models) if litellm_models is not None else None,
        "max_workers": args.max_workers,
        "output_dir": str(output_dir),
        "resume": args.resume,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    responses_path = output_dir / "responses.jsonl"
    results: list[dict[str, Any]] = load_jsonl(responses_path) if args.resume else []
    completed_keys = {make_eval_key(record) for record in results}
    eval_order_lookup = build_eval_order_lookup(sampled_items, selected_model_specs)
    total_evals = len(sampled_items) * len(selected_model_specs)
    completed_evals = len(completed_keys)
    progress = tqdm(
        total=total_evals,
        initial=completed_evals,
        desc="Evaluating static math",
        unit="eval",
    )

    pending_items = [
        item
        for item in sampled_items
        if any(make_eval_key(item, model_spec.label) not in completed_keys for model_spec in selected_model_specs)
    ]
    completed_keys_snapshot = set(completed_keys)

    def record_result(eval_key: str, record: dict[str, Any]) -> None:
        nonlocal completed_evals
        results.append(record)
        completed_keys.add(eval_key)
        progress.set_postfix_str(
            f"{record['model_label']} | {record['dataset']} #{record['sample_index'] + 1}/{dataset_counts[record['dataset']]}",
            refresh=False,
        )
        progress.update(1)
        completed_evals += 1
        if args.save_every > 0 and completed_evals % args.save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                results=results,
                eval_order_lookup=eval_order_lookup,
                completed_evals=completed_evals,
                total_evals=total_evals,
            )

    if args.max_workers <= 1 or len(pending_items) <= 1:
        for item in pending_items:
            item_results = evaluate_item(
                item,
                selected_model_specs=selected_model_specs,
                clients=clients,
                judge_model=args.judge_model,
                completed_keys=completed_keys_snapshot,
                generation_max_tokens=args.generation_max_tokens,
                generation_retries=args.generation_retries,
                model_max_tokens=args.model_max_tokens,
                use_litellm=args.use_litellm,
                litellm_models=litellm_models,
            )
            for eval_key, record in item_results:
                record_result(eval_key, record)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_item,
                    item,
                    selected_model_specs=selected_model_specs,
                    clients=clients,
                    judge_model=args.judge_model,
                    completed_keys=completed_keys_snapshot,
                    generation_max_tokens=args.generation_max_tokens,
                    generation_retries=args.generation_retries,
                    model_max_tokens=args.model_max_tokens,
                    use_litellm=args.use_litellm,
                    litellm_models=litellm_models,
                )
                for item in pending_items
            ]
            for future in concurrent.futures.as_completed(futures):
                item_results = future.result()
                for eval_key, record in item_results:
                    record_result(eval_key, record)

    progress.close()
    ordered_results = order_results(results, eval_order_lookup)
    summary_rows = summarize_results(ordered_results)
    save_checkpoint(
        output_dir=output_dir,
        results=results,
        eval_order_lookup=eval_order_lookup,
        completed_evals=completed_evals,
        total_evals=total_evals,
    )

    print_summary(summary_rows)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
