#!/usr/bin/env python3
"""
Static coding evaluation helpers and dataset definitions.

This module currently provides the shared sampling utilities used by
`sample_static_coding.py`, including dataset metadata, prompt builders,
and fixed benchmark loading logic for the coding-domain static suite.

The execution-based evaluator is intentionally left for a follow-up pass;
for now, running this file directly exits with a short explanatory message.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anthropic
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from openai import BadRequestError, OpenAI
from tqdm import tqdm
import yaml

try:
    from model_api_smoke_test import (
        MODEL_LOOKUP,
        MODEL_SPECS,
        extract_chat_content,
        get_env_value,
        get_routed_model_id,
        load_env_file,
        normalize_base_url,
        should_use_litellm_for_model,
    )
except ModuleNotFoundError:
    from eval_static.model_api_smoke_test import (
        MODEL_LOOKUP,
        MODEL_SPECS,
        extract_chat_content,
        get_env_value,
        get_routed_model_id,
        load_env_file,
        normalize_base_url,
        should_use_litellm_for_model,
    )


CODE_INSTRUCTION = (
    "Write a correct Python 3 solution. "
    "Return only executable Python code in a single ```python``` block, with no explanation."
)

EXECUTION_PREAMBLE = """\
from typing import *
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
    source_filename: str | None = None


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        "humaneval-plus",
        "evalplus/humanevalplus",
        None,
        "test",
        "humaneval",
        30,
        80,
    ),
    DatasetSpec(
        "mbpp-plus-sanitized",
        "evalplus/mbppplus",
        None,
        "test",
        "mbpp",
        30,
        140,
    ),
    DatasetSpec(
        "livecodebench-v6",
        "livecodebench/code_generation_lite",
        None,
        "test",
        "livecodebench",
        40,
        160,
        source_filename="test6.jsonl",
    ),
]

DATASET_LOOKUP = {spec.name: spec for spec in DATASET_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current model roster on static coding benchmarks.",
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
        "--max-items",
        type=int,
        default=None,
        help="Optional limit on the number of sampled items to evaluate, useful for smoke tests.",
    )
    parser.add_argument(
        "--use-litellm",
        action="store_true",
        help="Route model calls through a single OpenAI-compatible LiteLLM endpoint.",
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
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")
    if args.save_every == 50:
        args.save_every = int(section.get("save_every", 50))
    if not args.resume:
        args.resume = bool(section.get("resume", False))
    if args.max_workers == 4:
        args.max_workers = int(section.get("max_workers", 4))
    if args.max_items is None and section.get("max_items") is not None:
        args.max_items = int(section.get("max_items"))
    if not args.use_litellm:
        args.use_litellm = bool(section.get("use_litellm", False))

    if not args.sample_file:
        raise SystemExit("Error: --sample-file is required, either via CLI or config file.")
    return args


def call_openai_compatible(
    client: OpenAI,
    model_id: str,
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 4096,
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


def build_clients(selected_models: list[str]) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    providers = {MODEL_LOOKUP[name].provider for name in selected_models}

    if "openai" in providers or True:
        key = get_env_value("OPENAI_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("OPENAI_BASE_URL"), "openai")
            clients["openai"] = OpenAI(api_key=key, base_url=base_url, timeout=120)

    if "anthropic" in providers:
        key = get_env_value("ANTHROPIC_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("ANTHROPIC_BASE_URL"), "anthropic")
            clients["anthropic"] = anthropic.Anthropic(api_key=key, base_url=base_url, timeout=120)

    if "google" in providers:
        key = get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("GEMINI_BASE_URL"), "google")
            if base_url:
                clients["google"] = OpenAI(api_key=key, base_url=base_url, timeout=120)

    if "openrouter" in providers:
        key = get_env_value("OPENROUTER_API_KEY")
        if key:
            clients["openrouter"] = OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                timeout=120,
            )

    return clients


def build_clients_with_mode(selected_models: list[str], *, use_litellm: bool) -> dict[str, Any]:
    clients = build_clients(selected_models)
    routed_models = [
        label
        for label in selected_models
        if should_use_litellm_for_model(MODEL_LOOKUP[label], use_litellm=use_litellm)
    ]
    if routed_models:
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
        clients["litellm"] = OpenAI(api_key=key, base_url=base_url, timeout=120)
    return clients


def call_model(model_label: str, clients: dict[str, Any], prompt: str, *, use_litellm: bool = False) -> str:
    spec = MODEL_LOOKUP[model_label]
    if should_use_litellm_for_model(spec, use_litellm=use_litellm):
        return call_openai_compatible(
            clients["litellm"],
            get_routed_model_id(spec, use_litellm=True),
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    if spec.provider == "openai":
        return call_openai_compatible(
            clients["openai"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    if spec.provider == "google":
        return call_openai_compatible(
            clients["google"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    if spec.provider == "openrouter":
        return call_openai_compatible(
            clients["openrouter"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    if spec.provider == "anthropic":
        resp = clients["anthropic"].messages.create(
            model=spec.model_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
    raise ValueError(f"Unsupported provider: {spec.provider}")


def validate_clients_for_models(selected_models: list[str], clients: dict[str, Any], *, use_litellm: bool) -> None:
    missing: list[str] = []
    for label in selected_models:
        spec = MODEL_LOOKUP[label]
        if should_use_litellm_for_model(spec, use_litellm=use_litellm):
            if "litellm" not in clients:
                missing.append(f"{label} via LiteLLM")
            continue
        if spec.provider not in clients:
            missing.append(f"{label} via {spec.provider}")

    if missing:
        raise SystemExit(
            "Missing configured client(s) for selected models: "
            + ", ".join(missing)
            + ". Check the corresponding API keys/base URLs in your environment."
        )


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


def _load_livecodebench_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    if not spec.source_filename:
        raise ValueError(f"Dataset '{spec.name}' requires a source filename.")
    path = hf_hub_download(
        repo_id=spec.hf_path,
        filename=spec.source_filename,
        repo_type="dataset",
    )
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_raw_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    if spec.kind == "livecodebench":
        return _load_livecodebench_rows(spec)

    if spec.hf_config is None:
        dataset = load_dataset(spec.hf_path)[spec.split]
    else:
        dataset = load_dataset(spec.hf_path, spec.hf_config)[spec.split]
    return [dict(row) for row in dataset]


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


def _decode_livecodebench_private_tests(raw_value: str) -> list[dict[str, Any]]:
    decoded = zlib.decompress(base64.b64decode(raw_value))
    parsed = pickle.loads(decoded)
    if isinstance(parsed, bytes):
        parsed = parsed.decode("utf-8")
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return parsed


def _load_livecodebench_public_tests(item: dict[str, Any]) -> list[dict[str, Any]]:
    raw_cases = item.get("public_test_cases")
    if not raw_cases:
        return []
    if isinstance(raw_cases, str):
        return json.loads(raw_cases)
    if isinstance(raw_cases, list):
        return raw_cases
    return []


def _infer_livecodebench_test_type(item: dict[str, Any]) -> str:
    public_tests = _load_livecodebench_public_tests(item)
    if public_tests:
        return str(public_tests[0].get("testtype", "stdin"))
    return "functional" if item.get("starter_code") else "stdin"


def _build_livecodebench_interface_instruction(item: dict[str, Any]) -> str:
    test_type = _infer_livecodebench_test_type(item)
    starter_code = item.get("starter_code", "").strip()
    if test_type == "functional":
        if starter_code:
            return (
                "Implement the solution exactly within the provided starter code. "
                "Keep the same class name, function name, and signature."
            )
        return "Write the required Python function so it can be called directly by the tests."
    return (
        "Write a complete Python program that reads from standard input and writes to standard output. "
        "Do not print any extra text."
    )


def _format_livecodebench_public_examples(item: dict[str, Any]) -> str | None:
    public_tests = _load_livecodebench_public_tests(item)
    if not public_tests:
        return None

    test_type = _infer_livecodebench_test_type(item)
    lines: list[str] = ["Public examples:"]
    for idx, case in enumerate(public_tests, start=1):
        lines.append("")
        lines.append(f"Example {idx}:")
        input_label = "Input" if test_type == "stdin" else "Arguments"
        lines.append(f"{input_label}:")
        lines.append("```text")
        lines.append(str(case.get("input", "")).rstrip())
        lines.append("```")
        lines.append("Output:")
        lines.append("```text")
        lines.append(str(case.get("output", "")).rstrip())
        lines.append("```")
    return "\n".join(lines)


def build_raw_question(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "humaneval":
        return (
            "Complete the following Python function.\n\n"
            f"```python\n{item['prompt'].rstrip()}\n```"
        )
    if dataset_spec.kind == "mbpp":
        signature_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(.*?\)\s*:", item.get("code", ""), flags=re.DOTALL)
        required_name = signature_match.group(1) if signature_match else None
        sections = ["Write a Python function for this task:", "", item["prompt"].strip()]
        if required_name:
            sections.extend(
                [
                    "",
                    f"Your function must be named `{required_name}`.",
                ]
            )
        return "\n".join(sections)
    if dataset_spec.kind == "livecodebench":
        starter_code = item.get("starter_code", "").rstrip()
        interface_instruction = _build_livecodebench_interface_instruction(item)
        public_examples = _format_livecodebench_public_examples(item)
        sections = [
            f"Title: {item.get('question_title', '').strip()}",
            "",
            item.get("question_content", "").strip(),
            "",
            interface_instruction,
        ]
        if starter_code:
            sections.extend(
                [
                    "",
                    "Starter code:",
                    f"```python\n{starter_code}\n```",
                ]
            )
        if public_examples:
            sections.extend(["", public_examples])
        return "\n".join(part for part in sections if part is not None)
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def build_eval_prompt(dataset_spec: DatasetSpec, item: dict) -> str:
    return f"{build_raw_question(dataset_spec, item)}\n\n{CODE_INSTRUCTION}"


def build_gold_answer(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "humaneval":
        return f"{item['prompt'].rstrip()}\n{item['canonical_solution'].rstrip()}"
    if dataset_spec.kind == "mbpp":
        return item["code"]
    if dataset_spec.kind == "livecodebench":
        private_tests = _decode_livecodebench_private_tests(item["private_test_cases"])
        return json.dumps(
            {
                "public_tests": json.loads(item["public_test_cases"]),
                "private_test_count": len(private_tests),
            },
            ensure_ascii=False,
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def get_item_metadata(dataset_spec: DatasetSpec, item: dict) -> dict[str, Any]:
    if dataset_spec.kind == "humaneval":
        return {"level": "function", "subject": "python_function_synthesis"}
    if dataset_spec.kind == "mbpp":
        return {"level": "basic", "subject": "python_programming"}
    if dataset_spec.kind == "livecodebench":
        return {
            "level": item.get("difficulty") or "unknown",
            "subject": item.get("platform") or "competitive_programming",
        }
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


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


def extract_code_from_response(response_text: str) -> str:
    text = response_text.strip()
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[0].strip()
    return text


def _normalize_text_output(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _parse_structured_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return __import__("ast").literal_eval(text)
    except Exception:
        return text


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, tuple) and isinstance(expected, tuple) and len(actual) == len(expected):
        return all(_values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(_values_equal(actual[k], expected[k]) for k in actual)
    return actual == expected


def _run_python_script(script_text: str, *, stdin_text: str = "", timeout_s: int = 10) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="static_coding_eval_") as tmpdir:
        script_path = Path(tmpdir) / "runner.py"
        script_path.write_text(script_text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script_path)],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=tmpdir,
        )


def _grade_humaneval(item: dict[str, Any], code: str) -> dict[str, Any]:
    raw = item["raw_item"]
    script = (
        f"{EXECUTION_PREAMBLE}\n"
        f"{code}\n\n"
        f"{raw['test']}\n\n"
        f"check({raw['entry_point']})\n"
        "print('__EVAL_PASS__')\n"
    )
    completed = _run_python_script(script)
    passed = completed.returncode == 0 and "__EVAL_PASS__" in completed.stdout
    details = completed.stderr or completed.stdout
    return {"passed": passed, "details": details[:4000], "grader_raw": (completed.stdout + completed.stderr)[:8000]}


def _grade_mbpp(item: dict[str, Any], code: str) -> dict[str, Any]:
    raw = item["raw_item"]
    script = f"{EXECUTION_PREAMBLE}\n{code}\n\n{raw['test']}\nprint('__EVAL_PASS__')\n"
    completed = _run_python_script(script)
    passed = completed.returncode == 0 and "__EVAL_PASS__" in completed.stdout
    details = completed.stderr or completed.stdout
    return {"passed": passed, "details": details[:4000], "grader_raw": (completed.stdout + completed.stderr)[:8000]}


def _infer_livecodebench_callable(raw_item: dict[str, Any], code: str) -> tuple[str, str]:
    starter_code = raw_item.get("starter_code", "")
    class_match = re.search(r"class\s+Solution\s*:\s*(?:\n[ \t]+.*)*?\n[ \t]+def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    if class_match:
        return "solution_method", class_match.group(1)
    func_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    if func_match:
        return "function", func_match.group(1)
    code_func_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    if code_func_match:
        return "function", code_func_match.group(1)
    raise ValueError("Could not infer callable name for LiveCodeBench task.")


def _build_livecodebench_functional_harness(code: str, raw_item: dict[str, Any], tests: list[dict[str, Any]]) -> str:
    target_kind, target_name = _infer_livecodebench_callable(raw_item, code)
    return f"""\
import inspect
import json
import math

{EXECUTION_PREAMBLE}
{code}

TESTS = {json.dumps(tests, ensure_ascii=False)}
TARGET_KIND = {target_kind!r}
TARGET_NAME = {target_name!r}

def parse_value(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        import ast
        return ast.literal_eval(text)

def values_equal(actual, expected):
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, tuple) and isinstance(expected, tuple) and len(actual) == len(expected):
        return all(values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(values_equal(actual[k], expected[k]) for k in actual)
    return actual == expected

if TARGET_KIND == "solution_method":
    callable_obj = getattr(Solution(), TARGET_NAME)
else:
    callable_obj = globals()[TARGET_NAME]

signature = inspect.signature(callable_obj)
param_count = len(signature.parameters)

for idx, case in enumerate(TESTS):
    raw_input = parse_value(case["input"])
    expected = parse_value(case["output"])
    if param_count == 0:
        actual = callable_obj()
    elif param_count == 1:
        actual = callable_obj(raw_input)
    elif isinstance(raw_input, dict):
        actual = callable_obj(**raw_input)
    elif isinstance(raw_input, (list, tuple)):
        actual = callable_obj(*raw_input)
    else:
        raise AssertionError(f"Case {{idx}} input shape incompatible with signature: {{raw_input!r}}")
    if not values_equal(actual, expected):
        raise AssertionError(
            f"Case {{idx}} failed: expected={{expected!r}} actual={{actual!r}} input={{raw_input!r}}"
        )

print("__EVAL_PASS__")
"""


def _grade_livecodebench(item: dict[str, Any], code: str) -> dict[str, Any]:
    raw = item["raw_item"]
    public_tests = json.loads(raw["public_test_cases"])
    private_tests = _decode_livecodebench_private_tests(raw["private_test_cases"])
    tests = public_tests + private_tests
    test_type = tests[0].get("testtype", "stdin") if tests else "stdin"

    if test_type == "functional":
        script = _build_livecodebench_functional_harness(code, raw, tests)
        completed = _run_python_script(script)
        passed = completed.returncode == 0 and "__EVAL_PASS__" in completed.stdout
        details = completed.stderr or completed.stdout
        return {"passed": passed, "details": details[:4000], "grader_raw": (completed.stdout + completed.stderr)[:8000]}

    last_stdout = ""
    for idx, case in enumerate(tests):
        completed = _run_python_script(f"{EXECUTION_PREAMBLE}\n{code}", stdin_text=case.get("input", ""))
        actual = _normalize_text_output(completed.stdout)
        expected = _normalize_text_output(case.get("output", ""))
        last_stdout = completed.stdout + completed.stderr
        if completed.returncode != 0:
            return {
                "passed": False,
                "details": f"stdin case {idx} exited with code {completed.returncode}",
                "grader_raw": last_stdout[:8000],
            }
        if actual != expected:
            return {
                "passed": False,
                "details": f"stdin case {idx} mismatch: expected={expected!r} actual={actual!r}",
                "grader_raw": last_stdout[:8000],
            }
    return {"passed": True, "details": "", "grader_raw": last_stdout[:8000]}


def grade_code_response(item: dict[str, Any], response_text: str) -> dict[str, Any]:
    item = hydrate_sampled_item(item)
    code = extract_code_from_response(response_text)
    if not code.strip():
        return {
            "passed": False,
            "details": "No executable code was extracted from the model response.",
            "grader_raw": response_text[:8000],
            "extracted_code": code,
        }

    try:
        if item["dataset_kind"] == "humaneval":
            graded = _grade_humaneval(item, code)
        elif item["dataset_kind"] == "mbpp":
            graded = _grade_mbpp(item, code)
        elif item["dataset_kind"] == "livecodebench":
            graded = _grade_livecodebench(item, code)
        else:
            graded = {
                "passed": False,
                "details": f"Unsupported dataset kind: {item['dataset_kind']}",
                "grader_raw": "",
            }
    except subprocess.TimeoutExpired as exc:
        graded = {
            "passed": False,
            "details": f"Execution timed out after {exc.timeout}s.",
            "grader_raw": "",
        }
    except Exception as exc:
        graded = {
            "passed": False,
            "details": f"Execution harness error: {exc}",
            "grader_raw": "",
        }
    graded["extracted_code"] = code
    return graded


def hydrate_sampled_item(item: dict[str, Any]) -> dict[str, Any]:
    dataset_name = str(item["dataset"])
    spec = DATASET_LOOKUP[dataset_name]
    raw_item = item["raw_item"]
    metadata = get_item_metadata(spec, raw_item)
    return {
        "dataset": dataset_name,
        "dataset_kind": item.get("dataset_kind", spec.kind),
        "sample_index": item["sample_index"],
        "prompt": item["prompt"],
        "question": item.get("question", build_raw_question(spec, raw_item)),
        "gold_answer": item.get("gold_answer", build_gold_answer(spec, raw_item)),
        "level": item.get("level", metadata["level"]),
        "subject": item.get("subject", metadata["subject"]),
        "raw_item": raw_item,
    }


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
                "exec_graded",
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


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["dataset"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_label, dataset_name), group in sorted(grouped.items()):
        scored = [row for row in group if row["correct"] is not None]
        all_binary = [int(bool(row["correct"])) for row in scored]
        exec_count = sum(1 for row in group if row["grading_method"] == "exec_tests")
        error_count = sum(1 for row in group if row["status"] != "ok")
        summary_rows.append(
            {
                "model_label": model_label,
                "dataset": dataset_name,
                "num_items": len(group),
                "num_scored": len(scored),
                "num_correct": sum(all_binary),
                "accuracy": mean(all_binary) if all_binary else None,
                "exec_graded": exec_count,
                "generation_errors": error_count,
            }
        )
    return summary_rows


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


def evaluate_item(
    item: dict[str, Any],
    *,
    selected_model_specs: list[Any],
    clients: dict[str, Any],
    completed_keys: set[str],
    use_litellm: bool,
) -> list[tuple[str, dict[str, Any]]]:
    item = hydrate_sampled_item(item)
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
            "status": "ok",
            "correct": None,
            "grading_method": "exec_tests",
            "response_text": None,
            "extracted_code": None,
            "latency_s": None,
        }

        started = time.time()
        try:
            response_text = call_model(model_spec.label, clients, item["prompt"], use_litellm=use_litellm)
            record["response_text"] = response_text
            record["latency_s"] = round(time.time() - started, 2)
        except Exception as exc:
            record["status"] = "generation_error"
            record["latency_s"] = round(time.time() - started, 2)
            item_results.append((eval_key, record))
            continue

        graded = grade_code_response(item, record["response_text"] or "")
        record["correct"] = graded["passed"]
        record["extracted_code"] = graded.get("extracted_code")
        item_results.append((eval_key, record))

    return item_results


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nAccuracy summary")
    print("-" * 80)
    for row in summary_rows:
        accuracy = row["accuracy"]
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        print(
            f"{row['model_label']:24} {row['dataset']:22} "
            f"acc={accuracy_text:>8} "
            f"scored={row['num_scored']:4}/{row['num_items']:4} "
            f"exec={row['exec_graded']:4} errors={row['generation_errors']:3}"
        )
    print("-" * 80)


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)
    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_model_specs = [MODEL_LOOKUP[label] for label in args.models]
    clients = build_clients_with_mode(args.models, use_litellm=args.use_litellm)
    validate_clients_for_models(args.models, clients, use_litellm=args.use_litellm)
    sample_file = Path(args.sample_file)
    sampled_items = load_jsonl(sample_file)
    if args.max_items is not None:
        if args.max_items < 0:
            raise SystemExit("Error: --max-items must be non-negative.")
        sampled_items = sampled_items[: args.max_items]
    dataset_counts: dict[str, int] = defaultdict(int)
    for item in sampled_items:
        dataset_counts[str(item["dataset"])] += 1

    run_config = {
        "models": args.models,
        "sample_file": str(sample_file),
        "datasets": sorted(dataset_counts.keys()),
        "sample_plan": dict(dataset_counts),
        "use_litellm": args.use_litellm,
        "max_workers": args.max_workers,
        "max_items": args.max_items,
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
        desc="Evaluating static coding",
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
                completed_keys=completed_keys_snapshot,
                use_litellm=args.use_litellm,
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
                    completed_keys=completed_keys_snapshot,
                    use_litellm=args.use_litellm,
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
