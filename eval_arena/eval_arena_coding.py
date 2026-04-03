#!/usr/bin/env python3
"""
Static-style Arena coding evaluation with reward-model scoring.

For each Arena coding question (from arena_expert_5k_coding.jsonl), this script:
1. Generates one answer from each selected model.
2. Scores each generated answer with the reward model.
3. Writes one result row per `(question, model)` evaluation.

Config supports first_k to limit evaluation to the first k items (default: 500).

Example:
  python eval_arena/eval_arena_coding.py --config eval_arena/config_arena_coding.yaml
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anthropic
from openai import BadRequestError, OpenAI
from tqdm import tqdm
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_static.model_api_smoke_test import (  # noqa: E402
    MODEL_LOOKUP,
    MODEL_SPECS,
    extract_chat_content,
    get_routed_model_id,
    get_env_value,
    load_env_file,
    normalize_base_url,
    should_use_litellm_for_model,
)
from reward_client import RewardClient  # noqa: E402


DEFAULT_MODELS = [spec.label for spec in MODEL_SPECS]
ANSWER_INSTRUCTION = (
    "Provide a complete, correct solution. "
    "Include any necessary code, explanations, or steps as appropriate."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected models on Arena coding questions with a reward model.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Settings are read from the 'evaluation' section.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Arena coding input file (JSONL from arena_expert_5k_coding).",
    )
    parser.add_argument(
        "--first-k",
        type=int,
        default=None,
        dest="first_k",
        help="Use only first k items (default from config: 500).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=DEFAULT_MODELS,
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--rm-base-url",
        default=None,
        help="Reward model base URL. Falls back to ARENA_RM_BASE_URL or REWARD_MODEL_BASE_URL.",
    )
    parser.add_argument(
        "--rm-token",
        default=None,
        help="Reward model token. Falls back to ARENA_RM_TOKEN or REWARD_MODEL_TOKEN.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help=(
            "Direct OpenAI base URL for GPT models. Falls back to config, then "
            "OPENAI_BASE_URL, then https://api.openai.com/v1."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of Arena questions to evaluate in parallel.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Write checkpoint every N completed evals.",
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
        help="Maximum output tokens for generation requests. Omit or set to null in config to use no limit.",
    )
    parser.add_argument(
        "--reward-timeout",
        type=int,
        default=None,
        help="Reward model timeout seconds for each answer scoring request.",
    )
    parser.add_argument(
        "--reward-max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent reward-model requests.",
    )
    parser.add_argument(
        "--use-litellm",
        action="store_true",
        help="Route supported model calls through LiteLLM and fall back for the rest.",
    )
    parser.add_argument(
        "--litellm-models",
        nargs="+",
        default=None,
        choices=DEFAULT_MODELS,
        help=(
            "Explicit subset of model labels to route through LiteLLM. "
            "When set, this overrides the default Claude-only routing."
        ),
    )
    parser.add_argument(
        "--retry-statuses",
        nargs="*",
        default=None,
        choices=("generation_error", "reward_error"),
        help=(
            "When resuming, retry prior rows with these statuses instead of "
            "treating them as completed. Defaults to config or ['generation_error']."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing responses.jsonl in the output directory.",
    )
    args = parser.parse_args()
    if not hasattr(args, "model_max_tokens"):
        args.model_max_tokens = None
    return args


def resolve_env_path() -> Path:
    local_env = HERE / ".env"
    if local_env.exists():
        return local_env
    return REPO_ROOT / ".env"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: str(get_env_value(m.group(1)) or m.group(0)), value)
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
    section = config.get("evaluation", {})

    if args.input_file is None:
        args.input_file = section.get("input_file", "data/arena_expert_5k_coding.jsonl")
    if args.first_k is None:
        args.first_k = int(section.get("first_k", 500))
    if args.models is None:
        args.models = section.get("models", list(DEFAULT_MODELS))
    if args.output_dir is None:
        args.output_dir = section.get("output_dir", "results/arena_eval/arena_coding_v0")
    if args.rm_base_url is None:
        args.rm_base_url = section.get("rm_base_url")
    if args.rm_token is None:
        args.rm_token = section.get("rm_token")
    if args.openai_base_url is None:
        args.openai_base_url = section.get("openai_base_url")
    if args.max_workers is None:
        args.max_workers = int(section.get("max_workers", 4))
    if args.save_every is None:
        args.save_every = int(section.get("save_every", 50))
    if args.generation_timeout is None:
        args.generation_timeout = int(section.get("generation_timeout", 120))
    if args.generation_max_tokens is None:
        cfg_val = section.get("generation_max_tokens")
        args.generation_max_tokens = int(cfg_val) if cfg_val is not None else None
    if not hasattr(args, "model_max_tokens") or args.model_max_tokens is None:
        raw = section.get("model_max_tokens") or {}
        args.model_max_tokens = {k: (int(v) if v is not None else None) for k, v in raw.items()}
    if args.reward_timeout is None:
        args.reward_timeout = int(section.get("reward_timeout", 300))
    if args.reward_max_concurrency is None:
        args.reward_max_concurrency = int(section.get("reward_max_concurrency", 8))
    if not args.use_litellm:
        args.use_litellm = bool(section.get("use_litellm", False))
    if args.litellm_models is None and section.get("litellm_models") is not None:
        args.litellm_models = list(section.get("litellm_models", []))
    if args.retry_statuses is None:
        args.retry_statuses = list(section.get("retry_statuses", ["generation_error"]))
    if not args.resume:
        args.resume = bool(section.get("resume", False))
    return args


def load_input_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise SystemExit(f"Expected a JSON array in {path}.")
        return [dict(row) for row in parsed]

    # JSONL with possible multi-line objects (embedded newlines in strings)
    rows: list[dict[str, Any]] = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        buf = line
        while True:
            try:
                obj = json.loads(buf)
                rows.append(dict(obj))
                break
            except json.JSONDecodeError as e:
                if "Unterminated string" in str(e) and i < len(lines):
                    # Line was split by embedded newline; rejoin with escaped newline
                    buf += "\\n" + lines[i]
                    i += 1
                else:
                    raise SystemExit(f"JSON decode error in {path} at line {i}: {e}") from e
    return rows


def build_question_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build items from arena coding rows. Uses 'prompt' field as the question."""
    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        question = str(row.get("prompt") or row.get("question") or "").strip()
        if not question:
            continue
        item_id = str(row.get("id") or f"arena_{row_index:06d}")
        items.append(
            {
                "item_id": item_id,
                "row_index": row_index,
                "question": question,
                "prompt": f"{question}\n\n{ANSWER_INSTRUCTION}",
            }
        )
    return items


def should_route_via_litellm(
    spec: Any,
    *,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> bool:
    """
    Decide whether a model should be called through LiteLLM.

    Default behavior:
    - Claude/Anthropic models go through LiteLLM.
    - OpenAI and Gemini use their native clients.

    When litellm_models is set, it explicitly selects the model labels that
    should go through LiteLLM.

    When --use-litellm is enabled, preserve the old behavior and route every
    model with a LiteLLM mapping through LiteLLM.
    """
    if litellm_models is not None:
        return spec.label in litellm_models and spec.litellm_model_id is not None
    if use_litellm:
        return should_use_litellm_for_model(spec, use_litellm=True)
    return spec.provider == "anthropic" and spec.litellm_model_id is not None


def resolve_openai_direct_base_url(explicit_base_url: str | None = None) -> str:
    """
    Resolve the base URL for direct OpenAI calls.

    Guard against accidentally pointing the native OpenAI client at a LiteLLM
    gateway. LiteLLM should use LITELLM_BASE_URL instead.
    """
    base_url = normalize_base_url(explicit_base_url or get_env_value("OPENAI_BASE_URL"), "openai")
    if base_url and "litellm" in base_url.lower():
        raise SystemExit(
            "OPENAI_BASE_URL points to a LiteLLM endpoint. For direct OpenAI usage, "
            "set OPENAI_BASE_URL to an OpenAI hostname such as "
            "https://api.openai.com/v1 or https://us.api.openai.com/v1, and set "
            "LITELLM_BASE_URL separately for LiteLLM-routed models."
        )
    return base_url or "https://api.openai.com/v1"


def requires_litellm_client(
    selected_models: list[str],
    *,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> bool:
    """Return whether any selected model will be routed through LiteLLM."""
    return any(
        should_route_via_litellm(
            MODEL_LOOKUP[name],
            use_litellm=use_litellm,
            litellm_models=litellm_models,
        )
        for name in selected_models
    )


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


def build_clients(
    selected_models: list[str],
    generation_timeout: int,
    *,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    direct_providers = {
        MODEL_LOOKUP[name].provider
        for name in selected_models
        if not should_route_via_litellm(
            MODEL_LOOKUP[name],
            use_litellm=use_litellm,
            litellm_models=litellm_models,
        )
    }

    if "openai" in direct_providers:
        key = get_env_value("OPENAI_API_KEY")
        if key:
            base_url = resolve_openai_direct_base_url(openai_base_url)
            clients["openai"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)

    if "anthropic" in direct_providers:
        key = get_env_value("ANTHROPIC_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("ANTHROPIC_BASE_URL"), "anthropic")
            clients["anthropic"] = anthropic.Anthropic(
                api_key=key,
                base_url=base_url,
                timeout=generation_timeout,
            )

    if "google" in direct_providers:
        key = get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("GEMINI_BASE_URL"), "google")
            if base_url:
                clients["google"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)

    if "openrouter" in direct_providers:
        key = get_env_value("OPENROUTER_API_KEY")
        if key:
            clients["openrouter"] = OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                timeout=generation_timeout,
            )

    missing = sorted(direct_providers - set(clients.keys()) - {"openrouter"})
    if "openrouter" in direct_providers and "openrouter" not in clients:
        missing.append("openrouter")
    if missing:
        raise SystemExit(f"Missing API credentials/config for providers: {', '.join(sorted(set(missing)))}")
    return clients


def build_clients_with_mode(
    selected_models: list[str],
    generation_timeout: int,
    *,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    clients = build_clients(
        selected_models,
        generation_timeout,
        use_litellm=use_litellm,
        litellm_models=litellm_models,
        openai_base_url=openai_base_url,
    )
    if requires_litellm_client(
        selected_models,
        use_litellm=use_litellm,
        litellm_models=litellm_models,
    ):
        key = get_env_value("LITELLM_API_KEY", "OPENAI_API_KEY")
        base_url = normalize_base_url(get_env_value("LITELLM_BASE_URL"), "openai")
        if not key or not base_url:
            raise SystemExit(
                "LiteLLM is required for the selected routing mode. Set "
                "LITELLM_API_KEY (or OPENAI_API_KEY) and LITELLM_BASE_URL."
            )
        clients["litellm"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)
    return clients


def call_model(
    spec,
    clients: dict[str, Any],
    prompt: str,
    *,
    generation_max_tokens: int | None,
    use_litellm: bool = False,
    litellm_models: set[str] | None = None,
) -> str:
    # Gemini 2.5/3.1 are thinking models: they consume tokens for internal reasoning
    # before producing output, so the shared generation_max_tokens budget is insufficient.
    _GOOGLE_THINKING_PREFIXES = ("2.5", "3.1")
    google_max_tokens = (
        65536
        if any(p in spec.model_id for p in _GOOGLE_THINKING_PREFIXES)
        else generation_max_tokens
    )
    if should_route_via_litellm(
        spec,
        use_litellm=use_litellm,
        litellm_models=litellm_models,
    ):
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


def build_reward_conversation(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def normalize_reward(raw_reward: Any) -> tuple[float | None, str | None]:
    if isinstance(raw_reward, Exception):
        return None, str(raw_reward)
    if isinstance(raw_reward, (int, float)):
        value = float(raw_reward)
        if math.isfinite(value):
            return value, None
        return None, f"Non-finite reward: {value}"
    try:
        value = float(raw_reward)
        if math.isfinite(value):
            return value, None
    except (TypeError, ValueError):
        pass
    return None, f"Unexpected reward payload: {repr(raw_reward)[:500]}"


def make_eval_key(item_or_record: dict[str, Any], model_label: str | None = None) -> str:
    label = model_label if model_label is not None else str(item_or_record["model_label"])
    return f"{item_or_record['item_id']}::{label}"


def build_eval_order_lookup(
    items: list[dict[str, Any]],
    selected_model_specs: list[Any],
) -> dict[str, int]:
    order_lookup: dict[str, int] = {}
    order = 0
    for item in items:
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
            row["item_id"],
            row["model_label"],
        ),
    )


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


def split_resume_results(
    rows: list[dict[str, Any]],
    *,
    retry_statuses: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split prior rows into kept results and rows that should be retried."""
    if not retry_statuses:
        return rows, []
    kept: list[dict[str, Any]] = []
    retried: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") in retry_statuses:
            retried.append(row)
        else:
            kept.append(row)
    return kept, retried


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_label"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for model_label, group in sorted(grouped.items()):
        ok_rows = [row for row in group if row["status"] == "ok"]
        rewards = [float(row["reward"]) for row in ok_rows if row["reward"] is not None]
        summary_rows.append(
            {
                "model_label": model_label,
                "num_items": len(group),
                "num_scored": len(ok_rows),
                "avg_reward": mean(rewards) if rewards else None,
                "max_reward": max(rewards) if rewards else None,
                "min_reward": min(rewards) if rewards else None,
                "generation_errors": sum(1 for row in group if row["status"] == "generation_error"),
                "reward_errors": sum(1 for row in group if row["status"] == "reward_error"),
            }
        )
    return summary_rows


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "num_items",
                "num_scored",
                "avg_reward",
                "max_reward",
                "min_reward",
                "generation_errors",
                "reward_errors",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


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
    write_summary_csv(output_dir / "summary.csv", summarize_results(ordered))
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
    reward_client: RewardClient,
    reward_timeout: int,
    reward_semaphore: threading.Semaphore,
    completed_keys: set[str],
    generation_max_tokens: int | None,
    model_max_tokens: dict[str, int | None],
    use_litellm: bool,
    litellm_models: set[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    item_results: list[tuple[str, dict[str, Any]]] = []

    for model_spec in selected_model_specs:
        eval_key = make_eval_key(item, model_spec.label)
        if eval_key in completed_keys:
            continue

        record: dict[str, Any] = {
            "item_id": item["item_id"],
            "row_index": item["row_index"],
            "model_label": model_spec.label,
            "model_provider": model_spec.provider,
            "model_id": model_spec.model_id,
            "question": item["question"],
            "prompt": item["prompt"],
            "response_text": None,
            "reward": None,
            "status": "ok",
            "error": None,
            "generation_latency_s": None,
            "reward_latency_s": None,
            "total_latency_s": None,
            "processed_at": None,
        }

        started = time.time()
        generation_started = time.time()
        try:
            effective_max_tokens = model_max_tokens.get(model_spec.label, generation_max_tokens)
            response_text = call_model(
                model_spec,
                clients,
                item["prompt"],
                generation_max_tokens=effective_max_tokens,
                use_litellm=use_litellm,
                litellm_models=litellm_models,
            )
            record["response_text"] = response_text
            record["generation_latency_s"] = round(time.time() - generation_started, 2)
        except Exception as exc:
            record["status"] = "generation_error"
            record["generation_latency_s"] = round(time.time() - generation_started, 2)
            record["total_latency_s"] = round(time.time() - started, 2)
            record["error"] = str(exc)
            record["processed_at"] = datetime.now().isoformat()
            item_results.append((eval_key, record))
            continue

        reward_started = time.time()
        with reward_semaphore:
            reward_raw = reward_client.get_reward(
                build_reward_conversation(item["question"], record["response_text"] or ""),
                timeout=reward_timeout,
            )
        reward_value, reward_error = normalize_reward(reward_raw)
        record["reward_latency_s"] = round(time.time() - reward_started, 2)
        record["reward"] = reward_value
        if reward_error is not None:
            record["status"] = "reward_error"
            record["error"] = reward_error
        record["total_latency_s"] = round(time.time() - started, 2)
        record["processed_at"] = datetime.now().isoformat()
        item_results.append((eval_key, record))

    return item_results


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nReward summary")
    print("-" * 90)
    for row in summary_rows:
        avg_reward = row["avg_reward"]
        avg_text = f"{avg_reward:.4f}" if avg_reward is not None else "N/A"
        print(
            f"{row['model_label']:24} "
            f"avg_reward={avg_text:>8} "
            f"scored={row['num_scored']:4}/{row['num_items']:4} "
            f"gen_err={row['generation_errors']:3} reward_err={row['reward_errors']:3}"
        )
    print("-" * 90)


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)

    rm_base_url = args.rm_base_url or get_env_value("ARENA_RM_BASE_URL", "REWARD_MODEL_BASE_URL")
    rm_token = args.rm_token or get_env_value("ARENA_RM_TOKEN", "REWARD_MODEL_TOKEN")
    if not rm_base_url:
        raise SystemExit("Reward model base URL is required. Pass --rm-base-url or set ARENA_RM_BASE_URL.")
    if not rm_token:
        raise SystemExit("Reward model token is required. Pass --rm-token or set ARENA_RM_TOKEN.")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1.")
    if args.save_every < 1:
        raise SystemExit("--save-every must be at least 1.")
    if args.reward_max_concurrency < 1:
        raise SystemExit("--reward-max-concurrency must be at least 1.")
    if args.first_k < 1:
        raise SystemExit("--first-k must be at least 1.")
    if args.generation_max_tokens is not None and args.generation_max_tokens < 1:
        raise SystemExit("--generation-max-tokens must be at least 1, or omit to use no limit.")
    if args.litellm_models is not None:
        invalid_litellm_models = sorted(set(args.litellm_models) - set(DEFAULT_MODELS))
        if invalid_litellm_models:
            raise SystemExit(
                "Unknown litellm_models entries: " + ", ".join(invalid_litellm_models)
            )
    selected_model_specs = [MODEL_LOOKUP[label] for label in args.models]
    litellm_models = set(args.litellm_models) if args.litellm_models is not None else None
    retry_statuses = set(args.retry_statuses or [])
    uses_direct_openai = any(
        spec.provider == "openai"
        and not should_route_via_litellm(
            spec,
            use_litellm=args.use_litellm,
            litellm_models=litellm_models,
        )
        for spec in selected_model_specs
    )
    resolved_openai_base_url = (
        resolve_openai_direct_base_url(args.openai_base_url) if uses_direct_openai else None
    )
    resolved_litellm_base_url = normalize_base_url(get_env_value("LITELLM_BASE_URL"), "openai")

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clients = build_clients_with_mode(
        args.models,
        args.generation_timeout,
        use_litellm=args.use_litellm,
        litellm_models=litellm_models,
        openai_base_url=resolved_openai_base_url,
    )
    reward_client = RewardClient(base_url=rm_base_url, token=rm_token)
    reward_semaphore = threading.BoundedSemaphore(value=min(args.reward_max_concurrency, 8))

    raw_rows = load_input_rows(input_path)
    raw_rows = raw_rows[: args.first_k]
    items = build_question_items(raw_rows)

    eval_order_lookup = build_eval_order_lookup(items, selected_model_specs)
    responses_path = output_dir / "responses.jsonl"
    results: list[dict[str, Any]] = []
    retried_results: list[dict[str, Any]] = []
    if args.resume:
        existing_results = load_jsonl(responses_path)
        results, retried_results = split_resume_results(
            existing_results,
            retry_statuses=retry_statuses,
        )
    completed_keys = {make_eval_key(record) for record in results}
    total_evals = len(items) * len(selected_model_specs)
    completed_evals = len(completed_keys)

    run_config = {
        "config": args.config,
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "models": args.models,
        "resume": args.resume,
        "save_every": args.save_every,
        "max_workers": args.max_workers,
        "generation_timeout": args.generation_timeout,
        "generation_max_tokens": args.generation_max_tokens,
        "reward_timeout": args.reward_timeout,
        "reward_max_concurrency": min(args.reward_max_concurrency, 8),
        "use_litellm": args.use_litellm,
        "litellm_models": sorted(litellm_models) if litellm_models is not None else None,
        "openai_base_url": resolved_openai_base_url,
        "litellm_base_url": resolved_litellm_base_url,
        "retry_statuses": sorted(retry_statuses),
        "first_k": args.first_k,
        "rm_base_url": rm_base_url,
        "total_items": len(items),
        "total_evals": total_evals,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if retried_results:
        retry_counts = Counter(str(row.get("status", "unknown")) for row in retried_results)
        retry_summary = ", ".join(f"{status}={count}" for status, count in sorted(retry_counts.items()))
        print(f"Retrying prior failed rows on resume: {retry_summary}")

    progress = tqdm(
        total=total_evals,
        initial=completed_evals,
        desc="Evaluating Arena coding",
        unit="eval",
    )

    pending_items = [
        item
        for item in items
        if any(make_eval_key(item, model_spec.label) not in completed_keys for model_spec in selected_model_specs)
    ]
    completed_keys_snapshot = set(completed_keys)

    def record_result(eval_key: str, record: dict[str, Any]) -> None:
        nonlocal completed_evals
        results.append(record)
        completed_keys.add(eval_key)
        progress.set_postfix_str(
            f"{record['model_label']} | question #{record['row_index'] + 1}",
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
                reward_client=reward_client,
                reward_timeout=args.reward_timeout,
                reward_semaphore=reward_semaphore,
                completed_keys=completed_keys_snapshot,
                generation_max_tokens=args.generation_max_tokens,
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
                    reward_client=reward_client,
                    reward_timeout=args.reward_timeout,
                    reward_semaphore=reward_semaphore,
                    completed_keys=completed_keys_snapshot,
                    generation_max_tokens=args.generation_max_tokens,
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
    summary_rows = summarize_results(order_results(results, eval_order_lookup))
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
