#!/usr/bin/env python3
"""
Minimal connectivity test for the 16-model experiment roster.

This script:
1. Loads environment variables from a local .env file if present.
2. Sends one short prompt to each configured model.
3. Prints a compact pass/fail summary and optional JSON output.

Expected environment variables:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GOOGLE_API_KEY or GEMINI_API_KEY
    OPENROUTER_API_KEY

Usage examples:
    python model_api_smoke_test.py
    python model_api_smoke_test.py --models gpt-5.4 claude-sonnet-4-6
    python model_api_smoke_test.py --use-litellm
    python model_api_smoke_test.py --json-out smoke_test_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anthropic
from openai import BadRequestError, OpenAI


PROMPT = "Reply with exactly: OK"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    family: str
    provider: str
    model_id: str
    api_env: str
    litellm_model_id: str | None = None


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("gpt-5.4", "openai", "openai", "gpt-5.4", "OPENAI_API_KEY", "openai/gpt-5.4"),
    ModelSpec("gpt-5-mini", "openai", "openai", "gpt-5-mini", "OPENAI_API_KEY", "openai/gpt-5-mini"),
    ModelSpec("gpt-5.1-mini", "openai", "openai", "gpt-5.1-mini", "OPENAI_API_KEY", "openai/gpt-5.1-mini"),
    ModelSpec("gpt-4.1", "openai", "openai", "gpt-4.1", "OPENAI_API_KEY", "openai/gpt-4.1"),
    ModelSpec(
        "gpt-4.1-mini",
        "openai",
        "openai",
        "gpt-4.1-mini",
        "OPENAI_API_KEY",
        "openai/gpt-4.1-mini",
    ),
    ModelSpec(
        "claude-opus-4-6",
        "anthropic",
        "anthropic",
        "claude-opus-4-6",
        "ANTHROPIC_API_KEY",
        "claude-opus-4-6",
    ),
    ModelSpec(
        "claude-sonnet-4-6",
        "anthropic",
        "anthropic",
        "claude-sonnet-4-6",
        "ANTHROPIC_API_KEY",
        "claude-sonnet-4-6",
    ),
    ModelSpec(
        "claude-haiku-4-5",
        "anthropic",
        "anthropic",
        "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY",
        "vertex_ai/claude-haiku-4-5@20251001",
    ),
    ModelSpec(
        "gemini-3.1-pro",
        "google",
        "google",
        "gemini-3.1-pro-preview",
        "GOOGLE_API_KEY",
        "gemini/gemini-3.1-pro-preview",
    ),
    ModelSpec(
        "gemini-2.5-pro",
        "google",
        "google",
        "gemini-2.5-pro",
        "GOOGLE_API_KEY",
        "gemini/gemini-2.5-pro",
    ),
    ModelSpec(
        "gemini-2.5-flash",
        "google",
        "google",
        "gemini-2.5-flash",
        "GOOGLE_API_KEY",
        "gemini/gemini-2.5-flash",
    ),
    ModelSpec("grok-4", "xai", "openrouter", "x-ai/grok-4.20-beta", "OPENROUTER_API_KEY", None),
    ModelSpec(
        "deepseek-v3.2",
        "deepseek",
        "openrouter",
        "deepseek/deepseek-v3.2",
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "deepseek-r1", 
        "deepseek", 
        "openrouter", 
        "deepseek/deepseek-r1-0528", 
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "mistral-large-3",
        "mistral",
        "openrouter",
        "mistralai/mistral-large-2512",
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "qwen3-max-thinking",
        "qwen",
        "openrouter",
        "qwen/qwen3-max-thinking",
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "llama-4-maverick-instruct",
        "llama",
        "openrouter",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "OPENROUTER_API_KEY",
        None,
    ),
]

MODEL_LOOKUP = {spec.label: spec for spec in MODEL_SPECS}


def should_use_litellm_for_model(spec: ModelSpec, *, use_litellm: bool) -> bool:
    return use_litellm and spec.litellm_model_id is not None


def get_routed_model_id(spec: ModelSpec, *, use_litellm: bool) -> str:
    if use_litellm:
        if not spec.litellm_model_id:
            raise ValueError(f"LiteLLM mapping is not configured for model '{spec.label}'.")
        return spec.litellm_model_id
    return spec.model_id


@dataclass
class SmokeTestResult:
    label: str
    provider: str
    model_id: str
    ok: bool
    latency_s: float | None
    response_text: str | None
    error: str | None


def get_env_value(*names: str) -> str | None:
    """Return the first non-empty env value that is not a literal shell expression."""
    for name in names:
        value = os.environ.get(name)
        if value and "${" not in value:
            return value
    return None


def extract_chat_content(content: Any) -> str:
    """Best-effort extraction from OpenAI-compatible message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def normalize_base_url(base_url: str | None, provider: str) -> str | None:
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    if provider == "anthropic" and normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def call_openai_compatible(
    client: OpenAI,
    model_id: str,
    prompt: str,
    *,
    temperature: float = 0.0,
) -> str:
    request_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not model_id.startswith("gpt-5"):
        request_kwargs["temperature"] = temperature

    try:
        resp = client.chat.completions.create(**request_kwargs)
    except BadRequestError as exc:
        message = str(exc).lower()
        if "temperature" in request_kwargs and "unsupported" in message and "temperature" in message:
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**retry_kwargs)
        else:
            raise

    return extract_chat_content(resp.choices[0].message.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test API access for the planned model list.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[spec.label for spec in MODEL_SPECS],
        choices=[spec.label for spec in MODEL_SPECS],
        help="Subset of model labels to test. Defaults to all models.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write raw results as JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds for each call (default: 60).",
    )
    parser.add_argument(
        "--use-litellm",
        action="store_true",
        help="Route all model calls through a single OpenAI-compatible LiteLLM endpoint.",
    )
    return parser.parse_args()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
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


def build_clients(
    selected_specs: list[ModelSpec],
    *,
    timeout: int,
    use_litellm: bool,
) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    providers = {spec.provider for spec in selected_specs}

    if use_litellm:
        key = get_env_value("LITELLM_API_KEY", "OPENAI_API_KEY")
        base_url = normalize_base_url(
            get_env_value("LITELLM_BASE_URL", "OPENAI_BASE_URL"),
            "openai",
        )
        if key and base_url:
            clients["litellm"] = OpenAI(api_key=key, base_url=base_url, timeout=timeout)

    if "openai" in providers:
        key = get_env_value("OPENAI_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("OPENAI_BASE_URL"), "openai")
            clients["openai"] = OpenAI(api_key=key, base_url=base_url, timeout=timeout)

    if "anthropic" in providers:
        key = get_env_value("ANTHROPIC_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("ANTHROPIC_BASE_URL"), "anthropic")
            clients["anthropic"] = anthropic.Anthropic(api_key=key, base_url=base_url, timeout=timeout)

    if "google" in providers:
        key = get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if key:
            gemini_base_url = normalize_base_url(get_env_value("GEMINI_BASE_URL"), "google")
            if gemini_base_url:
                clients["google_openai_compat"] = OpenAI(
                    base_url=gemini_base_url,
                    api_key=key,
                    timeout=timeout,
                )

    if "openrouter" in providers:
        key = get_env_value("OPENROUTER_API_KEY")
        if key:
            clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                timeout=timeout,
            )

    return clients


def call_model(
    spec: ModelSpec,
    clients: dict[str, Any],
    timeout: int,
    *,
    use_litellm: bool,
) -> SmokeTestResult:
    start = time.time()

    routed_via_litellm = should_use_litellm_for_model(spec, use_litellm=use_litellm)

    if routed_via_litellm:
        has_required_key = bool(get_env_value("LITELLM_API_KEY", "OPENAI_API_KEY"))
        has_required_base_url = bool(get_env_value("LITELLM_BASE_URL", "OPENAI_BASE_URL"))
    elif spec.provider == "google":
        has_required_key = bool(get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        has_required_base_url = True
    else:
        has_required_key = bool(get_env_value(spec.api_env))
        has_required_base_url = True

    if not has_required_key or not has_required_base_url:
        return SmokeTestResult(
            label=spec.label,
            provider=spec.provider,
            model_id=spec.model_id,
            ok=False,
            latency_s=None,
            response_text=None,
            error=(
                "Missing LITELLM_API_KEY/OPENAI_API_KEY or LITELLM_BASE_URL/OPENAI_BASE_URL"
                if routed_via_litellm
                else
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY"
                if spec.provider == "google"
                else f"Missing {spec.api_env}"
            ),
        )

    try:
        if routed_via_litellm:
            routed_model_id = get_routed_model_id(spec, use_litellm=True)
            text = call_openai_compatible(clients["litellm"], routed_model_id, PROMPT)

        elif spec.provider == "openai":
            text = call_openai_compatible(clients["openai"], spec.model_id, PROMPT)

        elif spec.provider == "anthropic":
            resp = clients["anthropic"].messages.create(
                model=spec.model_id,
                max_tokens=32,
                messages=[{"role": "user", "content": PROMPT}],
            )
            text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()

        elif spec.provider == "google":
            if "google_openai_compat" not in clients:
                raise RuntimeError(
                    "Gemini testing requires GEMINI_BASE_URL configured for the OpenAI-compatible endpoint."
                )
            text = call_openai_compatible(clients["google_openai_compat"], spec.model_id, PROMPT)

        elif spec.provider == "openrouter":
            text = call_openai_compatible(clients["openrouter"], spec.model_id, PROMPT)

        else:
            raise ValueError(f"Unsupported provider: {spec.provider}")

        latency = time.time() - start
        return SmokeTestResult(
            label=spec.label,
            provider=spec.provider,
            model_id=spec.model_id,
            ok=bool(text),
            latency_s=round(latency, 2),
            response_text=text,
            error=None if text else "Empty response",
        )

    except Exception as exc:
        latency = time.time() - start
        return SmokeTestResult(
            label=spec.label,
            provider=spec.provider,
            model_id=spec.model_id,
            ok=False,
            latency_s=round(latency, 2),
            response_text=None,
            error=str(exc),
        )


def print_summary(results: list[SmokeTestResult]) -> None:
    print()
    print(f"{'MODEL':30} {'PROVIDER':12} {'STATUS':8} {'LATENCY':8} DETAILS")
    print("-" * 100)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        latency = f"{result.latency_s:.2f}s" if result.latency_s is not None else "-"
        detail = result.response_text if result.ok else result.error
        detail = (detail or "").replace("\n", " ")
        if len(detail) > 80:
            detail = detail[:77] + "..."
        print(f"{result.label:30} {result.provider:12} {status:8} {latency:8} {detail}")

    passed = sum(result.ok for result in results)
    print("-" * 100)
    print(f"Passed {passed}/{len(results)} model checks.")


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())

    selected_specs = [MODEL_LOOKUP[label] for label in args.models]
    clients = build_clients(selected_specs, timeout=args.timeout, use_litellm=args.use_litellm)
    results = [
        call_model(spec, clients, args.timeout, use_litellm=args.use_litellm)
        for spec in selected_specs
    ]

    print_summary(results)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(json.dumps([asdict(result) for result in results], indent=2))
        print(f"Wrote JSON results to {output_path}")

    if not all(result.ok for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
