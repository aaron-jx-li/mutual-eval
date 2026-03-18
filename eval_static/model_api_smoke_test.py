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


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("gpt-5.4", "openai", "openai", "gpt-5.4", "OPENAI_API_KEY"),
    ModelSpec("gpt-5-mini", "openai", "openai", "gpt-5-mini", "OPENAI_API_KEY"),
    ModelSpec("gpt-4.1", "openai", "openai", "gpt-4.1", "OPENAI_API_KEY"),
    ModelSpec("gpt-4.1-mini", "openai", "openai", "gpt-4.1-mini", "OPENAI_API_KEY"),
    ModelSpec("claude-opus-4-6", "anthropic", "anthropic", "claude-opus-4-6", "ANTHROPIC_API_KEY"),
    ModelSpec("claude-sonnet-4-6", "anthropic", "anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    ModelSpec(
        "claude-haiku-4-5",
        "anthropic",
        "anthropic",
        "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY",
    ),
    ModelSpec("gemini-3.1-pro", "google", "google", "gemini-3.1-pro-preview", "GOOGLE_API_KEY"),
    ModelSpec("gemini-2.5-pro", "google", "google", "gemini-2.5-pro", "GOOGLE_API_KEY"),
    ModelSpec("gemini-2.5-flash", "google", "google", "gemini-2.5-flash", "GOOGLE_API_KEY"),
    ModelSpec("grok-4", "xai", "openrouter", "x-ai/grok-4.20-beta", "OPENROUTER_API_KEY"),
    ModelSpec(
        "deepseek-v3.2",
        "deepseek",
        "openrouter",
        "deepseek/deepseek-v3.2",
        "OPENROUTER_API_KEY",
    ),
    ModelSpec(
        "deepseek-r1", 
        "deepseek", 
        "openrouter", 
        "deepseek/deepseek-r1-0528", 
        "OPENROUTER_API_KEY",
    ),
    ModelSpec(
        "mistral-large-3",
        "mistral",
        "openrouter",
        "mistralai/mistral-large-2512",
        "OPENROUTER_API_KEY",
    ),
    ModelSpec(
        "qwen3-max-thinking",
        "qwen",
        "openrouter",
        "qwen/qwen3-max-thinking",
        "OPENROUTER_API_KEY",
    ),
    ModelSpec(
        "llama-4-maverick-instruct",
        "llama",
        "openrouter",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "OPENROUTER_API_KEY",
    ),
]

MODEL_LOOKUP = {spec.label: spec for spec in MODEL_SPECS}


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


def build_clients(selected_specs: list[ModelSpec]) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    providers = {spec.provider for spec in selected_specs}

    if "openai" in providers:
        key = get_env_value("OPENAI_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("OPENAI_BASE_URL"), "openai")
            clients["openai"] = OpenAI(api_key=key, base_url=base_url, timeout=60)

    if "anthropic" in providers:
        key = get_env_value("ANTHROPIC_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("ANTHROPIC_BASE_URL"), "anthropic")
            clients["anthropic"] = anthropic.Anthropic(api_key=key, base_url=base_url, timeout=60)

    if "google" in providers:
        key = get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if key:
            gemini_base_url = normalize_base_url(get_env_value("GEMINI_BASE_URL"), "google")
            if gemini_base_url:
                clients["google_openai_compat"] = OpenAI(
                    base_url=gemini_base_url,
                    api_key=key,
                    timeout=60,
                )

    if "openrouter" in providers:
        key = get_env_value("OPENROUTER_API_KEY")
        if key:
            clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                timeout=60,
            )

    return clients


def call_model(spec: ModelSpec, clients: dict[str, Any], timeout: int) -> SmokeTestResult:
    start = time.time()

    if spec.provider == "google":
        has_required_key = bool(get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    else:
        has_required_key = bool(get_env_value(spec.api_env))

    if not has_required_key:
        return SmokeTestResult(
            label=spec.label,
            provider=spec.provider,
            model_id=spec.model_id,
            ok=False,
            latency_s=None,
            response_text=None,
            error=(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY"
                if spec.provider == "google"
                else f"Missing {spec.api_env}"
            ),
        )

    try:
        if spec.provider == "openai":
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
    load_env_file(Path(__file__).with_name(".env"))

    selected_specs = [MODEL_LOOKUP[label] for label in args.models]
    clients = build_clients(selected_specs)
    results = [call_model(spec, clients, args.timeout) for spec in selected_specs]

    print_summary(results)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(json.dumps([asdict(result) for result in results], indent=2))
        print(f"Wrote JSON results to {output_path}")

    if not all(result.ok for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
