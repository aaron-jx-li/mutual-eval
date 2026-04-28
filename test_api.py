#!/usr/bin/env python3
"""
Simple API key smoke tests for mutual-eval/.env.

Usage:
  python test_api.py
  python test_api.py --env .env
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path
from typing import Callable

import requests


def parse_export_env(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    export_re = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = export_re.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        lexer = shlex.shlex(value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
        if not tokens:
            continue
        value = tokens[0]
        env[key] = value
    return env


def mask_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def normalize_base(base_url: str) -> str:
    return base_url.rstrip("/")


def request_json(
    method: str, url: str, *, headers: dict[str, str] | None = None, timeout: int = 20
) -> tuple[bool, str]:
    try:
        resp = requests.request(method, url, headers=headers, timeout=timeout)
        if 200 <= resp.status_code < 300:
            return True, f"HTTP {resp.status_code}"
        body = (resp.text or "").strip().replace("\n", " ")
        if len(body) > 180:
            body = f"{body[:180]}..."
        return False, f"HTTP {resp.status_code}: {body}"
    except requests.RequestException as exc:
        return False, str(exc)


def test_openai(env: dict[str, str]) -> tuple[bool, str]:
    key = env.get("OPENAI_API_KEY")
    if not key:
        return False, "missing OPENAI_API_KEY"
    base = normalize_base(env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    return request_json("GET", f"{base}/models", headers={"Authorization": f"Bearer {key}"})


def test_gemini(env: dict[str, str]) -> tuple[bool, str]:
    key = env.get("GEMINI_API_KEY")
    if not key:
        return False, "missing GEMINI_API_KEY"
    base = normalize_base(
        env.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    )
    return request_json("GET", f"{base}/models", headers={"Authorization": f"Bearer {key}"})


def test_anthropic(env: dict[str, str]) -> tuple[bool, str]:
    key = env.get("ANTHROPIC_API_KEY")
    if not key:
        return False, "missing ANTHROPIC_API_KEY"
    base = normalize_base(env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"))
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    return request_json("GET", f"{base}/models", headers=headers)


def test_openrouter(env: dict[str, str]) -> tuple[bool, str]:
    key = env.get("OPENROUTER_API_KEY")
    if not key:
        return False, "missing OPENROUTER_API_KEY"
    headers = {"Authorization": f"Bearer {key}"}
    referer = env.get("OPENROUTER_HTTP_REFERER")
    title = env.get("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return request_json("GET", "https://openrouter.ai/api/v1/models", headers=headers)


def run_check(
    name: str,
    env: dict[str, str],
    key_name: str,
    test_fn: Callable[[dict[str, str]], tuple[bool, str]],
) -> bool:
    key = env.get(key_name)
    masked = mask_secret(key) if key else "N/A"
    ok, detail = test_fn(env)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:<12} key={masked}  {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Test provider API keys from .env")
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file that uses lines like: export KEY=value",
    )
    args = parser.parse_args()

    env_path = Path(args.env).expanduser().resolve()
    if not env_path.exists():
        print(f"ERROR: env file not found: {env_path}")
        return 2

    env = parse_export_env(env_path)
    os.environ.update(env)

    print(f"Testing API keys from: {env_path}")
    print("-" * 72)

    checks = [
        ("openai", "OPENAI_API_KEY", test_openai),
        ("gemini", "GEMINI_API_KEY", test_gemini),
        ("anthropic", "ANTHROPIC_API_KEY", test_anthropic),
        ("openrouter", "OPENROUTER_API_KEY", test_openrouter),
    ]

    passed = 0
    for name, key_name, fn in checks:
        if run_check(name, env, key_name, fn):
            passed += 1

    total = len(checks)
    print("-" * 72)
    print(f"Result: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
