"""Shared model roster for agentic evaluations.

Re-exports the 15-model roster defined in
``mutual-eval/eval_static/model_api_smoke_test.py`` so that the agentic
pipelines (SWE-bench Lite and Terminal-Bench) are guaranteed to score the
same set of models as the static-eval pipeline.

Also provides helpers that convert a roster label (e.g. ``gpt-5.4``) into
a LiteLLM-routable model id plus the provider credentials the underlying
agent (``mini-swe-agent`` or ``terminus-2`` via Harbor) should see in its
environment. Each model can be routed either:

* through a LiteLLM gateway (``route = "litellm"``), or
* against its native provider API (``route = "native"``).

This is per-model: a single run may send some models through LiteLLM and
others direct.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_STATIC_DIR = _REPO_ROOT / "eval_static"
if str(_EVAL_STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_STATIC_DIR))

from model_api_smoke_test import (  # noqa: E402  (import after sys.path tweak)
    MODEL_LOOKUP,
    MODEL_SPECS,
    ModelSpec,
    get_env_value,
    load_env_file,
    normalize_base_url,
    resolve_env_path,
)

__all__ = [
    "MODEL_LOOKUP",
    "MODEL_SPECS",
    "ModelSpec",
    "get_env_value",
    "load_env_file",
    "normalize_base_url",
    "resolve_env_path",
    "DEFAULT_ROSTER",
    "ROUTE_LITELLM",
    "ROUTE_NATIVE",
    "ROUTE_VALUES",
    "resolve_route",
    "resolve_model_routes",
    "resolve_litellm_model_id",
    "resolve_agent_env_for_model",
    "normalize_route_overrides",
]


def normalize_route_overrides(value: Any) -> dict[str, str]:
    """Validate and normalise ``route_overrides:`` from YAML.

    Accepts a ``{label: 'litellm'|'native'}`` mapping and returns a dict
    of strings. Raises ``SystemExit`` on invalid values so misconfigured
    runs fail fast.
    """
    try:
        raw = _as_mapping(value)
    except ValueError as exc:
        raise SystemExit(f"route_overrides must be a mapping: {exc}") from exc
    out: dict[str, str] = {}
    for label, route in raw.items():
        if label not in MODEL_LOOKUP:
            raise SystemExit(
                f"route_overrides references unknown model {label!r}. "
                f"Valid labels: {sorted(MODEL_LOOKUP)}"
            )
        if route not in ROUTE_VALUES:
            raise SystemExit(
                f"route_overrides[{label!r}]={route!r}; must be one of {ROUTE_VALUES}"
            )
        out[label] = route
    return out


def _normalize_model_label_list(value: Any, *, key_name: str) -> set[str]:
    """Validate ``litellm_models`` / ``native_models``-style lists."""
    if value is None:
        return set()
    if not isinstance(value, list):
        raise SystemExit(
            f"{key_name} must be a list of model labels; got {type(value).__name__}: {value!r}"
        )
    labels = {str(v) for v in value}
    unknown = sorted(label for label in labels if label not in MODEL_LOOKUP)
    if unknown:
        raise SystemExit(
            f"{key_name} references unknown model label(s): {unknown}. "
            f"Valid labels: {sorted(MODEL_LOOKUP)}"
        )
    return labels


ROUTE_LITELLM = "litellm"
ROUTE_NATIVE = "native"
ROUTE_VALUES = (ROUTE_LITELLM, ROUTE_NATIVE)


# Same 15 labels used by eval_static/config_static_coding.yaml.
DEFAULT_ROSTER: list[str] = [
    "gpt-5.4",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "gemini-3.1-pro",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "grok-4",
    "deepseek-v3.2",
    "mistral-large-3",
    "qwen3-max-thinking",
    "llama-4-maverick-instruct",
]


def resolve_route(
    label: str,
    *,
    default_route: str,
    route_overrides: Mapping[str, str] | None = None,
) -> str:
    """Return ``"litellm"`` or ``"native"`` for a roster label.

    Precedence: ``route_overrides[label]`` > ``default_route``.
    """
    if default_route not in ROUTE_VALUES:
        raise ValueError(f"default_route must be one of {ROUTE_VALUES}; got {default_route!r}")
    if route_overrides and label in route_overrides:
        route = route_overrides[label]
        if route not in ROUTE_VALUES:
            raise ValueError(
                f"route_overrides[{label!r}]={route!r}; must be one of {ROUTE_VALUES}"
            )
        return route
    return default_route


def resolve_model_routes(
    labels: list[str],
    *,
    use_litellm: bool | None = None,
    litellm_models: Any = None,
    native_models: Any = None,
    default_route: str | None = None,
    route_overrides: Any = None,
) -> dict[str, str]:
    """Resolve per-model routes from static-style config flags.

    Supported config knobs (all optional, precedence from low -> high):
      1) ``use_litellm`` (bool): global default route (static-style).
      2) ``default_route`` (``litellm``/``native``): newer synonym.
      3) ``litellm_models`` / ``native_models`` (lists):
         - if only one list is provided, it acts as a *subset selector* and
           non-listed selected models are routed to the opposite route.
         - if both are provided, they are treated as explicit per-model
           assignments on top of the global default.
      4) ``route_overrides`` mapping: highest precedence compatibility knob.
    """
    if default_route is not None:
        base_route = str(default_route)
        if base_route not in ROUTE_VALUES:
            raise SystemExit(
                f"default_route={base_route!r}; must be one of {ROUTE_VALUES}."
            )
    elif use_litellm is not None:
        base_route = ROUTE_LITELLM if bool(use_litellm) else ROUTE_NATIVE
    else:
        base_route = ROUTE_LITELLM

    litellm_set = _normalize_model_label_list(litellm_models, key_name="litellm_models")
    native_set = _normalize_model_label_list(native_models, key_name="native_models")
    overlap = sorted(litellm_set & native_set)
    if overlap:
        raise SystemExit(
            "litellm_models and native_models overlap for: "
            f"{overlap}. A model can only be assigned one route."
        )

    selected = list(labels)
    routes: dict[str, str] = {label: base_route for label in selected}

    only_litellm_subset = bool(litellm_set) and not bool(native_set)
    only_native_subset = bool(native_set) and not bool(litellm_set)

    if only_litellm_subset:
        for label in selected:
            routes[label] = ROUTE_LITELLM if label in litellm_set else ROUTE_NATIVE
    elif only_native_subset:
        for label in selected:
            routes[label] = ROUTE_NATIVE if label in native_set else ROUTE_LITELLM
    else:
        for label in selected:
            if label in litellm_set:
                routes[label] = ROUTE_LITELLM
            elif label in native_set:
                routes[label] = ROUTE_NATIVE

    normalized_overrides = normalize_route_overrides(route_overrides)
    for label in selected:
        if label in normalized_overrides:
            routes[label] = normalized_overrides[label]

    return routes


def resolve_litellm_model_id(
    label: str,
    *,
    overrides: Mapping[str, str] | None = None,
    route: str = ROUTE_LITELLM,
) -> str:
    """Map a roster label to a LiteLLM-compatible model id.

    The returned id is the provider-prefixed form that both
    ``mini-swe-agent`` v2 and Harbor/terminus-2 pass straight through to
    LiteLLM. The same id works whether routing is ``"litellm"``
    (gateway) or ``"native"`` (direct provider) -- in the ``native``
    case LiteLLM uses the prefix to pick the right provider creds, in
    the gateway case the gateway itself dispatches.

    Rules:
      * If ``overrides`` has an entry, use it verbatim.
      * Else prefer ``spec.litellm_model_id``; if that id lacks a ``/``
        prefix (e.g. ``claude-opus-4-6``), prepend the model family so we
        end up with ``anthropic/claude-opus-4-6``.
      * For OpenRouter-only specs (``litellm_model_id is None``) fall back
        to ``openrouter/<spec.model_id>``.
    """
    if overrides and label in overrides:
        return overrides[label]

    if label not in MODEL_LOOKUP:
        raise KeyError(
            f"Unknown model label: {label!r}. "
            f"Valid labels: {sorted(MODEL_LOOKUP)}"
        )
    spec = MODEL_LOOKUP[label]

    # Some roster mappings (notably Claude Haiku) may point at a Vertex route.
    # For native-provider runs, prefer the provider's own model id so we do not
    # require provider-specific SDK extras unrelated to the selected route.
    if route == ROUTE_NATIVE and spec.provider != "openrouter":
        if "/" in spec.model_id:
            return spec.model_id
        # Harbor/LiteLLM expects Google's Gemini provider prefix as "gemini/"
        # (not "google/") for direct model routing.
        if spec.provider == "google":
            return f"gemini/{spec.model_id}"
        return f"{spec.family}/{spec.model_id}"

    if spec.litellm_model_id:
        mid = spec.litellm_model_id
        return mid if "/" in mid else f"{spec.family}/{mid}"

    if spec.provider == "openrouter":
        return f"openrouter/{spec.model_id}"

    return spec.model_id


def resolve_agent_env_for_model(label: str, *, route: str) -> dict[str, str]:
    """Return env vars that should be exported for the agent subprocess.

    * ``route == "litellm"``: the agent talks to the LiteLLM gateway,
      reached via ``LITELLM_API_KEY`` / ``LITELLM_BASE_URL`` (with
      ``OPENAI_*`` fallbacks). Both names are set so downstream code
      that reads either pair finds the gateway.
    * ``route == "native"``: only the provider's own creds are exported:
        - openai     -> OPENAI_API_KEY (+ OPENAI_BASE_URL)
        - anthropic  -> ANTHROPIC_API_KEY (+ ANTHROPIC_BASE_URL)
        - google     -> GEMINI_API_KEY / GOOGLE_API_KEY (+ GEMINI_BASE_URL)
        - openrouter -> OPENROUTER_API_KEY

    Raises ``SystemExit`` with an actionable message when required creds
    are missing from the environment.
    """
    if route not in ROUTE_VALUES:
        raise ValueError(f"route must be one of {ROUTE_VALUES}; got {route!r}")
    if label not in MODEL_LOOKUP:
        raise KeyError(
            f"Unknown model label: {label!r}. Valid labels: {sorted(MODEL_LOOKUP)}"
        )
    spec = MODEL_LOOKUP[label]

    if route == ROUTE_LITELLM:
        key = get_env_value("LITELLM_API_KEY", "OPENAI_API_KEY")
        base_url = normalize_base_url(
            get_env_value("LITELLM_BASE_URL", "OPENAI_BASE_URL"),
            "openai",
        )
        if not key or not base_url:
            raise SystemExit(
                f"Model {label!r} is routed via LiteLLM but LITELLM_API_KEY/BASE_URL "
                "(or OPENAI_* fallbacks) are not set in the environment."
            )
        return {
            "OPENAI_API_KEY": key,
            "OPENAI_BASE_URL": base_url,
            "LITELLM_API_KEY": key,
            "LITELLM_BASE_URL": base_url,
        }

    env: dict[str, str] = {}
    if spec.provider == "openai":
        key = get_env_value("OPENAI_API_KEY")
        if not key:
            raise SystemExit(
                f"Model {label!r} is routed natively but OPENAI_API_KEY is not set."
            )
        env["OPENAI_API_KEY"] = key
        base_url = normalize_base_url(get_env_value("OPENAI_BASE_URL"), "openai")
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
    elif spec.provider == "anthropic":
        key = get_env_value("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                f"Model {label!r} is routed natively but ANTHROPIC_API_KEY is not set."
            )
        env["ANTHROPIC_API_KEY"] = key
        base_url = normalize_base_url(get_env_value("ANTHROPIC_BASE_URL"), "anthropic")
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
    elif spec.provider == "google":
        key = get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not key:
            raise SystemExit(
                f"Model {label!r} is routed natively but GEMINI_API_KEY / GOOGLE_API_KEY is not set."
            )
        env["GEMINI_API_KEY"] = key
        env["GOOGLE_API_KEY"] = key
        base_url = normalize_base_url(get_env_value("GEMINI_BASE_URL"), "google")
        if base_url:
            env["GEMINI_BASE_URL"] = base_url
    elif spec.provider == "openrouter":
        key = get_env_value("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit(
                f"Model {label!r} is routed natively but OPENROUTER_API_KEY is not set."
            )
        env["OPENROUTER_API_KEY"] = key
    else:
        raise SystemExit(
            f"Native routing is not supported for provider {spec.provider!r} "
            f"(model {label!r})."
        )
    return env


def _as_mapping(value: Any) -> dict[str, str]:
    """Defensive coercion for YAML-loaded route_overrides/model_overrides."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    raise ValueError(
        f"Expected a mapping (dict); got {type(value).__name__}: {value!r}"
    )
