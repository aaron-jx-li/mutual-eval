"""
Hardcoded reference rankings for IRT robustness experiments.

All rankings are sourced from the "Expected Results" table in robustness/README.md.
Keys are lowercase model names matching the actual values in the data files.
Values are integer ranks (1 = best).

For new JSONL-based data, dicts are looked up by file stem:
    os.path.splitext(os.path.basename(path))[0]

For joint both-reward mode, use _JOINT_REFERENCE_RANKINGS keyed by
    (static_stem, arena_stem) tuple.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Old CSV-based references (backwards compat)
# ---------------------------------------------------------------------------

# 2PL column (static dataset, old models)
_STATIC_REF: dict[str, int] = {
    "grok-3-mini-beta":           1,
    "gpt-4.1-mini-2025-04-14":    2,
    "deepseek-v3-0324":           3,
    "mistral-medium-2505":        4,
    "claude-3-5-haiku-20241022":  5,
    "llama-3.3-70b-it":           6,
    "gpt-4o-mini":                7,
    "gemini-2.0-flash":           8,
    "claude-3-7-sonnet-20250219": 9,
    "gemma-3-27b-it":             10,
}

# Pairwise IRT (Open-ended) column
_ARENA_REF: dict[str, int] = {
    "deepseek-v3-0324":           1,
    "gpt-4.1-mini-2025-04-14":    2,
    "mistral-medium-2505":        3,
    "claude-3-7-sonnet-20250219": 4,
    "grok-3-mini-beta":           5,
    "gpt-4o-mini":                6,
    "gemma-3-27b-it":             7,
    "claude-3-5-haiku-20241022":  8,
    "llama-3.3-70b-it":           9,
    "gemini-2.0-flash":           10,
}

# 2PL+Pairwise IRT (Overall) column
_BOTH_REF: dict[str, int] = {
    "deepseek-v3-0324":           1,
    "gpt-4.1-mini-2025-04-14":    2,
    "mistral-medium-2505":        3,
    "grok-3-mini-beta":           4,
    "claude-3-5-haiku-20241022":  5,
    "gpt-4o-mini":                6,
    "llama-3.3-70b-it":           7,
    "claude-3-7-sonnet-20250219": 8,
    "gemini-2.0-flash":           9,
    "gemma-3-27b-it":             10,
}

# ---------------------------------------------------------------------------
# New JSONL-based references (new 15-model data)
# ---------------------------------------------------------------------------

_REF_STATIC_MATH: dict[str, int] = {
    "claude-opus-4-6":           1,
    "claude-sonnet-4-6":         2,
    "gpt-5.4":                   3,
    "deepseek-v3.2":             4,
    "qwen3-max-thinking":        5,
    "grok-4":                    6,
    "gpt-4.1-mini":              7,
    "claude-haiku-4-5":          8,
    "gpt-5-mini":                9,
    "gpt-4.1":                   10,
    "mistral-large-3":           11,
    "llama-4-maverick-instruct": 12,
    "gemini-3.1-pro":            13,
    "gemini-2.5-flash":          14,
    "gemini-2.5-pro":            15,
}

_REF_STATIC_CODING: dict[str, int] = {
    "claude-sonnet-4-6":         1,
    "claude-opus-4-6":           2,
    "gemini-2.5-flash":          3,
    "gpt-5-mini":                4,
    "gpt-5.4":                   5,
    "gpt-4.1-mini":              6,
    "qwen3-max-thinking":        7,
    "gemini-3.1-pro":            8,
    "gpt-4.1":                   9,
    "claude-haiku-4-5":          10,
    "gemini-2.5-pro":            11,
    "deepseek-v3.2":             12,
    "mistral-large-3":           13,
    "grok-4":                    14,
    "llama-4-maverick-instruct": 15,
}

_REF_STATIC_ALL: dict[str, int] = {
    "claude-opus-4-6":           1,
    "claude-sonnet-4-6":         2,
    "gpt-5.4":                   3,
    "qwen3-max-thinking":        4,
    "gpt-4.1-mini":              5,
    "grok-4":                    6,
    "deepseek-v3.2":             7,
    "claude-haiku-4-5":          8,
    "gpt-5-mini":                9,
    "gpt-4.1":                   10,
    "llama-4-maverick-instruct": 11,
    "mistral-large-3":           12,
    "gemini-3.1-pro":            13,
    "gemini-2.5-flash":          14,
    "gemini-2.5-pro":            15,
}

_REF_ARENA_MATH: dict[str, int] = {
    "qwen3-max-thinking":        1,
    "gemini-3.1-pro":            2,
    "claude-opus-4-6":           3,
    "gpt-5.4":                   4,
    "deepseek-v3.2":             5,
    "claude-sonnet-4-6":         6,
    "gpt-4.1-mini":              7,
    "gpt-4.1":                   8,
    "gemini-2.5-pro":            9,
    "claude-haiku-4-5":          10,
    "gemini-2.5-flash":          11,
    "grok-4":                    12,
    "mistral-large-3":           13,
    "llama-4-maverick-instruct": 14,
    "gpt-5-mini":                15,
}

_REF_ARENA_CODING: dict[str, int] = {
    "gpt-5.4":                   1,
    "claude-opus-4-6":           2,
    "gemini-3.1-pro":            3,
    "claude-sonnet-4-6":         4,
    "gemini-2.5-pro":            5,
    "grok-4":                    6,
    "qwen3-max-thinking":        7,
    "gemini-2.5-flash":          8,
    "claude-haiku-4-5":          9,
    "deepseek-v3.2":             10,
    "gpt-4.1-mini":              11,
    "mistral-large-3":           12,
    "gpt-4.1":                   13,
    "gpt-5-mini":                14,
    "llama-4-maverick-instruct": 15,
}

_REF_ARENA_GENERIC: dict[str, int] = {
    "claude-opus-4-6":           1,
    "grok-4":                    2,
    "mistral-large-3":           3,
    "gemini-3.1-pro":            4,
    "claude-sonnet-4-6":         5,
    "gemini-2.5-pro":            6,
    "gpt-5.4":                   7,
    "deepseek-v3.2":             8,
    "qwen3-max-thinking":        9,
    "gemini-2.5-flash":          10,
    "gpt-4.1":                   11,
    "gpt-4.1-mini":              12,
    "claude-haiku-4-5":          13,
    "gpt-5-mini":                14,
    "llama-4-maverick-instruct": 15,
}

_REF_ARENA_ALL: dict[str, int] = {
    "claude-opus-4-6":           1,
    "gemini-3.1-pro":            2,
    "gpt-5.4":                   3,
    "claude-sonnet-4-6":         4,
    "gemini-2.5-pro":            5,
    "grok-4":                    6,
    "mistral-large-3":           7,
    "qwen3-max-thinking":        8,
    "deepseek-v3.2":             9,
    "gemini-2.5-flash":          10,
    "gpt-4.1":                   11,
    "gpt-4.1-mini":              12,
    "claude-haiku-4-5":          13,
    "gpt-5-mini":                14,
    "llama-4-maverick-instruct": 15,
}

_REF_BOTH_MATH: dict[str, int] = {
    "claude-opus-4-6":           1,
    "claude-sonnet-4-6":         2,
    "gpt-5.4":                   3,
    "deepseek-v3.2":             4,
    "qwen3-max-thinking":        5,
    "grok-4":                    6,
    "gpt-4.1-mini":              7,
    "claude-haiku-4-5":          8,
    "gpt-4.1":                   9,
    "mistral-large-3":           10,
    "gpt-5-mini":                11,
    "llama-4-maverick-instruct": 12,
    "gemini-3.1-pro":            13,
    "gemini-2.5-pro":            14,
    "gemini-2.5-flash":          15,
}

_REF_BOTH_CODING: dict[str, int] = {
    "claude-opus-4-6":           1,
    "claude-sonnet-4-6":         2,
    "gpt-5.4":                   3,
    "gemini-3.1-pro":            4,
    "gemini-2.5-pro":            5,
    "qwen3-max-thinking":        6,
    "gemini-2.5-flash":          7,
    "claude-haiku-4-5":          8,
    "gpt-5-mini":                9,
    "gpt-4.1-mini":              10,
    "gpt-4.1":                   11,
    "deepseek-v3.2":             12,
    "grok-4":                    13,
    "mistral-large-3":           14,
    "llama-4-maverick-instruct": 15,
}

_REF_BOTH_ALL: dict[str, int] = {
    "claude-opus-4-6":           1,
    "claude-sonnet-4-6":         2,
    "gpt-5.4":                   3,
    "qwen3-max-thinking":        4,
    "grok-4":                    5,
    "deepseek-v3.2":             6,
    "claude-haiku-4-5":          7,
    "gpt-4.1-mini":              8,
    "mistral-large-3":           9,
    "gpt-4.1":                   10,
    "gpt-5-mini":                11,
    "llama-4-maverick-instruct": 12,
    "gemini-3.1-pro":            13,
    "gemini-2.5-pro":            14,
    "gemini-2.5-flash":          15,
}

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Keyed by file stem: os.path.splitext(os.path.basename(path))[0]
REFERENCE_RANKINGS: dict[str, dict[str, int]] = {
    # Old CSV-based (backwards compat)
    "static": _STATIC_REF,
    "arena":  _ARENA_REF,
    "both":   _BOTH_REF,
    # New JSONL-based
    "static_math_v0":   _REF_STATIC_MATH,
    "static_coding_v0": _REF_STATIC_CODING,
    "static_all_v0":    _REF_STATIC_ALL,
    "arena_math_v0":    _REF_ARENA_MATH,
    "arena_coding_v0":  _REF_ARENA_CODING,
    "arena_generic_v0": _REF_ARENA_GENERIC,
    "arena_all_v0":     _REF_ARENA_ALL,
}

# Joint both-reward: (static_stem, arena_stem) → ref dict
JOINT_REFERENCE_RANKINGS: dict[tuple[str, str], dict[str, int]] = {
    ("static_math_v0",   "arena_math_v0"):   _REF_BOTH_MATH,
    ("static_coding_v0", "arena_coding_v0"): _REF_BOTH_CODING,
    ("static_all_v0",    "arena_all_v0"):    _REF_BOTH_ALL,
}
