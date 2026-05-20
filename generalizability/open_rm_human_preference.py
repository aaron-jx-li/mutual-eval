#!/usr/bin/env python3
"""
Evaluate open-source reward models against human pairwise preferences.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_rm_generalizability import MODEL_NAME_ALIASES, build_reward_model_runner, rm_slug


DEFAULT_DATASETS: dict[str, str] = {
    "math": "data/v1_math_200.jsonl",
    "coding": "data/arena_expert_5k_coding.jsonl",
    "misc": "data/v1_misc_300.jsonl",
    "generic": "data/v1_generic_1000.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare open reward models against human pairwise preference labels.",
    )
    parser.add_argument(
        "--reward-models",
        nargs="+",
        required=True,
        help="HF reward model names to run.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[f"{k}={v}" for k, v in DEFAULT_DATASETS.items()],
        help="Domain=path JSONL entries, e.g. math=data/v1_math_200.jsonl",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional subset of domains from --dataset to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="generalizability/results/open_rm_human_preference",
        help="Root output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Reward model batch size.")
    parser.add_argument("--max-length", type=int, default=4096, help="Tokenizer max length.")
    parser.add_argument("--max-rows-per-domain", type=int, default=None, help="Debug row cap.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype for HF loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for HF reward models.",
    )
    parser.add_argument(
        "--logit-index",
        type=int,
        default=-1,
        help="If model outputs multi-logit classification head, select this index.",
    )
    parser.add_argument(
        "--tie-threshold",
        type=float,
        default=0.0,
        help="Treat score_a - score_b within +/- threshold as tie.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_domain_paths(entries: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --dataset entry (expected domain=path): {entry}")
        domain, raw_path = entry.split("=", 1)
        domain = domain.strip()
        if not domain:
            raise ValueError(f"Invalid domain in --dataset entry: {entry}")
        out[domain] = resolve_path(raw_path.strip())
    return out


def _extract_python_quoted_literal(s: str, start_idx: int) -> tuple[str | None, int]:
    i = start_idx
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s) or s[i] not in ("'", '"'):
        return None, i
    quote = s[i]
    j = i + 1
    escaped = False
    while j < len(s):
        c = s[j]
        if escaped:
            escaped = False
        elif c == "\\":
            escaped = True
        elif c == quote:
            token = s[i : j + 1]
            try:
                return ast.literal_eval(token), j + 1
            except Exception:
                return None, j + 1
        j += 1
    return None, j


def _extract_assistant_text_from_repr(conversation_repr: str) -> str | None:
    role_iter = list(re.finditer(r"['\"]role['\"]\s*:\s*['\"]assistant['\"]", conversation_repr))
    if not role_iter:
        return None
    for role_match in reversed(role_iter):
        segment = conversation_repr[role_match.start() :]
        text_match = re.search(r"['\"]text['\"]\s*:\s*", segment)
        if not text_match:
            continue
        value, _ = _extract_python_quoted_literal(segment, text_match.end())
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_assistant_text_from_list(conversation_list: list[Any]) -> str | None:
    for msg in reversed(conversation_list):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role")) != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            joined = "\n".join(text_parts).strip()
            if joined:
                return joined
    return None


def extract_assistant_text(conversation: Any) -> str | None:
    if isinstance(conversation, list):
        return _extract_assistant_text_from_list(conversation)
    if isinstance(conversation, str):
        return _extract_assistant_text_from_repr(conversation)
    return None


def normalize_winner(raw_winner: Any) -> str | None:
    if raw_winner is None:
        return None
    winner = str(raw_winner).strip().lower()
    if winner in {"model_a", "a", "left"}:
        return "model_a"
    if winner in {"model_b", "b", "right"}:
        return "model_b"
    if winner in {"tie", "both_bad", "both_good"}:
        return "tie"
    return None


def load_human_pref_jsonl(path: Path, domain: str, max_rows: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset JSONL: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            winner = normalize_winner(d.get("winner"))
            if winner is None:
                continue
            prompt = d.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            response_a = extract_assistant_text(d.get("conversation_a"))
            response_b = extract_assistant_text(d.get("conversation_b"))
            if not response_a or not response_b:
                continue
            rows.append(
                {
                    "domain": domain,
                    "example_id": str(d.get("id", line_idx)),
                    "prompt": prompt,
                    "response_a": response_a,
                    "response_b": response_b,
                    "human_winner": winner,
                    "model_a": str(d.get("model_a", "")),
                    "model_b": str(d.get("model_b", "")),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break

    return pd.DataFrame(rows)


def compute_pairwise_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "n_human_non_tie": 0,
            "n_human_tie": 0,
            "n_pred_non_tie": 0,
            "n_pred_tie": 0,
            "strict_accuracy": float("nan"),
            "non_tie_accuracy": float("nan"),
            "non_tie_accuracy_excluding_pred_ties": float("nan"),
            "pred_tie_rate": float("nan"),
            "human_tie_rate": float("nan"),
        }

    n_rows = int(len(df))
    human_non_tie_mask = df["human_winner"].isin(["model_a", "model_b"])
    human_tie_mask = df["human_winner"].eq("tie")
    pred_non_tie_mask = df["pred_winner"].isin(["model_a", "model_b"])
    pred_tie_mask = df["pred_winner"].eq("tie")

    strict_accuracy = float((df["pred_winner"] == df["human_winner"]).mean())
    if human_non_tie_mask.any():
        non_tie_accuracy = float(
            (df.loc[human_non_tie_mask, "pred_winner"] == df.loc[human_non_tie_mask, "human_winner"]).mean()
        )
    else:
        non_tie_accuracy = float("nan")

    non_tie_non_pred_tie_mask = human_non_tie_mask & pred_non_tie_mask
    if non_tie_non_pred_tie_mask.any():
        non_tie_accuracy_excluding_pred_ties = float(
            (
                df.loc[non_tie_non_pred_tie_mask, "pred_winner"]
                == df.loc[non_tie_non_pred_tie_mask, "human_winner"]
            ).mean()
        )
    else:
        non_tie_accuracy_excluding_pred_ties = float("nan")

    return {
        "n_rows": n_rows,
        "n_human_non_tie": int(human_non_tie_mask.sum()),
        "n_human_tie": int(human_tie_mask.sum()),
        "n_pred_non_tie": int(pred_non_tie_mask.sum()),
        "n_pred_tie": int(pred_tie_mask.sum()),
        "strict_accuracy": strict_accuracy,
        "non_tie_accuracy": non_tie_accuracy,
        "non_tie_accuracy_excluding_pred_ties": non_tie_accuracy_excluding_pred_ties,
        "pred_tie_rate": float(pred_tie_mask.mean()),
        "human_tie_rate": float(human_tie_mask.mean()),
    }


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def main() -> None:
    args = parse_args()
    domain_paths = parse_domain_paths(args.dataset)
    domains = args.domains if args.domains is not None else list(domain_paths.keys())
    for d in domains:
        if d not in domain_paths:
            raise SystemExit(f"Requested domain '{d}' not found in --dataset entries")

    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    domain_data: dict[str, pd.DataFrame] = {}
    for domain in domains:
        df = load_human_pref_jsonl(domain_paths[domain], domain=domain, max_rows=args.max_rows_per_domain)
        if df.empty:
            raise SystemExit(f"No usable rows loaded for domain={domain} from {domain_paths[domain]}")
        domain_data[domain] = df
        if not args.quiet:
            print(
                f"Loaded domain={domain}: rows={len(df)}, "
                f"human_non_tie={int(df['human_winner'].isin(['model_a', 'model_b']).sum())}, "
                f"human_tie={int(df['human_winner'].eq('tie').sum())}"
            )

    aggregate_rows: list[dict[str, Any]] = []
    for raw_model_name in args.reward_models:
        model_name = MODEL_NAME_ALIASES.get(raw_model_name, raw_model_name)
        if not args.quiet:
            if model_name != raw_model_name:
                print(f"\nLoading reward model: {raw_model_name} (resolved to {model_name})")
            else:
                print(f"\nLoading reward model: {model_name}")
        rm = build_reward_model_runner(
            model_name=model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
            logit_index=args.logit_index,
            device=args.device,
            dtype=args.dtype,
        )
        model_slug = rm_slug(model_name)

        for domain in domains:
            if not args.quiet:
                print(f"\nScoring domain={domain} with model={model_name}")
            df = domain_data[domain].copy()

            prompts = df["prompt"].tolist()
            score_a = rm.score(prompts=prompts, responses=df["response_a"].tolist(), quiet=args.quiet)
            score_b = rm.score(prompts=prompts, responses=df["response_b"].tolist(), quiet=args.quiet)
            df["score_a"] = score_a
            df["score_b"] = score_b
            diff = score_a - score_b
            df["score_diff_a_minus_b"] = diff

            pred = np.where(
                np.abs(diff) <= args.tie_threshold,
                "tie",
                np.where(diff > 0, "model_a", "model_b"),
            )
            df["pred_winner"] = pred

            metrics = compute_pairwise_metrics(df)

            run_dir = output_root / model_slug / domain
            run_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(run_dir / "pairwise_scored.csv", index=False)

            summary = {
                "reward_model": model_name,
                "reward_model_slug": model_slug,
                "domain": domain,
                "tie_threshold": args.tie_threshold,
                "metrics": metrics,
                "input_dataset": str(domain_paths[domain]),
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            aggregate_rows.append(
                {
                    "reward_model": model_name,
                    "domain": domain,
                    "n_rows": metrics["n_rows"],
                    "n_human_non_tie": metrics["n_human_non_tie"],
                    "n_human_tie": metrics["n_human_tie"],
                    "strict_accuracy": metrics["strict_accuracy"],
                    "non_tie_accuracy": metrics["non_tie_accuracy"],
                    "non_tie_accuracy_excluding_pred_ties": metrics["non_tie_accuracy_excluding_pred_ties"],
                    "pred_tie_rate": metrics["pred_tie_rate"],
                    "human_tie_rate": metrics["human_tie_rate"],
                }
            )

            if not args.quiet:
                print(
                    "  "
                    f"strict_acc={_fmt(metrics['strict_accuracy'])}  "
                    f"non_tie_acc={_fmt(metrics['non_tie_accuracy'])}  "
                    f"pred_tie_rate={_fmt(metrics['pred_tie_rate'])}"
                )

    summary_df = pd.DataFrame(aggregate_rows)
    summary_df.to_csv(output_root / "aggregate_summary.csv", index=False)
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"\nSaved all outputs under: {output_root}")


if __name__ == "__main__":
    main()
