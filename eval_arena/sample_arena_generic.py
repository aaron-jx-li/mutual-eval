#!/usr/bin/env python3
"""
Sample generic (non-technical, everyday) prompts from Arena Human Preference 140k.

Two-phase process:
  1. Pre-filter using dataset labels (is_code and category_tag.math_v0.1.math)
     to exclude coding and math questions cheaply.
  2. LLM judge (gpt-4.1-mini) confirms each candidate is a non-technical,
     everyday task whose answer requires no code and no math.
  3. Select top-N by judge confidence (highest first).

Definition of generic used by the judge:
  A question is non-technical if a competent answer requires NO programming
  knowledge and NO mathematical computation. The topic can be anything —
  writing, language, advice, reasoning, creativity, knowledge, culture, etc. —
  as long as the answer is plain prose, not code or equations.

Dataset:
  https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k

Examples:
  python eval_arena/sample_arena_generic.py

  python eval_arena/sample_arena_generic.py \\
    --output data/arena_140k_generic_judged.jsonl \\
    --max-kept 1000 \\
    --min-confidence 0.85 \\
    --min-prompt-len 150 \\
    --min-prompt-words 20

  python eval_arena/sample_arena_generic.py --resume
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Judge prompts
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict quality filter for an LLM evaluation benchmark. "
    "Your job is to identify substantive, non-trivial everyday tasks where "
    "response quality varies meaningfully between capable models."
)

_JUDGE_USER_TMPL = """\
Decide whether this prompt is suitable for an LLM benchmark that tests \
everyday writing and reasoning quality.

ACCEPT (is_generic: true) only if ALL of the following hold:
1. No specialised domain knowledge is needed — a thoughtful non-expert can
   give a fully competent answer without training in medicine, law, finance,
   science, engineering, history, or any other professional field.
2. No programming knowledge or code is required.
3. No mathematical calculation, proof, or quantitative reasoning is required.
4. Style, tone, clarity, or creativity matters — a better writer produces a
   noticeably better answer than a mediocre one.
5. The prompt is SUBSTANTIVE: it provides enough context or constraints that
   two capable models would produce meaningfully different responses worth
   comparing. A benchmark judge could clearly distinguish a great answer from
   a merely adequate one.

Typical examples that PASS:
- Write / edit / improve an email, message, cover letter, essay, or story
  with specific context or constraints given.
- Give thoughtful advice on a concrete everyday situation (relationships,
  career decisions, travel planning, etc.) with enough detail to reason about.
- Summarise or rewrite a provided piece of text with a specific goal.
- Brainstorm with a clear creative brief (product names for X, slogans for Y).
- Explain a concept with a clear audience and purpose in mind.
- Role-play or conversational tasks with a well-defined scenario.

REJECT (is_generic: false) if ANY of the following is true:
- Answering well requires expert domain knowledge (medical diagnosis, legal
  analysis, financial modelling, scientific derivation, historical scholarship,
  engineering specs, etc.).
- The prompt is about software, code, or technical systems.
- The prompt requires arithmetic, statistics, or any mathematical reasoning.
- The question has a single correct factual answer (trivia, definitions,
  vocabulary lookups, spell-checks, etc.).
- The prompt is a greeting, pleasantry, or casual conversation opener with
  no substantive task ("How are you", "Hey, what's up", "Hello", etc.).
- The prompt is so short and vague that any competent model would give
  essentially the same response — there is nothing to differentiate quality
  (e.g. "Write a short story", "Tell me a joke", "Write a poem").
- The request is for a single joke, riddle, or punchline with no creative
  constraints or context.
- The request is for a single word, synonym, antonym, or trivial vocabulary
  lookup.
- The prompt could be fully answered in one or two sentences with no
  meaningful variation in quality between responses.

When in doubt, reject. This benchmark needs tasks where a great response is
clearly better than a mediocre one, and where that difference comes from
writing skill, judgment, or reasoning — not just from knowing the answer.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_generic": true or false,
  "confidence": 0.0 to 1.0,
  "category": "short label (e.g. email writing, creative writing, advice, summarisation, brainstorming, etc.)",
  "reason": "one sentence"
}}"""

_DONE_STATUSES = {"ok", "skipped_no_prompt", "skipped_language"}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample generic (non-technical, everyday) prompts from "
            "Arena Human Preference 140k using dataset labels + LLM judge."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="lmarena-ai/arena-human-preference-140k",
        help="HuggingFace dataset path.",
    )
    p.add_argument("--split", default="train", help="Dataset split.")
    p.add_argument(
        "--output",
        default="data/arena_140k_generic_judged.jsonl",
        metavar="PATH",
        help="Output JSONL for selected generic questions.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Checkpoint JSON path. Defaults to <output stem>_checkpoint.json.",
    )
    p.add_argument(
        "--judged-log",
        default=None,
        metavar="PATH",
        help=(
            "Intermediate JSONL logging all accepted judge results. "
            "Defaults to <output stem>_judged.jsonl. Used for resume."
        ),
    )
    p.add_argument(
        "--max-kept",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum number of generic questions to select.",
    )
    p.add_argument(
        "--allowed-languages",
        nargs="+",
        default=["en"],
        metavar="LANG",
        help="Language allowlist (ISO codes).",
    )
    p.add_argument(
        "--min-prompt-len",
        type=int,
        default=150,
        help="Minimum character length of extracted prompt.",
    )
    p.add_argument(
        "--min-prompt-words",
        type=int,
        default=20,
        help="Minimum word count of extracted prompt.",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="OpenAI-compatible judge model.",
    )
    p.add_argument(
        "--judge-api-key-env",
        default="OPENAI_API_KEY",
        metavar="ENV",
        help="Env var holding the judge API key.",
    )
    p.add_argument(
        "--judge-base-url",
        default=None,
        metavar="URL",
        help="Base URL for the judge endpoint.",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.85,
        help="Minimum judge confidence to accept a candidate.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Judge client timeout in seconds.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel judge threads.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Flush checkpoint every N judged rows.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling candidates before judging.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint/judged-log files.",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        metavar="N",
        help="Cap on candidates to judge (useful for dry runs).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Env / IO helpers
# ──────────────────────────────────────────────────────────────────────────────

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
            key = key[len("export "):].strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def resolve_env_path() -> Path:
    here = Path(__file__).resolve()
    local = here.with_name(".env")
    if local.exists():
        return local
    return here.parent.parent / ".env"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Prompt extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t.strip()
    return ""


def _extract_from_serialized(raw: str) -> str:
    if not raw:
        return ""
    m = re.search(
        r"'role':\s*'user'.*?'text':\s*(?P<text>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",
        raw,
        flags=re.DOTALL,
    )
    if not m:
        return ""
    try:
        value = ast.literal_eval(m.group("text"))
    except (SyntaxError, ValueError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def extract_user_prompt(row: dict[str, Any]) -> str:
    for key in ("conversation_a", "conversation_b"):
        conv = row.get(key)
        if isinstance(conv, list):
            for msg in conv:
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                    text = _extract_text(msg.get("content"))
                    if text:
                        return text
        elif isinstance(conv, str):
            text = _extract_from_serialized(conv)
            if text:
                return text
    full = row.get("full_conversation")
    if isinstance(full, list):
        for turn in full:
            if isinstance(turn, dict):
                user_blob = turn.get("user")
                if isinstance(user_blob, dict):
                    text = _extract_text(user_blob.get("content"))
                    if text:
                        return text
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Dataset label pre-filter
# ──────────────────────────────────────────────────────────────────────────────

def _category_tag(row: dict[str, Any]) -> dict[str, Any]:
    """Return the parsed category_tag dict, or {}."""
    ct = row.get("category_tag")
    if isinstance(ct, str):
        try:
            ct = json.loads(ct)
        except json.JSONDecodeError:
            return {}
    return ct if isinstance(ct, dict) else {}


def is_labelled_coding(row: dict[str, Any]) -> bool:
    """True if the dataset marks this row as a coding question."""
    if row.get("is_code"):
        return True
    ct = _category_tag(row)
    # Some rows carry an explicit coding tag under criteria
    return False


def is_labelled_math(row: dict[str, Any]) -> bool:
    """True if the dataset marks this row as a math question."""
    ct = _category_tag(row)
    math_tag = ct.get("math_v0.1") or {}
    if isinstance(math_tag, dict):
        return bool(math_tag.get("math"))
    return False


def is_generic_candidate(row: dict[str, Any]) -> bool:
    """Pass-through filter: exclude rows the dataset already labels as coding or math."""
    return not is_labelled_coding(row) and not is_labelled_math(row)


# ──────────────────────────────────────────────────────────────────────────────
# LLM judge
# ──────────────────────────────────────────────────────────────────────────────

def build_judge_client(args: argparse.Namespace) -> OpenAI:
    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise SystemExit(
            f"{args.judge_api_key_env} is required. "
            "Set it in the environment or in a .env file."
        )
    base_url = (
        args.judge_base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)


def call_judge(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    user_msg = _JUDGE_USER_TMPL.format(prompt=prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        lower = raw.lower()
        parsed = {
            "is_generic": ('"is_generic": true' in lower),
            "confidence": 0.0,
            "category": "parse_fallback",
            "reason": raw[:300],
        }
    return {
        "is_generic": bool(parsed.get("is_generic", False)),
        "confidence": float(parsed.get("confidence") or 0.0),
        "category": str(parsed.get("category", "")),
        "reason": str(parsed.get("reason", "")),
        "judge_raw": raw,
    }


def judge_row(
    client: OpenAI,
    judge_model: str,
    row_id: str,
    prompt: str,
    min_confidence: float,
) -> dict[str, Any]:
    if not prompt:
        return {
            "id": row_id, "prompt": "", "status": "skipped_no_prompt",
            "is_generic": False, "confidence": 0.0,
            "category": "", "reason": "", "judge_raw": "",
            "judge_model": judge_model, "selected": False,
        }
    try:
        result = call_judge(client, judge_model, prompt)
        selected = result["is_generic"] and result["confidence"] >= min_confidence
        return {
            "id": row_id, "prompt": prompt, "status": "ok",
            **result,
            "judge_model": judge_model, "selected": selected,
        }
    except Exception as exc:
        return {
            "id": row_id, "prompt": prompt, "status": "judge_error",
            "is_generic": False, "confidence": 0.0,
            "category": "judge_error", "reason": str(exc), "judge_raw": "",
            "judge_model": judge_model, "selected": False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Output record builder
# ──────────────────────────────────────────────────────────────────────────────

def build_output_record(
    raw_row: dict[str, Any],
    judged: dict[str, Any],
    prompt: str,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    return {
        "id": str(raw_row.get("id", "")).strip(),
        "dataset": dataset,
        "split": split,
        "model_a": raw_row.get("model_a"),
        "model_b": raw_row.get("model_b"),
        "winner": raw_row.get("winner"),
        "language": raw_row.get("language"),
        "is_code": raw_row.get("is_code"),
        "category_tag": raw_row.get("category_tag"),
        "prompt": prompt,
        "conversation_a": raw_row.get("conversation_a"),
        "conversation_b": raw_row.get("conversation_b"),
        "judge_model": judged["judge_model"],
        "is_generic": judged["is_generic"],
        "confidence": judged["confidence"],
        "category": judged["category"],
        "reason": judged["reason"],
        "judge_raw": judged["judge_raw"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())

    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("--min-confidence must be within [0.0, 1.0].")

    output_path = Path(args.output)
    judged_path = Path(args.judged_log) if args.judged_log else \
        output_path.with_name(f"{output_path.stem}_judged.jsonl")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        output_path.with_name(f"{output_path.stem}_checkpoint.json")

    if not args.resume:
        for p in (output_path, judged_path, ckpt_path):
            if p.exists():
                p.unlink()

    client = build_judge_client(args)

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading {args.dataset} ({args.split})…")
    ds = load_dataset(args.dataset, split=args.split)
    raw_rows = [dict(r) for r in ds]
    print(f"  Loaded {len(raw_rows):,} rows.")

    allowed = {v.strip().lower() for v in args.allowed_languages}
    raw_rows = [r for r in raw_rows if str(r.get("language") or "").strip().lower() in allowed]
    print(f"  After language filter {sorted(allowed)}: {len(raw_rows):,} rows remain.")

    # ── Domain-label pre-filter ───────────────────────────────────────────
    candidates_all = []
    skipped_coding = skipped_math = skipped_prompt = 0
    for row in raw_rows:
        if is_labelled_coding(row):
            skipped_coding += 1
            continue
        if is_labelled_math(row):
            skipped_math += 1
            continue
        prompt = extract_user_prompt(row)
        if len(prompt) < args.min_prompt_len:
            skipped_prompt += 1
            continue
        if len(prompt.split()) < args.min_prompt_words:
            skipped_prompt += 1
            continue
        candidates_all.append({
            "id": str(row.get("id", "")).strip(),
            "prompt": prompt,
            "raw_row": row,
        })

    print(
        f"  After domain-label pre-filter: {len(candidates_all):,} candidates "
        f"(skipped coding={skipped_coding}, math={skipped_math}, "
        f"short/sparse_prompt={skipped_prompt} "
        f"[min_len={args.min_prompt_len}, min_words={args.min_prompt_words}])."
    )

    # Shuffle for diversity before judging
    import random
    rng = random.Random(args.seed)
    rng.shuffle(candidates_all)

    if args.max_candidates is not None:
        candidates_all = candidates_all[: args.max_candidates]
        print(f"  Capped to {len(candidates_all):,} candidates via --max-candidates.")

    # ── Resume state ──────────────────────────────────────────────────────
    ckpt = load_checkpoint(ckpt_path) if args.resume else {}
    done_ids: set[str] = {str(v) for v in ckpt.get("done_ids", [])}
    judged_count = int(ckpt.get("judged_count", len(done_ids)))
    accepted_count = int(ckpt.get("accepted_count", 0))
    error_count = int(ckpt.get("error_count", 0))

    pending = [c for c in candidates_all if c["id"] not in done_ids]
    print(f"  Pending: {len(pending):,} (already done: {len(done_ids):,}).")

    # ── Parallel judging ──────────────────────────────────────────────────
    write_buffer: list[dict[str, Any]] = []
    processed_since_flush = 0

    def flush() -> None:
        nonlocal processed_since_flush
        append_jsonl(judged_path, write_buffer)
        write_buffer.clear()
        save_checkpoint(ckpt_path, {
            "done_ids": sorted(done_ids),
            "judged_count": judged_count,
            "accepted_count": accepted_count,
            "error_count": error_count,
        })
        processed_since_flush = 0

    pbar = tqdm(
        total=len(candidates_all),
        initial=judged_count,
        desc="Judging generic candidates",
        unit="row",
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
        pending_iter = iter(pending)

        def submit_next() -> bool:
            try:
                cand = next(pending_iter)
            except StopIteration:
                return False
            fut = executor.submit(
                judge_row,
                client,
                args.judge_model,
                cand["id"],
                cand["prompt"],
                args.min_confidence,
            )
            future_map[fut] = cand
            return True

        for _ in range(min(args.max_workers, len(pending))):
            submit_next()

        while future_map:
            done, _ = concurrent.futures.wait(
                future_map, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                cand = future_map.pop(fut)
                result = fut.result()
                status = result["status"]
                processed_since_flush += 1

                if status in _DONE_STATUSES:
                    done_ids.add(cand["id"])
                    judged_count += 1
                    pbar.update(1)
                elif status == "judge_error":
                    error_count += 1

                if result.get("selected"):
                    rec = build_output_record(
                        cand["raw_row"], result, cand["prompt"],
                        args.dataset, args.split,
                    )
                    write_buffer.append(rec)
                    accepted_count += 1

                pbar.set_postfix_str(
                    f"judged={judged_count} accepted={accepted_count} "
                    f"errors={error_count}",
                    refresh=False,
                )

                if processed_since_flush >= args.save_every:
                    flush()

                submit_next()

    pbar.close()
    flush()

    # ── Filter by confidence ≥ 0.9, random-sample up to max_kept ────────────
    all_accepted = load_jsonl(judged_path)

    # Drop trivial categories that survive the judge but add no benchmark value
    _TRIVIAL_CATEGORIES = {
        "casual conversation", "conversational", "conversational greeting",
        "greeting", "casual greeting", "casual question", "general inquiry",
        "casual response", "casual opinion",
    }
    before_trivial_filter = len(all_accepted)
    all_accepted = [
        r for r in all_accepted
        if r.get("category", "").strip().lower() not in _TRIVIAL_CATEGORIES
    ]
    if before_trivial_filter != len(all_accepted):
        print(
            f"  Removed {before_trivial_filter - len(all_accepted):,} rows "
            "with trivial categories."
        )

    # Keep only rows with confidence >= 0.9, then random-sample up to max_kept
    high_conf = [r for r in all_accepted if r.get("confidence", 0.0) >= 0.9]
    print(f"  Rows with confidence ≥ 0.9: {len(high_conf):,}")

    if len(high_conf) <= args.max_kept:
        selected = high_conf
        if len(selected) < args.max_kept:
            shortfall = args.max_kept - len(selected)
            print(
                f"WARNING [generic]: requested {args.max_kept}, "
                f"available {len(selected)} — {shortfall} short. "
                "Lower --min-confidence or increase --max-candidates.",
                file=sys.stderr,
            )
    else:
        rng.shuffle(high_conf)
        selected = high_conf[: args.max_kept]
        print(f"  Randomly sampled {len(selected):,} from {len(high_conf):,} high-confidence rows.")

    # ── Write final output ────────────────────────────────────────────────
    write_jsonl(output_path, selected)

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "allowed_languages": sorted(allowed),
        "judge_model": args.judge_model,
        "min_confidence": args.min_confidence,
        "seed": args.seed,
        "domain_label_pre_filter": {
            "excluded_coding": skipped_coding,
            "excluded_math": skipped_math,
            "excluded_short_prompt": skipped_prompt,
            "candidates_after_filter": len(candidates_all),
        },
        "judged": judged_count,
        "accepted": len(all_accepted),
        "selected": len(selected),
        "target": args.max_kept,
        "judge_errors": error_count,
        "output": str(output_path),
        "judged_log": str(judged_path),
        "checkpoint": str(ckpt_path),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. judged={judged_count} accepted={len(all_accepted)} "
          f"selected={len(selected)} errors={error_count}")
    print(f"Output:   {output_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
