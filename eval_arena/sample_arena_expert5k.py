#!/usr/bin/env python3
"""
Sample coding, math, and generic questions from Arena Expert 5k.

Two-phase process per category:
  1. Pre-filter by occupational_tags domain labels (no LLM calls).
  2. LLM judge (gpt-4.1-mini) to confirm domain and assign confidence.
  3. Select top-N candidates sorted by judge confidence (highest first).

Special handling for coding:
  - Existing questions from a prior evaluation run are preserved as-is
    (--coding-existing-responses, default: results/arena_eval/coding_v0/responses.jsonl).
  - If a pre-judged coding JSONL is available (--coding-judged, default:
    data/arena_expert_5k_coding.jsonl), it is reused to avoid re-calling the LLM.
  - Only the remaining quota is filled with new high-confidence candidates.

For math and generic, judging is always run from scratch (with --resume support).

Domain label pre-filters (occupational_tags):
  coding   →  software_and_it_services == True
  math     →  mathematical == True  (expandable via --math-extra-tags)
  generic  →  neither software_and_it_services nor mathematical is True

Dataset:
  https://huggingface.co/datasets/lmarena-ai/arena-expert-5k

Examples:
  # Quick start with defaults:
  python eval_arena/sample_arena_expert5k.py

  # Custom output paths:
  python eval_arena/sample_arena_expert5k.py \\
    --coding-output data/coding_500.jsonl \\
    --math-output   data/math_300.jsonl \\
    --generic-output data/generic_1000.jsonl

  # Resume a partial run:
  python eval_arena/sample_arena_expert5k.py --resume

  # Expand math candidate pool with extra domain tags:
  python eval_arena/sample_arena_expert5k.py \\
    --math-extra-tags life_and_physical_and_social_science engineering_and_architecture
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

_CODING_SYSTEM = (
    "You are a strict classifier. "
    "Your only job is to decide whether a complete, correct answer to the user "
    "prompt would require producing actual, runnable code. Be conservative: "
    "if prose alone suffices, the answer is false."
)
_CODING_USER_TMPL = """\
Decide whether a good answer to this prompt would need to contain actual code.

The key test: would a correct, complete answer require writing real, runnable
code (in any language — Python, JS, SQL, shell, regex, pseudocode does NOT
count, etc.)?

INCLUDE if the prompt asks to:
- Write, implement, or generate a function, class, script, or program.
- Debug, fix, or patch a specific piece of code.
- Refactor or optimise existing code.
- Add a feature to existing code.
- Translate code between languages.
- Write a query (SQL), shell command, regex, config file, or build script.
- Create a test or benchmark for code.
- Complete or fill in missing code in a snippet.

EXCLUDE even if the topic is technical:
- Conceptual explanations of algorithms, data structures, or CS theory where
  the answer is prose, not code (e.g. "explain how a B-tree works").
- High-level software design / architecture discussions with no coding output.
- Questions about which library/tool to choose.
- Debugging a problem described only in words with no code to fix.
- Career, workflow, or process advice for developers.
- Prompts whose best answer is a diagram, essay, or numbered list — not code.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_coding": true or false,
  "confidence": 0.0 to 1.0,
  "category": "short label (e.g. implementation, debugging, refactoring, SQL, shell, etc.)",
  "reason": "one sentence explaining why code is or is not required"
}}"""

_MATH_SYSTEM = (
    "You are a strict domain classifier. "
    "Decide whether the user prompt is primarily about mathematics."
)
_MATH_USER_TMPL = """\
Classify whether this user prompt is math-focused.

Definition of math-focused:
- The core task requires mathematical reasoning, calculation, proof, derivation,
  symbolic manipulation, quantitative word-problem solving, geometry, algebra,
  number theory, probability/statistics, calculus, optimization, or equation solving.
- Include competition math, applied math, and numerical analysis.
- Exclude pure coding questions unless mathematical reasoning is central.
- Exclude general science questions unless significant mathematical computation
  is required.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_math": true or false,
  "confidence": 0.0 to 1.0,
  "category": "short label",
  "reason": "short explanation"
}}"""

_GENERIC_SYSTEM = (
    "You are a strict domain classifier. "
    "Decide whether the user prompt is an everyday, style-driven task that "
    "requires no specialised domain knowledge."
)
_GENERIC_USER_TMPL = """\
Classify whether this user prompt is an everyday, style-driven task.

ACCEPT (is_generic: true) only if ALL of the following hold:
1. No specialised domain knowledge is needed — a thoughtful non-expert can
   give a fully competent answer without training in medicine, law, finance,
   science, engineering, history, or any other professional field.
2. No programming knowledge or code is required.
3. No mathematical calculation, proof, or quantitative reasoning is required.
4. Style, tone, clarity, or creativity matters at least as much as factual
   content — a better writer produces a noticeably better answer.

Typical examples that PASS:
- Write / edit / improve an email, message, cover letter, essay, or story.
- Give everyday advice (relationships, productivity, travel, food, etc.).
- Summarise or paraphrase a piece of text.
- Brainstorm names, slogans, gift ideas, or similar creative lists.
- Explain a concept the way you would to a friend (no expert depth required).
- Casual opinion or recommendation questions.
- Role-play or conversational tasks.

REJECT (is_generic: false) if any of the following is true:
- Answering well requires expert domain knowledge (medical diagnosis, legal
  analysis, financial modelling, scientific derivation, historical scholarship,
  engineering specs, etc.).
- The prompt is about software, code, or technical systems.
- The prompt requires arithmetic, statistics, or any mathematical reasoning.
- The question has a single correct factual answer that only an expert would
  know (trivia about niche fields, precise definitions, etc.).

When in doubt, reject — this category is for genuinely open, everyday tasks
where a good writer beats a domain expert.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_generic": true or false,
  "confidence": 0.0 to 1.0,
  "category": "short label (e.g. email writing, creative writing, advice, summarisation, brainstorming, etc.)",
  "reason": "one sentence"
}}"""

# occupational_tags key used as primary domain label per category
_CODING_TAG = "software_and_it_services"
_MATH_TAG = "mathematical"

_DONE_STATUSES = {"ok", "skipped_no_prompt", "skipped_language"}


def _warn_shortfall(category: str, available: int, requested: int, hint: str = "") -> None:
    """Emit a standardised shortfall warning to stderr."""
    shortfall = requested - available
    msg = (
        f"WARNING [{category}]: requested {requested}, "
        f"available {available} — {shortfall} short."
    )
    if hint:
        msg += f" {hint}"
    print(msg, file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample coding, math, and generic questions from Arena Expert 5k "
            "using domain labels + LLM judge filtering."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Output files ──
    out = p.add_argument_group("output files")
    out.add_argument(
        "--coding-output",
        default="data/arena_expert5k_coding_500.jsonl",
        metavar="PATH",
        help="Output JSONL for 500 coding questions.",
    )
    out.add_argument(
        "--math-output",
        default="data/arena_expert5k_math_300.jsonl",
        metavar="PATH",
        help="Output JSONL for 300 math questions.",
    )
    out.add_argument(
        "--generic-output",
        default="data/arena_expert5k_generic_1000.jsonl",
        metavar="PATH",
        help="Output JSONL for 1000 generic questions.",
    )

    # ── Target counts ──
    cnt = p.add_argument_group("target counts")
    cnt.add_argument("--coding-n", type=int, default=500, metavar="N",
                     help="Total coding questions to produce.")
    cnt.add_argument("--math-n", type=int, default=300, metavar="N",
                     help="Total math questions to produce.")
    cnt.add_argument("--generic-n", type=int, default=1000, metavar="N",
                     help=(
                         "Total generic questions to produce. Note: the "
                         "arena-expert-5k is heavily technical (~75%% of "
                         "English rows carry a coding or math domain label) "
                         "so the pre-filtered candidate pool is ~700-900 rows. "
                         "If you need more than that, lower --min-confidence."
                     ))

    # ── Existing coding data ──
    ex = p.add_argument_group("existing coding data (to preserve)")
    ex.add_argument(
        "--coding-existing-responses",
        default="results/arena_eval/coding_v0/responses.jsonl",
        metavar="PATH",
        help=(
            "Path to an existing eval responses.jsonl whose item_ids are kept "
            "as-is in the coding output."
        ),
    )
    ex.add_argument(
        "--coding-judged",
        default="data/arena_expert_5k_coding.jsonl",
        metavar="PATH",
        help=(
            "Pre-judged coding JSONL (from sample_arena_coding.py). "
            "If it exists the coding LLM judge is skipped."
        ),
    )

    # ── Dataset ──
    ds = p.add_argument_group("dataset")
    ds.add_argument("--dataset", default="lmarena-ai/arena-expert-5k",
                    help="HuggingFace dataset path.")
    ds.add_argument("--split", default="train", help="Dataset split.")
    ds.add_argument(
        "--allowed-languages", nargs="+", default=["en"],
        metavar="LANG",
        help="Language allowlist (ISO codes).",
    )

    # ── Judge ──
    jdg = p.add_argument_group("LLM judge")
    jdg.add_argument("--judge-model", default="gpt-4.1-mini",
                     help="OpenAI-compatible judge model.")
    jdg.add_argument("--judge-api-key-env", default="OPENAI_API_KEY",
                     metavar="ENV",
                     help="Env var holding the judge API key.")
    jdg.add_argument("--judge-base-url", default=None, metavar="URL",
                     help="Base URL for the judge endpoint.")
    jdg.add_argument("--min-confidence", type=float, default=0.7,
                     help="Minimum judge confidence to consider a candidate.")
    jdg.add_argument("--timeout", type=int, default=60,
                     help="Judge client timeout in seconds.")

    # ── Math extras ──
    p.add_argument(
        "--math-extra-tags", nargs="*", default=[],
        metavar="TAG",
        help=(
            "Additional occupational_tags (besides 'mathematical') to include "
            "as math pre-filter candidates, e.g. life_and_physical_and_social_science."
        ),
    )

    # ── Execution ──
    exe = p.add_argument_group("execution")
    exe.add_argument("--max-workers", type=int, default=8,
                     help="Parallel judge threads.")
    exe.add_argument("--save-every", type=int, default=100,
                     help="Flush checkpoint every N judged rows.")
    exe.add_argument("--seed", type=int, default=42,
                     help="RNG seed (unused – rows are sorted by confidence).")
    exe.add_argument("--resume", action="store_true",
                     help="Resume from existing checkpoint files.")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
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
# Occupational-tag helpers
# ──────────────────────────────────────────────────────────────────────────────

def _occ_tags(row: dict[str, Any]) -> dict[str, Any]:
    tags = row.get("occupational_tags") or {}
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = {}
    return tags if isinstance(tags, dict) else {}


def has_tag(row: dict[str, Any], tag: str) -> bool:
    return bool(_occ_tags(row).get(tag))


def is_coding_candidate(row: dict[str, Any]) -> bool:
    return has_tag(row, _CODING_TAG)


def is_math_candidate(row: dict[str, Any], extra_tags: list[str]) -> bool:
    tags = _occ_tags(row)
    if tags.get(_MATH_TAG):
        return True
    return any(tags.get(t) for t in extra_tags)


def is_generic_candidate(row: dict[str, Any]) -> bool:
    tags = _occ_tags(row)
    return not (tags.get(_CODING_TAG) or tags.get(_MATH_TAG))


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


def _call_judge(
    client: OpenAI,
    model: str,
    system: str,
    user_tmpl: str,
    prompt: str,
    domain_key: str,
) -> dict[str, Any]:
    """Call the LLM judge and return a normalised dict with domain_key + confidence."""
    user_msg = user_tmpl.format(prompt=prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
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
            domain_key: (f'"{domain_key}": true' in lower),
            "confidence": 0.0,
            "category": "parse_fallback",
            "reason": raw[:300],
        }
    return {
        domain_key: bool(parsed.get(domain_key, False)),
        "confidence": float(parsed.get("confidence") or 0.0),
        "category": str(parsed.get("category", "")),
        "reason": str(parsed.get("reason", "")),
        "judge_raw": raw,
    }


def judge_one(
    client: OpenAI,
    judge_model: str,
    row_id: str,
    prompt: str,
    category: str,          # "coding" | "math" | "generic"
) -> dict[str, Any]:
    """Run the appropriate judge for *category* and return a result dict."""
    if category == "coding":
        system, tmpl, domain_key = _CODING_SYSTEM, _CODING_USER_TMPL, "is_coding"
    elif category == "math":
        system, tmpl, domain_key = _MATH_SYSTEM, _MATH_USER_TMPL, "is_math"
    else:
        system, tmpl, domain_key = _GENERIC_SYSTEM, _GENERIC_USER_TMPL, "is_generic"

    try:
        result = _call_judge(client, judge_model, system, tmpl, prompt, domain_key)
        return {
            "id": row_id,
            "prompt": prompt,
            "status": "ok",
            **result,
            "judge_model": judge_model,
            "selected": bool(result[domain_key]),
        }
    except Exception as exc:
        return {
            "id": row_id,
            "prompt": prompt,
            "status": "judge_error",
            domain_key: False,
            "confidence": 0.0,
            "category": "judge_error",
            "reason": str(exc),
            "judge_raw": "",
            "judge_model": judge_model,
            "selected": False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core: judge a batch of candidates with resume support
# ──────────────────────────────────────────────────────────────────────────────

def run_judge_phase(
    *,
    candidates: list[dict[str, Any]],   # list of {"id", "prompt", "raw_row"}
    category: str,
    client: OpenAI,
    judge_model: str,
    max_workers: int,
    save_every: int,
    checkpoint_path: Path,
    judged_output_path: Path,
    resume: bool,
    min_confidence: float,
    desc: str,
) -> list[dict[str, Any]]:
    """
    Judge all candidates and write results incrementally.

    Returns the list of accepted records (domain flag True AND
    confidence >= min_confidence), ready to be sorted by confidence.
    """
    # Load resume state
    ckpt: dict[str, Any] = {}
    if resume:
        ckpt = load_checkpoint(checkpoint_path)

    done_ids: set[str] = {str(v) for v in ckpt.get("done_ids", [])}
    accepted_existing = load_jsonl(judged_output_path) if resume else []
    # Clear stale file if not resuming
    if not resume and judged_output_path.exists():
        judged_output_path.unlink()

    domain_key = {"coding": "is_coding", "math": "is_math", "generic": "is_generic"}[category]
    accepted_count = len(accepted_existing)
    judged_count = len(done_ids)
    error_count = int(ckpt.get("error_count", 0))

    pending = [c for c in candidates if c["id"] not in done_ids]

    write_buffer: list[dict[str, Any]] = []
    processed_since_flush = 0

    def flush() -> None:
        nonlocal processed_since_flush
        append_jsonl(judged_output_path, write_buffer)
        write_buffer.clear()
        save_checkpoint(
            checkpoint_path,
            {
                "done_ids": sorted(done_ids),
                "judged_count": judged_count,
                "accepted_count": accepted_count,
                "error_count": error_count,
            },
        )
        processed_since_flush = 0

    pbar = tqdm(
        total=len(candidates),
        initial=judged_count,
        desc=desc,
        unit="row",
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cand: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

        pending_iter = iter(pending)

        def submit_next() -> bool:
            try:
                cand = next(pending_iter)
            except StopIteration:
                return False
            fut = executor.submit(
                judge_one,
                client,
                judge_model,
                cand["id"],
                cand["prompt"],
                category,
            )
            future_to_cand[fut] = cand
            return True

        for _ in range(min(max_workers, len(pending))):
            submit_next()

        while future_to_cand:
            done, _ = concurrent.futures.wait(
                future_to_cand,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                cand = future_to_cand.pop(fut)
                result = fut.result()
                status = result["status"]
                processed_since_flush += 1

                if status in _DONE_STATUSES:
                    done_ids.add(cand["id"])
                    judged_count += 1
                    pbar.update(1)
                elif status == "judge_error":
                    error_count += 1

                if status == "ok" and result.get(domain_key) and result["confidence"] >= min_confidence:
                    rec = _build_output_record(cand["raw_row"], result, cand["prompt"])
                    write_buffer.append(rec)
                    accepted_count += 1

                pbar.set_postfix_str(
                    f"judged={judged_count} accepted={accepted_count} errors={error_count}",
                    refresh=False,
                )

                if processed_since_flush >= save_every:
                    flush()

                submit_next()

    pbar.close()
    flush()

    # Combine resumed existing + newly written
    all_accepted = load_jsonl(judged_output_path)
    print(
        f"  [{category}] judged={judged_count} accepted={len(all_accepted)} "
        f"errors={error_count}"
    )
    return all_accepted


def _build_output_record(
    raw_row: dict[str, Any],
    judged: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    """Build a standardised output record merging raw row metadata with judge output."""
    return {
        "id": str(raw_row.get("id", "")).strip(),
        "dataset": raw_row.get("dataset", "lmarena-ai/arena-expert-5k"),
        "split": raw_row.get("split", "train"),
        "model_a": raw_row.get("model_a"),
        "model_b": raw_row.get("model_b"),
        "winner": raw_row.get("winner"),
        "evaluation_order": raw_row.get("evaluation_order"),
        "language": raw_row.get("language"),
        "occupational_tags": raw_row.get("occupational_tags"),
        "prompt": prompt,
        "conversation_a": raw_row.get("conversation_a"),
        "conversation_b": raw_row.get("conversation_b"),
        "judge_model": judged.get("judge_model", ""),
        # domain flags – only the relevant one will be True
        "is_coding": bool(judged.get("is_coding", False)),
        "is_math": bool(judged.get("is_math", False)),
        "is_generic": bool(judged.get("is_generic", False)),
        "confidence": judged.get("confidence", 0.0),
        "category": judged.get("category", ""),
        "reason": judged.get("reason", ""),
        "judge_raw": judged.get("judge_raw", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase helpers
# ──────────────────────────────────────────────────────────────────────────────

def _row_id(raw_row: dict[str, Any], row_index: int) -> str:
    v = str(raw_row.get("id", "")).strip()
    return v if v else f"arena_expert_{row_index:06d}"


def load_existing_coding_ids(responses_path: Path) -> set[str]:
    """Return the set of item_ids already present in an eval responses.jsonl."""
    ids: set[str] = set()
    rows = load_jsonl(responses_path)
    for r in rows:
        rid = str(r.get("item_id") or r.get("id") or "").strip()
        if rid:
            ids.add(rid)
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 – Coding
# ──────────────────────────────────────────────────────────────────────────────

def build_coding_output(
    *,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    client: OpenAI,
) -> list[dict[str, Any]]:
    """
    Return exactly --coding-n records for the coding output file.

    Reuses data/arena_expert_5k_coding.jsonl if available; otherwise falls
    back to judging software_and_it_services-tagged rows from scratch.
    """
    target_n = args.coding_n
    existing_responses_path = Path(args.coding_existing_responses)
    judged_path = Path(args.coding_judged)
    output_path = Path(args.coding_output)

    # ── Load IDs to preserve ──────────────────────────────────────────────
    existing_ids = load_existing_coding_ids(existing_responses_path)
    if existing_ids:
        print(f"  [coding] Found {len(existing_ids)} existing questions to preserve "
              f"from {existing_responses_path}.")
    else:
        print(f"  [coding] No existing responses found at {existing_responses_path}.")

    # ── Reuse pre-judged coding JSONL if available ────────────────────────
    if judged_path.exists():
        print(f"  [coding] Reusing pre-judged coding data from {judged_path}.")
        all_judged = load_jsonl(judged_path)
        # Build id → record map; ensure all have required fields
        judged_map: dict[str, dict[str, Any]] = {}
        for rec in all_judged:
            rid = str(rec.get("id", "")).strip()
            if rid:
                # Normalise: add is_coding/is_math/is_generic if absent
                rec.setdefault("is_coding", True)
                rec.setdefault("is_math", False)
                rec.setdefault("is_generic", False)
                judged_map[rid] = rec
    else:
        print(
            f"  [coding] Pre-judged file not found at {judged_path}. "
            "Running LLM judge on software_and_it_services-tagged rows."
        )
        raw_map = {_row_id(r, i): r for i, r in enumerate(raw_rows)}
        coding_candidates = [
            {"id": rid, "prompt": extract_user_prompt(r), "raw_row": r}
            for rid, r in raw_map.items()
            if is_coding_candidate(r)
        ]
        coding_candidates = [c for c in coding_candidates if c["prompt"]]
        ckpt_path = output_path.with_name(f"{output_path.stem}_coding_judge_checkpoint.json")
        judged_path_tmp = output_path.with_name(f"{output_path.stem}_coding_judged.jsonl")
        accepted = run_judge_phase(
            candidates=coding_candidates,
            category="coding",
            client=client,
            judge_model=args.judge_model,
            max_workers=args.max_workers,
            save_every=args.save_every,
            checkpoint_path=ckpt_path,
            judged_output_path=judged_path_tmp,
            resume=args.resume,
            min_confidence=args.min_confidence,
            desc="Judging coding candidates",
        )
        judged_map = {str(r.get("id", "")).strip(): r for r in accepted if r.get("id")}

    # ── Split into existing and new pools ────────────────────────────────
    existing_records: list[dict[str, Any]] = []
    new_pool: list[dict[str, Any]] = []

    for rid, rec in judged_map.items():
        if rid in existing_ids:
            existing_records.append(rec)
        else:
            new_pool.append(rec)

    if len(existing_records) < len(existing_ids):
        missing = existing_ids - set(judged_map.keys())
        print(
            f"  [coding] Warning: {len(missing)} existing IDs not found in the "
            f"judged pool; they will be omitted."
        )

    still_needed = target_n - len(existing_records)
    if still_needed < 0:
        still_needed = 0

    # ── Select top-K new questions by confidence ─────────────────────────
    new_pool.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    selected_new = new_pool[:still_needed]

    if len(selected_new) < still_needed:
        _warn_shortfall("coding", len(selected_new), still_needed,
                        "Expand the pre-judged coding pool or lower --min-confidence.")

    final = existing_records + selected_new
    print(
        f"  [coding] Final: {len(existing_records)} preserved + "
        f"{len(selected_new)} new = {len(final)} total."
    )
    return final


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 – Math
# ──────────────────────────────────────────────────────────────────────────────

def build_math_output(
    *,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    client: OpenAI,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    target_n = args.math_n
    output_path = Path(args.math_output)
    extra_tags: list[str] = args.math_extra_tags or []

    # Pre-filter by domain labels; exclude IDs already committed to coding
    math_candidates = [
        {
            "id": _row_id(r, i),
            "prompt": extract_user_prompt(r),
            "raw_row": r,
        }
        for i, r in enumerate(raw_rows)
        if is_math_candidate(r, extra_tags)
        and _row_id(r, i) not in exclude_ids
    ]
    math_candidates = [c for c in math_candidates if c["prompt"]]
    print(
        f"  [math] {len(math_candidates)} candidates after domain-label pre-filter "
        f"(tag: {_MATH_TAG}" + (f" + {extra_tags}" if extra_tags else "") + ")."
    )

    if not math_candidates:
        print("  [math] No candidates found. Skipping math phase.")
        return []

    ckpt_path = output_path.with_name(f"{output_path.stem}_checkpoint.json")
    judged_path = output_path.with_name(f"{output_path.stem}_judged.jsonl")

    accepted = run_judge_phase(
        candidates=math_candidates,
        category="math",
        client=client,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
        save_every=args.save_every,
        checkpoint_path=ckpt_path,
        judged_output_path=judged_path,
        resume=args.resume,
        min_confidence=args.min_confidence,
        desc="Judging math candidates",
    )

    accepted.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    selected = accepted[:target_n]

    if len(selected) < target_n:
        _warn_shortfall("math", len(selected), target_n,
                        "Use --math-extra-tags to expand the candidate pool, "
                        "or lower --min-confidence.")
    else:
        print(f"  [math] Selected top {len(selected)} by confidence.")

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 – Generic
# ──────────────────────────────────────────────────────────────────────────────

def build_generic_output(
    *,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    client: OpenAI,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    target_n = args.generic_n
    output_path = Path(args.generic_output)

    # Pre-filter: exclude coding-tagged rows AND rows already selected
    generic_candidates = [
        {
            "id": _row_id(r, i),
            "prompt": extract_user_prompt(r),
            "raw_row": r,
        }
        for i, r in enumerate(raw_rows)
        if is_generic_candidate(r)
        and _row_id(r, i) not in exclude_ids
    ]
    generic_candidates = [c for c in generic_candidates if c["prompt"]]
    print(
        f"  [generic] {len(generic_candidates)} candidates after domain-label "
        f"pre-filter (excluding {_CODING_TAG} + {_MATH_TAG} tags)."
    )

    if not generic_candidates:
        print("  [generic] No candidates found. Skipping generic phase.")
        return []

    ckpt_path = output_path.with_name(f"{output_path.stem}_checkpoint.json")
    judged_path = output_path.with_name(f"{output_path.stem}_judged.jsonl")

    accepted = run_judge_phase(
        candidates=generic_candidates,
        category="generic",
        client=client,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
        save_every=args.save_every,
        checkpoint_path=ckpt_path,
        judged_output_path=judged_path,
        resume=args.resume,
        min_confidence=args.min_confidence,
        desc="Judging generic candidates",
    )

    accepted.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    selected = accepted[:target_n]

    if len(selected) < target_n:
        _warn_shortfall("generic", len(selected), target_n,
                        "Lower --min-confidence or reduce --generic-n.")
    else:
        print(f"  [generic] Selected top {len(selected)} by confidence.")

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())

    # Validation
    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("--min-confidence must be within [0.0, 1.0].")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1.")

    allowed = {v.strip().lower() for v in args.allowed_languages}

    # ── Build judge client ────────────────────────────────────────────────
    client = build_judge_client(args)

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading {args.dataset} ({args.split})…")
    ds = load_dataset(args.dataset, split=args.split)
    raw_rows = [dict(r) for r in ds]
    print(f"  Loaded {len(raw_rows):,} rows.")

    # Language filter
    raw_rows = [
        r for r in raw_rows
        if str(r.get("language") or "").strip().lower() in allowed
    ]
    print(f"  After language filter {sorted(allowed)}: {len(raw_rows):,} rows remain.")

    # Attach dataset/split metadata so _build_output_record can pick it up
    for r in raw_rows:
        r.setdefault("dataset", args.dataset)
        r.setdefault("split", args.split)

    # ── Phase 1: Coding ───────────────────────────────────────────────────
    print("\n=== Phase 1: Coding ===")
    coding_records = build_coding_output(
        args=args,
        raw_rows=raw_rows,
        client=client,
    )
    coding_ids = {str(r.get("id", "")).strip() for r in coding_records}

    # ── Phase 2: Math ─────────────────────────────────────────────────────
    print("\n=== Phase 2: Math ===")
    math_records = build_math_output(
        args=args,
        raw_rows=raw_rows,
        client=client,
        exclude_ids=coding_ids,
    )
    math_ids = {str(r.get("id", "")).strip() for r in math_records}

    # ── Phase 3: Generic ──────────────────────────────────────────────────
    print("\n=== Phase 3: Generic ===")
    generic_records = build_generic_output(
        args=args,
        raw_rows=raw_rows,
        client=client,
        exclude_ids=coding_ids | math_ids,
    )

    # ── Write outputs ─────────────────────────────────────────────────────
    print("\n=== Writing outputs ===")
    for path_str, records, label in [
        (args.coding_output, coding_records, "coding"),
        (args.math_output, math_records, "math"),
        (args.generic_output, generic_records, "generic"),
    ]:
        out_path = Path(path_str)
        write_jsonl(out_path, records)
        print(f"  {label:8s}  {len(records):5d} records  →  {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "allowed_languages": sorted(allowed),
        "judge_model": args.judge_model,
        "min_confidence": args.min_confidence,
        "max_workers": args.max_workers,
        "coding": {
            "output": args.coding_output,
            "count": len(coding_records),
            "target": args.coding_n,
        },
        "math": {
            "output": args.math_output,
            "count": len(math_records),
            "target": args.math_n,
            "extra_tags": args.math_extra_tags,
        },
        "generic": {
            "output": args.generic_output,
            "count": len(generic_records),
            "target": args.generic_n,
        },
    }
    summary_path = Path(args.coding_output).parent / "arena_expert5k_sample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
