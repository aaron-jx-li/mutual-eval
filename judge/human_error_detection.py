#!/usr/bin/env python3
"""
Experiments 4–5: Human Error Detection Sensitivity and Style Confounds.

Exp 4 — Error Severity vs Human Correctness Detection
    For each case where single_strong_judge detected an error, calls an LLM to
    rate error severity as minor (1), moderate (2), or critical (3). Then tests
    whether humans are more likely to avoid the error model when errors are severe.
    If human preference tracked correctness at all, severe errors should be
    penalised more. If the alignment rate is flat across severity levels, humans
    miss even critical errors.

Exp 5 — Style Confounds in Human Preference
    Identifies "Type-1 divergence" cases: human prefers model X, but the LLM judge
    (ssj) found an error in X and NOT in the other model. Extracts surface-level
    style signals from the raw responses (word count, bullet/header density,
    confidence language, numerical specificity) and compares the error model vs
    the correct model to reveal what drove the human's erroneous preference.

Usage:
    # Run both experiments (Exp 4 calls the LLM; Exp 5 is offline)
    python judge/human_error_detection.py \\
        --arena-data data/arena_140k_math_openai_single.json \\
        --ssj        results/single_strong_judge_140k_math_openai_single.json \\
        --output     results/error_severity_ratings.json

    # Skip LLM calls, only run Exp 5 style analysis
    python judge/human_error_detection.py \\
        --arena-data data/arena_140k_math_openai_single.json \\
        --ssj        results/single_strong_judge_140k_math_openai_single.json \\
        --output     results/error_severity_ratings.json \\
        --experiments 5

    # Re-use existing severity ratings (no API calls)
    python judge/human_error_detection.py \\
        --arena-data data/arena_140k_math_openai_single.json \\
        --ssj        results/single_strong_judge_140k_math_openai_single.json \\
        --output     results/error_severity_ratings.json \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from statistics import mean, stdev

from openai import OpenAI

# ── Constants ─────────────────────────────────────────────────────────

SEVERITY_SYSTEM = (
    "You are a mathematics error classifier. "
    "You read a description of a mathematical error and rate its severity."
)

SEVERITY_PROMPT = """\
A math response was found to contain an error. Here is the description of that error:

{explanation}

Rate the severity of this error on a 1–3 scale:
- 1 (minor): Small arithmetic slip, imprecise wording, notation inconsistency, or a \
peripheral claim that doesn't affect the core answer.
- 2 (moderate): A significant conceptual gap, wrong intermediate step, or incorrect \
sub-result that affects the reasoning but the final answer might still be coincidentally correct.
- 3 (critical): Fundamental flaw in reasoning, wrong formula/theorem, or incorrect \
final answer that makes the response misleading or harmful.

Respond with ONLY a JSON object (no markdown):
{{"severity": 1 or 2 or 3, "reason": "one-sentence justification"}}"""

CONFIDENCE_PATTERNS = [
    r"\bclearly\b", r"\bobviously\b", r"\btherefore\b", r"\bthus\b",
    r"\bhence\b", r"\bstraightforwardly\b", r"\bsimply\b", r"\beasily\b",
    r"\bof course\b", r"\bnote that\b", r"\bit follows\b", r"\bwe see that\b",
    r"\brecall that\b",
]

CONFIDENCE_RE = re.compile("|".join(CONFIDENCE_PATTERNS), re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def extract_response_text(conversation: list[dict]) -> str:
    """Concatenate all assistant turns from a conversation."""
    parts = []
    for msg in conversation:
        if msg["role"] == "assistant":
            parts.append(extract_text(msg["content"]))
    return "\n\n".join(parts)


def style_features(text: str) -> dict:
    """Extract surface-level style signals from response text."""
    words = text.split()
    lines = text.splitlines()
    return {
        "word_count":        len(words),
        "char_count":        len(text),
        "bullet_count":      sum(1 for l in lines if re.match(r"\s*[-*•]", l)),
        "header_count":      sum(1 for l in lines if re.match(r"\s*#{1,4}\s", l)),
        "number_count":      len(re.findall(r"\b\d+(?:\.\d+)?\b", text)),
        "latex_count":       text.count("$") // 2,
        "confidence_count":  len(CONFIDENCE_RE.findall(text)),
        "sentence_count":    len(re.split(r"[.!?]+", text)),
        "avg_word_len":      mean(len(w) for w in words) if words else 0.0,
    }


# ── Severity Rating (LLM) ─────────────────────────────────────────────

def call_severity_rater(
    client: OpenAI,
    model: str,
    explanation: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    prompt = SEVERITY_PROMPT.format(explanation=explanation)
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=SEVERITY_SYSTEM,
                input=prompt,
                max_output_tokens=256,
                store=False,
            )
            return resp.output_text.strip()
        except Exception as e:
            print(f"  [Attempt {attempt + 1}/{max_retries}] {model} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    return '{"severity": 2, "reason": "API_ERROR"}'


def parse_severity(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        sev = int(parsed.get("severity", 2))
        sev = max(1, min(3, sev))
        return {"severity": sev, "reason": str(parsed.get("reason", ""))}
    except (json.JSONDecodeError, ValueError):
        pass
    for digit in ["3", "2", "1"]:
        if digit in text:
            return {"severity": int(digit), "reason": f"PARSE_FALLBACK: {text[:200]}"}
    return {"severity": 2, "reason": f"PARSE_FALLBACK: {text[:200]}"}


# ── Experiment 4: Error Severity vs Human Detection ───────────────────

def exp4_severity_analysis(
    ssj_records: dict[str, dict],
    severity_data: dict[str, dict],
) -> None:
    print(f"\n{'='*70}")
    print("EXPERIMENT 4 — Error Severity vs Human Correctness Detection")
    print(f"{'='*70}")
    print(
        "For each asymmetric-error case (ssj finds error in exactly one model),\n"
        "error severity is rated 1–3 by an LLM rater. The correctness alignment\n"
        "rate (fraction of cases where human correctly avoids the error model)\n"
        "is computed per severity tier.\n"
        "If humans track correctness, alignment should increase with severity.\n"
    )

    severity_labels = {1: "minor", 2: "moderate", 3: "critical"}

    # Collect cases: (id, which_model_has_error, severity, human_label)
    cases = []
    for id_, rec in ssj_records.items():
        ea = rec["eval_a"]["error_found"]
        eb = rec["eval_b"]["error_found"]
        hl = rec["human_label"]
        # Only asymmetric-error cases
        if ea == eb:
            continue
        if id_ not in severity_data:
            continue
        if ea:
            error_model = "a"
            correct_label = "model_b"
            severity = severity_data[id_]["eval_a"]["severity"]
        else:
            error_model = "b"
            correct_label = "model_a"
            severity = severity_data[id_]["eval_b"]["severity"]
        cases.append({
            "id": id_,
            "error_model": error_model,
            "correct_label": correct_label,
            "severity": severity,
            "human_label": hl,
        })

    print(f"Asymmetric-error cases with severity ratings: {len(cases)}\n")

    if not cases:
        print("No rated cases found. Run without --resume to generate severity ratings.")
        return

    # Summary table per severity level
    print(f"{'Severity':>12}  {'N':>4}  {'Correct':>8}  {'Wrong':>8}  {'Neutral':>8}  {'Align%':>8}")
    print("-" * 58)
    combined_correct = combined_wrong = combined_neutral = 0
    for sev in [1, 2, 3]:
        group = [c for c in cases if c["severity"] == sev]
        correct = sum(1 for c in group if c["human_label"] == c["correct_label"])
        neutral = sum(1 for c in group if c["human_label"] in ("tie", "both_bad"))
        wrong = len(group) - correct - neutral
        combined_correct += correct
        combined_wrong += wrong
        combined_neutral += neutral
        pct = correct / len(group) * 100 if group else 0
        label = f"{sev} ({severity_labels[sev]})"
        print(f"{label:>12}  {len(group):>4}  {correct:>8}  {wrong:>8}  {neutral:>8}  {pct:>7.1f}%")
    n = len(cases)
    overall_pct = combined_correct / n * 100 if n else 0
    print("-" * 58)
    print(f"{'Overall':>12}  {n:>4}  {combined_correct:>8}  {combined_wrong:>8}  {combined_neutral:>8}  {overall_pct:>7.1f}%")

    # Show example cases for critical errors
    critical_cases = [c for c in cases if c["severity"] == 3]
    if critical_cases:
        print(f"\nCritical error cases (severity=3): {len(critical_cases)}")
        for c in critical_cases:
            sev_info = severity_data[c["id"]]
            error_key = f"eval_{c['error_model']}"
            reason = sev_info[error_key].get("reason", "")
            match = "✓" if c["human_label"] == c["correct_label"] else "✗"
            print(f"  {c['id'][:8]}: human={c['human_label']:>10} (expected {c['correct_label']:>7}) {match}")
            print(f"           reason: {reason[:80]}")


# ── Experiment 5: Style Confounds ─────────────────────────────────────

def exp5_style_analysis(
    ssj_records: dict[str, dict],
    arena_lookup: dict[str, dict],
) -> None:
    print(f"\n{'='*70}")
    print("EXPERIMENT 5 — Style Confounds in Human-Preferred Error Responses")
    print(f"{'='*70}")
    print(
        "Type-1 divergences: human prefers model X, but ssj found error only in X.\n"
        "This is the clearest case where style overrode correctness.\n"
        "We compare surface style features of the error model (human-preferred)\n"
        "vs the correct model (human-ignored) to reveal the confound.\n"
    )

    FEATURE_NAMES = [
        "word_count", "bullet_count", "header_count",
        "number_count", "latex_count", "confidence_count",
    ]
    FEATURE_LABELS = {
        "word_count":       "Word count",
        "bullet_count":     "Bullet points",
        "header_count":     "Headers (##)",
        "number_count":     "Numbers/values",
        "latex_count":      "LaTeX expressions ($)",
        "confidence_count": "Confidence markers",
    }

    type1_cases = []       # human prefers error model
    type2_cases = []       # human avoids error model (or neutral)

    for id_, rec in ssj_records.items():
        ea = rec["eval_a"]["error_found"]
        eb = rec["eval_b"]["error_found"]
        hl = rec["human_label"]
        if ea == eb:                  # not asymmetric
            continue
        if id_ not in arena_lookup:
            continue
        entry = arena_lookup[id_]

        # Which model has the error?
        if ea:                        # A has error
            error_model = "a"
            correct_model = "b"
            correct_label = "model_b"
        else:                         # B has error
            error_model = "b"
            correct_model = "a"
            correct_label = "model_a"

        error_conv_key = f"conversation_{error_model}"
        correct_conv_key = f"conversation_{correct_model}"
        error_text = extract_response_text(entry[error_conv_key])
        correct_text = extract_response_text(entry[correct_conv_key])

        record = {
            "id": id_,
            "human_label": hl,
            "error_model": error_model,
            "correct_model": correct_model,
            "error_features": style_features(error_text),
            "correct_features": style_features(correct_text),
            "error_explanation": rec[f"eval_{error_model}"]["explanation"],
        }

        if hl == f"model_{error_model}":   # human preferred the error model
            type1_cases.append(record)
        else:
            type2_cases.append(record)

    print(f"Type-1 cases (human prefers error model):   {len(type1_cases)}")
    print(f"Other asymmetric cases (human avoids/neutral): {len(type2_cases)}\n")

    if not type1_cases:
        print("No Type-1 cases found.")
        return

    def avg_feat(cases: list[dict], key1: str, key2: str) -> tuple[float, float]:
        vals = [c[key1][key2] for c in cases if key2 in c[key1]]
        return (mean(vals), stdev(vals) if len(vals) > 1 else 0.0) if vals else (0.0, 0.0)

    print("Style features: error model (human-preferred) vs correct model (human-ignored)")
    print("(within Type-1 cases only)\n")
    print(f"{'Feature':>25}  {'Error Model':>12}  {'Correct Model':>14}  {'Ratio':>7}")
    print("-" * 65)
    for feat in FEATURE_NAMES:
        err_mean, err_std = avg_feat(type1_cases, "error_features", feat)
        cor_mean, cor_std = avg_feat(type1_cases, "correct_features", feat)
        ratio = err_mean / cor_mean if cor_mean > 0 else float("inf")
        label = FEATURE_LABELS[feat]
        marker = " ← HIGHER" if ratio > 1.1 else (" ← LOWER" if ratio < 0.9 else "")
        print(f"{label:>25}  {err_mean:>9.1f}±{err_std:>4.1f}  {cor_mean:>11.1f}±{cor_std:>4.1f}  {ratio:>6.2f}x{marker}")

    # Compare word count across all asymmetric cases: does longer == more likely preferred?
    all_asymm = type1_cases + type2_cases
    if all_asymm:
        print()
        print("Across ALL asymmetric cases:")
        print("  Does longer response correlate with human preferring error model?")
        longer_is_error_and_preferred = sum(
            1 for c in all_asymm
            if c["error_features"]["word_count"] > c["correct_features"]["word_count"]
            and c["human_label"] == f"model_{c['error_model']}"
        )
        longer_is_error = sum(
            1 for c in all_asymm
            if c["error_features"]["word_count"] > c["correct_features"]["word_count"]
        )
        if longer_is_error > 0:
            print(f"  Error model is longer: {longer_is_error}/{len(all_asymm)} cases")
            print(
                f"  Of those, human prefers the longer (error) model: "
                f"{longer_is_error_and_preferred}/{longer_is_error} "
                f"({100*longer_is_error_and_preferred/longer_is_error:.0f}%)"
            )

    # Detailed case listing
    print(f"\nDetailed Type-1 cases ({len(type1_cases)} total):")
    for c in type1_cases:
        ewc = c["error_features"]["word_count"]
        cwc = c["correct_features"]["word_count"]
        ecc = c["error_features"]["confidence_count"]
        ccc = c["correct_features"]["confidence_count"]
        print(
            f"  {c['id'][:8]}: human={c['human_label']:>9}  "
            f"error_words={ewc:>5} vs correct_words={cwc:>5}  "
            f"error_conf={ecc:>2} vs correct_conf={ccc:>2}"
        )
        print(f"    error: {c['error_explanation'][:80]}...")


# ── Pipeline: severity rating ─────────────────────────────────────────

def rate_severities(
    client: OpenAI,
    judge_model: str,
    ssj_records: dict[str, dict],
    existing: dict[str, dict],
) -> dict[str, dict]:
    """Rate severity for all cases with errors. Returns dict keyed by ID."""
    results = dict(existing)
    ids_to_process = [
        id_ for id_, rec in ssj_records.items()
        if (rec["eval_a"]["error_found"] or rec["eval_b"]["error_found"])
        and id_ not in results
    ]
    total = len(ids_to_process)
    print(f"Rating error severity for {total} cases ({len(existing)} already done)...\n")

    for i, id_ in enumerate(ids_to_process):
        rec = ssj_records[id_]
        result = {"id": id_}
        for side, key in [("a", "eval_a"), ("b", "eval_b")]:
            eval_rec = rec[key]
            if eval_rec["error_found"]:
                raw = call_severity_rater(client, judge_model, eval_rec["explanation"])
                parsed = parse_severity(raw)
                result[key] = {"raw_response": raw, **parsed}
                print(
                    f"[{i+1}/{total}] {id_[:8]} eval_{side}: "
                    f"severity={parsed['severity']} ({parsed['reason'][:60]})"
                )
            else:
                result[key] = {"severity": 0, "reason": "no_error"}
        results[id_] = result

    return results


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse human error-detection sensitivity and style confounds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arena-data", required=True,
                   help="Arena raw data JSON (with conversation_a/b).")
    p.add_argument("--ssj", required=True,
                   help="single_strong_judge result JSON file.")
    p.add_argument("--output", required=True,
                   help="Output JSON file for severity ratings (used as resume cache).")
    p.add_argument("--judge-model", default="gpt-5.2",
                   help="Model to use for severity rating (default: gpt-5.2).")
    p.add_argument("--resume", action="store_true",
                   help="Load existing severity ratings; skip already-rated cases.")
    p.add_argument("--experiments", nargs="+", type=int, default=[4, 5],
                   choices=[4, 5],
                   help="Which experiments to run (default: 4 5).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.ssj) as f:
        ssj_list = json.load(f)
    ssj_records: dict[str, dict] = {r["id"]: r for r in ssj_list}
    print(f"Loaded {len(ssj_records)} ssj records from {args.ssj}")

    with open(args.arena_data) as f:
        arena_list = json.load(f)
    arena_lookup: dict[str, dict] = {r["id"]: r for r in arena_list}
    print(f"Loaded {len(arena_lookup)} arena entries from {args.arena_data}")

    severity_data: dict[str, dict] = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        severity_data = {r["id"]: r for r in existing}
        print(f"Loaded {len(severity_data)} existing severity ratings from {args.output}")

    if 4 in args.experiments:
        # Need LLM for severity rating (unless all already rated)
        error_ids = [
            id_ for id_, rec in ssj_records.items()
            if rec["eval_a"]["error_found"] or rec["eval_b"]["error_found"]
        ]
        unrated = [id_ for id_ in error_ids if id_ not in severity_data]

        if unrated:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit("Error: OPENAI_API_KEY not set. Required for severity rating.")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
            client = OpenAI(api_key=api_key, base_url=base_url)
            severity_data = rate_severities(client, args.judge_model, ssj_records, severity_data)
            with open(args.output, "w") as f:
                json.dump(list(severity_data.values()), f, indent=2)
            print(f"Severity ratings saved to {args.output}")
        else:
            print(f"All {len(severity_data)} error cases already rated.")

        exp4_severity_analysis(ssj_records, severity_data)

    if 5 in args.experiments:
        exp5_style_analysis(ssj_records, arena_lookup)

    print(f"\n{'='*70}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
