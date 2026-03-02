#!/usr/bin/env python3
"""
Experiment 6 — LLM Judges as Better Predictors of Model Capability.

Samples 300 problems from the MATH benchmark (Hendrycks et al.), generates
responses from multiple models, grades each against the ground truth, and then
evaluates all pairwise model combinations with two judge setups:

    single_strong_judge (SSJ) — independent per-response error detection
    single_strong_pair  (SSP) — direct pairwise comparison

For each pairwise battle the "correctness-implied label" is derived from which
models' responses are actually correct:

    A correct, B wrong  →  model_a
    A wrong,  B correct →  model_b
    Both correct        →  tie
    Both wrong          →  both_bad

This is the ground truth label for that battle.

Key metrics:
  • Judgment accuracy — how often each judge setup matches the correctness label
  • Directional accuracy — when one model is clearly correct, does the judge agree?
  • Model Elo rankings from (a) ground truth, (b) ssj, (c) ssp
  • Spearman correlation between judge rankings and ground truth accuracy ranking

Pipeline stages (each saves intermediate JSON and can resume):
  Stage 1  problems   — sample 300 MATH problems
  Stage 2  responses  — generate model responses
  Stage 3  grades     — grade each response against ground truth
  Stage 4  judge      — run SSJ + SSP on all pairwise battles
  Stage 5  analyze    — compute rankings, correlations, print results

Usage:
    python judge/compare_ranking.py \\
        --output-dir results/math_ranking \\
        --models gpt-4o-mini-2024-07-18 gpt-4o-2024-11-20 o3-mini \\
        --n-problems 300 \\
        --judge-model gpt-5.2

    # Resume from partial run
    python judge/compare_ranking.py \\
        --output-dir results/math_ranking \\
        --models gpt-4o-mini-2024-07-18 gpt-4o-2024-11-20 o3-mini \\
        --resume

    # Only print analysis of finished pipeline
    python judge/compare_ranking.py \\
        --output-dir results/math_ranking \\
        --eval-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean

from openai import OpenAI
from scipy.stats import spearmanr

# ── Constants ─────────────────────────────────────────────────────────

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

SOLVE_SYSTEM = (
    "You are an expert mathematician. Solve the problem step by step, "
    "showing your reasoning clearly. End your response with the final answer "
    r"enclosed in \boxed{} — for example: \boxed{42}."
)

SOLVE_PROMPT = """\
Solve the following math problem step by step. \
Put your final answer in \\boxed{{}}.

{problem}"""

# ── Error-detection judge (SSJ style) ────────────────────────────────

SSJ_SYSTEM = (
    "You are an expert mathematics judge. "
    "Check the given math response for correctness. "
    "Focus ONLY on mathematical correctness, not style or presentation."
)

SSJ_PROMPT = """\
Below is a math problem and a model response. Check the response for errors.

## Problem
{problem}

## Response
{response}

## Instructions
Identify any mathematical errors:
- Incorrect calculations or arithmetic
- Flawed logical reasoning or invalid proof steps
- Wrong formulas or theorems applied
- Incorrect final answer
- Missing critical cases or conditions

Respond with ONLY a JSON object (no markdown):
{{"error_found": true or false, "explanation": "brief description of errors found, or 'No errors found'"}}"""

# ── Pairwise judge (SSP style) ────────────────────────────────────────

SSP_SYSTEM = (
    "You are an expert mathematics judge. "
    "Compare two responses to the same math question. "
    "Focus ONLY on mathematical correctness, not style or presentation."
)

SSP_PROMPT = """\
Below are two responses to the same math problem. Which is mathematically better?

## Problem
{problem}

## Response A
{response_a}

## Response B
{response_b}

## Instructions
Compare on mathematical correctness:
- Incorrect calculations or arithmetic
- Flawed logical reasoning or invalid proof steps
- Wrong formulas or theorems applied
- Incorrect final answer
- Missing critical cases or conditions

Choose one of these four verdicts:
- "model_a": Response A is mathematically better
- "model_b": Response B is mathematically better
- "tie": Both are equally correct (or both wrong in the same way)
- "both_bad": Both contain mathematical errors

Respond with ONLY a JSON object (no markdown):
{{"verdict": "model_a" or "model_b" or "tie" or "both_bad", "explanation": "brief justification"}}"""

# ── Grader ────────────────────────────────────────────────────────────

GRADER_SYSTEM = "You are a math answer checker. Determine if a model's final answer is equivalent to the ground truth."

GRADER_PROMPT = """\
Problem: {problem}

Ground truth answer: {gt_answer}

Model response (last part): {response_tail}

Is the model's final answer mathematically equivalent to the ground truth?
Consider numerical equivalence, simplified forms, different but equal expressions.

Respond with ONLY a JSON object:
{{"correct": true or false, "reason": "brief explanation"}}"""


# ── Helpers ───────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{} in a string."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    # Take the last match and find its balanced closing brace
    last = matches[-1].end()
    depth = 1
    i = last
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[last : i - 1].strip() if depth == 0 else None


def normalize_answer(ans: str) -> str:
    """Normalize a math answer string for lightweight exact comparison."""
    ans = ans.strip()
    # Remove common LaTeX decorators that don't change value
    for pat in (r"\\text\{([^}]*)\}", r"\\mathrm\{([^}]*)\}", r"\\mathbf\{([^}]*)\}"):
        ans = re.sub(pat, r"\1", ans)
    ans = re.sub(r"[\$\s,]", "", ans)
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("{", "").replace("}", "")
    return ans.lower().strip()


def answers_match(a: str, b: str) -> bool:
    """Quick heuristic to compare two answer strings."""
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    # Try numeric comparison
    try:
        fa, fb = float(eval(na)), float(eval(nb))  # noqa: S307 – controlled inputs
        return abs(fa - fb) < 1e-6
    except Exception:
        pass
    return False


def call_api(
    client: OpenAI,
    model: str,
    system: str,
    prompt: str,
    *,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=system,
                input=prompt,
                max_output_tokens=max_tokens,
                store=False,
            )
            return resp.output_text.strip()
        except Exception as e:
            print(f"    [Attempt {attempt+1}/{max_retries}] {model} error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    return ""


def parse_json_safe(text: str, default: dict) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


# ── Stage 1: Sample MATH Problems ────────────────────────────────────

def stage_problems(n: int, seed: int, out_path: str) -> list[dict]:
    """Stratified sample of n problems from MATH test set."""
    if os.path.exists(out_path):
        with open(out_path) as f:
            problems = json.load(f)
        print(f"  Loaded {len(problems)} existing problems from {out_path}")
        return problems

    from datasets import load_dataset

    rng = random.Random(seed)
    all_problems: list[dict] = []
    for subj in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        for item in ds:
            gt = extract_boxed(item["solution"])
            all_problems.append({
                "subject": subj,
                "level": item["level"],
                "problem": item["problem"],
                "solution": item["solution"],
                "gt_answer": gt,
            })

    # Stratified sample: equal per subject
    per_subject = n // len(MATH_SUBJECTS)
    sampled: list[dict] = []
    for subj in MATH_SUBJECTS:
        subj_problems = [p for p in all_problems if p["subject"] == subj and p["gt_answer"]]
        k = min(per_subject, len(subj_problems))
        sampled.extend(rng.sample(subj_problems, k))

    # Top up to exactly n with remaining
    remaining = [p for p in all_problems if p not in sampled and p["gt_answer"]]
    rng.shuffle(remaining)
    sampled.extend(remaining[: n - len(sampled)])

    # Assign IDs
    for i, p in enumerate(sampled):
        p["math_id"] = f"math_{i:04d}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"  Sampled {len(sampled)} problems → {out_path}")
    subj_dist = Counter(p["subject"] for p in sampled)
    for s, c in sorted(subj_dist.items()):
        print(f"    {s}: {c}")
    return sampled


# ── Stage 2: Generate Responses ───────────────────────────────────────

def stage_responses(
    client: OpenAI,
    problems: list[dict],
    models: list[str],
    out_path: str,
) -> dict[str, dict[str, str]]:
    """Generate responses for each (problem, model). Returns {math_id: {model: response}}."""
    existing: dict[str, dict[str, str]] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    total_needed = sum(
        1 for p in problems for m in models
        if p["math_id"] not in existing or m not in existing[p["math_id"]]
    )
    print(f"  Generating {total_needed} model responses "
          f"({sum(len(v) for v in existing.values())} already done)...")

    done = 0
    for p in problems:
        mid = p["math_id"]
        if mid not in existing:
            existing[mid] = {}
        for model in models:
            if model in existing[mid]:
                continue
            prompt = SOLVE_PROMPT.format(problem=p["problem"])
            raw = call_api(client, model, SOLVE_SYSTEM, prompt, max_tokens=2048)
            existing[mid][model] = raw
            done += 1
            if done % 10 == 0 or done == total_needed:
                print(f"    [{done}/{total_needed}] responses generated")
                with open(out_path, "w") as f:
                    json.dump(existing, f)

    with open(out_path, "w") as f:
        json.dump(existing, f)
    print(f"  Responses saved to {out_path}")
    return existing


# ── Stage 3: Grade Responses ──────────────────────────────────────────

def grade_response(
    client: OpenAI,
    grader_model: str,
    problem: str,
    gt_answer: str,
    response: str,
) -> dict:
    """Grade a single response. Returns {correct, method, reason}."""
    model_answer = extract_boxed(response)

    if model_answer is not None:
        if answers_match(model_answer, gt_answer):
            return {"correct": True, "method": "boxed_match", "model_answer": model_answer, "reason": ""}

    # LLM grader fallback
    tail = response[-1200:] if len(response) > 1200 else response
    prompt = GRADER_PROMPT.format(
        problem=problem[:800],
        gt_answer=gt_answer,
        response_tail=tail,
    )
    raw = call_api(client, grader_model, GRADER_SYSTEM, prompt, max_tokens=256)
    parsed = parse_json_safe(raw, {"correct": False, "reason": "parse_error"})
    return {
        "correct": bool(parsed.get("correct", False)),
        "method": "llm_grader",
        "model_answer": model_answer or "",
        "reason": str(parsed.get("reason", "")),
    }


def stage_grades(
    client: OpenAI,
    problems: list[dict],
    responses: dict[str, dict[str, str]],
    models: list[str],
    grader_model: str,
    out_path: str,
) -> dict[str, dict[str, dict]]:
    """Grade all responses. Returns {math_id: {model: grade_dict}}."""
    existing: dict[str, dict[str, dict]] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    total_needed = sum(
        1 for p in problems for m in models
        if p["math_id"] not in existing or m not in existing[p["math_id"]]
    )
    print(f"  Grading {total_needed} responses "
          f"({sum(len(v) for v in existing.values())} already done)...")

    done = 0
    for p in problems:
        mid = p["math_id"]
        if mid not in existing:
            existing[mid] = {}
        for model in models:
            if model in existing[mid]:
                continue
            response = responses.get(mid, {}).get(model, "")
            grade = grade_response(client, grader_model, p["problem"], p["gt_answer"], response)
            existing[mid][model] = grade
            done += 1
            if done % 20 == 0 or done == total_needed:
                print(f"    [{done}/{total_needed}] graded")
                with open(out_path, "w") as f:
                    json.dump(existing, f)

    with open(out_path, "w") as f:
        json.dump(existing, f)
    print(f"  Grades saved to {out_path}")
    return existing


# ── Stage 4: Judge Pairs ──────────────────────────────────────────────

def correctness_implied_label(a_correct: bool, b_correct: bool) -> str:
    if a_correct and b_correct:
        return "tie"
    if a_correct and not b_correct:
        return "model_a"
    if not a_correct and b_correct:
        return "model_b"
    return "both_bad"


def judge_ssj(
    client: OpenAI,
    judge_model: str,
    problem: str,
    response_a: str,
    response_b: str,
) -> dict:
    """Single-strong judge: evaluate each response independently."""
    result = {}
    for label, resp in [("eval_a", response_a), ("eval_b", response_b)]:
        prompt = SSJ_PROMPT.format(problem=problem, response=resp)
        raw = call_api(client, judge_model, SSJ_SYSTEM, prompt, max_tokens=1024)
        parsed = parse_json_safe(raw, {"error_found": False, "explanation": "parse_error"})
        result[label] = {
            "raw_response": raw,
            "error_found": bool(parsed.get("error_found", False)),
            "explanation": str(parsed.get("explanation", "")),
        }
    ea = result["eval_a"]["error_found"]
    eb = result["eval_b"]["error_found"]
    if ea and eb:
        judge_label = "both_bad"
    elif ea:
        judge_label = "model_b"
    elif eb:
        judge_label = "model_a"
    else:
        judge_label = "tie"
    result["judge_label"] = judge_label
    return result


def judge_ssp(
    client: OpenAI,
    judge_model: str,
    problem: str,
    response_a: str,
    response_b: str,
) -> dict:
    """Single-strong pairwise judge: compare both responses at once."""
    prompt = SSP_PROMPT.format(
        problem=problem, response_a=response_a, response_b=response_b
    )
    raw = call_api(client, judge_model, SSP_SYSTEM, prompt, max_tokens=1024)
    parsed = parse_json_safe(raw, {"verdict": "tie", "explanation": "parse_error"})
    verdict = str(parsed.get("verdict", "tie")).lower()
    if verdict not in {"model_a", "model_b", "tie", "both_bad"}:
        verdict = "tie"
    return {
        "raw_response": raw,
        "judge_label": verdict,
        "explanation": str(parsed.get("explanation", "")),
    }


def stage_judge(
    client: OpenAI,
    problems: list[dict],
    responses: dict[str, dict[str, str]],
    grades: dict[str, dict[str, dict]],
    models: list[str],
    judge_model: str,
    out_path: str,
) -> list[dict]:
    """Run SSJ + SSP on all pairwise model battles for all problems."""
    existing_by_key: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_list = json.load(f)
        existing_by_key = {
            f"{r['math_id']}_{r['model_a']}_{r['model_b']}": r
            for r in existing_list
        }

    pairs = list(combinations(models, 2))
    total_battles = len(problems) * len(pairs)
    needed = sum(
        1 for p in problems for ma, mb in pairs
        if f"{p['math_id']}_{ma}_{mb}" not in existing_by_key
    )
    print(f"  Running judges on {needed} battles "
          f"({total_battles - needed} already done, {len(pairs)} model pairs × {len(problems)} problems)...")

    done = 0
    results = list(existing_by_key.values())

    for p in problems:
        mid = p["math_id"]
        for ma, mb in pairs:
            key = f"{mid}_{ma}_{mb}"
            if key in existing_by_key:
                continue
            resp_a = responses.get(mid, {}).get(ma, "")
            resp_b = responses.get(mid, {}).get(mb, "")
            g_a = grades.get(mid, {}).get(ma, {}).get("correct", False)
            g_b = grades.get(mid, {}).get(mb, {}).get("correct", False)
            gt_label = correctness_implied_label(g_a, g_b)

            ssj_result = judge_ssj(client, judge_model, p["problem"], resp_a, resp_b)
            ssp_result = judge_ssp(client, judge_model, p["problem"], resp_a, resp_b)

            record = {
                "math_id": mid,
                "subject": p["subject"],
                "level": p["level"],
                "model_a": ma,
                "model_b": mb,
                "gt_label": gt_label,
                "ssj_label": ssj_result["judge_label"],
                "ssp_label": ssp_result["judge_label"],
                "ssj_eval_a": ssj_result["eval_a"],
                "ssj_eval_b": ssj_result["eval_b"],
                "ssp_response": {k: v for k, v in ssp_result.items() if k != "judge_label"},
            }
            results.append(record)
            existing_by_key[key] = record
            done += 1
            if done % 20 == 0 or done == needed:
                print(f"    [{done}/{needed}] battles judged")
                with open(out_path, "w") as f:
                    json.dump(results, f)

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Judge results saved to {out_path}")
    return results


# ── Stage 5: Analysis ─────────────────────────────────────────────────

def compute_elo(battles: list[tuple[str, str, str]], k: float = 32.0, init: float = 1500.0) -> dict[str, float]:
    """Compute Elo ratings from (model_a, model_b, label) tuples."""
    ratings: dict[str, float] = defaultdict(lambda: init)
    for ma, mb, label in battles:
        ra, rb = ratings[ma], ratings[mb]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea
        if label == "model_a":
            sa, sb = 1.0, 0.0
        elif label == "model_b":
            sa, sb = 0.0, 1.0
        else:  # tie or both_bad
            sa, sb = 0.5, 0.5
        ratings[ma] = ra + k * (sa - ea)
        ratings[mb] = rb + k * (sb - eb)
    return dict(ratings)


def judge_accuracy(battles: list[dict], judge_key: str) -> dict:
    """Compute accuracy of a judge setup against ground truth labels."""
    exact = sum(1 for b in battles if b[judge_key] == b["gt_label"])
    total = len(battles)

    # Directional accuracy: when gt says one model is better
    dir_battles = [b for b in battles if b["gt_label"] in ("model_a", "model_b")]
    dir_correct = sum(1 for b in dir_battles if b[judge_key] == b["gt_label"])

    # False preference: when gt says no winner (tie/both_bad), judge picks one
    no_winner_battles = [b for b in battles if b["gt_label"] in ("tie", "both_bad")]
    false_pref = sum(
        1 for b in no_winner_battles if b[judge_key] in ("model_a", "model_b")
    )

    return {
        "exact_accuracy":    exact / total if total else 0,
        "exact_n":           exact,
        "total":             total,
        "dir_accuracy":      dir_correct / len(dir_battles) if dir_battles else 0,
        "dir_n":             dir_correct,
        "dir_total":         len(dir_battles),
        "false_pref_rate":   false_pref / len(no_winner_battles) if no_winner_battles else 0,
        "false_pref_n":      false_pref,
        "no_winner_total":   len(no_winner_battles),
    }


def stage_analyze(
    battles: list[dict],
    grades: dict[str, dict[str, dict]],
    models: list[str],
) -> None:
    print(f"\n{'='*70}")
    print("EXPERIMENT 6 — Ranking Analysis")
    print(f"{'='*70}")

    # ── Per-model ground-truth accuracy ──────────────────────────────
    print("\n--- Ground Truth Accuracy per Model ---")
    model_accuracy: dict[str, float] = {}
    for model in models:
        correct = sum(1 for grades_per_problem in grades.values()
                      if grades_per_problem.get(model, {}).get("correct", False))
        total = sum(1 for grades_per_problem in grades.values()
                    if model in grades_per_problem)
        acc = correct / total if total else 0.0
        model_accuracy[model] = acc
        print(f"  {model}: {correct}/{total} = {acc:.1%}")

    # ── Label distribution ────────────────────────────────────────────
    print(f"\n--- Label Distribution ({len(battles)} battles) ---")
    print(f"{'Label':>12}  {'GT':>8}  {'SSJ':>8}  {'SSP':>8}")
    for lbl in ["model_a", "model_b", "tie", "both_bad"]:
        gt_n  = sum(1 for b in battles if b["gt_label"]  == lbl)
        ssj_n = sum(1 for b in battles if b["ssj_label"] == lbl)
        ssp_n = sum(1 for b in battles if b["ssp_label"] == lbl)
        print(f"{lbl:>12}  {gt_n:>8}  {ssj_n:>8}  {ssp_n:>8}")

    # ── Judgment accuracy ─────────────────────────────────────────────
    print("\n--- Judgment Accuracy vs Ground Truth ---")
    for setup_name, judge_key in [("ssj", "ssj_label"), ("ssp", "ssp_label")]:
        m = judge_accuracy(battles, judge_key)
        print(f"\n  {setup_name.upper()}:")
        print(f"    Exact match accuracy:   {m['exact_n']}/{m['total']} = {m['exact_accuracy']:.1%}")
        print(f"    Directional accuracy:   {m['dir_n']}/{m['dir_total']} = {m['dir_accuracy']:.1%}")
        print(f"      (when GT has a clear winner)")
        print(f"    False preference rate:  {m['false_pref_n']}/{m['no_winner_total']} = {m['false_pref_rate']:.1%}")
        print(f"      (when GT says tie/both_bad, judge still picks winner)")

    # ── Confusion matrices ────────────────────────────────────────────
    LBLS = ["model_a", "model_b", "tie", "both_bad"]
    for setup_name, judge_key in [("ssj", "ssj_label"), ("ssp", "ssp_label")]:
        print(f"\n  {setup_name.upper()} Confusion Matrix (rows=GT, cols={setup_name}):")
        print(f"  {'':>10}" + "".join(f"{l:>10}" for l in LBLS))
        for gt_l in LBLS:
            row = "  " + f"{gt_l:>10}" + "".join(
                f"{sum(1 for b in battles if b['gt_label']==gt_l and b[judge_key]==jl):>10}"
                for jl in LBLS
            )
            print(row)

    # ── Per-subject accuracy ──────────────────────────────────────────
    subjects = sorted(set(b["subject"] for b in battles))
    print(f"\n--- Per-Subject Directional Accuracy ---")
    print(f"{'Subject':>30}  {'N':>5}  {'SSJ':>8}  {'SSP':>8}")
    for subj in subjects:
        subj_battles = [b for b in battles if b["subject"] == subj
                        and b["gt_label"] in ("model_a", "model_b")]
        if not subj_battles:
            continue
        ssj_d = sum(1 for b in subj_battles if b["ssj_label"] == b["gt_label"])
        ssp_d = sum(1 for b in subj_battles if b["ssp_label"] == b["gt_label"])
        n = len(subj_battles)
        print(f"{subj:>30}  {n:>5}  {ssj_d/n:>7.1%}  {ssp_d/n:>7.1%}")

    # ── Elo Rankings ─────────────────────────────────────────────────
    print(f"\n--- Model Elo Rankings ---")
    battle_tuples = {
        "gt":  [(b["model_a"], b["model_b"], b["gt_label"])  for b in battles],
        "ssj": [(b["model_a"], b["model_b"], b["ssj_label"]) for b in battles],
        "ssp": [(b["model_a"], b["model_b"], b["ssp_label"]) for b in battles],
    }
    elos: dict[str, dict[str, float]] = {}
    for src, tuples in battle_tuples.items():
        elos[src] = compute_elo(tuples)

    # Print combined ranking table
    gt_ranking = sorted(models, key=lambda m: elos["gt"].get(m, 1500), reverse=True)
    print(f"\n{'Model':>30}  {'GT Acc':>8}  {'GT Elo':>8}  {'SSJ Elo':>8}  {'SSP Elo':>8}")
    print("-" * 70)
    for model in gt_ranking:
        acc = model_accuracy.get(model, 0.0)
        gt_elo  = elos["gt"].get(model, 1500)
        ssj_elo = elos["ssj"].get(model, 1500)
        ssp_elo = elos["ssp"].get(model, 1500)
        print(f"{model:>30}  {acc:>7.1%}  {gt_elo:>8.1f}  {ssj_elo:>8.1f}  {ssp_elo:>8.1f}")

    # ── Spearman correlations ─────────────────────────────────────────
    print(f"\n--- Spearman Rank Correlation with GT Accuracy ---")
    acc_vec = [model_accuracy[m] for m in models]
    for src in ["gt", "ssj", "ssp"]:
        elo_vec = [elos[src].get(m, 1500) for m in models]
        rho, pval = spearmanr(acc_vec, elo_vec)
        print(f"  {src.upper()} Elo vs GT accuracy: ρ = {rho:.3f}  (p = {pval:.3f})")

    # ── Error detection accuracy by difficulty ────────────────────────
    levels = sorted(set(b["level"] for b in battles))
    print(f"\n--- SSJ Directional Accuracy by Difficulty Level ---")
    print(f"{'Level':>10}  {'N':>5}  {'SSJ':>8}  {'SSP':>8}")
    for lvl in levels:
        lvl_battles = [b for b in battles if b["level"] == lvl
                       and b["gt_label"] in ("model_a", "model_b")]
        if not lvl_battles:
            continue
        ssj_d = sum(1 for b in lvl_battles if b["ssj_label"] == b["gt_label"])
        ssp_d = sum(1 for b in lvl_battles if b["ssp_label"] == b["gt_label"])
        n = len(lvl_battles)
        print(f"{lvl:>10}  {n:>5}  {ssj_d/n:>7.1%}  {ssp_d/n:>7.1%}")


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 6: LLM judge rankings vs ground truth on MATH benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir", default="results/math_ranking",
                   help="Directory for all intermediate and final outputs.")
    p.add_argument("--models", nargs="+",
                   default=["gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "o3-mini"],
                   help="Models to evaluate.")
    p.add_argument("--n-problems", type=int, default=300,
                   help="Number of MATH problems to sample.")
    p.add_argument("--judge-model", default="gpt-5.2",
                   help="Model to use for judging and grading (default: gpt-5.2).")
    p.add_argument("--grader-model", default=None,
                   help="Override grader model (defaults to --judge-model).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for problem sampling.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing intermediate files.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip all API calls; only run analysis on existing results.")
    p.add_argument("--stages", nargs="+",
                   choices=["problems", "responses", "grades", "judge", "analyze"],
                   default=["problems", "responses", "grades", "judge", "analyze"],
                   help="Which pipeline stages to run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grader_model = args.grader_model or args.judge_model

    os.makedirs(args.output_dir, exist_ok=True)
    paths = {
        "problems":  os.path.join(args.output_dir, "math_problems.json"),
        "responses": os.path.join(args.output_dir, "math_responses.json"),
        "grades":    os.path.join(args.output_dir, "math_grades.json"),
        "judge":     os.path.join(args.output_dir, "math_judge_results.json"),
    }

    print(f"Output directory: {args.output_dir}")
    print(f"Models: {args.models}")
    print(f"N problems: {args.n_problems}")
    print(f"Judge model: {args.judge_model}")

    # ── Stage 1: Problems ─────────────────────────────────────────────
    print("\n[Stage 1] Loading/sampling MATH problems...")
    problems = stage_problems(args.n_problems, args.seed, paths["problems"])

    if args.eval_only:
        # Load all intermediate files and jump to analysis
        with open(paths["grades"]) as f:
            grades = json.load(f)
        with open(paths["judge"]) as f:
            battles = json.load(f)
        stage_analyze(battles, grades, args.models)
        return

    needs_api = any(s in args.stages for s in ("responses", "grades", "judge"))
    client = None
    if needs_api:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Error: OPENAI_API_KEY environment variable is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)

    # ── Stage 2: Responses ────────────────────────────────────────────
    STAGE_ORDER = ["problems", "responses", "grades", "judge", "analyze"]
    max_stage = max(STAGE_ORDER.index(s) for s in args.stages)

    if max_stage < 1:
        print(f"\n{'='*70}")
        print(f"Pipeline complete. Results in {args.output_dir}/")
        return

    if "responses" in args.stages:
        print("\n[Stage 2] Generating model responses...")
        responses = stage_responses(client, problems, args.models, paths["responses"])
    else:
        with open(paths["responses"]) as f:
            responses = json.load(f)
        print(f"[Stage 2] Loaded responses from {paths['responses']}")

    if max_stage < 2:
        print(f"\n{'='*70}")
        print(f"Pipeline complete. Results in {args.output_dir}/")
        return

    # ── Stage 3: Grades ───────────────────────────────────────────────
    if "grades" in args.stages:
        print("\n[Stage 3] Grading responses against ground truth...")
        grades = stage_grades(
            client, problems, responses, args.models, grader_model, paths["grades"]
        )
    else:
        with open(paths["grades"]) as f:
            grades = json.load(f)
        print(f"[Stage 3] Loaded grades from {paths['grades']}")

    if max_stage < 3:
        print(f"\n{'='*70}")
        print(f"Pipeline complete. Results in {args.output_dir}/")
        return

    # ── Stage 4: Judge ────────────────────────────────────────────────
    if "judge" in args.stages:
        print("\n[Stage 4] Running SSJ + SSP judges on all pairwise battles...")
        battles = stage_judge(
            client, problems, responses, grades, args.models, args.judge_model, paths["judge"]
        )
    else:
        with open(paths["judge"]) as f:
            battles = json.load(f)
        print(f"[Stage 4] Loaded battles from {paths['judge']}")

    # ── Stage 5: Analyze ──────────────────────────────────────────────
    if "analyze" in args.stages:
        stage_analyze(battles, grades, args.models)

    print(f"\n{'='*70}")
    print(f"Pipeline complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
