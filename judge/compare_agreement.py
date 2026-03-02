#!/usr/bin/env python3
"""
Experiments 1–3: Human vs LLM Judge Agreement Analysis.

Exp 1 — Correctness Alignment Rate
    Uses the single_strong_judge's per-model error flags as a correctness oracle.
    When the oracle identifies exactly one erroneous model, measures how often each
    label source (human, mutual_judge, mutual_pair, ssj, ssp) correctly prefers
    the non-error model.

Exp 2 — Cross-Judge Agreement Matrix (Cohen's κ)
    Computes pairwise Cohen's kappa between all five label sources:
    human, mutual_judge, mutual_pair, single_strong_judge, single_strong_pair.
    Shows whether LLM judges cluster together and how far human is from them.

Exp 3 — Correctness-Conditional Human Preference Distribution
    Groups examples by ssj error pattern (only_A_error, only_B_error, both_error,
    no_error) and shows the human label distribution in each group.
    Chi-squared test for independence between error pattern and human label.

Usage:
    python judge/compare_agreement.py \\
        --mutual-judge  results/mutual_judge_140k_math_openai_single.json \\
        --mutual-pair   results/mutual_pair_140k_math_openai_single.json \\
        --ssj           results/single_strong_judge_140k_math_openai_single.json \\
        --ssp           results/single_strong_pair_140k_math_openai_single.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

LABELS = ["model_a", "model_b", "tie", "both_bad"]
# Collapsed directional label: "a_better" | "b_better" | "neutral"
DIRECTION = {"model_a": "a_better", "model_b": "b_better", "tie": "neutral", "both_bad": "neutral"}


# ── Data loading ──────────────────────────────────────────────────────

def load_results(paths: dict[str, str]) -> dict[str, dict[str, dict]]:
    """Load each result file keyed by source name → {id: record}."""
    sources: dict[str, dict[str, dict]] = {}
    for name, path in paths.items():
        with open(path) as f:
            records = json.load(f)
        sources[name] = {r["id"]: r for r in records}
        print(f"  Loaded {len(records):>4} records from {name} ({path})")
    return sources


def common_ids(sources: dict[str, dict[str, dict]]) -> list[str]:
    """Return IDs present in all sources."""
    id_sets = [set(v.keys()) for v in sources.values()]
    shared = id_sets[0].intersection(*id_sets[1:])
    return sorted(shared)


def get_label(source_name: str, record: dict) -> str:
    """Extract the judge label from a record depending on source naming convention."""
    for key in ("mutual_label", "mutual_pair_label", "judge_label", "human_label"):
        if key in record:
            return record[key]
    raise KeyError(f"No label key found in record for source '{source_name}': {list(record.keys())}")


def ssj_error_pattern(record: dict) -> str:
    """Return the error pattern from single_strong_judge's binary error flags."""
    ea = record["eval_a"]["error_found"]
    eb = record["eval_b"]["error_found"]
    if ea and eb:
        return "both_error"
    if ea and not eb:
        return "only_A_error"     # A has error → B is correct
    if not ea and eb:
        return "only_B_error"     # B has error → A is correct
    return "no_error"


def correctness_implied_label(pattern: str) -> str | None:
    """Map an ssj error pattern to the correctness-implied preference label."""
    return {"only_A_error": "model_b", "only_B_error": "model_a"}.get(pattern)


# ── Experiment 1: Correctness Alignment Rate ─────────────────────────

def exp1_correctness_alignment(
    sources: dict[str, dict[str, dict]],
    ids: list[str],
    ssj_name: str = "ssj",
) -> None:
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 — Correctness Alignment Rate")
    print(f"{'='*70}")
    print(
        "Oracle: single_strong_judge binary error flags.\n"
        "When oracle finds error in exactly ONE model, the 'correct' preference\n"
        "is to pick the model WITHOUT the error.\n"
        "Metric: fraction of asymmetric-error cases each source gets right.\n"
    )

    ssj_records = sources[ssj_name]

    # Identify asymmetric-error cases
    asymmetric_ids = [
        i for i in ids
        if ssj_error_pattern(ssj_records[i]) in ("only_A_error", "only_B_error")
    ]
    print(f"Asymmetric-error cases (oracle): {len(asymmetric_ids)} / {len(ids)}")
    print(
        "  only_A_error (→ prefer model_b): "
        f"{sum(1 for i in asymmetric_ids if ssj_error_pattern(ssj_records[i]) == 'only_A_error')}"
    )
    print(
        "  only_B_error (→ prefer model_a): "
        f"{sum(1 for i in asymmetric_ids if ssj_error_pattern(ssj_records[i]) == 'only_B_error')}"
    )

    print(f"\n{'Source':>20}  {'Correct':>8}  {'Wrong':>8}  {'Neutral':>8}  {'Align%':>8}")
    print("-" * 62)
    for src_name, records in sources.items():
        correct = wrong = neutral = 0
        for aid in asymmetric_ids:
            if aid not in records:
                continue
            pattern = ssj_error_pattern(ssj_records[aid])
            expected = correctness_implied_label(pattern)
            actual = get_label(src_name, records[aid])
            if actual == expected:
                correct += 1
            elif actual in ("tie", "both_bad"):
                neutral += 1
            else:
                wrong += 1
        n = correct + wrong + neutral
        pct = correct / n * 100 if n else 0
        print(f"{src_name:>20}  {correct:>8}  {wrong:>8}  {neutral:>8}  {pct:>7.1f}%")

    print(
        "\nNote: ssj aligns 100% by construction (judge_label is derived from error flags)."
    )


# ── Experiment 2: Cross-Judge Agreement Matrix ────────────────────────

def exp2_agreement_matrix(
    sources: dict[str, dict[str, dict]],
    ids: list[str],
) -> None:
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 — Cross-Judge Agreement Matrix (Cohen's κ)")
    print(f"{'='*70}")
    print(
        "Pairwise Cohen's kappa between all five label sources.\n"
        "κ ≈ 1.0: perfect agreement; κ ≈ 0: chance-level agreement; κ < 0: worse than chance.\n"
    )

    source_names = list(sources.keys())

    # Build label vectors per source (over common IDs)
    label_vecs: dict[str, list[str]] = {}
    for name, records in sources.items():
        label_vecs[name] = [get_label(name, records[i]) for i in ids if i in records]

    # Some sources may have fewer records; use their intersection
    all_pairs: list[tuple[str, str]] = list(combinations(source_names, 2))

    # Build kappa matrix
    kappa_matrix = {n: {m: None for m in source_names} for n in source_names}
    for n in source_names:
        kappa_matrix[n][n] = 1.0

    print(f"{'':>22}" + "".join(f"{s:>14}" for s in source_names))
    print("-" * (22 + 14 * len(source_names)))

    for i, src_a in enumerate(source_names):
        row_str = f"{src_a:>22}"
        records_a = sources[src_a]
        for src_b in source_names:
            if src_a == src_b:
                row_str += f"{'1.000':>14}"
                continue
            records_b = sources[src_b]
            shared = [i for i in ids if i in records_a and i in records_b]
            if len(shared) < 2:
                row_str += f"{'N/A':>14}"
                continue
            labels_a = [get_label(src_a, records_a[i]) for i in shared]
            labels_b = [get_label(src_b, records_b[i]) for i in shared]
            try:
                kappa = cohen_kappa_score(labels_a, labels_b)
                kappa_matrix[src_a][src_b] = kappa
                row_str += f"{kappa:>14.3f}"
            except Exception as e:
                row_str += f"{'ERR':>14}"
        print(row_str)

    # Summary: average kappa of each source with all LLM judges (excluding human)
    print()
    llm_sources = [s for s in source_names if s != "human"]
    print("Average kappa with other LLM judges (excluding self and human):")
    for src in llm_sources:
        others = [m for m in llm_sources if m != src]
        kappas = [kappa_matrix[src][m] for m in others if kappa_matrix[src][m] is not None]
        avg_kappa = np.mean(kappas) if kappas else float("nan")
        print(f"  {src:>20}: {avg_kappa:.3f}")

    print("\nHuman's average kappa with each LLM judge:")
    for llm in llm_sources:
        k = kappa_matrix["human"].get(llm) or kappa_matrix[llm].get("human")
        print(f"  human vs {llm:>20}: {k:.3f}")

    # Also compute directional (3-way: a_better, b_better, neutral) kappa
    print("\nDirectional kappa (model_a/model_b → winner, tie+both_bad → neutral):")
    print(f"{'':>22}" + "".join(f"{s:>14}" for s in source_names))
    print("-" * (22 + 14 * len(source_names)))
    for src_a in source_names:
        row_str = f"{src_a:>22}"
        records_a = sources[src_a]
        for src_b in source_names:
            if src_a == src_b:
                row_str += f"{'1.000':>14}"
                continue
            records_b = sources[src_b]
            shared = [i for i in ids if i in records_a and i in records_b]
            if len(shared) < 2:
                row_str += f"{'N/A':>14}"
                continue
            labels_a = [DIRECTION[get_label(src_a, records_a[i])] for i in shared]
            labels_b = [DIRECTION[get_label(src_b, records_b[i])] for i in shared]
            try:
                kappa = cohen_kappa_score(labels_a, labels_b)
                row_str += f"{kappa:>14.3f}"
            except Exception:
                row_str += f"{'ERR':>14}"
        print(row_str)


# ── Experiment 3: Correctness-Conditional Human Distribution ─────────

def exp3_conditional_distribution(
    sources: dict[str, dict[str, dict]],
    ids: list[str],
    ssj_name: str = "ssj",
) -> None:
    print(f"\n{'='*70}")
    print("EXPERIMENT 3 — Correctness-Conditional Human Preference Distribution")
    print(f"{'='*70}")
    print(
        "Groups examples by ssj error pattern. Tests whether human label distribution\n"
        "shifts as a function of which model has errors (as detected by the LLM judge).\n"
        "If humans track correctness, the distributions should differ significantly.\n"
    )

    ssj_records = sources[ssj_name]
    human_records = sources["human"]

    patterns = ["only_A_error", "only_B_error", "both_error", "no_error"]
    pattern_labels = {
        "only_A_error": "Only A has error  (oracle→ model_b)",
        "only_B_error": "Only B has error  (oracle→ model_a)",
        "both_error":   "Both have errors  (oracle→ both_bad)",
        "no_error":     "Neither has error (oracle→ tie)",
    }

    # Build contingency table: rows = pattern, cols = human_label
    contingency: dict[str, Counter] = {p: Counter() for p in patterns}
    for i in ids:
        if i not in ssj_records or i not in human_records:
            continue
        pattern = ssj_error_pattern(ssj_records[i])
        human_lbl = get_label("human", human_records[i])
        contingency[pattern][human_lbl] += 1

    # Print distribution table
    print(f"\n{'Error Pattern':>40}  " + "  ".join(f"{l:>10}" for l in LABELS) + "  Total")
    print("-" * 100)
    ct_matrix = []
    for pat in patterns:
        counts = [contingency[pat].get(l, 0) for l in LABELS]
        total = sum(counts)
        ct_matrix.append(counts)
        counts_str = "  ".join(f"{c:>10}" for c in counts)
        # Show percentages of human label within this error pattern
        pct_str = "  ".join(f"{c/total*100:>9.1f}%" for c in counts)
        print(f"{pattern_labels[pat]:>40}  {counts_str}  {total}")
        print(f"{'':>40}  {pct_str}")
        print()

    # Chi-squared test for independence
    ct_array = np.array(ct_matrix)
    # Remove all-zero rows/cols to avoid issues
    row_mask = ct_array.sum(axis=1) > 0
    col_mask = ct_array.sum(axis=0) > 0
    ct_trimmed = ct_array[row_mask][:, col_mask]
    chi2, p_value, dof, expected = chi2_contingency(ct_trimmed)
    print(f"Chi-squared test for independence (error_pattern × human_label):")
    print(f"  χ² = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("  → SIGNIFICANT: human label distribution differs across error patterns (p < 0.05)")
    else:
        print("  → NOT SIGNIFICANT: human label distribution does NOT differ across error patterns")
        print("     This supports the hypothesis that human preference is independent of LLM-detected errors.")

    # Compute correctness alignment per pattern for human
    print()
    print("Human correctness alignment per error pattern:")
    print(f"  {'Pattern':>40}  {'Correct':>8}  {'Wrong':>8}  {'Neutral':>8}  {'Align%':>8}")
    for pat in ["only_A_error", "only_B_error"]:
        expected_lbl = correctness_implied_label(pat)
        counts = contingency[pat]
        total = sum(counts.values())
        correct = counts.get(expected_lbl, 0)
        neutral = counts.get("tie", 0) + counts.get("both_bad", 0)
        wrong = total - correct - neutral
        pct = correct / total * 100 if total else 0
        print(f"  {pattern_labels[pat]:>40}  {correct:>8}  {wrong:>8}  {neutral:>8}  {pct:>7.1f}%")

    # Compare error rate distribution: when human says X, what does oracle say?
    print()
    print("Reverse conditional: oracle error pattern distribution given human label")
    print(f"{'Human Label':>12}  " + "  ".join(f"{p[:15]:>16}" for p in patterns))
    print("-" * 80)
    human_given = defaultdict(Counter)
    for i in ids:
        if i not in ssj_records or i not in human_records:
            continue
        pat = ssj_error_pattern(ssj_records[i])
        hl = get_label("human", human_records[i])
        human_given[hl][pat] += 1
    for hl in LABELS:
        counts = human_given[hl]
        total = sum(counts.values())
        counts_str = "  ".join(f"{counts.get(p,0)/total*100:>15.1f}%" for p in patterns)
        print(f"{hl:>12}  {counts_str}  (n={total})")

    print(
        "\nInterpretation: if human preference tracked correctness, rows should differ\n"
        "substantially — e.g., human=model_a should have high 'only_B_error' and\n"
        "low 'only_A_error'. Uniform rows → human preference is correctness-blind."
    )


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare agreement between human and LLM judge labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mutual-judge", required=True,
                   help="mutual_judge result JSON file.")
    p.add_argument("--mutual-pair", required=True,
                   help="mutual_pair result JSON file.")
    p.add_argument("--ssj", required=True,
                   help="single_strong_judge result JSON file.")
    p.add_argument("--ssp", required=True,
                   help="single_strong_pair result JSON file.")
    p.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3],
                   choices=[1, 2, 3],
                   help="Which experiments to run (default: all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading result files...")
    paths = {
        "human":        args.ssj,  # human_label lives in ssj (all sources have it)
        "mutual_judge": args.mutual_judge,
        "mutual_pair":  args.mutual_pair,
        "ssj":          args.ssj,
        "ssp":          args.ssp,
    }

    # We use ssj as the canonical file for human labels (same human_label in all files).
    # Load separately so we can use the correct label key per source.
    raw: dict[str, dict[str, dict]] = {}
    for name, path in paths.items():
        if name == "human":
            continue
        with open(path) as f:
            records = json.load(f)
        raw[name] = {r["id"]: r for r in records}
        print(f"  {name}: {len(records)} records from {path}")

    # Build a pseudo-source "human" from any file that has human_label
    human_src: dict[str, dict] = {}
    with open(args.ssj) as f:
        ssj_records = json.load(f)
    for r in ssj_records:
        human_src[r["id"]] = {"human_label": r["human_label"]}
    raw["human"] = human_src
    print(f"  human: {len(human_src)} records (from ssj)")

    # Sort source order for readability
    ordered = {k: raw[k] for k in ["human", "mutual_judge", "mutual_pair", "ssj", "ssp"]}

    ids = common_ids(ordered)
    print(f"\nCommon IDs across all sources: {len(ids)}")

    if 1 in args.experiments:
        exp1_correctness_alignment(ordered, ids)
    if 2 in args.experiments:
        exp2_agreement_matrix(ordered, ids)
    if 3 in args.experiments:
        exp3_conditional_distribution(ordered, ids)

    print(f"\n{'='*70}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
