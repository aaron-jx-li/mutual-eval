#!/usr/bin/env python3
"""Analyze human-vs-judge consistency in one Arena-style JSON file."""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_LABELS = ("model_a", "model_b", "tie", "both_bad")
VALID_LABELS_SET = set(VALID_LABELS)
DECISIVE_LABELS = {"model_a", "model_b"}
NON_DECISIVE_LABELS = {"tie", "both_bad"}
MISSING = "<missing>"


@dataclass(frozen=True)
class Record:
    row_id: str
    human_label: str | None
    judge_label: str | None


@dataclass
class Dataset:
    path: Path
    rows: dict[str, Record]
    invalid_items: int
    missing_id_items: int
    duplicate_id_items: int
    missing_judge_key_items: int
    invalid_human_label_items: int
    invalid_judge_label_items: int


def normalize_label(value: Any) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, str):
        label = value.strip()
        if label == "":
            return None, False
        if label in VALID_LABELS_SET:
            return label, False
    return None, True


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def load_dataset(path: Path, id_key: str, human_key: str, judge_key: str) -> Dataset:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"{path} must be a JSON list, got {type(obj).__name__}")

    rows: dict[str, Record] = {}
    invalid_items = 0
    missing_id_items = 0
    duplicate_id_items = 0
    missing_judge_key_items = 0
    invalid_human_label_items = 0
    invalid_judge_label_items = 0

    for item in obj:
        if not isinstance(item, dict):
            invalid_items += 1
            continue

        raw_id = item.get(id_key)
        if raw_id is None or (isinstance(raw_id, str) and raw_id.strip() == ""):
            missing_id_items += 1
            continue
        row_id = str(raw_id)

        if row_id in rows:
            duplicate_id_items += 1
            continue

        human_label, human_invalid = normalize_label(item.get(human_key))
        if human_invalid:
            invalid_human_label_items += 1

        if judge_key in item:
            judge_label, judge_invalid = normalize_label(item.get(judge_key))
            if judge_invalid:
                invalid_judge_label_items += 1
        else:
            missing_judge_key_items += 1
            judge_label = None

        rows[row_id] = Record(
            row_id=row_id,
            human_label=human_label,
            judge_label=judge_label,
        )

    return Dataset(
        path=path,
        rows=rows,
        invalid_items=invalid_items,
        missing_id_items=missing_id_items,
        duplicate_id_items=duplicate_id_items,
        missing_judge_key_items=missing_judge_key_items,
        invalid_human_label_items=invalid_human_label_items,
        invalid_judge_label_items=invalid_judge_label_items,
    )


def counter_str(counter: Counter[str], total: int) -> str:
    labels = list(VALID_LABELS) + [MISSING]
    parts: list[str] = []
    for label in labels:
        count = counter.get(label, 0)
        if count == 0:
            continue
        parts.append(f"{label}={count} ({pct(count, total):.1f}%)")
    return ", ".join(parts) if parts else "n/a"


def classify_human_judge_inconsistency(human_label: str, judge_label: str) -> str:
    if human_label == judge_label:
        return "consistent"
    if {human_label, judge_label} == {"model_a", "model_b"}:
        return "decisive_flip_a_vs_b"
    if (human_label in DECISIVE_LABELS and judge_label in NON_DECISIVE_LABELS) or (
        judge_label in DECISIVE_LABELS and human_label in NON_DECISIVE_LABELS
    ):
        return "decisive_vs_tie_or_both_bad"
    if {human_label, judge_label} == {"tie", "both_bad"}:
        return "tie_vs_both_bad"
    return "other_labeled_mismatch"


def build_confusion_matrix(comparable_rows: list[Record]) -> list[list[int]]:
    label_to_idx = {label: idx for idx, label in enumerate(VALID_LABELS)}
    matrix = [[0 for _ in VALID_LABELS] for _ in VALID_LABELS]
    for row in comparable_rows:
        assert row.human_label is not None
        assert row.judge_label is not None
        i = label_to_idx[row.human_label]
        j = label_to_idx[row.judge_label]
        matrix[i][j] += 1
    return matrix


def save_confusion_heatmap(
    matrix: list[list[int]],
    output_path: Path,
    title: str | None = None,
) -> None:
    counts = np.array(matrix, dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_pct = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

    annot = np.empty_like(counts, dtype=object)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            annot[i, j] = f"{int(counts[i, j])}\n{row_pct[i, j] * 100:.1f}%"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        counts,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        xticklabels=VALID_LABELS,
        yticklabels=VALID_LABELS,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Judge label")
    plt.ylabel("Human label")
    plt.title(title or "Human vs Judge 4x4 Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def compute_metrics(ds: Dataset) -> dict[str, Any]:
    rows = list(ds.rows.values())
    n = len(rows)

    human_counts = Counter(r.human_label if r.human_label is not None else MISSING for r in rows)
    judge_counts = Counter(r.judge_label if r.judge_label is not None else MISSING for r in rows)

    comparable_rows = [
        r for r in rows if r.human_label in VALID_LABELS_SET and r.judge_label in VALID_LABELS_SET
    ]
    comparable_n = len(comparable_rows)
    excluded_n = n - comparable_n

    comparable_human_counts = Counter(r.human_label for r in comparable_rows)
    comparable_judge_counts = Counter(r.judge_label for r in comparable_rows)
    confusion_matrix = build_confusion_matrix(comparable_rows)

    legacy_subset = [
        r
        for r in comparable_rows
        if r.human_label in DECISIVE_LABELS and r.judge_label in DECISIVE_LABELS
    ]
    legacy_agree = sum(1 for r in legacy_subset if r.human_label == r.judge_label)
    full_agree = sum(1 for r in comparable_rows if r.human_label == r.judge_label)

    human_decisive = [r for r in comparable_rows if r.human_label in DECISIVE_LABELS]
    judge_decisive_on_human_decisive = [
        r for r in human_decisive if r.judge_label in DECISIVE_LABELS
    ]

    disagreements = [r for r in comparable_rows if r.human_label != r.judge_label]
    mismatch_counter = Counter(
        (r.human_label if r.human_label is not None else MISSING, r.judge_label if r.judge_label is not None else MISSING)
        for r in disagreements
    )
    inconsistency_counter = Counter(
        classify_human_judge_inconsistency(r.human_label, r.judge_label)
        for r in disagreements
        if r.human_label is not None and r.judge_label is not None
    )

    return {
        "n": n,
        "comparable_n": comparable_n,
        "excluded_n": excluded_n,
        "human_counts": human_counts,
        "judge_counts": judge_counts,
        "comparable_human_counts": comparable_human_counts,
        "comparable_judge_counts": comparable_judge_counts,
        "confusion_matrix": confusion_matrix,
        "judge_coverage_count": n - judge_counts.get(MISSING, 0),
        "legacy_subset_n": len(legacy_subset),
        "legacy_agree_n": legacy_agree,
        "full_subset_n": comparable_n,
        "full_agree_n": full_agree,
        "human_decisive_n": len(human_decisive),
        "judge_decisive_on_human_decisive_n": len(judge_decisive_on_human_decisive),
        "disagreements_n": len(disagreements),
        "mismatch_counter": mismatch_counter,
        "inconsistency_counter": inconsistency_counter,
    }


def render_report(ds: Dataset, metrics: dict[str, Any], top_k: int) -> str:
    lines: list[str] = []

    lines.append("# Arena Human-vs-Judge Consistency Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- File: {ds.path.name}")
    lines.append(f"- Rows loaded: {metrics['n']}")
    lines.append(
        "- Rows used for comparison (both human and judge labels present): "
        f"{metrics['comparable_n']}/{metrics['n']} "
        f"({pct(metrics['comparable_n'], metrics['n']):.1f}%)"
    )
    lines.append(f"- Rows excluded from comparison: {metrics['excluded_n']}")

    if ds.invalid_items or ds.missing_id_items or ds.duplicate_id_items:
        lines.append(
            "- Row quality:"
            f" invalid_non_dict={ds.invalid_items},"
            f" missing_id={ds.missing_id_items},"
            f" duplicate_id_skipped={ds.duplicate_id_items}"
        )
    if ds.missing_judge_key_items or ds.invalid_human_label_items or ds.invalid_judge_label_items:
        lines.append(
            "- Label quality:"
            f" missing_judge_key={ds.missing_judge_key_items},"
            f" invalid_human_label={ds.invalid_human_label_items},"
            f" invalid_judge_label={ds.invalid_judge_label_items}"
        )

    lines.append(
        f"- Human label distribution (all rows): {counter_str(metrics['human_counts'], metrics['n'])}"
    )
    lines.append(
        f"- Judge label distribution (all rows): {counter_str(metrics['judge_counts'], metrics['n'])}"
    )
    lines.append(
        f"- Judge coverage: {metrics['judge_coverage_count']}/{metrics['n']} "
        f"({pct(metrics['judge_coverage_count'], metrics['n']):.1f}%)"
    )
    lines.append(
        "- Human label distribution (comparison subset): "
        f"{counter_str(metrics['comparable_human_counts'], metrics['comparable_n'])}"
    )
    lines.append(
        "- Judge label distribution (comparison subset): "
        f"{counter_str(metrics['comparable_judge_counts'], metrics['comparable_n'])}"
    )
    lines.append("")

    lines.append("## Agreement Metrics")
    lines.append(
        "- Legacy decisive agreement (human/model_a|model_b vs judge/model_a|model_b): "
        f"{metrics['legacy_agree_n']}/{metrics['legacy_subset_n']} "
        f"({pct(metrics['legacy_agree_n'], metrics['legacy_subset_n']):.1f}%)"
    )
    lines.append(
        "- Full-label agreement (all non-missing human and judge labels): "
        f"{metrics['full_agree_n']}/{metrics['full_subset_n']} "
        f"({pct(metrics['full_agree_n'], metrics['full_subset_n']):.1f}%)"
    )
    lines.append(
        "- Decisive judge coverage on human-decisive items: "
        f"{metrics['judge_decisive_on_human_decisive_n']}/{metrics['human_decisive_n']} "
        f"({pct(metrics['judge_decisive_on_human_decisive_n'], metrics['human_decisive_n']):.1f}%)"
    )
    lines.append("")

    lines.append("## Typical Forms of Inconsistency (Human vs Judge)")
    lines.append(
        "- Total disagreements on comparison subset: "
        f"{metrics['disagreements_n']}/{metrics['full_subset_n']} "
        f"({pct(metrics['disagreements_n'], metrics['full_subset_n']):.1f}%)"
    )
    for key in [
        "decisive_flip_a_vs_b",
        "decisive_vs_tie_or_both_bad",
        "tie_vs_both_bad",
        "other_labeled_mismatch",
    ]:
        count = metrics["inconsistency_counter"].get(key, 0)
        lines.append(
            f"- {key}: {count} "
            f"({pct(count, metrics['disagreements_n']):.1f}% of comparison disagreements)"
        )

    top_mismatches = metrics["mismatch_counter"].most_common(top_k)
    if top_mismatches:
        formatted = ", ".join(f"{h}->{j}: {count}" for (h, j), count in top_mismatches)
        lines.append(f"- Top labeled mismatches (human->judge): {formatted}")
    lines.append("")

    lines.append("## Confusion Matrix")
    lines.append("- 4x4 layout: rows are human labels, columns are judge labels.")
    lines.append("- Label order: model_a, model_b, tie, both_bad.")
    lines.append("")

    lines.append("## Interpretation Note")
    lines.append(
        "- With one judge source and no external gold signal, this report quantifies disagreement patterns only; "
        "it does not prove whether human or judge labels are universally better."
    )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one Arena-style JSON file with human_label and judge_label, "
            "reporting agreement and inconsistency statistics."
        )
    )
    parser.add_argument("file", type=Path, help="Path to JSON file")
    parser.add_argument("--id-key", default="id", help="Record id key (default: id)")
    parser.add_argument(
        "--human-key", default="human_label", help="Human label key (default: human_label)"
    )
    parser.add_argument(
        "--judge-key", default="judge_label", help="Judge label key (default: judge_label)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top mismatch transitions to show (default: 10)",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path to save the markdown report",
    )
    parser.add_argument(
        "--save-heatmap",
        type=Path,
        default=None,
        help=(
            "Optional path to save a seaborn 4x4 confusion matrix heatmap "
            "(rows=human, cols=judge)."
        ),
    )
    parser.add_argument(
        "--heatmap-title",
        default=None,
        help="Optional custom title for the heatmap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.file, args.id_key, args.human_key, args.judge_key)
    metrics = compute_metrics(ds)
    report = render_report(ds, metrics, top_k=args.top_k)
    print(report, end="")

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(report, encoding="utf-8")
        print(f"Saved report to {args.save_report}")

    if args.save_heatmap is not None:
        title = args.heatmap_title or f"Human vs Judge Confusion: {ds.path.name}"
        save_confusion_heatmap(metrics["confusion_matrix"], args.save_heatmap, title=title)
        print(f"Saved heatmap to {args.save_heatmap}")


if __name__ == "__main__":
    main()
