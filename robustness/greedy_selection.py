"""
robustness/greedy_selection.py

(--strategy random):  Forward greedy with random candidate subsampling.
(--strategy diverse):  Forward greedy with level-diversity-based batch selection.

Both experiments incrementally build a question set whose IRT-derived model ranking
converges to the full-dataset reference ranking, measured by Kendall's tau.

Usage
-----
# Random candidate subsampling (default: math JSONL)
python robustness/greedy_selection.py --strategy random \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2

# With a specific JSONL file
python robustness/greedy_selection.py --strategy random \\
    --static-jsonl data/new/static_coding_v0.jsonl \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2

# Level-diversity batch greedy
python robustness/greedy_selection.py --strategy diverse \\
    --hard-cap 150 --n-batches 10 --seeds 0 1 2

# Old CSV mode still works
python robustness/greedy_selection.py --strategy random \\
    --static-csv data/static_10_models.csv \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ranking import fit_static_irt
from robustness.data_utils import load_static, load_static_jsonl
from robustness.metrics import save_results, compute_all_metrics
from robustness.common_cli import base_parser
from robustness.reference_rankings import REFERENCE_RANKINGS


def _file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _experiment_labels(static_csv: str | None, static_jsonl: str) -> tuple[str, str]:
    """Return (filetype, dataset) strings for folder naming."""
    if static_csv:
        return "static", "csv"
    stem = _file_stem(static_jsonl)
    parts = stem.split("_")
    return parts[0], (parts[1] if len(parts) >= 2 else stem)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _fit_metric(df: pd.DataFrame, ref_ranking: dict, num_epochs: int, metric: str = "kendall_tau") -> float:
    """
    Fit IRT on df and return the chosen metric vs. ref_ranking.
    Always runs silently (verbose=False). Returns -2.0 on failure.
    """
    try:
        mp, _ = fit_static_irt(df, num_epochs=num_epochs, verbose=False)
        return compute_all_metrics(mp, ref_ranking)[metric]
    except Exception:
        return -2.0


# ---------------------------------------------------------------------------
# Level-group helpers
# ---------------------------------------------------------------------------

def _norm_level(v) -> object:
    """Normalise a level value: NaN / empty string -> None (GSM-8k), else keep as-is."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if str(v).strip() == "":
        return None
    return v


def _build_level_groups(static_df: pd.DataFrame) -> dict[object, list]:
    """
    Return {level_key: [question_id, ...]} where level_key is None for GSM-8k
    and an int/str (1–5) for Hendrycks MATH levels.
    """
    q_level = static_df[["question_id", "level"]].drop_duplicates("question_id")
    groups: dict = {}
    for _, row in q_level.iterrows():
        key = _norm_level(row["level"])
        groups.setdefault(key, []).append(row["question_id"])
    return groups


def _build_qid_level_map(static_df: pd.DataFrame) -> dict:
    """Return {question_id: normalised_level} for printable labels. Returns empty dict if no level column."""
    if "level" not in static_df.columns:
        return {}
    q_level = static_df[["question_id", "level"]].drop_duplicates("question_id")
    return {row["question_id"]: _norm_level(row["level"]) for _, row in q_level.iterrows()}


def _level_label(v) -> str:
    """Human-readable level label."""
    return "gsm" if v is None else str(v)


# ---------------------------------------------------------------------------
# Strategy: random
# ---------------------------------------------------------------------------

def run_greedy_random(
    static_df: pd.DataFrame,
    ref_ranking: dict,
    *,
    hard_cap: int,
    candidates_per_step: int,
    threshold: float,
    consecutive_needed: int,
    seed: int,
    num_epochs: int,
    quiet: bool,
    metric: str = "kendall_tau",
) -> list[dict]:
    """
    Forward greedy with random candidate subsampling.

    At each step:
      1. Sample `candidates_per_step` questions at random from the remaining pool.
      2. Evaluate each by fitting IRT on (current set ∪ {candidate}).
      3. Add the candidate that maximises Kendall's tau.

    Stops when tau >= tau_threshold for `consecutive_needed` consecutive steps
    or when `hard_cap` questions have been added.

    Returns
    -------
    (rows, greedy_set) : tuple[list[dict], set]
    """
    rng = np.random.default_rng(seed)
    all_qids = static_df["question_id"].unique()
    qid_level = _build_qid_level_map(static_df)

    start_qid = rng.choice(all_qids)
    greedy_set: set = {start_qid}
    remaining: set = set(all_qids) - greedy_set

    rows: list[dict] = []
    consecutive_above = 0
    step = 0

    while len(greedy_set) < hard_cap and remaining:
        step += 1
        m = min(candidates_per_step, len(remaining))
        candidates = rng.choice(list(remaining), size=m, replace=False)

        best_q, best_score = None, -2.0
        for q in candidates:
            trial_df = static_df[static_df["question_id"].isin(greedy_set | {q})]
            score = _fit_metric(trial_df, ref_ranking, num_epochs, metric)
            if score > best_score:
                best_score, best_q = score, q

        greedy_set.add(best_q)
        remaining.discard(best_q)

        lvl = _level_label(qid_level.get(best_q))
        rows.append({
            "step":        step,
            "question_id": best_q,
            "level":       lvl,
            "score":       best_score,
            "n_questions": len(greedy_set),
            "seed":        seed,
            "strategy":    "random",
            "metric":      metric,
        })

        if not quiet:
            print(
                f"  [random] step {step:3d} | added {best_q} (level={lvl}) "
                f"| {metric}={best_score:.4f} | n={len(greedy_set)}"
            )

        if best_score >= threshold:
            consecutive_above += 1
            if consecutive_above >= consecutive_needed:
                if not quiet:
                    print(
                        f"  Converged: {metric} >= {threshold} for "
                        f"{consecutive_needed} consecutive steps."
                    )
                break
        else:
            consecutive_above = 0

    return rows, greedy_set


# ---------------------------------------------------------------------------
# Strategy: diverse
# ---------------------------------------------------------------------------

def run_greedy_diverse(
    static_df: pd.DataFrame,
    ref_ranking: dict,
    *,
    hard_cap: int,
    n_batches: int,
    threshold: float,
    consecutive_needed: int,
    seed: int,
    num_epochs: int,
    quiet: bool,
    metric: str = "kendall_tau",
) -> list[dict]:
    """
    Forward greedy with level-diversity-based batch selection.

    At each step:
      1. Build `n_batches` diverse candidate batches by sampling 1 question
         from each non-empty level group (GSM-8k + Levels 1–5) not yet in the set.
      2. Evaluate each batch by fitting IRT on (current set ∪ batch).
      3. Add all questions from the best-scoring batch.

    Enforces difficulty diversity by construction; each step adds up to 6 questions.
    Stops when tau >= tau_threshold for `consecutive_needed` consecutive steps
    or when `hard_cap` questions have been added.

    Returns
    -------
    (rows, greedy_set) : tuple[list[dict], set]
    """
    rng = np.random.default_rng(seed)
    all_qids = static_df["question_id"].unique()
    level_groups = _build_level_groups(static_df)

    start_qid = rng.choice(all_qids)
    greedy_set: set = {start_qid}

    rows: list[dict] = []
    consecutive_above = 0
    step = 0

    while len(greedy_set) < hard_cap:
        step += 1

        # Build n_batches diverse candidate batches
        batches: list[list] = []
        for _ in range(n_batches):
            batch = []
            for qids in level_groups.values():
                available = [q for q in qids if q not in greedy_set]
                if available:
                    batch.append(rng.choice(available))
            if batch:
                batches.append(batch)

        if not batches:
            break  # all questions exhausted

        best_batch, best_score = None, -2.0
        for batch in batches:
            trial_df = static_df[static_df["question_id"].isin(greedy_set | set(batch))]
            score = _fit_metric(trial_df, ref_ranking, num_epochs, metric)
            if score > best_score:
                best_score, best_batch = score, batch

        greedy_set.update(best_batch)

        # Summarise level composition of the added batch
        level_comp: dict[str, int] = {}
        for q in best_batch:
            lbl = _level_label(
                next((k for k, v in level_groups.items() if q in v), None)
            )
            level_comp[lbl] = level_comp.get(lbl, 0) + 1

        rows.append({
            "step":              step,
            "questions_added":   len(best_batch),
            "n_questions":       len(greedy_set),
            "score":             best_score,
            "level_composition": str(level_comp),
            "seed":              seed,
            "strategy":          "diverse",
            "metric":            metric,
        })

        if not quiet:
            print(
                f"  [diverse] step {step:3d} | added {len(best_batch)} questions "
                f"({level_comp}) | {metric}={best_score:.4f} | n={len(greedy_set)}"
            )

        if best_score >= threshold:
            consecutive_above += 1
            if consecutive_above >= consecutive_needed:
                if not quiet:
                    print(
                        f"  Converged: {metric} >= {threshold} for "
                        f"{consecutive_needed} consecutive steps."
                    )
                break
        else:
            consecutive_above = 0

    return rows, greedy_set


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = base_parser("Greedy question selection")
    parser.add_argument("--strategy", choices=["random", "diverse"], default="random", help="random: random candidate subsampling; diverse: level-diversity batch greedy")
    parser.add_argument("--hard-cap", type=int, default=150, help="Maximum total questions in the selected set")
    parser.add_argument("--candidates", type=int, default=50, help="[random] Candidate questions sampled per step")
    parser.add_argument("--n-batches", type=int, default=10, help="[diverse] Diverse batches evaluated per step")
    parser.add_argument("--metric", choices=["spearman_rho", "kendall_tau"], default="kendall_tau", help="Metric used for greedy selection and convergence")
    parser.add_argument("--threshold", type=float, default=0.95, help="Convergence threshold for the chosen metric")
    parser.add_argument("--consecutive", type=int, default=3, help="Consecutive steps above threshold to declare convergence")
    # Lower epoch default for speed; user can override with --num-epochs
    parser.set_defaults(num_epochs=500, seeds=[0, 1, 2])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if args.static_csv:
        static_df = load_static(args.static_csv)
    else:
        static_df = load_static_jsonl(args.static_jsonl)

    # ------------------------------------------------------------------
    # Resolve reference ranking
    # ------------------------------------------------------------------
    if args.static_csv:
        # CSV mode: fit reference dynamically (no hardcoded dict for arbitrary CSVs)
        if not args.quiet:
            print("Fitting reference IRT on full dataset ...")
        ref_mp, _ = fit_static_irt(static_df, num_epochs=args.num_epochs, verbose=not args.quiet)
        ref_ranking: dict = {m.lower(): i + 1 for i, m in enumerate(ref_mp["model_name"])}
        if not args.quiet:
            print("Reference ranking:")
            for name, rank in sorted(ref_ranking.items(), key=lambda x: x[1]):
                print(f"  {rank:2d}. {name}")
            print()
    else:
        # JSONL mode: use hardcoded reference ranking
        stem = _file_stem(args.static_jsonl)
        ref_ranking: dict = REFERENCE_RANKINGS[stem]
        if not args.quiet:
            print(f"Using hardcoded reference ranking for '{stem}'")

    # ------------------------------------------------------------------
    # Set up output folder before seed loop (needed for per-seed steps files)
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filetype, dataset = _experiment_labels(args.static_csv, args.static_jsonl)
    if args.strategy == "random":
        params = f"cap{args.hard_cap}_c{args.candidates}"
    else:
        params = f"cap{args.hard_cap}_b{args.n_batches}"
    folder_name = f"{args.strategy}_{filetype}_{dataset}_{params}_{timestamp}"
    out_folder = os.path.join(args.out_dir, "greedy", folder_name)
    os.makedirs(out_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Run greedy for each seed
    # ------------------------------------------------------------------
    total_q = len(static_df["question_id"].unique())
    seed_metric_rows: list[dict] = []

    for seed in args.seeds:
        if not args.quiet:
            print(f"=== Seed {seed} | strategy={args.strategy} | metric={args.metric} ===")

        if args.strategy == "random":
            rows, final_set = run_greedy_random(
                static_df,
                ref_ranking,
                hard_cap=args.hard_cap,
                candidates_per_step=args.candidates,
                threshold=args.threshold,
                consecutive_needed=args.consecutive,
                seed=seed,
                num_epochs=args.num_epochs,
                quiet=args.quiet,
                metric=args.metric,
            )
        else:
            rows, final_set = run_greedy_diverse(
                static_df,
                ref_ranking,
                hard_cap=args.hard_cap,
                n_batches=args.n_batches,
                threshold=args.threshold,
                consecutive_needed=args.consecutive,
                seed=seed,
                num_epochs=args.num_epochs,
                quiet=args.quiet,
                metric=args.metric,
            )

        # Create per-seed subfolder and save steps, ranking, sampled_questions
        seed_folder = os.path.join(out_folder, f"seed_{seed}")
        os.makedirs(seed_folder, exist_ok=True)
        save_results(rows, seed_folder, "steps.csv")

        # Fit IRT on final greedy question set and compute metrics
        final_df = static_df[static_df["question_id"].isin(final_set)]
        final_mp, _ = fit_static_irt(final_df, num_epochs=args.num_epochs, verbose=False)
        metrics = compute_all_metrics(final_mp, ref_ranking)
        frac = len(final_set) / total_q
        seed_metric_rows.append({"seed": seed, "fraction_of_data": frac, **metrics})
        if not args.quiet:
            print(
                f"  Final set ({len(final_set)} questions): "
                f"ρ={metrics['spearman_rho']:.3f}  τ={metrics['kendall_tau']:.3f}  "
                f"top3={metrics['top3_acc']:.2f}  top5={metrics['top5_acc']:.2f}  "
                f"exact={metrics['exact_matches']}"
            )

        rank_df = final_mp.copy().reset_index(drop=True)
        rank_df.insert(0, "rank", range(1, len(rank_df) + 1))
        rank_df.to_csv(os.path.join(seed_folder, "ranking.csv"), index=False)
        pd.DataFrame({"question_id": sorted(final_set)}).to_csv(
            os.path.join(seed_folder, "sampled_questions.csv"), index=False
        )
        print(f"  Saved: {seed_folder}/")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    numeric_cols = ["spearman_rho", "kendall_tau", "top3_acc", "top5_acc", "exact_matches"]
    metrics_df = pd.DataFrame(seed_metric_rows)
    if not args.quiet:
        avg_frac = metrics_df["fraction_of_data"].mean()
        avg_row = {"seed": "avg", "fraction_of_data": avg_frac,
                   **{c: metrics_df[c].mean() for c in numeric_cols}}
        display_df = pd.concat(
            [metrics_df, pd.DataFrame([avg_row])], ignore_index=True
        )
        display_df["fraction_of_data"] = display_df["fraction_of_data"].apply(
            lambda x: f"{x:.3g}"
        )
        display_df = display_df.set_index("seed")
        print("\n=== Summary ===")
        print(display_df.to_string(float_format=lambda x: f"{x:.3f}"))
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # metrics.csv — per-seed IRT metrics + aggregate
    avg_frac = metrics_df["fraction_of_data"].mean()
    mean_row = {"seed": "avg", "fraction_of_data": avg_frac, **{c: metrics_df[c].mean() for c in numeric_cols}}
    summary = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)
    summary.to_csv(os.path.join(out_folder, "metrics.csv"), index=False)
    print(f"  Saved: {out_folder}/metrics.csv")


if __name__ == "__main__":
    main()
