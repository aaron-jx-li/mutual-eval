"""
robustness/greedy_selection.py

(--strategy random):  Forward greedy with random candidate subsampling.
(--strategy diverse):  Forward greedy with level-diversity-based batch selection.

Both experiments incrementally build a question set whose IRT-derived model ranking
converges to the full-dataset reference ranking, measured by Kendall's tau.

Mode is auto-detected from the combination of arguments passed:
  --static-jsonl only          → static mode (fit_static_irt)
  --arena-jsonl only           → arena mode (fit_reward_irt)
  --static-jsonl + arena-jsonl → joint-reward mode (fit_joint_reward_irt)
  --static-csv                 → CSV static mode

Usage
-----
# Random candidate subsampling (default: math JSONL)
python robustness/greedy_selection.py --strategy random \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2

# With a specific static JSONL file
python robustness/greedy_selection.py --strategy random \\
    --static-jsonl data/new/static_coding_v0.jsonl \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2

# With an arena reward JSONL file (mode auto-detected)
python robustness/greedy_selection.py --strategy random \\
    --arena-jsonl data/new/arena_math_v0.jsonl \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2

# Joint static + arena reward (both files → joint-reward mode)
python robustness/greedy_selection.py --strategy random \\
    --static-jsonl data/new/static_math_v0.jsonl \\
    --arena-jsonl data/new/arena_math_v0.jsonl \\
    --hard-cap 150 --candidates 50 --seeds 0 1 2

# Level-diversity batch greedy (static only)
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

import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ranking import fit_static_irt, fit_reward_irt, fit_joint_reward_irt
from robustness.data_utils import load_static, load_static_jsonl, load_arena_reward
from robustness.metrics import save_results, compute_all_metrics
from robustness.common_cli import base_parser
from robustness.reference_rankings import REFERENCE_RANKINGS, JOINT_REFERENCE_RANKINGS


def _file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _experiment_labels(static_csv: str | None, jsonl_path: str) -> tuple[str, str]:
    """Return (filetype, dataset) strings for folder naming."""
    if static_csv:
        return "static", "csv"
    stem = _file_stem(jsonl_path)
    parts = stem.split("_")
    return parts[0], (parts[1] if len(parts) >= 2 else stem)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _fit_metric(
    df: pd.DataFrame,
    ref_ranking: dict,
    num_epochs: int,
    metric: str = "kendall_tau",
    irt_func=None,
) -> float:
    """
    Fit IRT on df and return the chosen metric vs. ref_ranking.
    Always runs silently (verbose=False). Returns -2.0 on failure.
    """
    if irt_func is None:
        irt_func = fit_static_irt
    try:
        mp, _ = irt_func(df, num_epochs=num_epochs, verbose=False)
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
# Joint metric helper
# ---------------------------------------------------------------------------

def _fit_metric_joint(
    static_df: pd.DataFrame,
    arena_df: pd.DataFrame,
    ref_ranking: dict,
    num_epochs: int,
    metric: str = "kendall_tau",
) -> float:
    """Fit joint IRT on (static_df, arena_df) and return the chosen metric vs. ref_ranking."""
    try:
        mp, _ = fit_joint_reward_irt(static_df, arena_df, num_epochs=num_epochs, verbose=False)
        return compute_all_metrics(mp, ref_ranking)[metric]
    except Exception:
        return -2.0


# ---------------------------------------------------------------------------
# Strategy: random
# ---------------------------------------------------------------------------

def run_greedy_random(
    data_df: pd.DataFrame,
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
    irt_func=None,
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
    if irt_func is None:
        irt_func = fit_static_irt
    rng = np.random.default_rng(seed)
    all_qids = data_df["question_id"].unique()
    qid_level = _build_qid_level_map(data_df)

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
            trial_df = data_df[data_df["question_id"].isin(greedy_set | {q})]
            score = _fit_metric(trial_df, ref_ranking, num_epochs, metric, irt_func)
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
# Strategy: joint-reward (static + arena reward, unified candidate pool)
# ---------------------------------------------------------------------------

def run_greedy_joint_reward(
    static_df: pd.DataFrame,
    arena_df: pd.DataFrame,
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
) -> tuple[list[dict], set, set]:
    """
    Forward greedy with joint static+arena-reward IRT and unified candidate sampling.

    Two separate question pools (static and arena) are maintained. At each step,
    candidates_per_step // 2 candidates are drawn from each remaining pool and
    evaluated by fitting fit_joint_reward_irt on the current selected sets plus
    the candidate. The single best candidate across both pools is added to its
    respective pool.

    Initializes with 1 question from each pool so both DataFrames are non-empty
    for fit_joint_reward_irt from step 1 onward.

    hard_cap applies to the total number of selected questions across both pools.

    Returns
    -------
    (rows, static_selected, arena_selected)
    """
    rng = np.random.default_rng(seed)
    all_static_qids = static_df["question_id"].unique()
    all_arena_qids = arena_df["question_id"].unique()

    # Seed with 1 question from each pool
    start_static = rng.choice(all_static_qids)
    start_arena = rng.choice(all_arena_qids)
    static_selected: set = {start_static}
    arena_selected: set = {start_arena}
    static_remaining: set = set(all_static_qids) - static_selected
    arena_remaining: set = set(all_arena_qids) - arena_selected

    rows: list[dict] = []
    consecutive_above = 0
    step = 0

    while (len(static_selected) + len(arena_selected)) < hard_cap and (
        static_remaining or arena_remaining
    ):
        step += 1
        half = max(1, candidates_per_step // 2)

        best_q, best_source, best_score = None, None, -2.0

        # Evaluate static candidates
        if static_remaining:
            m = min(half, len(static_remaining))
            for q in rng.choice(list(static_remaining), size=m, replace=False):
                trial_static = static_df[static_df["question_id"].isin(static_selected | {q})]
                trial_arena = arena_df[arena_df["question_id"].isin(arena_selected)]
                score = _fit_metric_joint(trial_static, trial_arena, ref_ranking, num_epochs, metric)
                if score > best_score:
                    best_score, best_q, best_source = score, q, "static"

        # Evaluate arena candidates
        if arena_remaining:
            m = min(half, len(arena_remaining))
            for q in rng.choice(list(arena_remaining), size=m, replace=False):
                trial_static = static_df[static_df["question_id"].isin(static_selected)]
                trial_arena = arena_df[arena_df["question_id"].isin(arena_selected | {q})]
                score = _fit_metric_joint(trial_static, trial_arena, ref_ranking, num_epochs, metric)
                if score > best_score:
                    best_score, best_q, best_source = score, q, "arena"

        if best_q is None:
            break

        if best_source == "static":
            static_selected.add(best_q)
            static_remaining.discard(best_q)
        else:
            arena_selected.add(best_q)
            arena_remaining.discard(best_q)

        n_total = len(static_selected) + len(arena_selected)
        rows.append({
            "step":        step,
            "source":      best_source,
            "question_id": best_q,
            "score":       best_score,
            "n_questions": n_total,
            "n_static":    len(static_selected),
            "n_arena":     len(arena_selected),
            "seed":        seed,
            "strategy":    "random",
            "metric":      metric,
        })

        if not quiet:
            print(
                f"  [joint] step {step:3d} | added {best_q} (source={best_source}) "
                f"| {metric}={best_score:.4f} | n={n_total} "
                f"(static={len(static_selected)}, arena={len(arena_selected)})"
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

    return rows, static_selected, arena_selected


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
    parser.add_argument("--candidates", "-c", type=int, default=50, help="[random] Candidate questions sampled per step")
    parser.add_argument("--n-batches", "-n", type=int, default=10, help="[diverse] Diverse batches evaluated per step")
    parser.add_argument("--metric", "-m", choices=["spearman_rho", "kendall_tau"], default="kendall_tau", help="Metric used for greedy selection and convergence")
    parser.add_argument("--threshold", "-t", type=float, default=0.95, help="Convergence threshold for the chosen metric")
    parser.add_argument("--consecutive", type=int, default=3, help="Consecutive steps above threshold to declare convergence")
    # Lower epoch default for speed; user can override with --num-epochs
    # Override base_parser JSONL defaults to None so we can detect which was explicitly passed
    parser.set_defaults(num_epochs=500, seeds=[0, 1, 2], static_jsonl=None, arena_jsonl=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Auto-detect mode from which JSONL argument was explicitly passed
    # ------------------------------------------------------------------
    if args.static_csv:
        mode = "static"
        data_df = load_static(args.static_csv)
    elif args.static_jsonl is not None and args.arena_jsonl is not None:
        mode = "joint-reward"
        data_df = None  # not used in joint mode; static_df/arena_df set below
        static_df = load_static_jsonl(args.static_jsonl)
        arena_df = load_arena_reward(args.arena_jsonl)
    elif args.arena_jsonl is not None:
        mode = "arena"
        data_df = load_arena_reward(args.arena_jsonl)
    else:
        # static_jsonl explicitly passed, or neither (fall back to default static)
        mode = "static"
        jsonl_path = args.static_jsonl or "data/new/static_math_v0.jsonl"
        data_df = load_static_jsonl(jsonl_path)
        args.static_jsonl = jsonl_path

    irt_func = fit_reward_irt if mode == "arena" else fit_static_irt

    # Diverse strategy requires a level column absent from arena/joint data
    if args.strategy == "diverse" and mode in ("arena", "joint-reward"):
        parser.error("--strategy diverse requires a 'level' column and is not supported for arena or joint-reward data")

    # ------------------------------------------------------------------
    # Resolve reference ranking
    # ------------------------------------------------------------------
    if args.static_csv:
        # CSV mode: fit reference dynamically (no hardcoded dict for arbitrary CSVs)
        if not args.quiet:
            print("Fitting reference IRT on full dataset ...")
        ref_mp, _ = fit_static_irt(data_df, num_epochs=args.num_epochs, verbose=not args.quiet)
        ref_ranking: dict = {m.lower(): i + 1 for i, m in enumerate(ref_mp["model_name"])}
        if not args.quiet:
            print("Reference ranking:")
            for name, rank in sorted(ref_ranking.items(), key=lambda x: x[1]):
                print(f"  {rank:2d}. {name}")
            print()
    elif mode == "joint-reward":
        static_stem = _file_stem(args.static_jsonl)
        arena_stem = _file_stem(args.arena_jsonl)
        ref_ranking: dict = JOINT_REFERENCE_RANKINGS[(static_stem, arena_stem)]
        if not args.quiet:
            print(f"Using hardcoded joint reference ranking for ('{static_stem}', '{arena_stem}')")
    elif mode == "arena":
        stem = _file_stem(args.arena_jsonl)
        ref_ranking: dict = REFERENCE_RANKINGS[stem]
        if not args.quiet:
            print(f"Using hardcoded reference ranking for '{stem}'")
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
    if mode == "joint-reward":
        static_stem = _file_stem(args.static_jsonl)
        arena_stem = _file_stem(args.arena_jsonl)
        params = f"cap{args.hard_cap}_c{args.candidates}"
        folder_name = f"{args.strategy}_joint_{static_stem}+{arena_stem}_{params}_{timestamp}"
    else:
        if mode == "arena":
            filetype, dataset = _experiment_labels(None, args.arena_jsonl)
        else:
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
    if mode == "joint-reward":
        total_q = (
            len(static_df["question_id"].unique()) + len(arena_df["question_id"].unique())
        )
    else:
        total_q = len(data_df["question_id"].unique())
    seed_metric_rows: list[dict] = []
    t_total_start = time.perf_counter()

    for seed in args.seeds:
        if not args.quiet:
            print(f"=== Seed {seed} | strategy={args.strategy} | mode={mode} | metric={args.metric} ===")
        t_seed_start = time.perf_counter()

        if mode == "joint-reward":
            rows, static_set, arena_set = run_greedy_joint_reward(
                static_df,
                arena_df,
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
            final_set = static_set | arena_set
        elif args.strategy == "random":
            rows, final_set = run_greedy_random(
                data_df,
                ref_ranking,
                hard_cap=args.hard_cap,
                candidates_per_step=args.candidates,
                threshold=args.threshold,
                consecutive_needed=args.consecutive,
                seed=seed,
                num_epochs=args.num_epochs,
                quiet=args.quiet,
                metric=args.metric,
                irt_func=irt_func,
            )
        else:
            rows, final_set = run_greedy_diverse(
                data_df,
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

        t_seed_elapsed = time.perf_counter() - t_seed_start
        print(f"  Seed {seed} elapsed: {t_seed_elapsed:.1f}s")

        # Create per-seed subfolder and save steps, ranking, sampled_questions
        seed_folder = os.path.join(out_folder, f"seed_{seed}")
        os.makedirs(seed_folder, exist_ok=True)
        save_results(rows, seed_folder, "steps.csv")

        # Fit IRT on final greedy question set and compute metrics
        if mode == "joint-reward":
            final_static_df = static_df[static_df["question_id"].isin(static_set)]
            final_arena_df = arena_df[arena_df["question_id"].isin(arena_set)]
            final_mp, _ = fit_joint_reward_irt(
                final_static_df, final_arena_df, num_epochs=args.num_epochs, verbose=False
            )
        else:
            final_df = data_df[data_df["question_id"].isin(final_set)]
            final_mp, _ = irt_func(final_df, num_epochs=args.num_epochs, verbose=False)
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
        if mode == "joint-reward":
            pd.DataFrame({"question_id": sorted(static_set)}).to_csv(
                os.path.join(seed_folder, "sampled_static_questions.csv"), index=False
            )
            pd.DataFrame({"question_id": sorted(arena_set)}).to_csv(
                os.path.join(seed_folder, "sampled_arena_questions.csv"), index=False
            )
        else:
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
    print(f"  Total elapsed: {time.perf_counter() - t_total_start:.1f}s")


if __name__ == "__main__":
    main()
