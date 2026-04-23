#!/usr/bin/env python3
"""Terminal-Bench 2.0 agentic evaluation driver.

For every roster model and every ``(task_id, attempt)`` we shell out to
Harbor once with a slate of remaining tasks, then parse the reward.txt
files Harbor writes under its jobs directory.

Layout produced under ``output_dir``::

    <output_dir>/run_config.json
    <output_dir>/summary.csv
    <output_dir>/responses.jsonl       # all models: one row per (model, task_id, attempt)
    <output_dir>/<model_label>/
        jobs/<run_id>/<task_id>__<h>/result.json  # Harbor 0.4+
        harbor_logs/<run_id>.stdout
        harbor_logs/<run_id>.stderr

With ``--resume`` (or ``evaluation.resume: true`` in YAML), existing
``(model_label, task_id, attempt)`` rows in ``responses.jsonl`` are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from collections import defaultdict
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml
from tqdm import tqdm

from model_roster import (
    DEFAULT_ROSTER,
    MODEL_LOOKUP,
    load_env_file,
    resolve_agent_env_for_model,
    resolve_env_path,
    resolve_litellm_model_id,
    resolve_model_routes,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def load_ids_from_file(path: str | None, *, label: str) -> list[str]:
    """Load newline-delimited IDs from a text file."""
    if not path:
        return []
    raw = Path(path).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        # Also resolve relative to repository root so configs are robust even
        # when launched from eval_agentic/ instead of repo root.
        candidates.append(Path(__file__).resolve().parent.parent / raw)
    p = next((c.resolve() for c in candidates if c.exists()), raw.resolve())
    if not p.exists():
        raise SystemExit(f"{label} does not exist: {p}")
    out: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


_EXCLUDED_DIR_ENTRIES = {"LICENSE", "README.md"}


def discover_local_tasks(dataset_path: Path) -> list[str]:
    """Return task_ids for a Terminal-Bench 2.0 task directory on disk.

    A valid task dir contains a ``task.toml`` plus ``tests/`` and
    ``environment/`` subdirectories. We treat any immediate subdirectory
    with a ``task.toml`` as a task.
    """
    if not dataset_path.is_dir():
        raise SystemExit(f"dataset_path does not exist or is not a directory: {dataset_path}")
    tasks: list[str] = []
    for child in sorted(dataset_path.iterdir()):
        if not child.is_dir() or child.name in _EXCLUDED_DIR_ENTRIES:
            continue
        if (child / "task.toml").exists():
            tasks.append(child.name)
    return tasks


def select_task_ids(
    *,
    dataset_path: Path | None,
    task_ids: list[str],
    exclude_task_ids: list[str],
    max_tasks: int | None,
) -> list[str]:
    if task_ids:
        ids = list(dict.fromkeys(task_ids))
    else:
        if dataset_path is None:
            raise SystemExit(
                "task_ids is empty and dataset_path is not set; cannot enumerate tasks. "
                "Either set dataset_path to a local Terminal-Bench 2.0 clone or provide task_ids."
            )
        ids = discover_local_tasks(dataset_path)

    excluded = set(exclude_task_ids or [])
    ids = [t for t in ids if t not in excluded]

    if max_tasks is not None:
        if max_tasks < 0:
            raise SystemExit("max_tasks must be non-negative.")
        ids = ids[:max_tasks]

    if not ids:
        raise SystemExit("No tasks selected; refusing to run an empty evaluation.")
    return ids


# ---------------------------------------------------------------------------
# Env / route helpers
# ---------------------------------------------------------------------------


# Every provider API key Harbor may forward to terminus-2 via --ae. We emit
# only the subset that the per-model route requires so creds for other
# providers don't leak across calls.
_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_BASE_URL",
    "OPENROUTER_API_KEY",
    "LITELLM_API_KEY",
    "LITELLM_BASE_URL",
)


def _route_api_base(route_env: dict[str, str]) -> str | None:
    """Pick the ``api_base=`` hint for terminus-2 based on route env."""
    # terminus-2 talks to the OpenAI-compatible surface for most providers;
    # LiteLLM (gateway or client) picks the right upstream via model prefix.
    for key in ("LITELLM_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "GEMINI_BASE_URL"):
        if key in route_env:
            return route_env[key]
    return None


# ---------------------------------------------------------------------------
# Harbor invocation
# ---------------------------------------------------------------------------


def _sanitize_run_id(stub: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stub)[:120]


def _harbor_binary() -> str:
    harbor = shutil.which("harbor")
    if not harbor:
        raise SystemExit(
            "`harbor` CLI not found in PATH. Install it with "
            "`pip install harbor` (or `uv tool install harbor`) and retry."
        )
    return harbor


def build_harbor_cmd(
    *,
    dataset: str | None,
    dataset_path: Path | None,
    agent_name: str,
    routed_model: str,
    task_ids: list[str],
    n_concurrent: int,
    jobs_dir: Path,
    run_id: str,
    timeout_multiplier: float,
    agent_kwargs: list[str],
    agent_env: list[str],
    extra_args: list[str],
) -> list[str]:
    cmd = [_harbor_binary(), "run"]
    # Harbor 0.4+ (`harbor run` / `harbor job start`):
    #   - Local clone: ``--path`` to the dataset root (e.g. terminal-bench-2/).
    #   - Registry:    ``--dataset`` name@version (e.g. terminal-bench@2.0).
    # Task filter: ``--include-task-name`` per task (replaces legacy ``--task-id``).
    if dataset_path is not None:
        cmd.extend(["--path", str(dataset_path)])
    elif dataset:
        cmd.extend(["--dataset", dataset])
    cmd.extend(["--agent", agent_name])
    cmd.extend(["--model", routed_model])
    for tid in task_ids:
        cmd.extend(["--include-task-name", tid])
    cmd.extend([
        "--n-concurrent", str(n_concurrent),
        "--jobs-dir", str(jobs_dir),
        "--job-name", run_id,
        "--timeout-multiplier", str(timeout_multiplier),
    ])
    for kv in agent_kwargs:
        cmd.extend(["--ak", kv])
    for kv in agent_env:
        cmd.extend(["--ae", kv])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _parse_trial_result_json(path: Path) -> tuple[float, bool]:
    """Read Harbor 0.4+ per-trial ``result.json``; return (reward, passed)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return 0.0, False
    vr = data.get("verifier_result")
    if vr and isinstance(vr, dict):
        rewards_map = vr.get("rewards") or {}
        try:
            reward = float(rewards_map.get("reward", 0) or 0)
        except (TypeError, ValueError):
            reward = 0.0
        return reward, reward >= 1.0
    # Agent or verifier crashed before scoring (see ``exception_info``).
    return 0.0, False


def _task_trial_finished(job_dir: Path, task_id: str) -> bool:
    """True once Harbor has written a terminal artifact for this task trial."""
    if any(job_dir.glob(f"{task_id}__*/result.json")):
        return True
    if any(job_dir.glob(f"{task_id}__*/verifier/reward.txt")):
        return True
    if any(job_dir.glob(f"{task_id}__*/reward.txt")):
        return True
    return False


def _quick_pass_for_finished_task(job_dir: Path, task_id: str) -> bool:
    """Best-effort pass/fail for postfix; False if only errors/unparseable."""
    rjs = sorted(job_dir.glob(f"{task_id}__*/result.json"))
    if rjs:
        _, passed = _parse_trial_result_json(rjs[-1])
        return passed
    for pattern in (f"{task_id}__*/verifier/reward.txt", f"{task_id}__*/reward.txt"):
        for rf in sorted(job_dir.glob(pattern)):
            try:
                reward = float(rf.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError):
                continue
            return reward >= 1.0
    return False


def _pipe_stream_to_file(stream: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("wb") as f:
            shutil.copyfileobj(stream, f)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def run_harbor(
    *,
    cmd: list[str],
    log_dir: Path,
    run_id: str,
    timeout_s: float,
    job_dir: Path | None = None,
    pending_task_ids: list[str] | None = None,
    progress_bar: tqdm | None = None,
    on_task_finished: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run ``harbor``; optionally emit per-task completion callbacks."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{run_id}.stdout"
    stderr_path = log_dir / f"{run_id}.stderr"

    print(f"  [harbor] {shlex.join(cmd)}", flush=True)
    start = time.time()
    timed_out = False

    use_poll = (
        job_dir is not None
        and pending_task_ids is not None
        and len(pending_task_ids) > 0
        and (progress_bar is not None or on_task_finished is not None)
    )

    if not use_poll:
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
            stdout = proc.stdout or b""
            stderr = proc.stderr or b""
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            rc = -1
            timed_out = True
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
    else:
        assert job_dir is not None and pending_task_ids is not None
        rc = -1
        job_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tout = threading.Thread(
            target=_pipe_stream_to_file,
            args=(proc.stdout, stdout_path),
            daemon=True,
        )
        terr = threading.Thread(
            target=_pipe_stream_to_file,
            args=(proc.stderr, stderr_path),
            daemon=True,
        )
        tout.start()
        terr.start()

        completed: set[str] = set()
        deadline = time.monotonic() + timeout_s

        def sweep_finished(*, update_bar: bool) -> None:
            for tid in pending_task_ids:
                if tid in completed:
                    continue
                if not _task_trial_finished(job_dir, tid):
                    continue
                completed.add(tid)
                if on_task_finished is not None:
                    on_task_finished(tid)
                if update_bar and progress_bar is not None:
                    passed = _quick_pass_for_finished_task(job_dir, tid)
                    tag = "pass" if passed else "fail"
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f"last={tid[:30]} {tag}", refresh=False)

        try:
            while True:
                rc = proc.poll()
                sweep_finished(update_bar=True)
                if rc is not None:
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    proc.kill()
                    break
                time.sleep(0.35)
            if proc.poll() is None:
                proc.wait(timeout=30)
            rc = proc.returncode if proc.returncode is not None else -1
        finally:
            tout.join(timeout=120)
            terr.join(timeout=120)
            sweep_finished(update_bar=True)
            for tid in pending_task_ids:
                if tid not in completed:
                    completed.add(tid)
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(f"last={tid[:30]} n/a", refresh=False)

        stdout = stdout_path.read_bytes() if stdout_path.exists() else b""
        stderr = stderr_path.read_bytes() if stderr_path.exists() else b""

    elapsed = time.time() - start
    status = "timeout" if timed_out else f"exit={rc}"
    print(f"  [harbor] finished ({status}) in {elapsed:.1f}s", flush=True)
    if rc != 0 and not timed_out:
        tail = stderr[-600:].decode("utf-8", errors="replace")
        print(f"  [harbor] stderr tail: {tail}", flush=True)

    return {
        "returncode": rc,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 2),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


# ---------------------------------------------------------------------------
# Reward ingestion
# ---------------------------------------------------------------------------


def read_rewards(job_dir: Path, task_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Map task_id -> {reward, passed, reward_path} by scanning the jobs dir.

    Harbor 0.3-style: ``<trial>/verifier/reward.txt`` or ``<trial>/reward.txt``.

    Harbor 0.4+: ``<task_id>__<hash>/result.json`` with
    ``verifier_result.rewards.reward``.
    """
    results: dict[str, dict[str, Any]] = {}

    reward_files = sorted(job_dir.glob("*/verifier/reward.txt"))
    if not reward_files:
        reward_files = sorted(job_dir.glob("*/reward.txt"))

    reward_by_task: dict[str, tuple[Path, float]] = {}
    for rf in reward_files:
        dir_name = rf.parent.parent.name if rf.parent.name == "verifier" else rf.parent.name
        task_id = dir_name.split("__", 1)[0] if "__" in dir_name else dir_name
        try:
            reward = float(rf.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            reward = 0.0
        prev = reward_by_task.get(task_id)
        if prev is None or reward > prev[1]:
            reward_by_task[task_id] = (rf, reward)

    for tid in task_ids:
        hit = reward_by_task.get(tid)
        if hit is not None:
            rf, reward = hit
            results[tid] = {"reward": reward, "passed": reward >= 1.0, "reward_path": str(rf)}
            continue
        trial_results = sorted(job_dir.glob(f"{tid}__*/result.json"))
        if not trial_results:
            results[tid] = {"reward": 0.0, "passed": False, "reward_path": None}
            continue
        best_path = trial_results[0]
        best_reward, best_passed = _parse_trial_result_json(best_path)
        for rp in trial_results[1:]:
            r, p = _parse_trial_result_json(rp)
            if r > best_reward:
                best_reward, best_passed, best_path = r, p, rp
        results[tid] = {
            "reward": best_reward,
            "passed": best_passed,
            "reward_path": str(best_path),
        }
    return results


def read_reward_for_task(job_dir: Path, task_id: str) -> dict[str, Any]:
    """Best-effort reward snapshot for one finished task trial."""
    return read_rewards(job_dir, [task_id]).get(
        task_id,
        {"reward": 0.0, "passed": False, "reward_path": None},
    )


def recover_rows_from_existing_jobs(
    *,
    label: str,
    attempt: int,
    task_ids: list[str],
    jobs_root: Path,
    log_dir: Path,
    routed_model: str,
    route: str,
    agent_name: str,
    dataset: str | None,
    dataset_path: Path | None,
    existing: dict[str, dict[str, Any]],
    global_rows: dict[str, dict[str, Any]],
) -> int:
    """Backfill per-task rows from prior Harbor job dirs for resume."""
    recovered = 0
    run_dirs = sorted(jobs_root.glob(f"eval_agentic_tb__{label}__a{attempt}__*/"))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        candidates = [
            tid
            for tid in task_ids
            if _row_key(tid, attempt) not in existing and _task_trial_finished(run_dir, tid)
        ]
        if not candidates:
            continue
        rewards = read_rewards(run_dir, candidates)
        for tid in candidates:
            r = rewards.get(tid, {"reward": 0.0, "passed": False, "reward_path": None})
            row = {
                "task_id": tid,
                "attempt": attempt,
                "model_label": label,
                "model_routed": routed_model,
                "route": route,
                "agent": agent_name,
                "dataset": dataset,
                "dataset_path": str(dataset_path) if dataset_path else None,
                "passed": bool(r.get("passed", False)),
                "reward": float(r.get("reward", 0.0)),
                "reward_path": r.get("reward_path"),
                "run_id": run_dir.name,
                # Prior process metadata may not be recoverable if interrupted.
                "harbor_returncode": None,
                "harbor_timed_out": False,
                "harbor_elapsed_s": None,
                "stdout_path": str(log_dir / f"{run_dir.name}.stdout") if (log_dir / f"{run_dir.name}.stdout").exists() else None,
                "stderr_path": str(log_dir / f"{run_dir.name}.stderr") if (log_dir / f"{run_dir.name}.stderr").exists() else None,
                "job_dir": str(run_dir),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            sk = _trial_storage_key(label, tid, attempt)
            if sk in global_rows:
                continue
            global_rows[sk] = row
            existing[_row_key(tid, attempt)] = row
            recovered += 1
    return recovered


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def _row_key(task_id: str, attempt: int) -> str:
    return f"{task_id}::{attempt}"


def _trial_storage_key(model_label: str, task_id: str, attempt: int) -> str:
    """Stable key for the shared ``responses.jsonl`` (model × task × attempt)."""
    return f"{model_label}::{task_id}::{attempt}"


def load_global_responses(path: Path) -> dict[str, dict[str, Any]]:
    """Load ``responses.jsonl`` into a dict keyed by ``_trial_storage_key``."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = rec.get("model_label")
            tid = rec.get("task_id")
            att = rec.get("attempt")
            if label is None or tid is None or att is None:
                continue
            out[_trial_storage_key(str(label), str(tid), int(att))] = rec
    return out


def save_global_responses(path: Path, by_key: dict[str, dict[str, Any]]) -> None:
    """Write all rows to a single JSONL with deterministic ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        by_key.values(),
        key=lambda r: (
            str(r.get("model_label", "")),
            str(r.get("task_id", "")),
            int(r.get("attempt", 0)),
        ),
    )
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def global_rows_by_model(global_rows: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group shared JSONL rows by ``model_label`` for summary stats."""
    d: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in global_rows.values():
        d[str(row["model_label"])].append(row)
    for label in d:
        d[label].sort(key=lambda r: (str(r["task_id"]), int(r["attempt"])))
    return dict(d)


def write_summary_csv(
    output_dir: Path,
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    eval_runs: int,
    print_table: bool = False,
) -> list[dict[str, Any]]:
    """Write ``summary.csv``; optionally print the same table as ``summarize``."""
    summary: list[dict[str, Any]] = []
    for label, rows in sorted(rows_by_model.items()):
        by_task: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_task.setdefault(row["task_id"], []).append(row)

        attempts = len(rows)
        passes = sum(1 for r in rows if r.get("passed"))
        pass_at_1 = sum(1 for runs in by_task.values() if runs and runs[0].get("passed"))
        pass_at_k = sum(1 for runs in by_task.values() if any(r.get("passed") for r in runs))
        timeouts = sum(1 for r in rows if r.get("harbor_timed_out"))
        errors = sum(1 for r in rows if not r.get("harbor_timed_out") and (r.get("harbor_returncode") or 0) != 0)

        summary.append({
            "model_label": label,
            "num_tasks": len(by_task),
            "num_attempts": attempts,
            "num_passes": passes,
            "pass_rate": (passes / attempts) if attempts else None,
            "pass@1": (pass_at_1 / len(by_task)) if by_task else None,
            f"pass@{eval_runs}": (pass_at_k / len(by_task)) if by_task else None,
            "harbor_timeouts": timeouts,
            "harbor_errors": errors,
        })

    fieldnames = [
        "model_label",
        "num_tasks",
        "num_attempts",
        "num_passes",
        "pass_rate",
        "pass@1",
        f"pass@{eval_runs}",
        "harbor_timeouts",
        "harbor_errors",
    ]
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    if print_table:
        print("\nPass rate per model")
        print("-" * 84)
        for row in summary:
            pr = f"{row['pass_rate']:.2%}" if row["pass_rate"] is not None else "N/A"
            p1 = f"{row['pass@1']:.2%}" if row["pass@1"] is not None else "N/A"
            pk = f"{row[f'pass@{eval_runs}']:.2%}" if row[f"pass@{eval_runs}"] is not None else "N/A"
            print(
                f"{row['model_label']:28} attempts={row['num_attempts']:>4} "
                f"pass_rate={pr:>7} pass@1={p1:>7} pass@{eval_runs}={pk:>7} "
                f"timeouts={row['harbor_timeouts']} errors={row['harbor_errors']}"
            )
        print("-" * 84)
        print(f"Wrote {summary_path}")

    return summary


def apply_save_every_checkpoint(
    *,
    global_rows: dict[str, dict[str, Any]],
    shared_responses_path: Path,
    output_dir: Path,
    eval_runs: int,
    save_every: int,
    checkpoint_state: dict[str, int],
    n_new_rows: int,
    print_summary: bool = False,
) -> None:
    """Persist ``responses.jsonl`` and ``summary.csv`` per ``save_every`` new rows.

    * ``save_every <= 1``: checkpoint after every Harbor batch that adds rows.
    * ``save_every >= 2``: checkpoint each time at least ``save_every`` new rows
      have been appended since the last checkpoint (carries across models).
    * ``save_every == 0``: disable intermediate checkpoints (final save still done
      in ``main()``).
    """
    if save_every == 0:
        return
    if n_new_rows <= 0:
        return

    if save_every <= 1:
        save_global_responses(shared_responses_path, global_rows)
        write_summary_csv(
            output_dir,
            global_rows_by_model(global_rows),
            eval_runs=eval_runs,
            print_table=print_summary,
        )
        return

    checkpoint_state["since_last"] = checkpoint_state.get("since_last", 0) + n_new_rows
    while checkpoint_state["since_last"] >= save_every:
        save_global_responses(shared_responses_path, global_rows)
        write_summary_csv(
            output_dir,
            global_rows_by_model(global_rows),
            eval_runs=eval_runs,
            print_table=print_summary,
        )
        checkpoint_state["since_last"] -= save_every


def evaluate_model(
    *,
    label: str,
    model_output_dir: Path,
    task_ids: list[str],
    eval_runs: int,
    dataset: str | None,
    dataset_path: Path | None,
    agent_cfg: dict[str, Any],
    model_agent_kwargs_overrides: dict[str, dict[str, Any]],
    n_concurrent: int,
    timeout_multiplier: float,
    harbor_timeout_s: float,
    route: str,
    model_overrides: dict[str, str],
    global_rows: dict[str, dict[str, Any]],
    shared_responses_path: Path,
    output_dir: Path,
    save_every: int,
    checkpoint_state: dict[str, int],
    show_progress: bool = True,
    recover_from_jobs: bool = True,
) -> list[dict[str, Any]]:
    model_output_dir.mkdir(parents=True, exist_ok=True)
    jobs_root = model_output_dir / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    log_dir = model_output_dir / "harbor_logs"

    existing: dict[str, dict[str, Any]] = {}
    for _k, rec in global_rows.items():
        if rec.get("model_label") != label:
            continue
        existing[_row_key(str(rec["task_id"]), int(rec["attempt"]))] = rec

    routed_model = resolve_litellm_model_id(label, overrides=model_overrides, route=route)
    route_env = resolve_agent_env_for_model(label, route=route)
    api_base = _route_api_base(route_env)

    # When GEMINI_BASE_URL is the api_base, terminus-2 routes calls through an
    # OpenAI-compatible Gemini endpoint. LiteLLM's "gemini/" prefix triggers its
    # native Gemini/Vertex AI handler (which requires google-cloud-aiplatform and
    # is unrelated to the OpenAI-compat endpoint). Replace it with "openai/" so
    # LiteLLM uses its OpenAI handler pointed at GEMINI_BASE_URL, sending the
    # bare model name (e.g. "gemini-3.1-pro-preview") to the endpoint — exactly
    # how model_api_smoke_test.py calls the same endpoint via the OpenAI SDK.
    # Harbor's --ae flag sets env vars inside the agent's Docker container, NOT
    # in the LiteLLM client's process env, so we cannot override OPENAI_API_KEY
    # via --ae. Instead we pass api_key directly as a LiteLLM kwarg (--ak),
    # which flows through Harbor's LiteLLM wrapper into litellm.acompletion().
    gemini_via_openai_compat = (
        api_base
        and "GEMINI_BASE_URL" in route_env
        and routed_model.startswith("gemini/")
    )
    gemini_api_key_override: str | None = None
    if gemini_via_openai_compat:
        routed_model = "openai/" + routed_model[len("gemini/"):]
        gemini_api_key_override = route_env["GEMINI_API_KEY"]

    # Agent kwargs forwarded via --ak.
    # Per-model overrides let us tune settings like reasoning effort for a
    # single model label without affecting the rest of the roster.
    merged_agent_kwargs: dict[str, Any] = dict(agent_cfg.get("kwargs") or {})
    merged_agent_kwargs.update(model_agent_kwargs_overrides.get(label, {}))
    kwargs_flags: list[str] = []
    for key, val in merged_agent_kwargs.items():
        kwargs_flags.append(f"{key}={val}")
    if api_base:
        # terminus-2 points LiteLLM at this OpenAI-compatible endpoint.
        kwargs_flags.append(f"api_base={api_base}")
    if gemini_api_key_override is not None:
        # Force LiteLLM's OpenAI handler to authenticate with the Gemini key
        # instead of the OPENAI_API_KEY inherited from the eval's process env.
        # terminus-2 only forwards extra auth kwargs to LiteLLM via its
        # `llm_kwargs` parameter; `api_key` as a top-level --ak lands in the
        # agent base class and is ignored by LiteLLM. Nest it under llm_kwargs.
        kwargs_flags.append(
            f"llm_kwargs={json.dumps({'api_key': gemini_api_key_override})}"
        )

    # Agent env forwarded via --ae. Start with user-declared env, then layer
    # the per-model route creds -- they take precedence.
    env_map: dict[str, str] = {}
    for key, val in (agent_cfg.get("env") or {}).items():
        if str(key) in _PROVIDER_ENV_KEYS:
            # Skip user-declared provider creds so we don't leak the wrong
            # provider's key for this route; route_env below will set them.
            continue
        env_map[str(key)] = str(val)
    env_map.update(route_env)
    env_flags: list[str] = [f"{k}={v}" for k, v in env_map.items()]

    extra_args: list[str] = list(agent_cfg.get("extra_harbor_args") or [])
    agent_name = str(agent_cfg.get("name", "terminus-2"))

    trials_total = len(task_ids) * eval_runs
    trials_done_initial = sum(
        1 for r in global_rows.values() if r.get("model_label") == label
    )

    bar_cm: Any = (
        tqdm(
            total=trials_total,
            initial=min(trials_done_initial, trials_total),
            desc=f"{label}"[:48],
            unit="trial",
            dynamic_ncols=True,
            mininterval=0.4,
            file=sys.stderr,
        )
        if show_progress
        else contextlib.nullcontext()
    )

    with bar_cm as trial_bar:
        for attempt in range(eval_runs):
            if recover_from_jobs:
                recovered = recover_rows_from_existing_jobs(
                    label=label,
                    attempt=attempt,
                    task_ids=task_ids,
                    jobs_root=jobs_root,
                    log_dir=log_dir,
                    routed_model=routed_model,
                    route=route,
                    agent_name=agent_name,
                    dataset=dataset,
                    dataset_path=dataset_path,
                    existing=existing,
                    global_rows=global_rows,
                )
                if recovered > 0:
                    if trial_bar is not None:
                        trial_bar.update(recovered)
                    apply_save_every_checkpoint(
                        global_rows=global_rows,
                        shared_responses_path=shared_responses_path,
                        output_dir=output_dir,
                        eval_runs=eval_runs,
                        save_every=save_every,
                        checkpoint_state=checkpoint_state,
                        n_new_rows=recovered,
                        print_summary=False,
                    )
                    print(
                        f"[{label}] attempt={attempt}: recovered {recovered} completed task(s) from prior jobs.",
                        flush=True,
                    )
            pending = [tid for tid in task_ids if _row_key(tid, attempt) not in existing]
            if not pending:
                print(f"[{label}] attempt={attempt}: all {len(task_ids)} tasks cached; skipping.", flush=True)
                continue

            run_id = _sanitize_run_id(f"eval_agentic_tb__{label}__a{attempt}__{uuid.uuid4().hex[:6]}")
            job_dir = jobs_root / run_id
            print(
                f"[{label}] route={route} attempt={attempt} routed_model={routed_model} "
                f"pending={len(pending)}/{len(task_ids)} run_id={run_id}",
                flush=True,
            )

            cmd = build_harbor_cmd(
                dataset=dataset,
                dataset_path=dataset_path,
                agent_name=agent_name,
                routed_model=routed_model,
                task_ids=pending,
                n_concurrent=n_concurrent,
                jobs_dir=jobs_root,
                run_id=run_id,
                timeout_multiplier=timeout_multiplier,
                agent_kwargs=kwargs_flags,
                agent_env=env_flags,
                extra_args=extra_args,
            )
            progress_kw: dict[str, Any] = {}
            interim_written: set[str] = set()
            if save_every > 0:
                def _checkpoint_finished_task(tid: str) -> None:
                    r = read_reward_for_task(job_dir, tid)
                    row = {
                        "task_id": tid,
                        "attempt": attempt,
                        "model_label": label,
                        "model_routed": routed_model,
                        "route": route,
                        "agent": agent_name,
                        "dataset": dataset,
                        "dataset_path": str(dataset_path) if dataset_path else None,
                        "passed": bool(r.get("passed", False)),
                        "reward": float(r.get("reward", 0.0)),
                        "reward_path": r.get("reward_path"),
                        "run_id": run_id,
                        # Filled with provisional values while Harbor is still running.
                        "harbor_returncode": None,
                        "harbor_timed_out": False,
                        "harbor_elapsed_s": None,
                        "stdout_path": None,
                        "stderr_path": None,
                        "job_dir": str(job_dir),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                    sk = _trial_storage_key(label, tid, attempt)
                    first_write = sk not in global_rows
                    global_rows[sk] = row
                    existing[_row_key(tid, attempt)] = row
                    interim_written.add(tid)
                    if first_write:
                        apply_save_every_checkpoint(
                            global_rows=global_rows,
                            shared_responses_path=shared_responses_path,
                            output_dir=output_dir,
                            eval_runs=eval_runs,
                            save_every=save_every,
                            checkpoint_state=checkpoint_state,
                            n_new_rows=1,
                            print_summary=False,
                        )

                progress_kw["on_task_finished"] = _checkpoint_finished_task
            if show_progress and trial_bar is not None:
                progress_kw = {
                    "job_dir": job_dir,
                    "pending_task_ids": pending,
                    "progress_bar": trial_bar,
                    **progress_kw,
                }
            run_meta = run_harbor(
                cmd=cmd,
                log_dir=log_dir,
                run_id=run_id,
                timeout_s=harbor_timeout_s,
                **progress_kw,
            )

            rewards = read_rewards(job_dir, pending)
            for tid in pending:
                r = rewards.get(tid, {"reward": 0.0, "passed": False, "reward_path": None})
                row = {
                    "task_id": tid,
                    "attempt": attempt,
                    "model_label": label,
                    "model_routed": routed_model,
                    "route": route,
                    "agent": agent_name,
                    "dataset": dataset,
                    "dataset_path": str(dataset_path) if dataset_path else None,
                    "passed": bool(r.get("passed", False)),
                    "reward": float(r.get("reward", 0.0)),
                    "reward_path": r.get("reward_path"),
                    "run_id": run_id,
                    "harbor_returncode": run_meta["returncode"],
                    "harbor_timed_out": run_meta["timed_out"],
                    "harbor_elapsed_s": run_meta["elapsed_s"],
                    "stdout_path": run_meta["stdout_path"],
                    "stderr_path": run_meta["stderr_path"],
                    "job_dir": str(job_dir),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                sk = _trial_storage_key(label, tid, attempt)
                global_rows[sk] = row
                existing[_row_key(tid, attempt)] = row

            apply_save_every_checkpoint(
                global_rows=global_rows,
                shared_responses_path=shared_responses_path,
                output_dir=output_dir,
                eval_runs=eval_runs,
                save_every=save_every,
                checkpoint_state=checkpoint_state,
                n_new_rows=sum(1 for tid in pending if tid not in interim_written),
                print_summary=False,
            )

            n_pass = sum(1 for t in pending if rewards.get(t, {}).get("passed"))
            print(f"[{label}] attempt={attempt} passed {n_pass}/{len(pending)} newly graded.", flush=True)

    # Return rows in a deterministic order (task first, then attempt).
    return [existing[_row_key(t, a)] for t in task_ids for a in range(eval_runs) if _row_key(t, a) in existing]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(
    output_dir: Path,
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    eval_runs: int,
    num_tasks: int,
) -> list[dict[str, Any]]:
    del num_tasks  # reserved for API parity with callers
    return write_summary_csv(
        output_dir,
        rows_by_model,
        eval_runs=eval_runs,
        print_table=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Terminal-Bench 2.0 evaluation via Harbor + terminus-2.")
    p.add_argument("--config", default=None, help="Path to YAML config.")
    p.add_argument("--models", nargs="+", default=None, help="Subset of roster labels to run.")
    p.add_argument("--tasks", nargs="+", default=None, help="Specific task_ids to run.")
    p.add_argument("--max-tasks", type=int, default=None, help="Cap tasks (smoke-test).")
    p.add_argument("--eval-runs", type=int, default=None, help="Override attempts per (model, task).")
    p.add_argument("--output-dir", default=None, help="Override evaluation.output_dir.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output_dir>/responses.jsonl; skip (model, task, attempt) rows already present.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume: re-run all trials (ignore existing responses.jsonl).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm trial progress bar (for logs / non-interactive pipelines).",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Checkpoint responses.jsonl + summary.csv every N new rows (1=every Harbor batch; 0=only at end).",
    )
    p.add_argument(
        "--no-resume-recover-jobs",
        action="store_true",
        help="When resuming, do not recover cached rows from existing Harbor jobs/ artifacts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    config = load_yaml_config(args.config)
    section = config.get("evaluation", {}) or {}

    models = args.models or section.get("models") or list(DEFAULT_ROSTER)
    unknown = [m for m in models if m not in MODEL_LOOKUP]
    if unknown:
        raise SystemExit(
            f"Unknown model labels: {unknown}. Valid labels: {sorted(MODEL_LOOKUP)}"
        )

    dataset = section.get("dataset")
    dataset_path_str = section.get("dataset_path")
    dataset_path = Path(dataset_path_str).resolve() if dataset_path_str else None

    task_ids_file = section.get("task_ids_file")
    file_task_ids = load_ids_from_file(task_ids_file, label="task_ids_file")
    task_ids = args.tasks or file_task_ids or list(section.get("task_ids") or [])
    exclude_task_ids = list(section.get("exclude_task_ids") or [])
    max_tasks = args.max_tasks if args.max_tasks is not None else section.get("max_tasks")
    task_ids = select_task_ids(
        dataset_path=dataset_path,
        task_ids=task_ids,
        exclude_task_ids=exclude_task_ids,
        max_tasks=max_tasks,
    )

    eval_runs = int(args.eval_runs if args.eval_runs is not None else section.get("eval_runs", 3))
    if eval_runs <= 0:
        raise SystemExit("eval_runs must be >= 1.")
    n_concurrent = int(section.get("n_concurrent", 4))
    timeout_multiplier = float(section.get("global_timeout_multiplier", 1.0))
    harbor_timeout_s = float(section.get("harbor_subprocess_timeout_s", 3 * 3600))

    output_dir = Path(
        args.output_dir
        or section.get("output_dir")
        or f"results/agentic_eval/terminal_bench/{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = dict(section.get("agent") or {"name": "terminus-2"})
    model_agent_kwargs_overrides = {
        str(label): dict(kwargs)
        for label, kwargs in (section.get("model_agent_kwargs_overrides") or {}).items()
        if isinstance(kwargs, dict)
    }
    use_litellm = section.get("use_litellm")
    litellm_models = section.get("litellm_models")
    native_models = section.get("native_models")
    default_route = section.get("default_route")
    route_overrides = section.get("route_overrides")
    model_overrides = dict(section.get("model_overrides") or {})
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True
    else:
        resume = bool(section.get("resume", True))
    show_progress = not args.no_progress and bool(section.get("show_progress", True))
    resume_recover_jobs = bool(section.get("resume_recover_jobs", True))
    if args.no_resume_recover_jobs:
        resume_recover_jobs = False
    save_every = (
        args.save_every
        if args.save_every is not None
        else int(section.get("save_every", 1))
    )
    if save_every < 0:
        raise SystemExit("save_every must be >= 0 (use 0 for checkpoints only at the end of the run).")

    model_routes = resolve_model_routes(
        models,
        use_litellm=bool(use_litellm) if use_litellm is not None else None,
        litellm_models=litellm_models,
        native_models=native_models,
        default_route=str(default_route) if default_route is not None else None,
        route_overrides=route_overrides,
    )

    run_config = {
        "models": models,
        "dataset": dataset,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "task_ids_file": task_ids_file,
        "num_tasks": len(task_ids),
        "tasks_sample": task_ids[:10],
        "eval_runs": eval_runs,
        "n_concurrent": n_concurrent,
        "timeout_multiplier": timeout_multiplier,
        "output_dir": str(output_dir),
        "agent": agent_cfg,
        "model_agent_kwargs_overrides": model_agent_kwargs_overrides,
        "use_litellm": use_litellm,
        "litellm_models": litellm_models or [],
        "native_models": native_models or [],
        "default_route": default_route,
        "route_overrides": route_overrides,
        "model_routes": model_routes,
        "resume": resume,
        "resume_recover_jobs": resume_recover_jobs,
        "save_every": save_every,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    run_config["responses_jsonl"] = str(output_dir / "responses.jsonl")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    shared_responses_path = output_dir / "responses.jsonl"
    global_rows: dict[str, dict[str, Any]] = {}
    if resume:
        global_rows = load_global_responses(shared_responses_path)

    checkpoint_state: dict[str, int] = {"since_last": 0}

    for label in models:
        model_output_dir = output_dir / label
        evaluate_model(
            label=label,
            model_output_dir=model_output_dir,
            task_ids=task_ids,
            eval_runs=eval_runs,
            dataset=dataset,
            dataset_path=dataset_path,
            agent_cfg=agent_cfg,
            model_agent_kwargs_overrides=model_agent_kwargs_overrides,
            n_concurrent=n_concurrent,
            timeout_multiplier=timeout_multiplier,
            harbor_timeout_s=harbor_timeout_s,
            route=model_routes[label],
            model_overrides=model_overrides,
            global_rows=global_rows,
            shared_responses_path=shared_responses_path,
            output_dir=output_dir,
            save_every=save_every,
            checkpoint_state=checkpoint_state,
            show_progress=show_progress,
            recover_from_jobs=resume and resume_recover_jobs,
        )

    save_global_responses(shared_responses_path, global_rows)
    summarize(
        output_dir,
        global_rows_by_model(global_rows),
        eval_runs=eval_runs,
        num_tasks=len(task_ids),
    )


if __name__ == "__main__":
    main()
