#!/usr/bin/env python3
"""SWE-bench Lite agentic evaluation driver.

For each roster model:
  1. Run ``mini-swe-agent`` v2 against SWE-bench Lite
       python -m minisweagent.run.utilities.mini_extra swebench \
           --subset lite --split test \
           [--filter <regex>] \
           --model <routed_model> \
           --output <per_model>/agent_run \
           --workers <N>
     which emits <per_model>/agent_run/preds.json.
  2. Convert preds.json -> predictions.jsonl for the grading harness.
  3. Grade with ``swebench.harness.run_evaluation`` (from the local clone).
  4. Write per-instance results to responses.jsonl and update summary.csv.

Everything is resumable: a model whose final responses.jsonl already lists
every requested instance is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from datasets import load_dataset

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
# Environment preflight checks
# ---------------------------------------------------------------------------


def _is_rootless_docker() -> bool:
    """Return True when docker is running in rootless mode."""
    try:
        out = subprocess.check_output(
            ["docker", "info", "--format", "{{json .SecurityOptions}}"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return False
    return "rootless" in out.lower()


def _subid_range_contains(path: str, user: str, uid_or_gid: int) -> bool:
    """Return True if any subordinate id range for ``user`` contains id."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) != 3 or parts[0] != user:
                    continue
                start = int(parts[1])
                length = int(parts[2])
                if start <= uid_or_gid < start + length:
                    return True
    except Exception:
        # If we cannot inspect the mapping files, stay conservative and skip the warning.
        return True
    return False


def warn_rootless_matplotlib_subid_risk(instance_ids: list[str]) -> None:
    """Warn early when rootless Docker is likely to fail matplotlib images."""
    if not any(iid.startswith("matplotlib__matplotlib-") for iid in instance_ids):
        return
    if not _is_rootless_docker():
        return

    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if not user:
        return

    # Observed ownership in SWE-bench matplotlib layers that can fail rootless extraction.
    needs_uid = 197609
    needs_gid = 197121
    uid_ok = _subid_range_contains("/etc/subuid", user, needs_uid)
    gid_ok = _subid_range_contains("/etc/subgid", user, needs_gid)
    if uid_ok and gid_ok:
        return

    print(
        (
            "WARNING: rootless Docker + matplotlib SWE-bench images may fail to start.\n"
            f"  Missing subordinate ID coverage for uid={needs_uid} gid={needs_gid} "
            f"(user={user}).\n"
            "  Typical symptom: docker run exits 125 with 'failed to register layer ... "
            "try increasing the number of subordinate IDs'.\n"
            "  Fix options:\n"
            "    1) add/expand this user's ranges in /etc/subuid and /etc/subgid,\n"
            "    2) run on a host/user with sufficient ranges,\n"
            "    3) use a non-rootless Docker daemon if available."
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Config loading
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


def load_ids_from_file(path: str | None) -> list[str]:
    """Load newline-delimited ids, ignoring blank/comment lines."""
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
        raise SystemExit(f"instance_ids_file does not exist: {p}")
    ids: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


# ---------------------------------------------------------------------------
# Instance selection
# ---------------------------------------------------------------------------


def load_instance_ids(
    *,
    dataset_name: str,
    dataset_split: str,
    explicit_ids: list[str],
    max_instances: int | None,
) -> list[str]:
    if explicit_ids:
        ids = list(dict.fromkeys(explicit_ids))
    else:
        print(f"Loading {dataset_name} [{dataset_split}] ...", flush=True)
        ds = load_dataset(dataset_name, split=dataset_split)
        ids = [row["instance_id"] for row in ds]
    if max_instances is not None:
        if max_instances < 0:
            raise SystemExit("max_instances must be non-negative.")
        ids = ids[:max_instances]
    if not ids:
        raise SystemExit("No instance_ids selected; refusing to run an empty evaluation.")
    return ids


# ---------------------------------------------------------------------------
# mini-swe-agent v2 subprocess wrapper
# ---------------------------------------------------------------------------


_DATASET_TO_SUBSET = {
    "princeton-nlp/SWE-bench_Lite": "lite",
    "princeton-nlp/SWE-bench": "full",
    "princeton-nlp/SWE-bench_Verified": "verified",
    "princeton-nlp/SWE-bench_Multimodal": "multimodal",
    "swe-bench/SWE-Bench_Multilingual": "multilingual",
    "SWE-bench/SWE-smith": "smith",
}

# mini-swe-agent CLI does not currently accept these keys as top-level flags.
# Keep them in config for future compatibility, but avoid hard failures now.
_UNSUPPORTED_MINI_SWE_AGENT_ARGS = {"reasoning_effort"}


def run_mini_swe_agent(
    *,
    agent_cfg: dict[str, Any],
    dataset_name: str,
    dataset_split: str,
    instance_ids: list[str],
    routed_model: str,
    output_dir: Path,
    env_overrides: Mapping[str, str] | None = None,
    replace_env: bool = False,
) -> None:
    """Invoke mini-swe-agent v2 swebench command as a subprocess.

    When ``replace_env`` is ``True``, the subprocess inherits exactly
    ``env_overrides`` (already built by the caller, typically via
    ``build_model_subprocess_env``). Otherwise ``env_overrides`` is merged
    on top of ``os.environ``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = _DATASET_TO_SUBSET.get(dataset_name, dataset_name)
    module = str(agent_cfg.get("module", "minisweagent.run.utilities.mini_extra"))
    subcommand = str(agent_cfg.get("subcommand", "swebench"))
    workers = str(agent_cfg.get("workers", 1))

    filter_pattern = "^(?:" + "|".join(re.escape(i) for i in instance_ids) + ")$"

    cmd = [
        sys.executable, "-m", module, subcommand,
        "--subset", subset,
        "--split", dataset_split,
        "--filter", filter_pattern,
        "--model", routed_model,
        "--output", str(output_dir),
        "--workers", workers,
    ]
    extra_args_cfg = agent_cfg.get("extra_args") or {}
    if extra_args_cfg:
        # mini-swe-agent v2 stops loading its default swebench config as soon as
        # any --config is provided; re-add it explicitly before overrides.
        cmd.extend(["--config", "swebench.yaml"])

    for key, val in extra_args_cfg.items():
        key_str = str(key)
        if key_str in _UNSUPPORTED_MINI_SWE_AGENT_ARGS:
            print(
                f"  [agent] WARNING: mini-swe-agent does not support --{key_str.replace('_', '-')}; "
                f"ignoring {key_str}={val!r}.",
                flush=True,
            )
            continue
        # mini-swe-agent v2 CLI expects custom settings as --config key=value.
        # Passing unknown --flag args causes an argparse exit(2) before any
        # predictions are produced.
        cmd.extend(["--config", f"{key_str}={val}"])

    if replace_env:
        env = dict(env_overrides or {})
    else:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

    # mini-swe-agent v2.2.8 may still raise on LiteLLM model-price lookups for
    # unmapped models even when --config cost_tracking=ignore_errors is passed.
    # Mirror this setting into the documented env var as a reliable fallback.
    cost_tracking = extra_args_cfg.get("cost_tracking")
    if cost_tracking is not None:
        env["MSWEA_COST_TRACKING"] = str(cost_tracking)

    print(f"  [agent] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start
    print(f"  [agent] exit={result.returncode} in {elapsed:.1f}s", flush=True)
    if result.returncode != 0:
        preds_json = output_dir / "preds.json"
        preds_jsonl = output_dir / "preds-trial.jsonl"
        # Fail fast when the runner cannot start (e.g. wrong Python env missing
        # mini-swe-agent). Continuing in this case silently writes empty rows
        # for every instance, which looks like a model failure instead of an
        # environment/setup issue.
        if not preds_json.exists() and not preds_jsonl.exists():
            raise RuntimeError(
                "mini-swe-agent exited non-zero before producing predictions. "
                f"python={sys.executable!r} returncode={result.returncode}. "
                f"command={' '.join(cmd)!r}. "
                "Common causes: missing mini-swe-agent in this interpreter, or "
                "CLI arg/version mismatch (exit code 2 is often argument parsing)."
            )
        print(
            f"  [agent] WARNING: mini-swe-agent exited non-zero (code={result.returncode}); "
            "will still try to ingest any predictions it produced.",
            flush=True,
        )


def parse_mini_swe_agent_preds(agent_output_dir: Path, instance_ids: list[str]) -> dict[str, str]:
    """Return instance_id -> model_patch, defaulting missing to empty string."""
    preds_json = agent_output_dir / "preds.json"
    preds_jsonl = agent_output_dir / "preds-trial.jsonl"
    patches: dict[str, str] = {iid: "" for iid in instance_ids}

    if preds_json.exists():
        try:
            data = json.loads(preds_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for iid, rec in data.items():
                    if not isinstance(rec, dict) or iid not in patches:
                        continue
                    patches[iid] = rec.get("model_patch") or rec.get("prediction") or rec.get("patch") or ""
        except json.JSONDecodeError:
            print(f"  [agent] Could not parse {preds_json}", flush=True)
    elif preds_jsonl.exists():
        with preds_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = rec.get("instance_id") or rec.get("id") or ""
                if iid not in patches:
                    continue
                patches[iid] = rec.get("model_patch") or rec.get("prediction") or rec.get("patch") or ""
    else:
        print(
            f"  [agent] WARNING: no preds.json or preds-trial.jsonl in {agent_output_dir}; "
            "every instance will be graded with an empty patch.",
            flush=True,
        )
    return patches


# ---------------------------------------------------------------------------
# SWE-bench grading harness
# ---------------------------------------------------------------------------


def write_predictions_jsonl(
    path: Path,
    *,
    instance_to_patch: dict[str, str],
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for iid, patch in instance_to_patch.items():
            f.write(json.dumps({
                "instance_id": iid,
                "model_patch": patch,
                "model_name_or_path": model_name,
            }) + "\n")


def run_swebench_grader(
    *,
    grading_cfg: dict[str, Any],
    swebench_path: Path,
    dataset_name: str,
    dataset_split: str,
    predictions_path: Path,
    instance_ids: list[str],
    run_id: str,
    report_dir: Path,
    model_name: str,
    on_instance_graded: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Call swebench.harness.run_evaluation and return per-instance results.

    Returns a map ``instance_id -> {resolved, completed, error}``.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    if str(swebench_path) and str(swebench_path) not in sys.path:
        sys.path.insert(0, str(swebench_path))
    try:
        from swebench.harness.run_evaluation import main as run_eval_main  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import swebench.harness.run_evaluation from {swebench_path}. "
            "Either pip install the local clone (`pip install -e <swebench_path>`) "
            "or ensure it is on PYTHONPATH."
        ) from exc

    print(f"  [grader] running harness on {len(instance_ids)} predictions (run_id={run_id})", flush=True)
    run_eval_kwargs = dict(
        dataset_name=dataset_name,
        split=dataset_split,
        instance_ids=list(instance_ids),
        predictions_path=str(predictions_path),
        max_workers=int(grading_cfg.get("max_workers", 4)),
        force_rebuild=bool(grading_cfg.get("force_rebuild", False)),
        cache_level=str(grading_cfg.get("cache_level", "env")),
        clean=bool(grading_cfg.get("clean", False)),
        open_file_limit=int(grading_cfg.get("open_file_limit", 4096)),
        run_id=run_id,
        timeout=int(grading_cfg.get("timeout", 1800)),
        namespace=grading_cfg.get("namespace"),
        rewrite_reports=False,
        modal=False,
        instance_image_tag=str(grading_cfg.get("instance_image_tag", "latest")),
        env_image_tag=str(grading_cfg.get("env_image_tag", "latest")),
        report_dir=str(report_dir),
    )

    aggregate: Any = None
    if on_instance_graded is None:
        try:
            aggregate = run_eval_main(**run_eval_kwargs)
        except Exception as exc:
            print(f"  [grader] harness raised: {exc}", flush=True)
            return {iid: {"resolved": False, "completed": False, "error": True, "error_msg": str(exc)} for iid in instance_ids}
    else:
        done = threading.Event()
        state: dict[str, Any] = {"aggregate": None, "error": None}
        pending = set(instance_ids)

        def _runner() -> None:
            try:
                state["aggregate"] = run_eval_main(**run_eval_kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                state["error"] = exc
            finally:
                done.set()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        while not done.is_set():
            if pending:
                partial = _fallback_per_instance_reports(report_dir=report_dir, instance_ids=list(pending))
                for iid in list(pending):
                    rec = partial.get(iid, {})
                    if rec.get("error_msg") == "report not found":
                        continue
                    on_instance_graded(iid, rec)
                    pending.remove(iid)
            time.sleep(1.0)
        t.join(timeout=1.0)

        if state["error"] is not None:
            exc = state["error"]
            print(f"  [grader] harness raised: {exc}", flush=True)
            fail = {iid: {"resolved": False, "completed": False, "error": True, "error_msg": str(exc)} for iid in instance_ids}
            for iid in list(pending):
                on_instance_graded(iid, fail[iid])
            return fail

        aggregate = state["aggregate"]

    if isinstance(aggregate, dict) and {"resolved_ids", "completed_ids", "error_ids"} <= aggregate.keys():
        resolved = set(aggregate.get("resolved_ids") or [])
        completed = set(aggregate.get("completed_ids") or [])
        errored = set(aggregate.get("error_ids") or [])
        out = {
            iid: {"resolved": iid in resolved, "completed": iid in completed, "error": iid in errored}
            for iid in instance_ids
        }
        if on_instance_graded is not None:
            for iid, rec in out.items():
                on_instance_graded(iid, rec)
        return out

    aggregate_file = _discover_aggregate_report(
        run_id=run_id,
        model_name=model_name,
        run_dir=predictions_path.parent,
        report_dir=report_dir,
    )
    if aggregate_file is not None:
        data = json.loads(aggregate_file.read_text(encoding="utf-8"))
        resolved = set(data.get("resolved_ids") or [])
        completed = set(data.get("completed_ids") or [])
        errored = set(data.get("error_ids") or [])
        out = {
            iid: {"resolved": iid in resolved, "completed": iid in completed, "error": iid in errored}
            for iid in instance_ids
        }
        if on_instance_graded is not None:
            for iid, rec in out.items():
                on_instance_graded(iid, rec)
        return out

    out = _fallback_per_instance_reports(
        report_dir=report_dir,
        instance_ids=instance_ids,
    )
    if on_instance_graded is not None:
        for iid, rec in out.items():
            on_instance_graded(iid, rec)
    return out


def _discover_aggregate_report(
    *,
    run_id: str,
    model_name: str,
    run_dir: Path,
    report_dir: Path,
) -> Path | None:
    candidates: list[Path] = [
        run_dir / f"{run_id}.aggregate.json",
        report_dir / f"{run_id}.aggregate.json",
        run_dir / f"{model_name}.{run_id}.json",
        Path.cwd() / f"{model_name}.{run_id}.json",
        report_dir / f"{model_name}.{run_id}.json",
    ]
    candidates += list(run_dir.glob(f"*.{run_id}.json"))
    candidates += list(report_dir.glob(f"*.{run_id}.json"))
    candidates += list(Path.cwd().glob(f"*.{run_id}.json"))
    seen: set[Path] = set()
    for cpath in candidates:
        if cpath in seen or not cpath.exists():
            continue
        seen.add(cpath)
        try:
            data = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and {"resolved_ids", "completed_ids", "error_ids"} <= data.keys():
            return cpath
    return None


def _fallback_per_instance_reports(
    *,
    report_dir: Path,
    instance_ids: list[str],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for iid in instance_ids:
        matches = list(report_dir.rglob(f"{iid}*report.json")) or list(report_dir.rglob(f"**/{iid}/*.json"))
        if not matches:
            results[iid] = {"resolved": False, "completed": False, "error": True, "error_msg": "report not found"}
            continue
        try:
            data = json.loads(matches[0].read_text(encoding="utf-8"))
        except Exception as exc:
            results[iid] = {"resolved": False, "completed": False, "error": True, "error_msg": f"bad report json: {exc}"}
            continue
        if {"resolved_ids", "completed_ids", "error_ids"} <= data.keys():
            results[iid] = {
                "resolved": iid in data.get("resolved_ids", []),
                "completed": iid in data.get("completed_ids", []),
                "error": iid in data.get("error_ids", []),
            }
        else:
            results[iid] = {
                "resolved": bool(data.get("resolved", False)),
                "completed": bool(data.get("completed", False)),
                "error": bool(data.get("error", False)),
            }
    return results


# ---------------------------------------------------------------------------
# Env routing (per-model)
# ---------------------------------------------------------------------------


_LITELLM_ENV_KEYS = ("LITELLM_API_KEY", "LITELLM_BASE_URL")
_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_BASE_URL",
    "OPENROUTER_API_KEY",
)
_ROUTE_SCRUB_KEYS = set(_LITELLM_ENV_KEYS) | set(_PROVIDER_ENV_KEYS)


def build_model_subprocess_env(base_env: Mapping[str, str], overrides: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``base_env`` with only the route-specific creds set.

    This wipes all known model-auth env vars (so a stale OPENAI_BASE_URL
    pointing at LiteLLM can't leak into a native Anthropic call) and then
    applies ``overrides``.
    """
    env = dict(base_env)
    for key in _ROUTE_SCRUB_KEYS:
        env.pop(key, None)
    env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------


def _shared_row_key(model_label: str, instance_id: str) -> str:
    return f"{model_label}::{instance_id}"


def load_global_responses(path: Path) -> dict[str, dict[str, Any]]:
    """Load shared responses.jsonl keyed by model+instance."""
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
            iid = rec.get("instance_id")
            if label is None or iid is None:
                continue
            out[_shared_row_key(str(label), str(iid))] = rec
    return out


def save_global_responses(path: Path, by_key: dict[str, dict[str, Any]]) -> None:
    """Write shared responses.jsonl in deterministic model/instance order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        by_key.values(),
        key=lambda r: (
            str(r.get("model_label", "")),
            str(r.get("instance_id", "")),
        ),
    )
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def global_rows_by_model(global_rows: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in global_rows.values():
        label = str(row.get("model_label", ""))
        out.setdefault(label, []).append(row)
    for label, rows in out.items():
        rows.sort(key=lambda r: str(r.get("instance_id", "")))
    return out


def evaluate_model(
    *,
    label: str,
    agent_cfg: dict[str, Any],
    model_agent_extra_args_overrides: dict[str, dict[str, Any]],
    grading_cfg: dict[str, Any],
    swebench_path: Path,
    dataset_name: str,
    dataset_split: str,
    instance_ids: list[str],
    model_output_dir: Path,
    route: str,
    model_overrides: dict[str, str],
    resume: bool,
    resume_recover_artifacts: bool,
    save_every: int,
    base_env: Mapping[str, str],
    shared_responses_path: Path,
    global_rows: dict[str, dict[str, Any]],
    on_checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    model_output_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    for rec in global_rows.values():
        if rec.get("model_label") != label:
            continue
        iid = rec.get("instance_id")
        if iid is None:
            continue
        existing[str(iid)] = rec

    def _persist_responses() -> None:
        ordered_rows = [existing[iid] for iid in instance_ids if iid in existing]
        for row in ordered_rows:
            global_rows[_shared_row_key(label, str(row["instance_id"]))] = row
        save_global_responses(shared_responses_path, global_rows)
        if on_checkpoint is not None:
            on_checkpoint(ordered_rows)

    routed_model = resolve_litellm_model_id(label, overrides=model_overrides, route=route)

    if resume and resume_recover_artifacts:
        recover_ids = [iid for iid in instance_ids if iid not in existing]
        if recover_ids:
            agent_output_dir = model_output_dir / "agent_run"
            report_dir = model_output_dir / "reports"
            recovered = 0
            if report_dir.exists():
                patches = parse_mini_swe_agent_preds(agent_output_dir, recover_ids)
                graded = _fallback_per_instance_reports(report_dir=report_dir, instance_ids=recover_ids)
                for iid in recover_ids:
                    g = graded.get(iid, {})
                    if g.get("error_msg") == "report not found":
                        continue
                    row = {
                        "instance_id": iid,
                        "model_label": label,
                        "model_routed": routed_model,
                        "dataset_name": dataset_name,
                        "dataset_split": dataset_split,
                        "model_patch": patches.get(iid, ""),
                        "patch_non_empty": bool((patches.get(iid) or "").strip()),
                        "resolved": bool(g.get("resolved", False)),
                        "completed": bool(g.get("completed", False)),
                        "error": bool(g.get("error", False)),
                        "error_msg": g.get("error_msg"),
                        "agent_elapsed_s": None,
                        "run_id": "recovered_from_artifacts",
                        "agent_output_dir": str(agent_output_dir),
                        "report_dir": str(report_dir),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                    existing[iid] = row
                    recovered += 1
            if recovered > 0:
                _persist_responses()
                print(
                    f"[{label}] recovered {recovered} cached instance(s) from agent/reports artifacts.",
                    flush=True,
                )

    pending_ids = [iid for iid in instance_ids if iid not in existing]
    if not pending_ids:
        print(f"[{label}] all {len(instance_ids)} instances already evaluated; skipping.", flush=True)
        return [existing[iid] for iid in instance_ids if iid in existing]

    route_env = resolve_agent_env_for_model(label, route=route)
    subprocess_env = build_model_subprocess_env(base_env, route_env)
    merged_agent_extra_args = dict(agent_cfg.get("extra_args") or {})
    merged_agent_extra_args.update(model_agent_extra_args_overrides.get(label, {}))
    agent_cfg_effective = dict(agent_cfg)
    agent_cfg_effective["extra_args"] = merged_agent_extra_args
    print(
        f"[{label}] route={route} routed_model={routed_model} "
        f"pending={len(pending_ids)}/{len(instance_ids)}",
        flush=True,
    )

    agent_output_dir = model_output_dir / "agent_run"
    started = time.time()
    run_mini_swe_agent(
        agent_cfg=agent_cfg_effective,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        instance_ids=pending_ids,
        routed_model=routed_model,
        output_dir=agent_output_dir,
        env_overrides=subprocess_env,
        replace_env=True,
    )
    agent_elapsed = time.time() - started

    patches = parse_mini_swe_agent_preds(agent_output_dir, pending_ids)
    non_empty = sum(1 for p in patches.values() if (p or "").strip())
    print(f"[{label}] agent produced {non_empty}/{len(pending_ids)} non-empty patches", flush=True)

    model_name = f"eval_agentic__{label}"
    run_id = f"agentic_swebl__{label.replace('/', '_')}__{int(time.time())}"
    predictions_path = model_output_dir / "predictions.jsonl"
    write_predictions_jsonl(predictions_path, instance_to_patch=patches, model_name=model_name)

    report_dir = model_output_dir / "reports"
    new_rows: list[dict[str, Any]] = []
    since_last_save = 0

    def _upsert_instance_row(iid: str, g: dict[str, Any]) -> None:
        nonlocal since_last_save
        row = {
            "instance_id": iid,
            "model_label": label,
            "model_routed": routed_model,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "model_patch": patches.get(iid, ""),
            "patch_non_empty": bool((patches.get(iid) or "").strip()),
            "resolved": bool(g.get("resolved", False)),
            "completed": bool(g.get("completed", False)),
            "error": bool(g.get("error", False)),
            "error_msg": g.get("error_msg"),
            "agent_elapsed_s": round(agent_elapsed, 2),
            "run_id": run_id,
            "agent_output_dir": str(agent_output_dir),
            "report_dir": str(report_dir),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        first_write = iid not in existing
        existing[iid] = row
        if save_every != 0 and first_write:
            since_last_save += 1
            if save_every <= 1 or since_last_save >= save_every:
                _persist_responses()
                since_last_save = 0

    def _on_instance_graded(iid: str, g: dict[str, Any]) -> None:
        if iid not in pending_ids:
            return
        _upsert_instance_row(iid, g)

    graded = run_swebench_grader(
        grading_cfg=grading_cfg,
        swebench_path=swebench_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        predictions_path=predictions_path,
        instance_ids=pending_ids,
        run_id=run_id,
        report_dir=report_dir,
        model_name=model_name,
        on_instance_graded=_on_instance_graded,
    )

    for iid in pending_ids:
        g = graded.get(iid, {"resolved": False, "completed": False, "error": True, "error_msg": "missing result"})
        _upsert_instance_row(iid, g)
        new_rows.append(existing[iid])

    _persist_responses()

    n_res = sum(1 for r in new_rows if r["resolved"])
    print(f"[{label}] resolved {n_res}/{len(new_rows)} newly graded instances.", flush=True)
    return [existing[iid] for iid in instance_ids if iid in existing]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(
    output_dir: Path,
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    print_table: bool = True,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for label, rows in sorted(rows_by_model.items()):
        n = len(rows)
        n_resolved = sum(1 for r in rows if r.get("resolved"))
        n_completed = sum(1 for r in rows if r.get("completed"))
        n_errors = sum(1 for r in rows if r.get("error"))
        n_non_empty = sum(1 for r in rows if r.get("patch_non_empty"))
        summary.append({
            "model_label": label,
            "num_instances": n,
            "num_non_empty_patches": n_non_empty,
            "num_completed": n_completed,
            "num_resolved": n_resolved,
            "num_errors": n_errors,
            "resolve_rate": (n_resolved / n) if n else None,
        })
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "num_instances",
                "num_non_empty_patches",
                "num_completed",
                "num_resolved",
                "num_errors",
                "resolve_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    if print_table:
        print("\nResolve rate per model")
        print("-" * 72)
        for row in summary:
            rr = f"{row['resolve_rate']:.2%}" if row["resolve_rate"] is not None else "N/A"
            print(
                f"{row['model_label']:28} resolved={row['num_resolved']:>3}/{row['num_instances']:<3} "
                f"resolve_rate={rr:>7} errors={row['num_errors']}"
            )
        print("-" * 72)
        print(f"Wrote {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SWE-bench Lite evaluation via mini-swe-agent v2 + SWE-bench harness.")
    p.add_argument("--config", default=None, help="Path to YAML config.")
    p.add_argument("--models", nargs="+", default=None, help="Subset of roster labels to run.")
    p.add_argument("--instance-ids", nargs="+", default=None, help="Specific SWE-bench Lite instance_ids.")
    p.add_argument("--max-instances", type=int, default=None, help="Cap the number of instances (smoke-test).")
    p.add_argument("--output-dir", default=None, help="Override evaluation.output_dir in the config.")
    p.add_argument("--no-resume", action="store_true", help="Ignore cached responses.jsonl and re-run.")
    p.add_argument(
        "--no-resume-recover-artifacts",
        action="store_true",
        help="When resuming, do not recover cached rows from existing agent/reports artifacts.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Checkpoint model responses.jsonl every N newly graded instances (1=each; 0=only final save).",
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

    dataset_name = str(section.get("dataset_name", "princeton-nlp/SWE-bench_Lite"))
    dataset_split = str(section.get("dataset_split", "test"))
    instance_ids_file = section.get("instance_ids_file")
    file_ids = load_ids_from_file(instance_ids_file)
    explicit_ids = args.instance_ids or file_ids or list(section.get("instance_ids") or [])
    max_instances = args.max_instances if args.max_instances is not None else section.get("max_instances")

    instance_ids = load_instance_ids(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        explicit_ids=explicit_ids,
        max_instances=max_instances,
    )
    warn_rootless_matplotlib_subid_risk(instance_ids)

    output_dir = Path(args.output_dir or section.get("output_dir") or f"results/agentic_eval/swe_bench_lite/{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)

    swebench_path = Path(section.get("swebench_path", "/home/aaronjli/SWE-bench")).resolve()
    if not swebench_path.exists():
        raise SystemExit(f"swebench_path does not exist: {swebench_path}")

    use_litellm = section.get("use_litellm")
    litellm_models = section.get("litellm_models")
    native_models = section.get("native_models")
    default_route = section.get("default_route")
    route_overrides = section.get("route_overrides")
    model_overrides = dict(section.get("model_overrides") or {})
    agent_cfg = dict(section.get("agent") or {})
    model_agent_extra_args_overrides = {
        str(label): dict(extra)
        for label, extra in (section.get("model_agent_extra_args_overrides") or {}).items()
        if isinstance(extra, dict)
    }
    grading_cfg = dict(section.get("grading") or {})
    resume = not args.no_resume and bool(section.get("resume", True))
    resume_recover_artifacts = bool(section.get("resume_recover_artifacts", True))
    if args.no_resume_recover_artifacts:
        resume_recover_artifacts = False
    save_every = args.save_every if args.save_every is not None else int(section.get("save_every", 1))
    if save_every < 0:
        raise SystemExit("save_every must be >= 0 (use 0 for checkpoints only at final save).")

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
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "num_instances": len(instance_ids),
        "instance_ids_sample": instance_ids[:5],
        "instance_ids_file": instance_ids_file,
        "output_dir": str(output_dir),
        "swebench_path": str(swebench_path),
        "use_litellm": use_litellm,
        "litellm_models": litellm_models or [],
        "native_models": native_models or [],
        "default_route": default_route,
        "route_overrides": route_overrides,
        "model_routes": model_routes,
        "model_agent_extra_args_overrides": model_agent_extra_args_overrides,
        "agent": agent_cfg,
        "grading": grading_cfg,
        "resume": resume,
        "resume_recover_artifacts": resume_recover_artifacts,
        "save_every": save_every,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    run_config["responses_jsonl"] = str(output_dir / "responses.jsonl")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    base_env = os.environ.copy()
    shared_responses_path = output_dir / "responses.jsonl"
    global_rows: dict[str, dict[str, Any]] = {}
    if resume:
        global_rows = load_global_responses(shared_responses_path)

    rows_by_model: dict[str, list[dict[str, Any]]] = global_rows_by_model(global_rows)
    for label in models:
        model_output_dir = output_dir / label
        def _checkpoint_update(rows: list[dict[str, Any]], *, _label: str = label) -> None:
            rows_by_model[_label] = rows
            summarize(output_dir, rows_by_model, print_table=False)

        rows = evaluate_model(
            label=label,
            agent_cfg=agent_cfg,
            model_agent_extra_args_overrides=model_agent_extra_args_overrides,
            grading_cfg=grading_cfg,
            swebench_path=swebench_path,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            instance_ids=instance_ids,
            model_output_dir=model_output_dir,
            route=model_routes[label],
            model_overrides=model_overrides,
            resume=resume,
            resume_recover_artifacts=resume and resume_recover_artifacts,
            save_every=save_every,
            base_env=base_env,
            shared_responses_path=shared_responses_path,
            global_rows=global_rows,
            on_checkpoint=_checkpoint_update,
        )
        rows_by_model[label] = rows

    summarize(output_dir, rows_by_model, print_table=True)


if __name__ == "__main__":
    main()
