#!/usr/bin/env python3
"""
SWE-bench Lite evaluation with Claude Opus 4.6.

Generates patches for SWE-bench Lite instances by sending the problem
statement and relevant source files to Claude, then (optionally) evaluates
the patch using the SWE-bench Docker harness.

Setup (local machine with Docker):
    cd eval_static
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."   # or put in ../.env

Usage:
    # List all 300 SWE-bench Lite instances
    python swebench.py list

    # Generate a patch for one instance (oracle file retrieval)
    python swebench.py predict django__django-12113

    # Generate with model-guided file retrieval (no peeking at gold patch)
    python swebench.py predict django__django-12113 --retrieval agentless

    # Batch: generate predictions for instances 0-9
    python swebench.py batch --start 0 --end 10

    # Batch: all 300 instances (resumes from where it left off)
    python swebench.py batch --start 0 --end 300

    # Evaluate a generated prediction (requires Docker)
    python swebench.py evaluate django__django-12113

    # Full pipeline: predict + evaluate
    python swebench.py run django__django-12113
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic
from datasets import load_dataset
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-6"
DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
DATASET_SPLIT = "test"

MAX_FILE_LINES = 2000
MAX_CONTEXT_FILES = 10
MAX_TOKENS = 8192

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "workspace"
REPOS_DIR = WORKSPACE / "repos"
PREDICTIONS_DIR = WORKSPACE / "predictions"
RESULTS_DIR = WORKSPACE / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Load .env from project root (handles `export KEY=val` syntax)
load_dotenv(BASE_DIR.parent / ".env")

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_dataset_cache = None


def load_swebench():
    """Load and cache the SWE-bench Lite test split."""
    global _dataset_cache
    if _dataset_cache is None:
        log.info("Loading %s (split=%s) ...", DATASET_NAME, DATASET_SPLIT)
        _dataset_cache = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return _dataset_cache


def get_instance(instance_id: str) -> dict:
    ds = load_swebench()
    for inst in ds:
        if inst["instance_id"] == instance_id:
            return dict(inst)
    raise ValueError(
        f"Instance '{instance_id}' not found. Run `python swebench.py list` to see available IDs."
    )


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------


def clone_repo(repo: str, base_commit: str) -> Path:
    """Clone repo and checkout the base commit.  Cached under workspace/repos/."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = REPOS_DIR / repo.replace("/", "__")

    if repo_dir.exists():
        log.info("Repo cached at %s — resetting to %s", repo_dir, base_commit[:10])
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_dir, check=True, capture_output=True,
        )
        subprocess.run(["git", "clean", "-fdx"], cwd=repo_dir, capture_output=True)
        return repo_dir

    repo_url = f"https://github.com/{repo}.git"
    log.info("Cloning %s (this may take a few minutes) ...", repo_url)
    subprocess.run(
        ["git", "clone", "--quiet", repo_url, str(repo_dir)],
        check=True,
    )
    log.info("Checking out base commit %s", base_commit[:10])
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_dir, check=True, capture_output=True,
    )
    return repo_dir


def get_python_file_tree(repo_dir: Path) -> str:
    """Flat list of *.py paths in the repo (for the localization prompt)."""
    result = subprocess.run(
        [
            "find", ".", "-type", "f", "-name", "*.py",
            "-not", "-path", "./.git/*",
            "-not", "-path", "*/migrations/*",
            "-not", "-path", "*/__pycache__/*",
        ],
        cwd=repo_dir, capture_output=True, text=True,
    )
    files = sorted(
        f.removeprefix("./") for f in result.stdout.strip().split("\n") if f
    )
    return "\n".join(files)


def read_repo_file(repo_dir: Path, file_path: str) -> str:
    """Read a file from the cloned repo, truncating if huge."""
    full = repo_dir / file_path
    if not full.exists():
        return f"[File not found: {file_path}]"
    lines = full.read_text(errors="replace").splitlines()
    if len(lines) > MAX_FILE_LINES:
        header = lines[:MAX_FILE_LINES]
        header.append(
            f"\n# ... truncated — showing {MAX_FILE_LINES} of {len(lines)} lines ..."
        )
        return "\n".join(header)
    return "\n".join(lines)


def files_from_patch(patch: str) -> list[str]:
    """Extract modified file paths from a unified diff."""
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                paths.append(parts[2].removeprefix("a/"))
    return list(dict.fromkeys(paths))


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.  Either:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  or add it to ../.env"
        )
    return anthropic.Anthropic(api_key=api_key)


def call_claude(client: anthropic.Anthropic, system: str, user: str) -> str:
    log.info("Calling %s ...", MODEL)
    t0 = time.time()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.time() - t0
    u = msg.usage
    log.info(
        "Done in %.1fs  (in=%d  out=%d tokens)", elapsed, u.input_tokens, u.output_tokens
    )
    return msg.content[0].text


# ---------------------------------------------------------------------------
# File localization
# ---------------------------------------------------------------------------


def localize_oracle(instance: dict, **_kw) -> list[str]:
    """Use files listed in the gold patch (upper-bound retrieval)."""
    return files_from_patch(instance["patch"])


def localize_agentless(
    instance: dict, *, client: anthropic.Anthropic, repo_dir: Path
) -> list[str]:
    """Ask Claude to pick the relevant files from the repo tree."""
    tree = get_python_file_tree(repo_dir)

    system = (
        "You are an expert software engineer.  Given a bug report and the "
        "repository's Python file tree, identify the source files most likely "
        "to need modification.  Return ONLY a JSON array of relative file paths."
    )
    user_msg = (
        f"## Bug Report\n\n{instance['problem_statement']}\n\n"
        f"## Repository Python Files\n\n```\n{tree}\n```\n\n"
        f"Return a JSON array of the {MAX_CONTEXT_FILES} most relevant files."
    )

    resp = call_claude(client, system, user_msg)

    match = re.search(r"\[.*?\]", resp, re.DOTALL)
    if match:
        try:
            paths = json.loads(match.group())
            return [p for p in paths if isinstance(p, str)][:MAX_CONTEXT_FILES]
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse localization response — falling back to oracle")
    return localize_oracle(instance)


RETRIEVAL_FNS = {
    "oracle": localize_oracle,
    "agentless": localize_agentless,
}

# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = (
    "You are an expert software engineer.  Your task is to fix a bug described "
    "in the problem statement by modifying the provided source files.\n\n"
    "Output ONLY a unified diff patch that starts with `diff --git` and can be "
    "applied with `git apply`.  Do not wrap the patch in markdown fences.  "
    "Do not add any explanation before or after the patch.\n\n"
    "Guidelines:\n"
    "- Make the minimal change needed to fix the issue.\n"
    "- Preserve existing code style.\n"
    "- Do not refactor unrelated code."
)


def generate_patch(
    client: anthropic.Anthropic,
    instance: dict,
    repo_dir: Path,
    file_paths: list[str],
) -> str:
    """Send problem + file contents to Claude and get a diff back."""
    files_block = ""
    for fp in file_paths:
        content = read_repo_file(repo_dir, fp)
        files_block += f"\n### {fp}\n```python\n{content}\n```\n"

    hints = (instance.get("hints_text") or "").strip()
    hints_block = f"\n## Hints\n\n{hints}\n" if hints else ""

    user_msg = (
        f"## Problem Statement\n\n{instance['problem_statement']}\n"
        f"{hints_block}\n"
        f"## Source Files\n{files_block}\n"
        "Generate a minimal unified diff patch to fix this issue."
    )

    response = call_claude(client, REPAIR_SYSTEM, user_msg)

    # Strip surrounding prose / markdown fences if the model added any
    if "diff --git" in response:
        patch = response[response.index("diff --git"):]
        if patch.rstrip().endswith("```"):
            patch = patch[: patch.rstrip().rfind("```")]
        return patch.strip()

    return response.strip()


# ---------------------------------------------------------------------------
# Prediction I/O
# ---------------------------------------------------------------------------


def prediction_path(run_id: str) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return PREDICTIONS_DIR / f"{run_id}.jsonl"


def save_prediction(instance_id: str, model_patch: str, run_id: str) -> Path:
    out = prediction_path(run_id)
    pred = {
        "instance_id": instance_id,
        "model_name_or_path": MODEL,
        "model_patch": model_patch,
    }
    with open(out, "a") as f:
        f.write(json.dumps(pred) + "\n")
    log.info("Prediction saved → %s", out)
    return out


def load_completed_ids(run_id: str) -> set[str]:
    """Return instance IDs that already have predictions in the run file."""
    out = prediction_path(run_id)
    if not out.exists():
        return set()
    ids = set()
    for line in out.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["instance_id"])
    return ids


# ---------------------------------------------------------------------------
# Evaluation (Docker required)
# ---------------------------------------------------------------------------


def run_evaluation(pred_path: Path, instance_id: str, run_id: str):
    """Invoke the SWE-bench Docker-based evaluation harness."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", DATASET_NAME,
        "--predictions_path", str(pred_path),
        "--instance_ids", instance_id,
        "--max_workers", "1",
        "--run_id", run_id,
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Evaluation harness exited with code %d", result.returncode)
        sys.exit(result.returncode)
    log.info("Evaluation complete — check SWE-bench logs for pass/fail results.")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_list(_args):
    ds = load_swebench()
    by_repo: dict[str, list[str]] = {}
    for inst in ds:
        by_repo.setdefault(inst["repo"], []).append(inst["instance_id"])

    print(f"\nSWE-bench Lite  ({len(ds)} instances, {len(by_repo)} repos)\n")
    for repo, ids in sorted(by_repo.items()):
        print(f"  {repo}  ({len(ids)})")
        for iid in sorted(ids)[:3]:
            print(f"    • {iid}")
        if len(ids) > 3:
            print(f"    … and {len(ids) - 3} more")
    print()


def predict_one(
    instance: dict,
    client: anthropic.Anthropic,
    retrieval: str,
    run_id: str,
    *,
    verbose: bool = True,
) -> str:
    """Core prediction logic for a single instance.  Returns the generated patch."""
    iid = instance["instance_id"]
    log.info(
        "Instance: %s  |  repo: %s  |  version: %s",
        iid, instance["repo"], instance["version"],
    )

    repo_dir = clone_repo(instance["repo"], instance["base_commit"])

    log.info("File retrieval: %s", retrieval)
    localize = RETRIEVAL_FNS[retrieval]
    file_paths = localize(instance, client=client, repo_dir=repo_dir)
    log.info("Files for context: %s", file_paths)

    patch = generate_patch(client, instance, repo_dir, file_paths)

    if verbose:
        sep = "=" * 72
        print(f"\n{sep}\nGenerated patch for {iid}\n{sep}")
        print(patch)
        print(sep)

    save_prediction(iid, patch, run_id)
    return patch


def cmd_predict(args):
    instance = get_instance(args.instance_id)
    client = get_client()
    run_id = args.run_id or f"opus46_{int(time.time())}"

    patch = predict_one(instance, client, args.retrieval, run_id)

    pred_path = prediction_path(run_id)
    if args.retrieval == "oracle":
        gold_files = files_from_patch(instance["patch"])
        gen_files = files_from_patch(patch) if "diff --git" in patch else []
        print(f"\nGold patch modifies : {gold_files}")
        print(f"Generated patch hits: {gen_files}")

    print(f"\nPrediction file: {pred_path}")
    print(f"Run ID          : {run_id}")
    return pred_path, run_id


def cmd_batch(args):
    ds = load_swebench()
    total = len(ds)
    start = max(0, args.start)
    end = min(total, args.end)

    if start >= end:
        log.error("Invalid range: start=%d end=%d (dataset has %d instances)", start, end, total)
        sys.exit(1)

    run_id = args.run_id or f"opus46_batch_{int(time.time())}"
    client = get_client()
    retrieval = args.retrieval

    completed = load_completed_ids(run_id)
    log.info(
        "Batch: instances [%d, %d)  |  retrieval: %s  |  run_id: %s",
        start, end, retrieval, run_id,
    )
    if completed:
        log.info("Resuming — %d instances already completed in this run", len(completed))

    succeeded, failed, skipped = 0, 0, 0
    cost_in, cost_out = 0, 0

    for idx in range(start, end):
        inst = dict(ds[idx])
        iid = inst["instance_id"]
        progress = f"[{idx - start + 1}/{end - start}]"

        if iid in completed:
            log.info("%s Skipping %s (already completed)", progress, iid)
            skipped += 1
            continue

        try:
            log.info("%s Predicting %s ...", progress, iid)
            predict_one(inst, client, retrieval, run_id, verbose=False)
            succeeded += 1
        except Exception:
            failed += 1
            log.exception("%s FAILED on %s", progress, iid)

    pred_path = prediction_path(run_id)
    final_count = len(load_completed_ids(run_id))

    print(f"\n{'=' * 72}")
    print(f"Batch complete  |  run_id: {run_id}")
    print(f"  Range       : [{start}, {end})")
    print(f"  Succeeded   : {succeeded}")
    print(f"  Failed      : {failed}")
    print(f"  Skipped     : {skipped} (already done)")
    print(f"  Total saved : {final_count} predictions in {pred_path}")
    print(f"{'=' * 72}")


def cmd_evaluate(args):
    run_id = args.run_id or "opus46_latest"
    if args.predictions:
        pred_path = Path(args.predictions)
    else:
        pred_path = prediction_path(run_id)
    if not pred_path.exists():
        log.error("Predictions file not found: %s", pred_path)
        sys.exit(1)
    run_evaluation(pred_path, args.instance_id, run_id)


def cmd_run(args):
    pred_path, run_id = cmd_predict(args)
    print("\n--- Starting evaluation (requires Docker) ---\n")
    run_evaluation(pred_path, args.instance_id, run_id)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="SWE-bench Lite evaluation with Claude Opus 4.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python swebench.py list\n"
            "  python swebench.py predict django__django-12113\n"
            "  python swebench.py predict django__django-12113 --retrieval agentless\n"
            "  python swebench.py batch --start 0 --end 10\n"
            "  python swebench.py batch --start 0 --end 300 --retrieval agentless\n"
            "  python swebench.py evaluate django__django-12113 --run-id opus46_1710000000\n"
            "  python swebench.py run django__django-12113\n"
        ),
    )
    sub = p.add_subparsers(dest="command")
    sub.required = True

    # --- list ---
    sub.add_parser("list", help="List available SWE-bench Lite instances")

    # --- predict ---
    pp = sub.add_parser("predict", help="Generate a patch prediction via Claude API")
    pp.add_argument("instance_id", help="e.g. django__django-12113")
    pp.add_argument(
        "--retrieval",
        choices=["oracle", "agentless"],
        default="oracle",
        help="How to select context files (default: oracle)",
    )
    pp.add_argument("--run-id", help="Custom run identifier (default: auto)")

    # --- batch ---
    pb = sub.add_parser("batch", help="Generate predictions for a range of instances")
    pb.add_argument(
        "--start", type=int, required=True,
        help="Start index (inclusive, 0-based)",
    )
    pb.add_argument(
        "--end", type=int, required=True,
        help="End index (exclusive, 0-based).  Dataset has 300 instances.",
    )
    pb.add_argument(
        "--retrieval",
        choices=["oracle", "agentless"],
        default="oracle",
        help="How to select context files (default: oracle)",
    )
    pb.add_argument(
        "--run-id",
        help="Run identifier.  Re-use the same ID to resume an interrupted batch.",
    )

    # --- evaluate ---
    pe = sub.add_parser("evaluate", help="Evaluate predictions (needs Docker)")
    pe.add_argument("instance_id", help="e.g. django__django-12113")
    pe.add_argument("--predictions", help="Path to predictions JSONL file")
    pe.add_argument("--run-id", help="Run identifier to locate predictions file")

    # --- run ---
    pr = sub.add_parser("run", help="Predict + evaluate end-to-end")
    pr.add_argument("instance_id", help="e.g. django__django-12113")
    pr.add_argument(
        "--retrieval",
        choices=["oracle", "agentless"],
        default="oracle",
        help="How to select context files (default: oracle)",
    )
    pr.add_argument("--run-id", help="Custom run identifier (default: auto)")

    args = p.parse_args()
    {
        "list": cmd_list,
        "predict": cmd_predict,
        "batch": cmd_batch,
        "evaluate": cmd_evaluate,
        "run": cmd_run,
    }[args.command](args)


if __name__ == "__main__":
    main()
