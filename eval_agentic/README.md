# eval_agentic/

Agentic evaluations for the MutualEval roster. Two pipelines live here:

| Benchmark | Tasks | Agent (pinned) | Grader |
| --- | --- | --- | --- |
| SWE-bench Lite | 300 Python bug-fix instances (`princeton-nlp/SWE-bench_Lite`) | `mini-swe-agent` v2 | `swebench.harness.run_evaluation` from a local SWE-bench clone |
| Terminal-Bench 2.0 | 89 hand-authored container tasks (`terminal-bench@2.0`) | `terminus-2` | Harbor's built-in verifier (`reward.txt` per task) |

Both pipelines evaluate the same 15-model roster that `eval_static/` uses
(`mutual-eval/eval_static/model_api_smoke_test.py :: MODEL_SPECS`) and
route every call through a single LiteLLM gateway by default.

---

## 1. One-time setup

### 1.1 Python deps

```bash
cd mutual-eval
pip install -r eval_agentic/requirements.txt
# also pulls in the eval_static deps if you haven't already:
pip install -r eval_static/requirements.txt
```

### 1.2 SWE-bench grader

The grader is the official SWE-bench harness. Install the local clone in
editable mode (its package name is `swebench`):

```bash
pip install -e /home/aaronjli/SWE-bench
```

### 1.3 mini-swe-agent v2 (pinned SHA)

```bash
pip install "git+https://github.com/SWE-agent/mini-swe-agent.git@bc85a45654e6348dcc6e4c5a40ad146ed0bb144d"
```

Verify the v2 CLI is reachable:

```bash
python -m minisweagent.run.utilities.mini_extra swebench --help | head -20
```

You should see `--subset`, `--split`, `--filter`, `--model`, `--output`,
`--workers` flags.

### 1.4 Harbor + Terminal-Bench 2.0

```bash
pip install harbor       # or: uv tool install harbor
which harbor
harbor run --help | head -20
```

The Terminal-Bench 2.0 task set already lives at
`/home/aaronjli/terminal-bench-2/` (cloned by the user). The default
`config_terminal_bench.yaml` points `dataset_path` at that clone so
Harbor doesn't re-download.

### 1.5 Secrets

Populate `mutual-eval/.env` (or `mutual-eval/eval_agentic/.env`) with at
least:

```env
# Gateway route (default for every model).
LITELLM_API_KEY=sk-...
LITELLM_BASE_URL=https://your-litellm-gateway/v1
# (fallbacks if LITELLM_* is missing)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://...

# Native route (only needed for models marked route=native).
ANTHROPIC_API_KEY=sk-ant-...
# ANTHROPIC_BASE_URL=...
GEMINI_API_KEY=AIza...
# GEMINI_BASE_URL=...
OPENROUTER_API_KEY=sk-or-...
```

Docker must be running for both benchmarks (SWE-bench builds instance
images, Terminal-Bench spins up per-task containers).

---

## 2. Run it

### 2.1 SWE-bench Lite

```bash
python eval_agentic/eval_swe_bench_lite.py \
    --config eval_agentic/config_swe_bench_lite.yaml
```

Smoke test on three instances with a single cheap model:

```bash
python eval_agentic/eval_swe_bench_lite.py \
    --config eval_agentic/config_swe_bench_lite.yaml \
    --models gpt-4.1-mini --max-instances 3
```

Output layout:

```text
results/agentic_eval/swe_bench_lite_v0/
    run_config.json
    summary.csv                           # per-model resolve rate
    <model_label>/
        agent_run/preds.json              # mini-swe-agent output
        predictions.jsonl                 # what the grader consumed
        reports/                          # swebench harness report dir
        responses.jsonl                   # per-instance resolved/completed/error
```

### 2.2 Terminal-Bench 2.0

```bash
python eval_agentic/eval_terminal_bench.py \
    --config eval_agentic/config_terminal_bench.yaml
```

Smoke test on three tasks, one attempt, one model:

```bash
python eval_agentic/eval_terminal_bench.py \
    --config eval_agentic/config_terminal_bench.yaml \
    --models gpt-4.1-mini --max-tasks 3 --eval-runs 1
```

Output layout:

```text
results/agentic_eval/terminal_bench_v0/
    run_config.json
    summary.csv                           # per-model pass_rate, pass@1, pass@k
    responses.jsonl                       # all models: one row per (model, task_id, attempt)
    <model_label>/
        jobs/<run_id>/<task>__<h>/...
        harbor_logs/<run_id>.stdout|.stderr
```

---

## 3. Resuming and overrides

**SWE-bench Lite** writes `responses.jsonl` per model under `<output_dir>/<model>/`.
It skips already-graded instances when `resume` is enabled (YAML `evaluation.resume`,
default true; CLI `--no-resume` disables).

**Terminal-Bench** writes a **single** `<output_dir>/responses.jsonl` for every model.
Resume skips rows whose `(model_label, task_id, attempt)` key already exists. Use
`--resume` to force loading that file (like `eval_static`), or rely on
`evaluation.resume: true` (default in `config_terminal_bench.yaml`). Use `--no-resume`
for a full re-run.

`evaluation.save_every` (or `--save-every N`) rewrites `responses.jsonl` and `summary.csv`
periodically: `1` = after each Harbor batch (default); `50` = every 50 new rows accumulated;
`0` = disable intermediate checkpoints (still writes both files at exit).

Common overrides:

```bash
# Different output dir (e.g. for a paper revision run)
python eval_agentic/eval_swe_bench_lite.py --config ... \
    --output-dir results/agentic_eval/swe_bench_lite_v1

# Model subset
python eval_agentic/eval_terminal_bench.py --config ... \
    --models claude-opus-4-6 gpt-5.4

# Force a fresh SWE-bench run (ignore per-model cache)
python eval_agentic/eval_swe_bench_lite.py --config ... --no-resume

# Terminal-Bench: explicit resume from shared responses.jsonl
python eval_agentic/eval_terminal_bench.py --config ... --resume --output-dir results/agentic_eval/terminal_bench_2
```

### Per-model routing (LiteLLM vs. native)

Each model can be independently routed through the LiteLLM gateway or
against its native provider API, in the same run. The config knobs are:

```yaml
evaluation:
  # Static-style global default (same naming as eval_static).
  use_litellm: true

  # Subset selectors:
  # - If only litellm_models is set, listed models use litellm and all
  #   other selected models use native.
  # - If only native_models is set, listed models use native and all
  #   other selected models use litellm.
  litellm_models:
    - gpt-5.4
    - gpt-5-mini
  native_models:
    - claude-opus-4-6
    - grok-4

  # Optional advanced compatibility knobs:
  default_route: null           # litellm|native; synonym for use_litellm
  route_overrides: {}           # highest precedence {"label": "litellm|native"}

  # (optional) Override the routed model id string itself, e.g. to pin a
  # specific OpenRouter slug.
  model_overrides:
    grok-4: "openrouter/x-ai/grok-4.20-beta"
```

What each route reads from the environment (per model):

| Provider   | `route: litellm`                            | `route: native`                                 |
|------------|---------------------------------------------|-------------------------------------------------|
| openai     | `LITELLM_API_KEY`/`LITELLM_BASE_URL` (or `OPENAI_*`) | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`)  |
| anthropic  | same as above                               | `ANTHROPIC_API_KEY` (+ optional `ANTHROPIC_BASE_URL`) |
| google     | same as above                               | `GEMINI_API_KEY` / `GOOGLE_API_KEY` (+ `GEMINI_BASE_URL`) |
| openrouter | same as above                               | `OPENROUTER_API_KEY`                            |

If a required env var is missing at launch, the driver fails fast with
an actionable error that names both the model and the missing variable.

Precedence for the final route per model:
1. `route_overrides[label]` (highest)
2. `litellm_models` / `native_models`
3. `default_route` (if provided)
4. `use_litellm` (fallback/global default)

---

## 4. Design notes

- **Single source of truth for models**: `model_roster.py` re-exports
  `MODEL_LOOKUP` / `DEFAULT_ROSTER` from `eval_static/model_api_smoke_test.py`,
  so the 15-model list in the configs here is identical to the one used
  by `eval_static/config_static_coding.yaml`. Swapping roster members in
  one place propagates to both.
- **Pinned agents**: the default configs hard-code `mini-swe-agent`
  (v2 CLI entrypoint) and `terminus-2`. If you want a different agent
  you'll need to fork the relevant YAML and eval script.
- **Resume granularity**:
  - SWE-bench: per instance (`responses.jsonl` keyed by `instance_id`).
  - Terminal-Bench: per `(task_id, attempt)` pair.
- **Per-model routing**: every model carries its own route (`litellm`
  or `native`). Only the creds matching the active route are exported
  to the agent subprocess -- other provider keys are scrubbed from the
  env so, for example, an `OPENAI_BASE_URL` pointing at a LiteLLM
  gateway never leaks into a native Anthropic call. OpenRouter-only
  roster entries (e.g. `grok-4`) are auto-prefixed with `openrouter/`
  unless `model_overrides` says otherwise.

---

## 5. Troubleshooting

- `harbor` not found → `pip install harbor` (or `uv tool install harbor`)
  and re-open the shell so `PATH` updates.
- `ModuleNotFoundError: swebench` → `pip install -e /home/aaronjli/SWE-bench`.
- `ModuleNotFoundError: minisweagent` → install mini-swe-agent at the
  pinned SHA (see 1.3).
- Empty patches on every instance → check `<model>/agent_run/stdout.log`
  (mini-swe-agent) or `<model>/harbor_logs/<run_id>.stderr` (Harbor)
  for auth/rate-limit errors.
- Docker permission issues → run `sudo usermod -aG docker $USER` and
  re-login, or prepend `sudo` to the command.
