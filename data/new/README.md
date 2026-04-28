# New Datasets

Download the datasets from here: https://huggingface.co/datasets/aaronjli/mutualeval/tree/main

Put them in the corresponding versioned sub-directories:
- `./data/new/v0/` — original v0 files
- `./data/new/v1/` — v1 files (new datasets, additional domains)

## Models (15, shared across all files and versions)

| Model | Provider |
|---|---|
| claude-haiku-4-5 | Anthropic |
| claude-sonnet-4-6 | Anthropic |
| claude-opus-4-6 | Anthropic |
| deepseek-v3.2 | DeepSeek |
| gemini-2.5-flash | Google |
| gemini-2.5-pro | Google |
| gemini-3.1-pro | Google |
| gpt-4.1 | OpenAI |
| gpt-4.1-mini | OpenAI |
| gpt-5-mini | OpenAI |
| gpt-5.4 | OpenAI |
| grok-4 | xAI |
| llama-4-maverick-instruct | Meta |
| mistral-large-3 | Mistral |
| qwen3-max-thinking | Alibaba |

## V0 Datasets

| File | Type | Domain | Records | Unique Questions | Grading | Label |
|---|---|---|---|---|---|---|
| `v0/static_math_v0.jsonl` | static | math (gsm8k) | 7,500 | 500 | `llm_judge` | binary `correct` (bool) |
| `v0/static_coding_v0.jsonl` | static | coding (humaneval-plus) | 5,700 | 380 | `exec_tests` | binary `correct` (bool) |
| `v0/arena_math_v0.jsonl` | arena | math | 6,000 | 400 | reward model | continuous `reward` (float) |
| `v0/arena_coding_v0.jsonl` | arena | coding | 2,250 | 150 | reward model | continuous `reward` (float) |
| `v0/arena_generic_v0.jsonl` | arena | generic | 7,500 | 500 | reward model | continuous `reward` (float) |

## V1 Datasets

V1 expands coverage with harder benchmarks and two new **misc** domains (static + arena).

| File | Type | Domain | Records | Unique Questions | Grading | Label |
|---|---|---|---|---|---|---|
| `v1/v1_static_math.jsonl` | static | math (aime-2025, etc.) | 3,000 | 200 | `llm_judge` / `string_match` | binary `correct` (bool) |
| `v1/v1_static_coding.jsonl` | static | coding (mbpp-plus-sanitized) | 4,500 | 300 | `exec_tests` | binary `correct` (bool) |
| `v1/v1_static_misc.jsonl` | static | misc (hle, etc.) | 4,500 | 300 | `llm_judge` | binary `correct` (bool) |
| `v1/v1_arena_math.jsonl` | arena | math | 3,000 | 200 | reward model | continuous `reward` (float) |
| `v1/v1_arena_coding.jsonl` | arena | coding | 4,500 | 300 | reward model | continuous `reward` (float) |
| `v1/v1_arena_generic.jsonl` | arena | generic | 7,500 | 500 | reward model | continuous `reward` (float) |
| `v1/v1_arena_misc.jsonl` | arena | misc | 4,500 | 300 | reward model | continuous `reward` (float) |

All files have one record per (question, model) pair. Records with `status != "ok"` should be filtered out.

## Key Field Differences

### V0 Format
- **Static files**: `model_label` (not `model_name`), `sample_index` + `dataset` (not `question_id`), no `correct_answer` for coding.
- **Arena files**: `item_id` as question identifier, continuous `reward` instead of discrete 0/1/2/3 pairwise labels.
- `ranking.py` has native loaders for both formats: `load_static_jsonl()` and `load_arena_reward_jsonl()`.

### V1 Format
V1 uses the same core fields as V0 and is compatible with the existing `load_static_jsonl()` and `load_arena_reward_jsonl()` loaders in `ranking.py`. Notable additions:

- **Static files**: additional metadata fields `question`, `prompt`, `gold_answer`, `judge_reason`, `grading_method`, `generation_attempts`, `level`, `subject`.
- **Arena files**: per-request latency fields `generation_latency_s`, `reward_latency_s`, `total_latency_s`.
- **Status values** in v1 static files: `ok`, `error`, `timeout`, `generation_error` (same filter rule: keep only `status == "ok"`).
- `v1_static_coding.jsonl` uses the `mbpp-plus-sanitized` dataset (vs. `humaneval-plus` in v0).
