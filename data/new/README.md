# New Datasets (v0)

Download the new datasets from here: https://huggingface.co/datasets/aaronjli/mutualeval/tree/main

Put them in this directory: `./data/new/`

## Models (15, shared across all files)

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

## Datasets

| File | Type | Domain | Records | Unique Questions | Grading | Label |
|---|---|---|---|---|---|---|
| `static_math_v0.jsonl` | static | math (gsm8k) | 7,500 | 500 | `llm_judge` | binary `correct` (bool) |
| `static_coding_v0.jsonl` | static | coding (humaneval-plus) | 5,700 | 380 | `exec_tests` | binary `correct` (bool) |
| `arena_math_v0.jsonl` | arena | math | 6,000 | 400 | reward model | continuous `reward` (float) |
| `arena_coding_v0.jsonl` | arena | coding | 2,250 | 150 | reward model | continuous `reward` (float) |
| `arena_generic_v0.jsonl` | arena | generic | 7,500 | 500 | reward model | continuous `reward` (float) |

All files have one record per (question, model) pair. Records with `status != "ok"` should be filtered out.

## Key Field Differences from Original CSVs

- **Static files**: `model_label` (not `model_name`), `sample_index` + `dataset` (not `question_id`), no `correct_answer` for coding.
- **Arena files**: `item_id` as question identifier, continuous `reward` instead of discrete 0/1/2/3 pairwise labels.
- `ranking.py` has native loaders for both formats: `load_static_jsonl()` and `load_arena_reward_jsonl()`.
