#!/usr/bin/env python3
"""
Static miscellaneous evaluation (HLE non-math strata + SimpleQA) for the model roster.

Breakdown (misc v1 — 300 items):
  - HLE Humanities/Social Science: 90 (text-only, exactMatch, non-numeric gold)
  - HLE Other: 83 (text-only)
  - HLE Biology/Medicine: 52 (text-only)
  - HLE Engineering: 25 (text-only)
  - SimpleQA: 50 stratified by topic (see config)

Usage:
    python eval_static/sample_static_misc.py --config eval_static/config_static_misc.yaml
    python eval_static/eval_static_misc.py --config eval_static/config_static_misc.yaml
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import BadRequestError, OpenAI
from tqdm import tqdm

from model_api_smoke_test import (
    MODEL_LOOKUP,
    MODEL_SPECS,
    extract_chat_content,
    get_routed_model_id,
    should_use_litellm_for_model,
    get_env_value,
    load_env_file,
    normalize_base_url,
)

from eval_static_math import (
    _sanitize_for_json,
    _should_use_litellm,
    build_clients_with_mode,
    build_eval_order_lookup,
    build_output_dir,
    call_model,
    load_jsonl,
    load_yaml_config,
    make_eval_key,
    order_results,
    resolve_env_path,
    save_checkpoint,
    summarize_results,
    write_jsonl,
    _expand_env,
)

ANSWER_INSTRUCTION_MISC = (
    "Answer the question concisely and correctly. "
    "End your response with a final line of the form 'Final answer: <answer>'."
)

JUDGE_SYSTEM_INSTRUCTIONS_MISC = (
    "You are a careful grader for short factual and technical answers. "
    "Decide whether the model response is correct with respect to the reference answer. "
    "Treat equivalent phrasings, common aliases, rounding-equivalent numbers, and equivalent units as correct. "
    "For multiple acceptable formulations of the same fact, mark correct. "
    "Do not reward style or verbosity. Focus on factual correctness."
)

JUDGE_PROMPT_TEMPLATE_MISC = """\
Question:
{question}

Reference answer:
{gold_answer}

Model response:
{model_answer}

Decide whether the model response is factually correct (including acceptable paraphrases and equivalents).

Return ONLY a JSON object:
{{"correct": true or false, "reason": "brief explanation"}}
"""


# --- HLE eligibility (aligned with misc v1 spec) ---

HLE_HF_PATH = "cais/hle"
HLE_SPLIT = "test"


def is_hle_text_only(row: dict[str, Any]) -> bool:
    return not (row.get("image") and str(row["image"]).strip())


def is_non_numeric_gold(answer: str) -> bool:
    """Heuristic: exclude pure numbers / simple fractions (for Humanities stratum)."""
    s = str(answer).strip()
    if not s:
        return False
    s_plain = re.sub(r"[\$\\]", "", s)
    compact = s_plain.replace(",", "").replace(" ", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", compact):
        return False
    if re.fullmatch(r"-?\d+/\d+", compact):
        return False
    return True


def filter_hle_humanities_misc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("category") != "Humanities/Social Science":
            continue
        if row.get("answer_type") != "exactMatch":
            continue
        if not is_hle_text_only(row):
            continue
        if not is_non_numeric_gold(str(row.get("answer", ""))):
            continue
        out.append(row)
    return out


def filter_hle_category_text_only(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("category") == category and is_hle_text_only(r)]


def sample_deterministic(rows: list[dict[str, Any]], n: int, seed: int, salt: str) -> list[dict[str, Any]]:
    rng = random.Random(seed + sum(ord(c) for c in salt))
    pool = list(rows)
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"Need {n} items for {salt!r} but only {len(pool)} eligible rows.")
    return pool[:n]


HLE_DATASET_LABELS: dict[str, str] = {
    "humanities_social_science": "hle-humanities-social-science",
    "other": "hle-other",
    "biology_medicine": "hle-biology-medicine",
    "engineering": "hle-engineering",
}


def build_hle_item(dataset_key: str, row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    dataset = HLE_DATASET_LABELS[dataset_key]
    raw = _sanitize_for_json(dict(row))
    question = str(row["question"])
    return {
        "dataset": dataset,
        "dataset_kind": "hle",
        "hle_category": row.get("category"),
        "sample_index": sample_index,
        "question": question,
        "prompt": f"{question}\n\n{ANSWER_INSTRUCTION_MISC}",
        "gold_answer": str(row.get("answer", "")),
        "level": "Expert",
        "subject": str(row.get("category", "")),
        "hle_answer_type": row.get("answer_type"),
        "raw_item": raw,
    }


def build_simpleqa_item(row: dict[str, Any], meta: dict[str, Any], sample_index: int) -> dict[str, Any]:
    topic = str(meta.get("topic", "unknown"))
    q = str(row["problem"])
    return {
        "dataset": "simpleqa",
        "dataset_kind": "simpleqa",
        "sample_index": sample_index,
        "question": q,
        "prompt": f"{q}\n\n{ANSWER_INSTRUCTION_MISC}",
        "gold_answer": str(row.get("answer", "")),
        "level": None,
        "subject": topic,
        "topic": topic,
        "raw_item": {"problem": q, "answer": row.get("answer"), "metadata": meta},
    }


def build_misc_sampled_items(sampling_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(sampling_cfg.get("seed", 0))
    hle_cfg = sampling_cfg.get("hle", {})
    simpleqa_cfg = sampling_cfg.get("simpleqa", {})

    ds_hle = load_dataset(HLE_HF_PATH, split=HLE_SPLIT)
    hle_rows = [dict(r) for r in ds_hle]

    items: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = {}

    # HLE strata (fixed order)
    hs_n = int(hle_cfg.get("humanities_social_science", 90))
    pool_hs = filter_hle_humanities_misc(hle_rows)
    sampled_hs = sample_deterministic(pool_hs, hs_n, seed, "hle_humanities")
    for row in sampled_hs:
        items.append(build_hle_item("humanities_social_science", row, len(items)))
    stratum_counts["hle_humanities_social_science"] = len(sampled_hs)

    o_n = int(hle_cfg.get("other", 83))
    pool_o = filter_hle_category_text_only(hle_rows, "Other")
    sampled_o = sample_deterministic(pool_o, o_n, seed, "hle_other")
    for row in sampled_o:
        items.append(build_hle_item("other", row, len(items)))
    stratum_counts["hle_other"] = len(sampled_o)

    b_n = int(hle_cfg.get("biology_medicine", 52))
    pool_b = filter_hle_category_text_only(hle_rows, "Biology/Medicine")
    sampled_b = sample_deterministic(pool_b, b_n, seed, "hle_biology")
    for row in sampled_b:
        items.append(build_hle_item("biology_medicine", row, len(items)))
    stratum_counts["hle_biology_medicine"] = len(sampled_b)

    e_n = int(hle_cfg.get("engineering", 25))
    pool_e = filter_hle_category_text_only(hle_rows, "Engineering")
    sampled_e = sample_deterministic(pool_e, e_n, seed, "hle_engineering")
    for row in sampled_e:
        items.append(build_hle_item("engineering", row, len(items)))
    stratum_counts["hle_engineering"] = len(sampled_e)

    # SimpleQA stratified
    sq_hf = str(simpleqa_cfg.get("hf_path", "basicv8vc/SimpleQA"))
    sq_split = str(simpleqa_cfg.get("split", "test"))
    topic_counts: dict[str, int] = {
        str(k): int(v) for k, v in (simpleqa_cfg.get("topics") or {}).items()
    }
    if not topic_counts:
        raise ValueError("simpleqa.topics is empty: add topic counts under sampling.simpleqa in the YAML config.")
    ds_sq = load_dataset(sq_hf, split=sq_split)

    # Index rows by topic
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in ds_sq:
        r = dict(row)
        meta = ast.literal_eval(r["metadata"])
        topic = str(meta["topic"])
        by_topic.setdefault(topic, []).append((r, meta))

    simpleqa_sampled: dict[str, int] = {}
    for topic, need in topic_counts.items():
        pool = by_topic.get(topic, [])
        if need > len(pool):
            raise ValueError(f"SimpleQA topic {topic!r}: need {need}, only {len(pool)} available.")
        # Deterministic per-topic sample
        rng = random.Random(seed + sum(ord(c) for c in f"simpleqa::{topic}"))
        rng.shuffle(pool)
        chosen = pool[:need]
        for r, meta in chosen:
            items.append(build_simpleqa_item(r, meta, len(items)))
        simpleqa_sampled[topic] = len(chosen)

    meta_out = {
        "seed": seed,
        "hle_stratum_counts": stratum_counts,
        "simpleqa_topic_counts": simpleqa_sampled,
        "total_items": len(items),
    }
    return items, meta_out


# --- Evaluation (LLM judge) ---


def judge_correctness_misc(
    judge_client: OpenAI,
    judge_model: str,
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    use_litellm: bool = False,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT_TEMPLATE_MISC.format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
    )
    if use_litellm:
        request_kwargs: dict[str, Any] = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_INSTRUCTIONS_MISC},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = judge_client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "response_format" in request_kwargs and "unsupported" in message and "response_format" in message:
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("response_format", None)
                resp = judge_client.chat.completions.create(**retry_kwargs)
            else:
                raise
        raw_text = extract_chat_content(resp.choices[0].message.content).strip()
    else:
        resp = judge_client.responses.create(
            model=judge_model,
            instructions=JUDGE_SYSTEM_INSTRUCTIONS_MISC,
            input=prompt,
            max_output_tokens=512,
            store=False,
        )
        raw_text = resp.output_text.strip()
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        correct = bool(parsed.get("correct", False))
        reason = str(parsed.get("reason", ""))
    except json.JSONDecodeError:
        lower = cleaned.lower()
        correct = '"correct": true' in lower or '"correct":true' in lower
        reason = cleaned[:500]
    return {
        "correct": correct,
        "method": "llm_judge",
        "judge_raw": raw_text,
        "judge_reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate models on static misc benchmarks (HLE strata + SimpleQA).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config; evaluation settings come from the 'evaluation' section.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=[spec.label for spec in MODEL_SPECS],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--sample-file",
        default=None,
        help="Path to sampled_items.jsonl from sample_static_misc.py.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenAI (or routed) judge model.",
    )
    parser.add_argument(
        "--generation-timeout",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--generation-max-tokens",
        type=int,
        default=None,
        dest="generation_max_tokens",
    )
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--use-litellm",
        action="store_true",
    )
    parser.add_argument(
        "--litellm-models",
        nargs="+",
        default=None,
        choices=[spec.label for spec in MODEL_SPECS],
    )
    return parser.parse_args()


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    section = config.get("evaluation", {})

    if args.models is None:
        args.models = section.get("models", [spec.label for spec in MODEL_SPECS])
    if args.sample_file is None:
        args.sample_file = section.get("sample_file")
        if args.sample_file is None:
            sampling_output_dir = config.get("sampling", {}).get("output_dir")
            if sampling_output_dir:
                args.sample_file = str(Path(sampling_output_dir) / "sampled_items.jsonl")
    if args.judge_model is None:
        args.judge_model = section.get("judge_model", "gpt-4.1-mini")
    if args.generation_timeout is None:
        args.generation_timeout = int(section.get("generation_timeout", 120))
    if args.generation_max_tokens is None:
        args.generation_max_tokens = int(section.get("generation_max_tokens", 32768))
    if args.generation_retries is None:
        args.generation_retries = int(section.get("generation_retries", 2))
    args.model_max_tokens = {
        key: (int(value) if value is not None else None)
        for key, value in section.get("model_max_tokens", {}).items()
    }
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")
    if args.save_every == 50:
        args.save_every = int(section.get("save_every", 50))
    if not args.resume:
        args.resume = bool(section.get("resume", False))
    if args.max_workers == 4:
        args.max_workers = int(section.get("max_workers", 4))
    if not args.use_litellm:
        args.use_litellm = bool(section.get("use_litellm", False))
    if args.litellm_models is None and section.get("litellm_models") is not None:
        args.litellm_models = list(section.get("litellm_models", []))

    if not args.sample_file:
        raise SystemExit("Error: --sample-file is required, either via CLI or config file.")
    return args


def evaluate_item(
    item: dict[str, Any],
    *,
    selected_model_specs: list[Any],
    clients: dict[str, Any],
    judge_model: str,
    completed_keys: set[str],
    generation_max_tokens: int | None,
    generation_retries: int,
    model_max_tokens: dict[str, int | None] | None,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    item_results: list[tuple[str, dict[str, Any]]] = []

    for model_spec in selected_model_specs:
        eval_key = make_eval_key(item, model_spec.label)
        if eval_key in completed_keys:
            continue

        record: dict[str, Any] = {
            "dataset": item["dataset"],
            "dataset_kind": item["dataset_kind"],
            "sample_index": item["sample_index"],
            "model_label": model_spec.label,
            "model_provider": model_spec.provider,
            "model_id": model_spec.model_id,
            "level": item.get("level"),
            "subject": item.get("subject"),
            "question": item["question"],
            "prompt": item["prompt"],
            "gold_answer": item["gold_answer"],
            "status": "ok",
            "correct": None,
            "grading_method": "llm_judge",
            "response_text": None,
            "judge_reason": None,
            "judge_raw": None,
            "latency_s": None,
            "generation_attempts": 0,
        }

        started = time.time()
        effective_max_tokens = (model_max_tokens or {}).get(model_spec.label, generation_max_tokens)
        response_text: str | None = None
        last_generation_error: str | None = None
        for attempt in range(generation_retries + 1):
            record["generation_attempts"] = attempt + 1
            try:
                candidate = call_model(
                    model_spec,
                    clients,
                    item["prompt"],
                    generation_max_tokens=effective_max_tokens,
                    use_litellm=use_litellm,
                    litellm_models=litellm_models,
                )
                if candidate.strip():
                    response_text = candidate
                    break
                last_generation_error = "Model returned an empty response after extraction."
            except Exception as exc:
                last_generation_error = str(exc)
            if attempt < generation_retries:
                time.sleep(min(2**attempt, 4))

        record["latency_s"] = round(time.time() - started, 2)
        if response_text is None:
            record["status"] = "generation_error"
            record["correct"] = False
            record["grading_method"] = "generation_error"
            record["judge_reason"] = last_generation_error or "Unknown generation failure."
            item_results.append((eval_key, record))
            continue
        record["response_text"] = response_text

        judge_spec = MODEL_LOOKUP.get(judge_model)
        judge_uses_litellm = (
            judge_spec is not None
            and _should_use_litellm(judge_spec, use_litellm=use_litellm, litellm_models=litellm_models)
        )
        effective_judge_model = (
            get_routed_model_id(judge_spec, use_litellm=True)
            if judge_uses_litellm and judge_spec is not None
            else judge_model
        )
        judged = judge_correctness_misc(
            clients["litellm"] if judge_uses_litellm else clients["openai"],
            effective_judge_model,
            question=item["question"],
            gold_answer=item["gold_answer"],
            model_answer=record["response_text"] or "",
            use_litellm=judge_uses_litellm,
        )
        record["correct"] = judged["correct"]
        record["grading_method"] = judged["method"]
        record["judge_reason"] = judged["judge_reason"]
        record["judge_raw"] = judged["judge_raw"]
        item_results.append((eval_key, record))

    return item_results


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    from statistics import mean

    print("\nAccuracy summary")
    print("-" * 80)
    for row in summary_rows:
        accuracy = row["accuracy"]
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        print(
            f"{row['model_label']:24} {row['dataset']:32} "
            f"acc={accuracy_text:>8} "
            f"scored={row['num_scored']:4}/{row['num_items']:4} "
            f"judge={row['judge_graded']:4} errors={row['generation_errors']:3}"
        )
    print("-" * 80)


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)

    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_model_specs = [MODEL_LOOKUP[label] for label in args.models]
    litellm_models = set(args.litellm_models) if args.litellm_models else None
    selected_labels = {spec.label for spec in selected_model_specs}
    if litellm_models is not None:
        unselected = sorted(litellm_models - selected_labels)
        if unselected:
            sys.exit(
                "LiteLLM subset includes models not in --models/config: " + ", ".join(unselected)
            )
        unsupported_subset = sorted(
            label for label in litellm_models if MODEL_LOOKUP[label].litellm_model_id is None
        )
        if unsupported_subset:
            sys.exit(
                "LiteLLM is not configured for these subset models: " + ", ".join(unsupported_subset)
            )
    elif args.use_litellm:
        unsupported = [spec.label for spec in selected_model_specs if spec.litellm_model_id is None]
        if unsupported:
            sys.exit("LiteLLM is not configured for: " + ", ".join(unsupported))

    clients = build_clients_with_mode(
        args.models,
        args.generation_timeout,
        use_litellm=args.use_litellm,
        litellm_models=litellm_models,
    )
    sample_file = Path(args.sample_file)
    sampled_items = load_jsonl(sample_file)
    dataset_counts: dict[str, int] = {}
    for item in sampled_items:
        dataset_counts[item["dataset"]] = dataset_counts.get(item["dataset"], 0) + 1

    run_config = {
        "models": args.models,
        "sample_file": str(sample_file),
        "datasets": sorted(dataset_counts.keys()),
        "sample_plan": dataset_counts,
        "judge_model": args.judge_model,
        "generation_timeout": args.generation_timeout,
        "generation_max_tokens": args.generation_max_tokens,
        "generation_retries": args.generation_retries,
        "model_max_tokens": args.model_max_tokens,
        "use_litellm": args.use_litellm,
        "litellm_models": sorted(litellm_models) if litellm_models is not None else None,
        "max_workers": args.max_workers,
        "output_dir": str(output_dir),
        "resume": args.resume,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    responses_path = output_dir / "responses.jsonl"
    results: list[dict[str, Any]] = load_jsonl(responses_path) if args.resume else []
    completed_keys = {make_eval_key(record) for record in results}
    eval_order_lookup = build_eval_order_lookup(sampled_items, selected_model_specs)
    total_evals = len(sampled_items) * len(selected_model_specs)
    completed_evals = len(completed_keys)
    progress = tqdm(
        total=total_evals,
        initial=completed_evals,
        desc="Evaluating static misc",
        unit="eval",
    )

    pending_items = [
        item
        for item in sampled_items
        if any(
            make_eval_key(item, model_spec.label) not in completed_keys
            for model_spec in selected_model_specs
        )
    ]
    completed_keys_snapshot = set(completed_keys)

    def record_result(eval_key: str, record: dict[str, Any]) -> None:
        nonlocal completed_evals
        results.append(record)
        completed_keys.add(eval_key)
        progress.set_postfix_str(
            f"{record['model_label']} | {record['dataset']} #{record['sample_index'] + 1}",
            refresh=False,
        )
        progress.update(1)
        completed_evals += 1
        if args.save_every > 0 and completed_evals % args.save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                results=results,
                eval_order_lookup=eval_order_lookup,
                completed_evals=completed_evals,
                total_evals=total_evals,
            )

    if args.max_workers <= 1 or len(pending_items) <= 1:
        for item in pending_items:
            item_results = evaluate_item(
                item,
                selected_model_specs=selected_model_specs,
                clients=clients,
                judge_model=args.judge_model,
                completed_keys=completed_keys_snapshot,
                generation_max_tokens=args.generation_max_tokens,
                generation_retries=args.generation_retries,
                model_max_tokens=args.model_max_tokens,
                use_litellm=args.use_litellm,
                litellm_models=litellm_models,
            )
            for eval_key, record in item_results:
                record_result(eval_key, record)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_item,
                    item,
                    selected_model_specs=selected_model_specs,
                    clients=clients,
                    judge_model=args.judge_model,
                    completed_keys=completed_keys_snapshot,
                    generation_max_tokens=args.generation_max_tokens,
                    generation_retries=args.generation_retries,
                    model_max_tokens=args.model_max_tokens,
                    use_litellm=args.use_litellm,
                    litellm_models=litellm_models,
                )
                for item in pending_items
            ]
            for future in concurrent.futures.as_completed(futures):
                item_results = future.result()
                for eval_key, record in item_results:
                    record_result(eval_key, record)

    progress.close()
    ordered_results = order_results(results, eval_order_lookup)
    summary_rows = summarize_results(ordered_results)
    save_checkpoint(
        output_dir=output_dir,
        results=results,
        eval_order_lookup=eval_order_lookup,
        completed_evals=completed_evals,
        total_evals=total_evals,
    )
    print_summary(summary_rows)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
