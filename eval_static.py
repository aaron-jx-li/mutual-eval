"""
Evaluate multiple LLMs on math benchmark datasets using an LLM judge.

Usage examples:
    # Run with defaults (all models, all datasets)
    python eval_static.py

    # Evaluate specific models on specific datasets
    python eval_static.py --models gpt-4o-mini grok-3-mini-beta --datasets MMLU-abstract GSM-8k

    # Custom output path, judge model, and random seed
    python eval_static.py --output results.csv --judge-model gpt-4.1-mini --seed 42

    # Limit Hendrycks sample size
    python eval_static.py --hendrycks-samples 50

Required environment variables (set the ones needed for your chosen models):
    OPENAI_API_KEY      - For gpt-4o-mini and the judge model
    XAI_API_KEY         - For grok-3-mini-beta
    ANTHROPIC_API_KEY   - For claude-3-5-haiku-20241022
    GOOGLE_API_KEY      - For gemini-2.0-flash
    HF_TOKEN            - For llama-3.3-70b-it
"""

import argparse
import csv
import os
import random
import sys
from statistics import mean

import anthropic
import google.generativeai as genai
from datasets import load_dataset
from huggingface_hub import InferenceClient
from openai import OpenAI


# ── All available models ──────────────────────────────────────────────
ALL_MODELS = [
    "gpt-4o-mini",
    "grok-3-mini-beta",
    "claude-3-5-haiku-20241022",
    "gemini-2.0-flash",
    "llama-3.3-70b-it",
]

# ── All available datasets ────────────────────────────────────────────
ALL_DATASETS = [
    "MMLU-abstract",
    "MMLU-college",
    "GSM-8k",
    "Hendrycks-Algebra",
    "Hendrycks-Counting",
]


# ── CLI argument parsing ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on math benchmarks with an LLM judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        metavar="MODEL",
        help=f"Models to evaluate (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        metavar="DATASET",
        help=f"Datasets to use (default: all). Choices: {', '.join(ALL_DATASETS)}",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="Model used as the judge (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.csv",
        help="Output CSV file path (default: evaluation_results.csv)",
    )
    parser.add_argument(
        "--hendrycks-samples",
        type=int,
        default=100,
        help="Number of samples to draw from each Hendrycks dataset (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    return parser.parse_args()


# ── Client initialisation ────────────────────────────────────────────
def init_clients(models: list[str]) -> dict:
    """Lazily initialise only the API clients required by the chosen models."""
    clients: dict = {}

    needs_openai = any("gpt" in m for m in models)
    needs_xai = any("grok" in m for m in models)
    needs_anthropic = any("claude" in m for m in models)
    needs_google = any("gemini" in m for m in models)
    needs_hf = any("llama" in m for m in models)

    if needs_openai:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("Error: OPENAI_API_KEY environment variable is not set.")
        clients["openai"] = OpenAI(api_key=key)

    if needs_xai:
        key = os.environ.get("XAI_API_KEY")
        if not key:
            sys.exit("Error: XAI_API_KEY environment variable is not set.")
        clients["xai"] = OpenAI(base_url="https://api.x.ai/v1", api_key=key)

    if needs_anthropic:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")
        clients["anthropic"] = anthropic.Anthropic(api_key=key)

    if needs_google:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("Error: GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=key)

    if needs_hf:
        key = os.environ.get("HF_TOKEN")
        if not key:
            sys.exit("Error: HF_TOKEN environment variable is not set.")
        clients["hf"] = InferenceClient(provider="novita", api_key=key)

    # The judge always uses OpenAI
    if "openai" not in clients:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("Error: OPENAI_API_KEY environment variable is required for the judge model.")
        clients["openai"] = OpenAI(api_key=key)

    return clients


# ── Answer generators ─────────────────────────────────────────────────
def generate_model_answer(clients: dict, model_name: str, question: str) -> str:
    """Route a question to the correct API and return the model's answer."""
    try:
        if "gpt" in model_name:
            resp = clients["openai"].responses.create(
                model="gpt-4o-mini",
                instructions="Answer concisely with only the final result.",
                input=question,
                max_output_tokens=1000,
                store=False,
            )
            return resp.output_text.strip()

        elif "grok" in model_name:
            resp = clients["xai"].chat.completions.create(
                model="grok-3-mini-beta",
                messages=[{"role": "user", "content": question}],
                max_tokens=1000,
            )
            return resp.choices[0].message.content.strip()

        elif "claude" in model_name:
            resp = clients["anthropic"].messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"{question}\nAnswer concisely with only the final answer."}
                ],
            )
            return resp.content[0].text.strip()

        elif "gemini" in model_name:
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(
                f"{question}\nAnswer concisely with only the final answer.",
                generation_config=genai.types.GenerationConfig(max_output_tokens=1000),
            )
            return resp.text.strip()

        elif "llama" in model_name:
            resp = clients["hf"].chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=[
                    {"role": "system", "content": "Answer concisely with only the final result."},
                    {"role": "user", "content": question},
                ],
                max_tokens=1000,
            )
            return resp.choices[0].message.content.strip()

        else:
            return "UNKNOWN MODEL"

    except Exception as e:
        print(f"[Warning] {model_name} failed: {e}")
        return "ERROR"


# ── Judge ─────────────────────────────────────────────────────────────
def judge_answer(
    clients: dict,
    judge_model: str,
    question: str,
    correct_answer: str,
    model_answer: str,
) -> int:
    """Use the judge model to decide correctness (1 = correct, 0 = incorrect)."""
    prompt = (
        "You are an accurate grader.\n\n"
        f"Question: {question}\n"
        f"Ground Truth Answer: {correct_answer}\n"
        f"Model Answer: {model_answer}\n\n"
        "Decide if the model answer means the same as the correct one.\n"
        'If correct, return ONLY "1".\n'
        'If incorrect, return ONLY "0".\n'
        "No explanations."
    )
    try:
        resp = clients["openai"].responses.create(
            model=judge_model,
            input=prompt,
            max_output_tokens=5,
            store=False,
        )
        result = resp.output_text.strip()
        return int(result) if result in ("0", "1") else 0
    except Exception as e:
        print(f"[Warning] Judge failed: {e}")
        return 0


# ── Dataset helpers ───────────────────────────────────────────────────
def sample_balanced(dataset, n_samples: int = 100):
    """Sample evenly across difficulty levels."""
    levels = list(set(dataset["level"]))
    per_level = max(1, n_samples // len(levels))
    samples = []
    for lvl in levels:
        subset = [ex for ex in dataset if ex["level"] == lvl]
        samples.extend(random.sample(subset, min(per_level, len(subset))))
    return samples[:n_samples]


def load_datasets(
    selected: list[str], hendrycks_samples: int
) -> dict[str, tuple[list, int]]:
    """Load and return only the requested datasets."""
    print("Loading datasets...")
    cfg: dict[str, tuple[list, int]] = {}

    if "MMLU-abstract" in selected:
        ds = load_dataset("brucewlee1/mmlu-abstract-algebra")["test"]
        cfg["MMLU-abstract"] = (ds, len(ds))

    if "MMLU-college" in selected:
        ds = load_dataset("brucewlee1/mmlu-college-mathematics")["test"]
        cfg["MMLU-college"] = (ds, len(ds))

    if "GSM-8k" in selected:
        ds = load_dataset("openai/gsm8k", "main")["test"]
        cfg["GSM-8k"] = (ds, len(ds))

    if "Hendrycks-Algebra" in selected:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra")["test"]
        cfg["Hendrycks-Algebra"] = (sample_balanced(ds, hendrycks_samples), hendrycks_samples)

    if "Hendrycks-Counting" in selected:
        ds = load_dataset("EleutherAI/hendrycks_math", "counting_and_probability")["test"]
        cfg["Hendrycks-Counting"] = (sample_balanced(ds, hendrycks_samples), hendrycks_samples)

    return cfg


def load_question(dataset_name: str, ds, idx: int) -> tuple[str, str, str]:
    """Extract (question, correct_answer, level) from a dataset row."""
    item = ds[idx % len(ds)]
    level = item.get("level", "N/A") if isinstance(item, dict) else "N/A"

    if "GSM" in dataset_name:
        question = item["question"]
        correct_answer = item["answer"]
    elif "MMLU" in dataset_name:
        question = item["centerpiece"]
        correct_answer = item["correct_options_literal"][0]
    else:  # Hendrycks
        question = item["problem"]
        correct_answer = item["solution"]

    return question, correct_answer, level


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Models   : {', '.join(args.models)}")
    print(f"Datasets : {', '.join(args.datasets)}")
    print(f"Judge    : {args.judge_model}")
    print(f"Output   : {args.output}")
    print()

    clients = init_clients(args.models)
    datasets_cfg = load_datasets(args.datasets, args.hendrycks_samples)

    results: list[dict] = []

    for dataset_name, (dataset, n) in datasets_cfg.items():
        for i in range(n):
            question, correct_answer, level = load_question(dataset_name, dataset, i)
            qid = f"{dataset_name}_{i + 1}"

            for model_name in args.models:
                print(f"Evaluating {model_name} on {dataset_name} Q{i + 1}/{n}")
                model_answer = generate_model_answer(clients, model_name, question)
                score = judge_answer(
                    clients, args.judge_model, question, correct_answer, model_answer
                )

                results.append(
                    {
                        "question_id": qid,
                        "question": question,
                        "correct_answer": correct_answer,
                        "model_name": model_name,
                        "model_answer": model_answer,
                        "judge_result": score,
                        "level": level,
                    }
                )

    # ── Save CSV ──────────────────────────────────────────────────────
    fieldnames = [
        "question_id",
        "question",
        "model_name",
        "correct_answer",
        "model_answer",
        "judge_result",
        "level",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation complete. Results saved to {args.output}")

    # ── Accuracy Summary ──────────────────────────────────────────────
    print("\nAccuracy Summary:")
    print("-" * 45)
    model_groups: dict[str, list[int]] = {}
    for r in results:
        model_groups.setdefault(r["model_name"], []).append(r["judge_result"])
    for m, vals in model_groups.items():
        acc = mean(vals)
        print(f"  {m:30s} {acc:.2%}")
    print("-" * 45)


if __name__ == "__main__":
    main()
