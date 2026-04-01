# robustness/

This directory contains experiments that stress-test IRT-based model rankings under data sparsity and label corruption. The core question is: **how sparse can the evaluation dataset be while still producing reliable model rankings?**

All experiments run on the **math task only** (`data/static_10_models.csv`: 500 questions × 10 models). The reference ranking is always a full-data IRT fit from `ranking.fit_static_irt`.


## Files

| File | Purpose |
|---|---|
| `data_utils.py` | Data loading and subsampling helpers (random rows, questions, pairs, model drops) |
| `metrics.py` | Wrappers around `ranking.py`: fit IRT on a sparse df and compute Spearman ρ vs. reference |
| `common_cli.py` | Shared argparse flags (`--static-csv`, `--arena-csv`, `--out-dir`, `--seeds`, etc.) |
| `seed_stability.py` | Verify IRT is stable across random seeds before running any experiments |
| `sparsity_random.py` | Exp 1: randomly subsample questions and measure rank correlation vs. fraction kept |
| `new_model_ranking.py` | Exp 2: LOO simulation — how few questions are needed to rank a new (held-out) model? |
| `new_question_calibration.py` | Exp 3: how few model responses are needed to estimate a new question's b_q and a_q? |
| `sparsity_stratified.py` | Exp 4: level-stratified sampling — which difficulty levels are most informative? |
| `label_noise.py` | Exp 5: label corruption — which corruptions most destabilize rankings? |
| `add_hendrycks_metadata.py` | Exp 6 (stretch): extract Hendrycks MATH categories and add `category` column for category-stratified sampling |
| `greedy_selection.py` | Exp 7: forward greedy question selection — finds minimal question set that recovers full-data rankings |
| `run_experiments.py` | Top-level CLI orchestrator to run any or all experiments |
| `results/` | CSV outputs and plots from each experiment |


## Expected Results

| Model | Accuracy | 2PL | Pairwise IRT (Static) | Pairwise IRT (Open-ended) | 2PL+Pairwise IRT (Overall) |
|---|---|---|---|---|---|
| Deepseek-v3-0324              | 3  | 3  | 3  | 1  | 1  |
| GPT-4.1-mini-2025-04-14       | 2  | 2  | 2  | 2  | 2  |
| Mistral-medium-2505           | 4  | 4  | 4  | 3  | 3  |
| Grok-3-mini-beta              | 1  | 1  | 1  | 5  | 4  |
| Claude-3.5-haiku-20241022     | 5  | 5  | 5  | 8  | 5  |
| GPT-4o-mini                   | 6  | 7  | 7  | 6  | 6  |
| Llama-3.3-70b-it              | 8  | 6  | 6  | 9  | 7  |
| Gemini-2.0-flash              | 7  | 8  | 8  | 10 | 8  |
| Claude-3.7-sonnet-20250219    | 9  | 9  | 9  | 4  | 9  |
| Gemma-3-27b-it                | 10 | 10 | 10 | 7  | 10 |

- 2PL: Uses data in `data/static_10_models.csv` to generate rankings (500 questions, 10 models, 5000 rows - 1 row per question per model)
- Pairwise IRT (Open-ended): Uses data in `data/pairwise_results_900.csv` to generate rankings (892 questions, 10 models, 892 rows - 1 row per question)
- 2PL+Pairwise IRT (Overall): Gives equal 0.50 weightage to both 2PL and Pairwise IRT to generate rankings


## Experiments

### Exp 0 — Seed Stability
Runs `fit_static_irt` on the full dataset with multiple random seeds and computes pairwise Spearman ρ across seeds. This confirms that IRT optimization converges to the same solution regardless of initialization, which is a prerequisite for treating any single fit as a reliable reference.

### Exp 1 — Random Question Subsampling
Randomly keeps a fraction `f` of the 500 questions (all 10 model responses retained per question) and re-fits IRT. Produces a ρ-vs-fraction curve to find the "elbow" — the minimum number of questions that still recovers the full-data ranking. This is the baseline sparsification result.

### Exp 2 — New Model Ranking
Simulates adding an 11th model to an existing benchmark. Uses leave-one-out: hold out model `i`, vary the question budget `k ∈ [10, 25, 50, 100, 200, 300, 500]`, fit IRT on all 10 models with only `k` questions, and measure how accurately the held-out model's rank is recovered. Repeated across all 10 models and 5 seeds per `(model, k)` pair to get stable estimates of rank error.

### Exp 3 — New Question Calibration
Simulates adding a new question to an existing item bank. Holds out one question at a time, varies how many of the 10 models answer it (`k ∈ [2, 3, 4, 5, 6, 8, 10]`), re-fits IRT, and measures MAE of the estimated `b_q` (difficulty) and `a_q` (discrimination) vs. the full-data reference. Identifies the minimum number of model responses needed to reliably characterize a question's psychometric properties.

### Exp 4 — Level-Stratified Sampling
Tests whether the difficulty composition of the question set matters. Uses the pre-existing `level` column (NaN = GSM-8k word problems; Levels 1–5 = Hendrycks MATH) to construct subsets with different level mixes (e.g., GSM-only, Hendrycks-only, hard-only, uniform across levels). Answers: "if you have a fixed question budget, which difficulty levels should you prioritize?"

### Exp 5 — Label Corruption
Deliberately corrupts `judge_result` labels and measures ranking degradation. Tests both random flips (uniform noise) and targeted flips (easy-only, hard-only, or all responses for one model). The per-model inflation/deflation variants identify which models' positions are most sensitive to systematic annotation errors.

### Exp 6 — Hendrycks Category Metadata *(stretch)*
Tests whether category-stratified sampling (Algebra, Number Theory, Geometry, etc.) beats level-stratified sampling for a fixed question budget. First checks whether Hendrycks MATH categories can be recovered from `question_id` structure or by matching question text against the public Hendrycks MATH dataset. If feasible, adds a `category` column via `add_hendrycks_metadata.py` and reruns Exp 4-style stratification schemes by category instead of level.

### Exp 7 — Greedy Question Selection
Forward greedy search for the minimal question set that recovers the full-data model ranking. Starting from one random question, iteratively adds the question whose inclusion maximizes Kendall's τ against the full-data reference ranking. Terminates when τ > 0.95 for 3 consecutive steps or a hard cap is reached (150 for 2PL, 200 for pairwise IRT). Produces a τ-vs-question-count curve that serves as a lower bound on the question budget required for reliable rankings, directly comparable to the Exp 1 random baseline. The level distribution of greedily selected questions also informs which difficulty levels are most informative (connecting to Exp 4).


## How to Run

```bash
# 0. Baseline: verify seed stability
python robustness/seed_stability.py --static-csv data/static_10_models.csv

# 1. Random question subsampling
python robustness/sparsity_random.py \
    --mode static --sparsity-type questions \
    --fractions 0.05 0.1 0.2 0.3 0.5 0.7 \
    --seeds 0 1 2 3 4

# 2. New model ranking (Scenario A)
python robustness/new_model_ranking.py --static-csv data/static_10_models.csv

# 3. New question calibration (Scenario B)
python robustness/new_question_calibration.py --static-csv data/static_10_models.csv

# 4. Level-stratified sampling
python robustness/sparsity_stratified.py --static-csv data/static_10_models.csv

# 5. Label corruption
python robustness/label_noise.py --static-csv data/static_10_models.csv

# 6. Hendrycks category metadata (stretch — check feasibility first)
python robustness/add_hendrycks_metadata.py --static-csv data/static_10_models.csv

# 7. Greedy question selection
python robustness/greedy_selection.py --static-csv data/static_10_models.csv --hard-cap 150

# All results are written to robustness/results/
```
