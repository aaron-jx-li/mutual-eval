# Method

## Motivation

Standard LLM benchmarks produce a single aggregate accuracy score per model, collapsing all questions into one number. This ignores the fact that questions differ in difficulty and discriminating power — an easy question that every model answers correctly carries no ranking signal, while a hard question that separates the top tier is highly informative. We adopt **Item Response Theory (IRT)**, a psychometric framework originally designed to estimate student ability and question difficulty from test responses, to jointly infer latent model ability and per-question parameters from two complementary signal sources: static benchmarks with binary correctness labels, and arena-style evaluations with continuous reward scores.

---

## Technical Formulation

### Parameters

Every model $i$ is assigned a scalar **ability** $\theta_i \in \mathbb{R}$. Every question $q$ has two item parameters:

- $b_q$ — **difficulty**: the ability level at which a model's expected performance is at the neutral point
- $a_q = \exp(k_q) > 0$ — **discrimination**: how sharply the question separates high- and low-ability models (parameterised via $k_q$ to enforce positivity)

All parameters are learned jointly by gradient descent. After each update, $\theta$ is re-centred to zero mean (and $b$ shifted by the same amount) to fix the scale.

### Static benchmark signal (binary correct/incorrect)

The 2-parameter logistic (2PL) IRT model treats correctness as a Bernoulli observation:

$$P(\text{correct}_{i,q}) = \sigma\!\left(a_q\,(\theta_i - b_q)\right)$$

Loss: binary cross-entropy (BCE).

### Arena reward signal

A reward model scores each response on a continuous scale. We consider two ways to incorporate this signal into the IRT framework.

**Option A — Direct reward regression.** Because arena reward scores lie on a roughly logit-like scale (empirically $\approx [-2.5,\ 2.8]$, centred near 0), they map naturally onto the IRT logit axis. We treat the reward as a direct noisy observation of the latent performance:

$$r_{i,q} = a_q\,(\theta_i - b_q) + \varepsilon$$

Loss: mean squared error (MSE). This is the most faithful use of the cardinal reward signal, but is sensitive to reward miscalibration or scale inconsistencies across questions.

**Option B — Soft pairwise distillation (alternative).** Rather than regressing on raw values, rewards are first standardised into global z-scores $z_{i,q} = (r_{i,q} - \bar{r}) / \sigma_r$, then converted into soft pairwise preference targets for every model pair on each question:

$$p^*_{ijq} = \sigma(z_{i,q} - z_{j,q})$$

The IRT model then fits these soft targets via a nested pairwise likelihood, with an additional learned scalar $\gamma > 0$ that controls the sharpness of the preference signal:

$$\hat{P}(i \succ j \mid q) = \sigma\!\left(\gamma \cdot a_q\,(\pi_{i,q} - \pi_{j,q})\right), \quad \pi_{i,q} = \sigma(\theta_i - b_q)$$

Loss: BCE against $p^*_{ijq}$. The sigmoid compression of z-score differences makes this formulation more robust to reward outliers than direct MSE, while still preserving magnitude information (unlike hard win/loss labels): a reward gap of 0.01 produces a target near 0.5, a gap of 2.0 produces a target near 1.0. The both-bad penalty is applied when both models' z-scores fall below a threshold $\tau$, pushing $\pi_{i,q}$ and $\pi_{j,q}$ below the question's difficulty level.

### Joint model

When both signal sources are available they share the same $(\theta, b, k)$ embeddings and are trained with a weighted composite loss. For Option A:

$$\mathcal{L} = \lambda_{\text{static}}\,\mathcal{L}_{\text{BCE}} + \lambda_{\text{reward}}\,\mathcal{L}_{\text{MSE}} + \lambda_{\text{reg}}\left(\|\theta\|^2 + \|b\|^2 + \|k\|^2\right)$$

For Option B, $\mathcal{L}_{\text{MSE}}$ is replaced by the soft pairwise BCE loss, and $\gamma$ is included in the regularisation term.

The reward data improves ability and difficulty estimates even for (model, question) pairs not covered by the static benchmark. The final ranking is the sorted $\theta$ vector.

---

## Experiment Setup

All three experiments evaluate the same set of **15 frontier models**: GPT-5.4, GPT-5-mini, GPT-4.1, GPT-4.1-mini, Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5, Gemini 3.1 Pro, Gemini 2.5 Pro, Gemini 2.5 Flash, Grok-4, DeepSeek-V3.2, Mistral Large 3, Qwen3-Max (thinking), and LLaMA-4 Maverick Instruct.

### math_v0 — joint static + arena reward

**Static benchmark (500 questions across 13 datasets):**

| Dataset | Questions |
|---|---|
| GSM8K | 50 |
| MATH (algebra, counting, geometry, intermediate, number theory, prealgebra, precalculus) | 270 |
| MMLU (abstract algebra, college mathematics) | 40 |
| AIME 2025 / 2026 | 60 |
| Olympiad-Math | 80 |

Correctness is graded by GPT-4.1-mini as judge. The joint loss combines BCE (static) and MSE (reward).

**Arena reward (894 questions):** Questions sourced from Chatbot Arena math conversations. Each model generates a response, which is scored by a reward model. The reward signal supplements the static benchmark, especially on the hardest questions where most models fail.

### coding_v0 — joint static + arena reward

**Static benchmark (380 questions across 3 datasets):**

| Dataset | Questions |
|---|---|
| HumanEval-Plus | 80 |
| MBPP-Plus (sanitized) | 140 |
| LiveCodeBench v6 | 160 |

Correctness is determined by execution against test suites (no LLM judge). The joint loss combines BCE (static) and MSE (reward).

**Arena reward (500 questions):** Questions sourced from Chatbot Arena coding conversations (expert-annotated subset). Each model generates a response scored by the reward model.

### generic_v0 — arena reward only

No static benchmark is available for general-purpose instruction following. Ranking is based purely on reward regression IRT.

**Arena reward (500 questions):** Questions sampled from a 140k Chatbot Arena conversation pool covering diverse real-user prompts. Each model generates a response scored by the reward model. The IRT model infers $\theta_i$, $b_q$, and $a_q$ from the reward observations alone.
