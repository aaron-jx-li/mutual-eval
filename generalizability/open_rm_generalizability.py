#!/usr/bin/env python3
"""
Generalizability check: open-source reward models vs Arena RM.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RANKING_DIR = REPO_ROOT / "ranking"
if str(RANKING_DIR) not in sys.path:
    sys.path.insert(0, str(RANKING_DIR))

from rank_v1 import build_soft_pairwise_targets, fit_irt_v1, resolve_pairwise_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate open-source reward models against Arena RM across domains.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["math", "coding", "misc", "generic"],
        choices=["math", "coding", "misc", "generic"],
        help="Domains to evaluate.",
    )
    parser.add_argument(
        "--reward-models",
        nargs="+",
        required=True,
        help="HF reward model names to run.",
    )
    parser.add_argument(
        "--arena-jsonl-dir",
        default="differential_test/hf_dataset/v1",
        help="Directory containing v1_arena_<domain>.jsonl files.",
    )
    parser.add_argument(
        "--arena-ranking-dir",
        default="differential_test/ranking_rm_results/v1",
        help="Directory containing Arena-RM IRT outputs per domain.",
    )
    parser.add_argument(
        "--output-dir",
        default="generalizability/results/open_rm",
        help="Root output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Reward model batch size.")
    parser.add_argument("--max-length", type=int, default=4096, help="Tokenizer max length.")
    parser.add_argument("--max-rows-per-domain", type=int, default=None, help="Debug/smoke-test row cap.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype for HF loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for HF reward models.",
    )
    parser.add_argument(
        "--logit-index",
        type=int,
        default=-1,
        help="If model outputs multi-logit classification head, select this index.",
    )
    parser.add_argument("--num-epochs", type=int, default=2500, help="IRT training epochs.")
    parser.add_argument("--lr", type=float, default=0.02, help="IRT learning rate.")
    parser.add_argument("--reg-lambda", type=float, default=1e-4, help="IRT regularization.")
    parser.add_argument("--lambda-arena", type=float, default=1.0, help="Pairwise IRT loss weight.")
    parser.add_argument("--lambda-reg", type=float, default=1.0, help="Regression IRT loss weight.")
    parser.add_argument(
        "--bb-ratio",
        type=float,
        default=None,
        help="Target both-bad pair ratio for pairwise target construction.",
    )
    parser.add_argument(
        "--tie-ratio",
        type=float,
        default=None,
        help="Target tie pair ratio for pairwise target construction.",
    )
    parser.add_argument(
        "--both-bad-threshold",
        type=float,
        default=-0.5,
        help="Fallback both-bad threshold when bb-ratio is unset.",
    )
    parser.add_argument(
        "--arena-tie-threshold",
        type=float,
        default=0.0,
        help="Tie threshold for Arena RM pairwise sign checks.",
    )
    parser.add_argument(
        "--open-tie-threshold",
        type=float,
        default=0.0,
        help="Tie threshold for open RM pairwise sign checks.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rm_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")


MODEL_NAME_ALIASES: dict[str, str] = {
    "NCSOFT_Llama-3-OffsetBias-RM-8B": "NCSOFT/Llama-3-OffsetBias-RM-8B",
    "Skywork_Skywork-Reward-V2-Qwen3-8B": "Skywork/Skywork-Reward-V2-Qwen3-8B",
    "OpenAssistant_reward-model-deberta-v3-large-v2": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "RLHFlow_ArmoRM-Llama3-8B-v0.1": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
}


def _corr_pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    x_vals = x.to_numpy(dtype=float)
    y_vals = y.to_numpy(dtype=float)
    if np.isclose(x_vals.std(ddof=0), 0.0) or np.isclose(y_vals.std(ddof=0), 0.0):
        return float("nan")
    return float(np.corrcoef(x_vals, y_vals)[0, 1])


def _corr_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    return float(x.rank(method="average").corr(y.rank(method="average"), method="pearson"))


def _rank_topk_overlap(
    ranking_a: pd.DataFrame,
    ranking_b: pd.DataFrame,
    key_col: str,
    score_col: str,
    k: int,
) -> float:
    top_a = set(ranking_a.sort_values(score_col, ascending=False).head(k)[key_col].tolist())
    top_b = set(ranking_b.sort_values(score_col, ascending=False).head(k)[key_col].tolist())
    if not top_a or not top_b:
        return float("nan")
    return float(len(top_a & top_b) / float(k))


def _question_level_spearman(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    corrs: list[float] = []
    for _, g in df.groupby("item_id", sort=False):
        if len(g) < 3:
            continue
        corr = _corr_spearman(g[col_a], g[col_b])
        if math.isfinite(corr):
            corrs.append(corr)
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def _pairwise_preference_consistency(
    df: pd.DataFrame,
    arena_col: str,
    open_col: str,
    arena_tie_threshold: float,
    open_tie_threshold: float,
) -> dict[str, Any]:
    considered = 0
    agree = 0
    arena_ties = 0
    open_ties = 0
    both_ties = 0
    total_pairs = 0

    for _, g in df.groupby("item_id", sort=False):
        if len(g) < 2:
            continue
        arena_vals = g[arena_col].to_numpy(dtype=float)
        open_vals = g[open_col].to_numpy(dtype=float)
        n = len(g)
        for i, j in combinations(range(n), 2):
            total_pairs += 1
            da = float(arena_vals[i] - arena_vals[j])
            do = float(open_vals[i] - open_vals[j])
            arena_tie = abs(da) <= arena_tie_threshold
            open_tie = abs(do) <= open_tie_threshold
            arena_ties += int(arena_tie)
            open_ties += int(open_tie)
            both_ties += int(arena_tie and open_tie)
            if arena_tie or open_tie:
                continue
            considered += 1
            agree += int((da > 0) == (do > 0))

    return {
        "pairwise_total_pairs": int(total_pairs),
        "pairwise_considered_non_tie_pairs": int(considered),
        "pairwise_preference_consistency_non_tie": (float(agree / considered) if considered else float("nan")),
        "arena_tie_pairs": int(arena_ties),
        "open_tie_pairs": int(open_ties),
        "both_tie_pairs": int(both_ties),
    }


def load_domain_arena_data(arena_jsonl_dir: Path, domain: str, max_rows: int | None) -> pd.DataFrame:
    path = arena_jsonl_dir / f"v1_arena_{domain}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing arena JSONL: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("status") != "ok":
                continue
            if d.get("reward") is None:
                continue
            response_text = d.get("response_text")
            if not response_text:
                continue
            prompt = d.get("prompt") or d.get("question")
            if not prompt:
                continue
            rows.append(
                {
                    "domain": domain,
                    "item_id": str(d["item_id"]),
                    "model_name": str(d["model_label"]),
                    "prompt": str(prompt),
                    "response_text": str(response_text),
                    "arena_reward": float(d["reward"]),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["item_id", "model_name"], keep="last").reset_index(drop=True)


def build_reward_df_for_irt(domain_df: pd.DataFrame) -> pd.DataFrame:
    reward_df = domain_df.copy()
    reward_df = reward_df.rename(columns={"model_name": "model_name", "item_id": "item_id"})
    reward_df["source"] = "v1"
    reward_df["benchmark"] = "Arena"
    reward_df["question_id"] = reward_df["item_id"].map(lambda x: f"v1::{x}")
    reward_df = reward_df.rename(columns={"open_reward": "reward_raw"})
    reward_mean = float(reward_df["reward_raw"].mean())
    reward_std = float(reward_df["reward_raw"].std(ddof=0))
    if not math.isfinite(reward_std) or reward_std < 1e-8:
        reward_std = 1.0
    reward_df["reward_z"] = (reward_df["reward_raw"] - reward_mean) / reward_std
    return reward_df[["source", "benchmark", "model_name", "question_id", "reward_raw", "reward_z"]]


def compare_irt_with_arena(
    domain: str,
    open_model_ranking: pd.DataFrame,
    open_question_params: pd.DataFrame,
    arena_ranking_dir: Path,
) -> dict[str, Any]:
    domain_dir = arena_ranking_dir / domain
    arena_model_path = domain_dir / "model_ranking.csv"
    arena_question_path = domain_dir / "arena_question_params.csv"
    if not arena_model_path.exists() or not arena_question_path.exists():
        raise FileNotFoundError(f"Missing Arena IRT files in {domain_dir}")

    arena_model = pd.read_csv(arena_model_path)
    arena_q = pd.read_csv(arena_question_path)

    merged_model = open_model_ranking.merge(
        arena_model,
        how="inner",
        on="model_name",
        suffixes=("_open", "_arena"),
    )

    model_theta_pearson = _corr_pearson(merged_model["theta_open"], merged_model["theta_arena"])
    model_theta_spearman = _corr_spearman(merged_model["theta_open"], merged_model["theta_arena"])
    top3_overlap = _rank_topk_overlap(open_model_ranking, arena_model, "model_name", "theta", 3)
    top5_overlap = _rank_topk_overlap(open_model_ranking, arena_model, "model_name", "theta", 5)

    merged_q = open_question_params.merge(
        arena_q,
        how="inner",
        on="question_id",
        suffixes=("_open", "_arena"),
    )

    q_disc_pearson = _corr_pearson(
        merged_q["discrimination_exp_k_open"],
        merged_q["discrimination_exp_k_arena"],
    )
    q_disc_spearman = _corr_spearman(
        merged_q["discrimination_exp_k_open"],
        merged_q["discrimination_exp_k_arena"],
    )
    q_diff_pearson = (
        _corr_pearson(merged_q["difficulty_b_open"], merged_q["difficulty_b_arena"])
        if "difficulty_b_open" in merged_q.columns and "difficulty_b_arena" in merged_q.columns
        else float("nan")
    )
    q_diff_spearman = (
        _corr_spearman(merged_q["difficulty_b_open"], merged_q["difficulty_b_arena"])
        if "difficulty_b_open" in merged_q.columns and "difficulty_b_arena" in merged_q.columns
        else float("nan")
    )

    return {
        "domain": domain,
        "shared_models": int(len(merged_model)),
        "model_theta_pearson": model_theta_pearson,
        "model_theta_spearman": model_theta_spearman,
        "model_top3_overlap": top3_overlap,
        "model_top5_overlap": top5_overlap,
        "shared_items": int(len(merged_q)),
        "item_discrimination_pearson": q_disc_pearson,
        "item_discrimination_spearman": q_disc_spearman,
        "item_difficulty_pearson": q_diff_pearson,
        "item_difficulty_spearman": q_diff_spearman,
    }


@dataclass
class BaseRewardModelRunner:
    model_name: str
    batch_size: int
    max_length: int
    trust_remote_code: bool
    logit_index: int
    device: str | None
    dtype: str

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'transformers'. Install it with `pip install transformers` "
                "before running this script."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = self._resolve_torch_dtype()
        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.model = self._load_reward_model(
            AutoModelForSequenceClassification,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self.model.to(self.device)

    def _patch_transformers_remote_compat(self) -> None:
        """Patch minor transformers API drifts used by some remote-code RMs."""
        try:
            llama_mod = importlib.import_module("transformers.models.llama.modeling_llama")
        except Exception:
            return
        # Some custom RM repos import this symbol from older/newer transformers APIs.
        if not hasattr(llama_mod, "LLAMA_INPUTS_DOCSTRING"):
            setattr(llama_mod, "LLAMA_INPUTS_DOCSTRING", "")

    def _load_reward_model(
        self,
        auto_cls: Any,
        *,
        torch_dtype: torch.dtype | None,
    ) -> Any:
        if self.trust_remote_code:
            self._patch_transformers_remote_compat()
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if torch_dtype is not None:
            # Preferred argument name in newer transformers.
            kwargs["dtype"] = torch_dtype
        try:
            return auto_cls.from_pretrained(self.model_name, **kwargs)
        except TypeError:
            # Backward compatibility for older transformers that only accept torch_dtype.
            if torch_dtype is not None:
                kwargs.pop("dtype", None)
                kwargs["torch_dtype"] = torch_dtype
                return auto_cls.from_pretrained(self.model_name, **kwargs)
            raise
        except ImportError:
            raise

    def _resolve_torch_dtype(self) -> torch.dtype | None:
        if self.dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32
        if self.dtype == "float32":
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return None

    def _uses_direct_chat_template_ids(self) -> bool:
        return False

    def _is_missing_gating_token_error(self, exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "Token pattern not found in the list." in str(exc)

    def _format_pair(self, prompt: str, response: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass
        return f"User: {prompt}\nAssistant: {response}"

    def _normalize_chat_template_encoding(self, encoded: Any) -> dict[str, list[int]]:
        """
        Normalize tokenizer.apply_chat_template(...) outputs into plain list-valued
        fields suitable for tokenizer.pad(...).
        """
        def _unwrap_named(value: Any, field: str) -> Any:
            """Unwrap nested BatchEncoding/dict wrappers for a named field."""
            current = value
            for _ in range(8):
                if torch.is_tensor(current):
                    current = current.detach().cpu().tolist()
                    continue
                if isinstance(current, np.ndarray):
                    current = current.tolist()
                    continue
                if isinstance(current, dict):
                    if field in current:
                        current = current[field]
                        continue
                    if len(current) == 1:
                        current = next(iter(current.values()))
                        continue
                    break
                data_attr = getattr(current, "data", None)
                if isinstance(data_attr, dict):
                    if field in data_attr:
                        current = data_attr[field]
                        continue
                    if len(data_attr) == 1:
                        current = next(iter(data_attr.values()))
                        continue
                to_dict = getattr(current, "to_dict", None)
                if callable(to_dict):
                    try:
                        mapped = to_dict()
                    except Exception:
                        mapped = None
                    if isinstance(mapped, dict):
                        current = mapped
                        continue
                break
            return current

        input_ids_raw = _unwrap_named(encoded, "input_ids")
        attention_mask_raw = _unwrap_named(encoded, "attention_mask")

        if isinstance(input_ids_raw, tuple):
            input_ids_raw = list(input_ids_raw)
        while isinstance(input_ids_raw, list) and input_ids_raw and isinstance(input_ids_raw[0], (list, tuple)):
            input_ids_raw = list(input_ids_raw[0])
        if not isinstance(input_ids_raw, list):
            raise TypeError(f"Unexpected input_ids type from chat template: {type(input_ids_raw)}")

        row: dict[str, list[int]] = {"input_ids": [int(x) for x in input_ids_raw]}

        if isinstance(attention_mask_raw, tuple):
            attention_mask_raw = list(attention_mask_raw)
        while (
            isinstance(attention_mask_raw, list)
            and attention_mask_raw
            and isinstance(attention_mask_raw[0], (list, tuple))
        ):
            attention_mask_raw = list(attention_mask_raw[0])
        if isinstance(attention_mask_raw, list):
            row["attention_mask"] = [int(x) for x in attention_mask_raw]

        return row

    def _encode_chat_batch_direct(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> dict[str, torch.Tensor]:
        batch_messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(prompts, responses, strict=True)
        ]
        return self.tokenizer.apply_chat_template(
            batch_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=False,
        )

    def _encode_chat_batch_direct_fallback(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> dict[str, torch.Tensor]:
        encoded_rows: list[dict[str, list[int]]] = []
        for prompt, response in zip(prompts, responses, strict=True):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            row = self._normalize_chat_template_encoding(encoded)
            if self.max_length and len(row["input_ids"]) > self.max_length:
                row["input_ids"] = row["input_ids"][-self.max_length :]
                if "attention_mask" in row:
                    row["attention_mask"] = row["attention_mask"][-self.max_length :]
            encoded_rows.append(row)
        return self.tokenizer.pad(
            encoded_rows,
            return_tensors="pt",
            padding=True,
        )

    def _extract_scores(self, outputs: Any) -> torch.Tensor:
        logits: torch.Tensor | None = None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple) and outputs:
            if torch.is_tensor(outputs[0]):
                logits = outputs[0]
        if logits is None:
            raise RuntimeError(
                f"Cannot extract reward scores from model output for {self.model_name}. "
                "Expected `.logits` or tensor-like first output."
            )
        if logits.ndim == 1:
            return logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits.squeeze(1)
        return logits[:, self.logit_index]

    def _encode_batch(self, prompts: list[str], responses: list[str]) -> dict[str, torch.Tensor]:
        if self._uses_direct_chat_template_ids():
            try:
                return self._encode_chat_batch_direct(prompts, responses)
            except Exception:
                return self._encode_chat_batch_direct_fallback(prompts, responses)
        texts = [self._format_pair(p, r) for p, r in zip(prompts, responses, strict=True)]
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def _run_model_on_encoded(self, enc: dict[str, torch.Tensor]) -> np.ndarray:
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc)
            return self._extract_scores(out).detach().float().cpu().numpy()

    def score(self, prompts: list[str], responses: list[str], quiet: bool = False) -> np.ndarray:
        scores: list[np.ndarray] = []
        total = len(prompts)
        for start in range(0, total, self.batch_size):
            end = min(total, start + self.batch_size)
            batch_prompts = prompts[start:end]
            batch_responses = responses[start:end]
            enc = self._encode_batch(batch_prompts, batch_responses)
            try:
                batch_scores = self._run_model_on_encoded(enc)
            except ValueError as exc:
                if not self._is_missing_gating_token_error(exc):
                    raise
                if not quiet:
                    print(
                        f"  [{self.model_name}] warning: missing gating token pattern in "
                        f"batch {start + 1}-{end}; retrying per-sample and skipping bad rows."
                    )
                recovered_scores: list[float] = []
                for idx, (prompt, response) in enumerate(
                    zip(batch_prompts, batch_responses, strict=True),
                    start=start + 1,
                ):
                    try:
                        single_enc = self._encode_batch([prompt], [response])
                        one_score = self._run_model_on_encoded(single_enc)
                        recovered_scores.append(float(one_score[0]))
                    except ValueError as inner_exc:
                        if not self._is_missing_gating_token_error(inner_exc):
                            raise
                        recovered_scores.append(float("nan"))
                        if not quiet:
                            print(
                                f"  [{self.model_name}] warning: skipped sample index={idx} "
                                "due to missing gating token pattern after truncation."
                            )
                batch_scores = np.asarray(recovered_scores, dtype=float)
            scores.append(batch_scores)
            if not quiet:
                print(f"  [{self.model_name}] scored {end}/{total}")
        return np.concatenate(scores, axis=0) if scores else np.array([], dtype=float)


@dataclass
class NCSOFTRunner(BaseRewardModelRunner):
    pass


@dataclass
class SkyworkRunner(BaseRewardModelRunner):
    pass


@dataclass
class OpenAssistantRunner(BaseRewardModelRunner):
    pass


@dataclass
class ArmoRMRunner(BaseRewardModelRunner):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer.truncation_side = "left"

    def _uses_direct_chat_template_ids(self) -> bool:
        return True


def build_reward_model_runner(
    model_name: str,
    *,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    logit_index: int,
    device: str | None,
    dtype: str,
) -> BaseRewardModelRunner:
    if model_name == "NCSOFT/Llama-3-OffsetBias-RM-8B":
        runner_cls: type[BaseRewardModelRunner] = NCSOFTRunner
    elif model_name == "Skywork/Skywork-Reward-V2-Qwen3-8B":
        runner_cls = SkyworkRunner
    elif model_name == "OpenAssistant/reward-model-deberta-v3-large-v2":
        runner_cls = OpenAssistantRunner
    elif model_name == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
        runner_cls = ArmoRMRunner
    else:
        runner_cls = BaseRewardModelRunner
    return runner_cls(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        logit_index=logit_index,
        device=device,
        dtype=dtype,
    )


def main() -> None:
    args = parse_args()
    arena_jsonl_dir = resolve_path(args.arena_jsonl_dir)
    arena_ranking_dir = resolve_path(args.arena_ranking_dir)
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    domain_data: dict[str, pd.DataFrame] = {}
    for domain in args.domains:
        df = load_domain_arena_data(arena_jsonl_dir, domain, args.max_rows_per_domain)
        if df.empty:
            raise SystemExit(f"No usable rows loaded for domain={domain}")
        domain_data[domain] = df
        if not args.quiet:
            print(
                f"Loaded domain={domain}: rows={len(df)}, "
                f"items={df['item_id'].nunique()}, models={df['model_name'].nunique()}"
            )

    aggregate_rows: list[dict[str, Any]] = []
    for raw_model_name in args.reward_models:
        model_name = MODEL_NAME_ALIASES.get(raw_model_name, raw_model_name)
        if not args.quiet:
            if model_name != raw_model_name:
                print(f"\nLoading reward model: {raw_model_name} (resolved to {model_name})")
            else:
                print(f"\nLoading reward model: {model_name}")
        rm = build_reward_model_runner(
            model_name=model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
            logit_index=args.logit_index,
            device=args.device,
            dtype=args.dtype,
        )
        model_slug = rm_slug(model_name)

        for domain in args.domains:
            if not args.quiet:
                print(f"\nScoring domain={domain} with model={model_name}")
            df = domain_data[domain].copy()
            scores = rm.score(
                prompts=df["prompt"].tolist(),
                responses=df["response_text"].tolist(),
                quiet=args.quiet,
            )
            df["open_reward"] = scores

            corr_metrics = {
                "reward_pearson": _corr_pearson(df["open_reward"], df["arena_reward"]),
                "reward_spearman": _corr_spearman(df["open_reward"], df["arena_reward"]),
                "question_level_spearman_mean": _question_level_spearman(df, "open_reward", "arena_reward"),
                "n_rows": int(len(df)),
                "n_items": int(df["item_id"].nunique()),
                "n_models": int(df["model_name"].nunique()),
            }
            pair_metrics = _pairwise_preference_consistency(
                df,
                arena_col="arena_reward",
                open_col="open_reward",
                arena_tie_threshold=args.arena_tie_threshold,
                open_tie_threshold=args.open_tie_threshold,
            )

            reward_df = build_reward_df_for_irt(df)
            both_bad_threshold, tie_delta = resolve_pairwise_thresholds(
                reward_df,
                bb_ratio=args.bb_ratio,
                tie_ratio=args.tie_ratio,
                both_bad_threshold=args.both_bad_threshold,
                both_bad_use_zscore=True,
            )
            pairwise_df = build_soft_pairwise_targets(
                reward_df,
                both_bad_threshold=both_bad_threshold,
                both_bad_use_zscore=True,
                tie_delta=tie_delta,
            )
            model_params, _static_qp, arena_qp, fit_meta = fit_irt_v1(
                static_df=None,
                pairwise_df=pairwise_df if not pairwise_df.empty else None,
                reward_df=reward_df if not reward_df.empty else None,
                arena_mode="pairwise+regression",
                num_epochs=args.num_epochs,
                lr=args.lr,
                lambda_static=0.0,
                lambda_arena=args.lambda_arena,
                lambda_reg=args.lambda_reg,
                reg_lambda=args.reg_lambda,
                verbose=not args.quiet,
            )

            irt_consistency = compare_irt_with_arena(
                domain=domain,
                open_model_ranking=model_params,
                open_question_params=arena_qp,
                arena_ranking_dir=arena_ranking_dir,
            )

            run_dir = output_root / model_slug / domain
            run_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(run_dir / "rescored_responses.csv", index=False)
            reward_df.to_csv(run_dir / "open_reward_table_for_irt.csv", index=False)
            pairwise_df.to_csv(run_dir / "open_reward_pairwise_targets.csv", index=False)
            model_params.to_csv(run_dir / "open_model_ranking.csv", index=False)
            arena_qp.to_csv(run_dir / "open_arena_question_params.csv", index=False)

            result = {
                "reward_model": model_name,
                "reward_model_slug": model_slug,
                "domain": domain,
                "correlation_metrics": corr_metrics,
                "pairwise_metrics": pair_metrics,
                "irt_fit_meta": fit_meta,
                "pairwise_target_meta": {
                    "both_bad_threshold": both_bad_threshold,
                    "tie_delta": tie_delta,
                    "bb_ratio": args.bb_ratio,
                    "tie_ratio": args.tie_ratio,
                },
                "irt_consistency_vs_arena": irt_consistency,
            }
            (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            aggregate_rows.append(
                {
                    "reward_model": model_name,
                    "domain": domain,
                    "reward_pearson": corr_metrics["reward_pearson"],
                    "reward_spearman": corr_metrics["reward_spearman"],
                    "pairwise_pref_consistency": pair_metrics["pairwise_preference_consistency_non_tie"],
                    "irt_model_theta_spearman": irt_consistency["model_theta_spearman"],
                    "irt_item_disc_spearman": irt_consistency["item_discrimination_spearman"],
                    "irt_item_diff_spearman": irt_consistency["item_difficulty_spearman"],
                }
            )

            if not args.quiet:
                print(
                    "  "
                    f"reward_spearman={corr_metrics['reward_spearman']:.4f}  "
                    f"pref_consistency={pair_metrics['pairwise_preference_consistency_non_tie']:.4f}  "
                    f"irt_model_theta_spearman={irt_consistency['model_theta_spearman']:.4f}"
                )

    summary_df = pd.DataFrame(aggregate_rows)
    summary_df.to_csv(output_root / "aggregate_summary.csv", index=False)
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"\nSaved all outputs under: {output_root}")


if __name__ == "__main__":
    main()
