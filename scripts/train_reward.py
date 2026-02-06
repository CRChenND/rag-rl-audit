import argparse
import inspect
import json
import random
from pathlib import Path
import sys

from datasets import load_dataset
import numpy as np
from peft import LoraConfig
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer
try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_reward_data import build_reward_datasets
from scripts.train import load_config
from src.train.reward_postprocess import apply_reward_postprocess_numpy, normalize_reward_postprocess_cfg


def _to_bool(v, default=False):
    if v is None:
        return default
    return bool(v)


class RegularizedRewardTrainer(RewardTrainer):
    def __init__(self, *args, regularization_cfg: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        reg_cfg = regularization_cfg or {}
        self.margin_target = float(reg_cfg.get("margin_target", 1.0))
        self.margin_weight = float(reg_cfg.get("margin_weight", 0.0))
        self.max_abs_reward = float(reg_cfg.get("max_abs_reward", 8.0))
        self.reward_clamp_weight = float(reg_cfg.get("reward_clamp_weight", 0.0))
        self._warned_missing_rewards = False

    @staticmethod
    def _get_from_outputs(outputs, key: str):
        if isinstance(outputs, dict):
            return outputs.get(key)
        return getattr(outputs, key, None)

    def _extract_rewards(self, outputs):
        candidate_pairs = [
            ("rewards_chosen", "rewards_rejected"),
            ("chosen_rewards", "rejected_rewards"),
            ("chosen_logits", "rejected_logits"),
            ("logits_chosen", "logits_rejected"),
        ]
        for left_key, right_key in candidate_pairs:
            left = self._get_from_outputs(outputs, left_key)
            right = self._get_from_outputs(outputs, right_key)
            if left is not None and right is not None:
                return left.float(), right.float()
        return None, None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kwargs = {"return_outputs": True}
        if "num_items_in_batch" in inspect.signature(super().compute_loss).parameters:
            kwargs["num_items_in_batch"] = num_items_in_batch
        base = super().compute_loss(model, inputs, **kwargs)
        if isinstance(base, tuple):
            loss, outputs = base
        else:
            loss, outputs = base, {}

        chosen_reward, rejected_reward = self._extract_rewards(outputs)
        if chosen_reward is None or rejected_reward is None:
            if not self._warned_missing_rewards and (self.margin_weight > 0 or self.reward_clamp_weight > 0):
                self._warned_missing_rewards = True
                print(
                    "[reward_regularization] could not find chosen/rejected rewards in trainer outputs; "
                    "margin/clamp regularization skipped."
                )
            return (loss, outputs) if return_outputs else loss

        reg = 0.0
        if self.margin_weight > 0.0:
            margin = chosen_reward - rejected_reward
            margin_deficit = torch.relu(self.margin_target - margin)
            reg = reg + self.margin_weight * torch.mean(margin_deficit.pow(2))

        if self.reward_clamp_weight > 0.0:
            all_rewards = torch.cat([chosen_reward, rejected_reward], dim=0)
            overflow = torch.relu(torch.abs(all_rewards) - self.max_abs_reward)
            reg = reg + self.reward_clamp_weight * torch.mean(overflow.pow(2))

        total_loss = loss + reg
        return (total_loss, outputs) if return_outputs else total_loss


def _join_prompt_response(prompt: str, response: str) -> str:
    return f"{prompt.rstrip()}\n{response.lstrip()}"


def _infer_seqcls_head_modules(model) -> list[str]:
    preferred = ("score", "classifier", "v_head")
    module_names = [name for name, _ in model.named_modules()]
    inferred = []
    for base_name in preferred:
        exact = [name for name in module_names if name == base_name]
        suffix = [name for name in module_names if name.endswith(f".{base_name}")]
        candidates = exact or suffix
        if candidates:
            inferred.append(sorted(candidates, key=len)[0])
    return inferred


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _score_texts(model, tokenizer, texts, max_length: int, batch_size: int) -> list[float]:
    device = _get_model_device(model)
    scores: list[float] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            if logits.ndim == 2:
                batch_scores = logits[:, 0]
            else:
                batch_scores = logits
            scores.extend(batch_scores.detach().cpu().float().tolist())

    return scores


def _token_lengths(tokenizer, texts, batch_size: int) -> list[int]:
    lengths: list[int] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _histogram(values: np.ndarray, bins: int = 20) -> dict:
    if len(values) == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"bins": [float(v) for v in edges.tolist()], "counts": [int(v) for v in counts.tolist()]}


def _compute_pairwise_metrics(
    model,
    tokenizer,
    prompts: list[str],
    chosens: list[str],
    rejecteds: list[str],
    max_length: int,
    batch_size: int,
    postprocess_cfg: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen_texts = [_join_prompt_response(p, c) for p, c in zip(prompts, chosens)]
    rejected_texts = [_join_prompt_response(p, r) for p, r in zip(prompts, rejecteds)]

    chosen_scores = np.array(_score_texts(model, tokenizer, chosen_texts, max_length, batch_size), dtype=np.float64)
    rejected_scores = np.array(_score_texts(model, tokenizer, rejected_texts, max_length, batch_size), dtype=np.float64)
    if postprocess_cfg:
        all_scores = np.concatenate([chosen_scores, rejected_scores], axis=0)
        all_scores = apply_reward_postprocess_numpy(all_scores, postprocess_cfg)
        split = len(chosen_scores)
        chosen_scores = all_scores[:split]
        rejected_scores = all_scores[split:]
    margins = chosen_scores - rejected_scores

    chosen_resp_lens = np.array(_token_lengths(tokenizer, chosens, batch_size), dtype=np.float64)
    rejected_resp_lens = np.array(_token_lengths(tokenizer, rejecteds, batch_size), dtype=np.float64)
    chosen_total_lens = np.array(_token_lengths(tokenizer, chosen_texts, batch_size), dtype=np.float64)
    rejected_total_lens = np.array(_token_lengths(tokenizer, rejected_texts, batch_size), dtype=np.float64)

    all_rewards = np.concatenate([chosen_scores, rejected_scores], axis=0)
    all_total_lengths = np.concatenate([chosen_total_lens, rejected_total_lens], axis=0)
    all_resp_lengths = np.concatenate([chosen_resp_lens, rejected_resp_lens], axis=0)

    metrics = {
        "num_pairs": int(len(margins)),
        "pairwise_accuracy": float(np.mean(margins > 0.0)),
        "tie_rate": float(np.mean(margins == 0.0)),
        "margin_mean": float(np.mean(margins)),
        "margin_variance": float(np.var(margins)),
        "margin_std": float(np.std(margins)),
        "chosen_reward_mean": float(np.mean(chosen_scores)),
        "rejected_reward_mean": float(np.mean(rejected_scores)),
        "reward_total_length_pearson": _pearson_corr(all_rewards, all_total_lengths),
        "reward_response_length_pearson": _pearson_corr(all_rewards, all_resp_lengths),
        "chosen_reward_total_length_pearson": _pearson_corr(chosen_scores, chosen_total_lens),
        "rejected_reward_total_length_pearson": _pearson_corr(rejected_scores, rejected_total_lens),
        "chosen_total_length_mean": float(np.mean(chosen_total_lens)),
        "rejected_total_length_mean": float(np.mean(rejected_total_lens)),
        "chosen_response_length_mean": float(np.mean(chosen_resp_lens)),
        "rejected_response_length_mean": float(np.mean(rejected_resp_lens)),
        "chosen_reward_histogram": _histogram(chosen_scores, bins=20),
        "rejected_reward_histogram": _histogram(rejected_scores, bins=20),
        "reward_histogram": _histogram(all_rewards, bins=20),
        "margin_histogram": _histogram(margins, bins=20),
    }
    return metrics, margins, chosen_scores, rejected_scores, chosen_total_lens


def _bootstrap_stability(margins: np.ndarray, num_samples: int, seed: int) -> dict:
    n = len(margins)
    if n == 0 or num_samples <= 0:
        return {
            "bootstrap_samples": int(num_samples),
            "pairwise_accuracy_std": float("nan"),
            "pairwise_accuracy_ci95_low": float("nan"),
            "pairwise_accuracy_ci95_high": float("nan"),
            "margin_mean_std": float("nan"),
            "margin_mean_ci95_low": float("nan"),
            "margin_mean_ci95_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    acc_values = np.zeros(num_samples, dtype=np.float64)
    margin_mean_values = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        sample = margins[rng.integers(0, n, size=n)]
        acc_values[i] = np.mean(sample > 0.0)
        margin_mean_values[i] = np.mean(sample)

    return {
        "bootstrap_samples": int(num_samples),
        "pairwise_accuracy_std": float(np.std(acc_values)),
        "pairwise_accuracy_ci95_low": float(np.percentile(acc_values, 2.5)),
        "pairwise_accuracy_ci95_high": float(np.percentile(acc_values, 97.5)),
        "margin_mean_std": float(np.std(margin_mean_values)),
        "margin_mean_ci95_low": float(np.percentile(margin_mean_values, 2.5)),
        "margin_mean_ci95_high": float(np.percentile(margin_mean_values, 97.5)),
    }


def _run_reward_diagnostics(cfg: dict, train_cfg: dict, eval_ds, model, tokenizer) -> dict:
    diag_cfg = cfg.get("reward_diagnostics", {})
    enabled = _to_bool(diag_cfg.get("enabled"), default=True)
    if not enabled:
        return {"enabled": False}

    batch_size = int(diag_cfg.get("batch_size", train_cfg.get("per_device_eval_batch_size", 2)))
    max_length = int(diag_cfg.get("max_length", train_cfg.get("max_length", 1024)))
    dev_size = int(diag_cfg.get("dev_size", 256))
    dev_seed = int(diag_cfg.get("dev_seed", 42))
    bootstrap_samples = int(diag_cfg.get("bootstrap_samples", 200))
    postprocess_cfg = normalize_reward_postprocess_cfg(diag_cfg.get("postprocess", cfg.get("reward_postprocess", {})))
    ppo_postprocess_cfg = normalize_reward_postprocess_cfg(cfg.get("reward_postprocess", {}))

    num_rows = len(eval_ds)
    if num_rows == 0:
        return {"enabled": True, "num_eval_rows": 0, "error": "Empty eval set"}

    prompts_all = list(eval_ds["prompt"])
    chosen_all = list(eval_ds["chosen"])
    rejected_all = list(eval_ds["rejected"])

    overall_metrics, _, _, _, _ = _compute_pairwise_metrics(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_all,
        chosens=chosen_all,
        rejecteds=rejected_all,
        max_length=max_length,
        batch_size=batch_size,
        postprocess_cfg=postprocess_cfg,
    )

    indices = list(range(num_rows))
    random.Random(dev_seed).shuffle(indices)
    use_dev_size = num_rows if dev_size <= 0 else min(dev_size, num_rows)
    dev_indices = indices[:use_dev_size]

    dev_prompts = [prompts_all[i] for i in dev_indices]
    dev_chosen = [chosen_all[i] for i in dev_indices]
    dev_rejected = [rejected_all[i] for i in dev_indices]

    dev_metrics, dev_margins, chosen_scores, rejected_scores, chosen_lens = _compute_pairwise_metrics(
        model=model,
        tokenizer=tokenizer,
        prompts=dev_prompts,
        chosens=dev_chosen,
        rejecteds=dev_rejected,
        max_length=max_length,
        batch_size=batch_size,
        postprocess_cfg=postprocess_cfg,
    )
    stability = _bootstrap_stability(dev_margins, num_samples=bootstrap_samples, seed=dev_seed)

    warnings = []
    if postprocess_cfg != ppo_postprocess_cfg and ppo_postprocess_cfg:
        warnings.append(
            "diagnostics.postprocess differs from reward_postprocess used by PPO; "
            "metrics may not match online PPO reward scale."
        )

    return {
        "enabled": True,
        "num_eval_rows": int(num_rows),
        "overall": overall_metrics,
        "postprocess": postprocess_cfg,
        "warnings": warnings,
        "fixed_dev": {
            "size": int(use_dev_size),
            "seed": int(dev_seed),
            "metrics": dev_metrics,
            "stability": stability,
            "score_preview": [
                {
                    "chosen_score": float(chosen_scores[i]),
                    "rejected_score": float(rejected_scores[i]),
                    "margin": float(dev_margins[i]),
                    "chosen_total_length": int(chosen_lens[i]),
                }
                for i in range(min(5, len(dev_margins)))
            ],
        },
    }


def run_reward_training(cfg: dict) -> None:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    force_rebuild = _to_bool(cfg.get("reward_data", {}).get("force_rebuild"), default=False)
    reward_train_path, reward_eval_path = build_reward_datasets(cfg, force=force_rebuild)

    reward_model_name = cfg.get("reward_model", {}).get("model_name", model_cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(reward_model_name, trust_remote_code=True)
    tokenizer.truncation_side = str(model_cfg.get("truncation_side", "left"))
    tokenizer.padding_side = str(model_cfg.get("padding_side", "right"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_dataset("json", data_files=reward_train_path)["train"]
    eval_ds = load_dataset("json", data_files=reward_eval_path)["train"]

    reward_kwargs = {
        "output_dir": train_cfg["output_dir"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(
            train_cfg.get("per_device_eval_batch_size", train_cfg["per_device_train_batch_size"])
        ),
        "max_length": int(train_cfg["max_length"]),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "eval_strategy": str(train_cfg.get("eval_strategy", "steps")),
        "eval_steps": int(train_cfg.get("eval_steps", 100)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "report_to": train_cfg.get("report_to", "none"),
        "gradient_checkpointing": _to_bool(train_cfg.get("gradient_checkpointing"), default=True),
        "dataset_num_proc": int(train_cfg.get("dataset_num_proc", 1)),
        "center_rewards_coefficient": train_cfg.get("center_rewards_coefficient", 0.0),
        "remove_unused_columns": False,
        "load_best_model_at_end": _to_bool(train_cfg.get("load_best_model_at_end"), default=True),
        "metric_for_best_model": str(train_cfg.get("metric_for_best_model", "eval_loss")),
        "greater_is_better": _to_bool(train_cfg.get("greater_is_better"), default=False),
        "model_init_kwargs": {
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "num_labels": 1,
        },
    }
    if "seed" in train_cfg:
        reward_kwargs["seed"] = int(train_cfg["seed"])
    if "bf16" in train_cfg:
        reward_kwargs["bf16"] = _to_bool(train_cfg["bf16"])
    if "fp16" in train_cfg:
        reward_kwargs["fp16"] = _to_bool(train_cfg["fp16"])

    reward_config_params = inspect.signature(RewardConfig.__init__).parameters
    reward_kwargs = {k: v for k, v in reward_kwargs.items() if k in reward_config_params}
    reward_config = RewardConfig(**reward_kwargs)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        num_labels=1,
    )

    peft_config = None
    if model_cfg.get("use_lora", False):
        lora_cfg = cfg.get("reward_model", {}).get("lora", cfg.get("lora", {}))
        if not lora_cfg:
            raise ValueError("model.use_lora=true but no LoRA config found in reward_model.lora or lora.")
        modules_to_save = lora_cfg.get("modules_to_save") or _infer_seqcls_head_modules(reward_model)
        if not modules_to_save:
            raise ValueError("Could not infer reward head modules for LoRA. Set reward_model.lora.modules_to_save.")
        print(f"[reward] modules_to_save={modules_to_save}")
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=modules_to_save,
        )

    trainer = RegularizedRewardTrainer(
        model=reward_model,
        args=reward_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        regularization_cfg=cfg.get("reward_regularization", {}),
    )
    early_stop_cfg = cfg.get("early_stopping", {})
    if _to_bool(early_stop_cfg.get("enabled"), default=True):
        if EarlyStoppingCallback is None:
            print("[early_stopping] transformers.EarlyStoppingCallback not available; skipping.")
        else:
            patience = int(early_stop_cfg.get("patience", 2))
            threshold = float(early_stop_cfg.get("threshold", 0.0))
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=threshold))
            print(f"[early_stopping] enabled (patience={patience}, threshold={threshold})")

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])

    diagnostics_model = trainer.model
    if peft_config is not None and _to_bool(train_cfg.get("merge_lora_on_save"), default=True):
        merged_path = str(Path(train_cfg["output_dir"]) / "merged")
        base_model = trainer.model.merge_and_unload()
        base_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        diagnostics_model = base_model
        print(f"Saved merged reward model: {merged_path}")

    diagnostics = _run_reward_diagnostics(
        cfg=cfg,
        train_cfg=train_cfg,
        eval_ds=eval_ds,
        model=diagnostics_model,
        tokenizer=tokenizer,
    )
    diagnostics_path = Path(train_cfg["output_dir"]) / "reward_diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    if diagnostics.get("enabled", False):
        overall = diagnostics.get("overall", {})
        fixed_dev = diagnostics.get("fixed_dev", {})
        fixed_dev_metrics = fixed_dev.get("metrics", {})
        stability = fixed_dev.get("stability", {})
        print(
            "[reward_diagnostics] "
            f"overall_pairwise_acc={overall.get('pairwise_accuracy', float('nan')):.4f}, "
            f"overall_margin_mean={overall.get('margin_mean', float('nan')):.4f}, "
            f"overall_reward_total_len_corr={overall.get('reward_total_length_pearson', float('nan')):.4f}"
        )
        print(
            "[reward_diagnostics] "
            f"fixed_dev_pairwise_acc={fixed_dev_metrics.get('pairwise_accuracy', float('nan')):.4f}, "
            f"fixed_dev_margin_mean={fixed_dev_metrics.get('margin_mean', float('nan')):.4f}, "
            f"fixed_dev_acc_ci95=[{stability.get('pairwise_accuracy_ci95_low', float('nan')):.4f}, "
            f"{stability.get('pairwise_accuracy_ci95_high', float('nan')):.4f}]"
        )
    print(f"Saved reward diagnostics: {diagnostics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to reward experiment YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_reward_training(cfg)


if __name__ == "__main__":
    main()
