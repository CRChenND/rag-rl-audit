import argparse
import inspect
import json
import random
from pathlib import Path
import sys

from datasets import load_dataset
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
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


class ScalarRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss


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


def _score_texts(model, tokenizer, texts, max_length: int, batch_size: int) -> np.ndarray:
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

    return np.asarray(scores, dtype=np.float64)


def _token_lengths(tokenizer, texts, batch_size: int) -> np.ndarray:
    lengths: list[int] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.float64)


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


def _tokenize_scalar_dataset(ds, tokenizer, max_length: int):
    def _fn(batch):
        texts = [_join_prompt_response(p, r) for p, r in zip(batch["prompt"], batch["response"])]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [float(x) for x in batch["label"]]
        return encoded

    return ds.map(_fn, batched=True)


def _compute_scalar_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    if logits.ndim > 1:
        logits = logits[:, 0]
    labels = np.asarray(labels).astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float64)
    acc = float(np.mean(preds == labels)) if labels.size > 0 else float("nan")
    bce = float(
        np.mean(
            -(
                labels * np.log(np.clip(probs, 1e-8, 1.0))
                + (1.0 - labels) * np.log(np.clip(1.0 - probs, 1e-8, 1.0))
            )
        )
    )
    return {
        "accuracy": acc,
        "bce": bce,
        "positive_rate_pred": float(np.mean(preds)) if preds.size > 0 else float("nan"),
        "positive_rate_label": float(np.mean(labels)) if labels.size > 0 else float("nan"),
    }


def _score_scalar_dataset(model, tokenizer, prompts, responses, max_length: int, batch_size: int) -> np.ndarray:
    texts = [_join_prompt_response(p, r) for p, r in zip(prompts, responses)]
    return _score_texts(model, tokenizer, texts, max_length=max_length, batch_size=batch_size)


def _bootstrap_scalar_stability(labels: np.ndarray, probs: np.ndarray, num_samples: int, seed: int) -> dict:
    n = len(labels)
    if n == 0 or num_samples <= 0:
        return {
            "bootstrap_samples": int(num_samples),
            "accuracy_std": float("nan"),
            "accuracy_ci95_low": float("nan"),
            "accuracy_ci95_high": float("nan"),
            "bce_std": float("nan"),
            "bce_ci95_low": float("nan"),
            "bce_ci95_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    acc_values = np.zeros(num_samples, dtype=np.float64)
    bce_values = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        idx = rng.integers(0, n, size=n)
        sample_labels = labels[idx]
        sample_probs = probs[idx]
        sample_preds = (sample_probs >= 0.5).astype(np.float64)
        acc_values[i] = np.mean(sample_preds == sample_labels)
        bce_values[i] = np.mean(
            -(
                sample_labels * np.log(np.clip(sample_probs, 1e-8, 1.0))
                + (1.0 - sample_labels) * np.log(np.clip(1.0 - sample_probs, 1e-8, 1.0))
            )
        )

    return {
        "bootstrap_samples": int(num_samples),
        "accuracy_std": float(np.std(acc_values)),
        "accuracy_ci95_low": float(np.percentile(acc_values, 2.5)),
        "accuracy_ci95_high": float(np.percentile(acc_values, 97.5)),
        "bce_std": float(np.std(bce_values)),
        "bce_ci95_low": float(np.percentile(bce_values, 2.5)),
        "bce_ci95_high": float(np.percentile(bce_values, 97.5)),
    }


def _run_reward_diagnostics(cfg: dict, train_cfg: dict, eval_ds, model, tokenizer) -> dict:
    diag_cfg = cfg.get("reward_diagnostics", {})
    enabled = _to_bool(diag_cfg.get("enabled"), default=True)
    if not enabled:
        return {"enabled": False}

    num_rows = len(eval_ds)
    if num_rows == 0:
        return {"enabled": True, "num_eval_rows": 0, "error": "Empty eval set"}

    batch_size = int(diag_cfg.get("batch_size", train_cfg.get("per_device_eval_batch_size", 2)))
    max_length = int(diag_cfg.get("max_length", train_cfg.get("max_length", 1024)))
    dev_size = int(diag_cfg.get("dev_size", 256))
    dev_seed = int(diag_cfg.get("dev_seed", 42))
    bootstrap_samples = int(diag_cfg.get("bootstrap_samples", 200))
    postprocess_cfg = normalize_reward_postprocess_cfg(diag_cfg.get("postprocess", cfg.get("reward_postprocess", {})))

    prompts_all = list(eval_ds["prompt"])
    responses_all = list(eval_ds["response"])
    labels_all = np.asarray(eval_ds["label"], dtype=np.float64)
    texts_all = [_join_prompt_response(p, r) for p, r in zip(prompts_all, responses_all)]

    logits_all = _score_scalar_dataset(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_all,
        responses=responses_all,
        max_length=max_length,
        batch_size=batch_size,
    )
    if postprocess_cfg:
        logits_all = apply_reward_postprocess_numpy(logits_all, postprocess_cfg)
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))
    preds_all = (probs_all >= 0.5).astype(np.float64)

    response_lengths_all = _token_lengths(tokenizer, responses_all, batch_size=batch_size)
    total_lengths_all = _token_lengths(tokenizer, texts_all, batch_size=batch_size)

    def _metrics_for_subset(logits: np.ndarray, labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict:
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0
        return {
            "num_examples": int(len(labels)),
            "accuracy": float(np.mean(preds == labels)),
            "bce": float(
                np.mean(
                    -(
                        labels * np.log(np.clip(probs, 1e-8, 1.0))
                        + (1.0 - labels) * np.log(np.clip(1.0 - probs, 1e-8, 1.0))
                    )
                )
            ),
            "positive_rate_pred": float(np.mean(preds)),
            "positive_rate_label": float(np.mean(labels)),
            "logit_mean": float(np.mean(logits)),
            "prob_mean": float(np.mean(probs)),
            "positive_logit_mean": float(np.mean(logits[pos_mask])) if np.any(pos_mask) else float("nan"),
            "negative_logit_mean": float(np.mean(logits[neg_mask])) if np.any(neg_mask) else float("nan"),
            "positive_prob_mean": float(np.mean(probs[pos_mask])) if np.any(pos_mask) else float("nan"),
            "negative_prob_mean": float(np.mean(probs[neg_mask])) if np.any(neg_mask) else float("nan"),
            "logit_histogram": _histogram(logits, bins=20),
            "prob_histogram": _histogram(probs, bins=20),
        }

    overall = _metrics_for_subset(logits_all, labels_all, probs_all, preds_all)
    overall.update(
        {
            "logit_total_length_pearson": _pearson_corr(logits_all, total_lengths_all),
            "logit_response_length_pearson": _pearson_corr(logits_all, response_lengths_all),
            "prob_total_length_pearson": _pearson_corr(probs_all, total_lengths_all),
            "prob_response_length_pearson": _pearson_corr(probs_all, response_lengths_all),
            "response_length_mean": float(np.mean(response_lengths_all)),
            "total_length_mean": float(np.mean(total_lengths_all)),
        }
    )

    indices = list(range(num_rows))
    random.Random(dev_seed).shuffle(indices)
    use_dev_size = num_rows if dev_size <= 0 else min(dev_size, num_rows)
    dev_indices = indices[:use_dev_size]
    dev_idx = np.asarray(dev_indices, dtype=np.int64)

    dev_logits = logits_all[dev_idx]
    dev_labels = labels_all[dev_idx]
    dev_probs = probs_all[dev_idx]
    dev_preds = preds_all[dev_idx]
    dev_total_lengths = total_lengths_all[dev_idx]
    stability = _bootstrap_scalar_stability(dev_labels, dev_probs, num_samples=bootstrap_samples, seed=dev_seed)

    fixed_dev = _metrics_for_subset(dev_logits, dev_labels, dev_probs, dev_preds)
    fixed_dev.update(
        {
            "logit_total_length_pearson": _pearson_corr(dev_logits, dev_total_lengths),
            "score_preview": [
                {
                    "label": int(dev_labels[i]),
                    "logit": float(dev_logits[i]),
                    "prob": float(dev_probs[i]),
                    "pred": int(dev_preds[i]),
                    "total_length": int(dev_total_lengths[i]),
                }
                for i in range(min(5, len(dev_logits)))
            ],
        }
    )

    return {
        "enabled": True,
        "num_eval_rows": int(num_rows),
        "postprocess": postprocess_cfg,
        "overall": overall,
        "fixed_dev": {
            "size": int(use_dev_size),
            "seed": int(dev_seed),
            "metrics": fixed_dev,
            "stability": stability,
        },
    }


def run_reward_training(cfg: dict) -> None:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    reward_train_cfg = cfg.get("reward_training", {})
    objective = str(reward_train_cfg.get("objective", "scalar_regression")).strip().lower()
    if objective != "scalar_regression":
        raise ValueError(
            "This training script is scalar-only. Set reward_training.objective='scalar_regression'."
        )

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
    raw_eval_ds = eval_ds

    required_columns = {"prompt", "response", "label"}
    missing_train = required_columns - set(train_ds.column_names)
    missing_eval = required_columns - set(eval_ds.column_names)
    if missing_train or missing_eval:
        raise ValueError(
            "Scalar reward data must contain columns prompt/response/label. "
            f"missing_train={sorted(missing_train)}, missing_eval={sorted(missing_eval)}"
        )

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

    if peft_config is not None:
        reward_model = get_peft_model(reward_model, peft_config)

    train_ds = _tokenize_scalar_dataset(train_ds, tokenizer=tokenizer, max_length=int(train_cfg["max_length"]))
    eval_ds = _tokenize_scalar_dataset(eval_ds, tokenizer=tokenizer, max_length=int(train_cfg["max_length"]))
    keep_cols = {"input_ids", "attention_mask", "labels"}
    train_drop = [c for c in train_ds.column_names if c not in keep_cols]
    eval_drop = [c for c in eval_ds.column_names if c not in keep_cols]
    if train_drop:
        train_ds = train_ds.remove_columns(train_drop)
    if eval_drop:
        eval_ds = eval_ds.remove_columns(eval_drop)

    training_kwargs = {
        "output_dir": train_cfg["output_dir"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(
            train_cfg.get("per_device_eval_batch_size", train_cfg["per_device_train_batch_size"])
        ),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "eval_steps": int(train_cfg.get("eval_steps", 100)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "report_to": train_cfg.get("report_to", "none"),
        "gradient_checkpointing": _to_bool(train_cfg.get("gradient_checkpointing"), default=True),
        "remove_unused_columns": False,
        "load_best_model_at_end": _to_bool(train_cfg.get("load_best_model_at_end"), default=True),
        "metric_for_best_model": str(train_cfg.get("metric_for_best_model", "eval_loss")),
        "greater_is_better": _to_bool(train_cfg.get("greater_is_better"), default=False),
    }
    if "seed" in train_cfg:
        training_kwargs["seed"] = int(train_cfg["seed"])
    if "bf16" in train_cfg:
        training_kwargs["bf16"] = _to_bool(train_cfg["bf16"])
    if "fp16" in train_cfg:
        training_kwargs["fp16"] = _to_bool(train_cfg["fp16"])

    eval_strategy_value = str(train_cfg.get("eval_strategy", "steps"))
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = eval_strategy_value

    trainer_kwargs = {
        "model": reward_model,
        "args": TrainingArguments(**training_kwargs),
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": _compute_scalar_metrics,
    }
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = ScalarRewardTrainer(**trainer_kwargs)

    early_stop_cfg = cfg.get("early_stopping", {})
    if _to_bool(early_stop_cfg.get("enabled"), default=True):
        if EarlyStoppingCallback is None:
            print("[early_stopping] transformers.EarlyStoppingCallback not available; skipping.")
        else:
            patience = int(early_stop_cfg.get("patience", 2))
            threshold = float(early_stop_cfg.get("threshold", 0.0))
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=threshold,
                )
            )
            print(f"[early_stopping] enabled (patience={patience}, threshold={threshold})")

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

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
        eval_ds=raw_eval_ds,
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
            f"overall_acc={overall.get('accuracy', float('nan')):.4f}, "
            f"overall_bce={overall.get('bce', float('nan')):.4f}, "
            f"overall_logit_total_len_corr={overall.get('logit_total_length_pearson', float('nan')):.4f}"
        )
        print(
            "[reward_diagnostics] "
            f"fixed_dev_acc={fixed_dev_metrics.get('accuracy', float('nan')):.4f}, "
            f"fixed_dev_bce={fixed_dev_metrics.get('bce', float('nan')):.4f}, "
            f"fixed_dev_acc_ci95=[{stability.get('accuracy_ci95_low', float('nan')):.4f}, "
            f"{stability.get('accuracy_ci95_high', float('nan')):.4f}]"
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
