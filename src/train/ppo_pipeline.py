import inspect
import warnings

import torch
from torch import nn
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel, get_peft_model

try:
    from trl.experimental.ppo import PPOConfig, PPOTrainer
except ImportError:
    from trl import PPOConfig, PPOTrainer

from src.train.common import (
    attach_context,
    build_prompt,
    load_document_store,
    load_jsonl,
)
from src.train.reward_postprocess import (
    apply_length_penalty_torch,
    apply_squash_clip_torch,
    normalize_reward_postprocess_cfg,
)


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


def _load_seqcls_with_optional_adapter(model_cfg: dict, fallback_model_name: str, role: str):
    model_name = model_cfg.get("model_name", fallback_model_name)
    base_model_name = model_cfg.get("base_model_name")
    adapter_path = model_cfg.get("adapter_path")
    adapter_trainable_default = role == "value"
    adapter_trainable = bool(model_cfg.get("adapter_trainable", adapter_trainable_default))

    if not adapter_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=1,
            torch_dtype="auto",
        )
        return model, {"load_mode": "single_checkpoint", "model_name": model_name}

    if not base_model_name:
        raise ValueError(
            f"{role}_model.adapter_path is set but {role}_model.base_model_name is missing."
        )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=adapter_trainable)
    return model, {
        "load_mode": "base_plus_adapter",
        "base_model_name": base_model_name,
        "adapter_path": adapter_path,
        "adapter_trainable": adapter_trainable,
    }


class RewardPostProcessWrapper(nn.Module):
    def __init__(self, base_model, cfg: dict):
        super().__init__()
        self.base_model = base_model
        self.cfg = normalize_reward_postprocess_cfg(cfg)
        self.enabled = self.cfg["enabled"]
        self.temperature = self.cfg["temperature"]
        self.normalize = self.cfg["normalize"]
        self.apply_tanh = self.cfg["apply_tanh"]
        self.clip_min = self.cfg["clip_min"]
        self.clip_max = self.cfg["clip_max"]
        self.eps = self.cfg["eps"]
        self.running_momentum = self.cfg["running_momentum"]
        self.length_penalty = self.cfg["length_penalty"]
        self.length_penalty_mode = self.cfg["length_penalty_mode"]
        self.length_penalty_scale = self.cfg["length_penalty_scale"]
        self.update_stats_in_eval = self.cfg["update_stats_in_eval"]
        self.min_count_for_running = self.cfg["min_count_for_running"]

        self.register_buffer("running_mean", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("running_var", torch.ones(1, dtype=torch.float32), persistent=False)
        self.register_buffer("running_inited", torch.zeros(1, dtype=torch.bool), persistent=False)
        self.register_buffer("running_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def _extract_response_lengths(self, kwargs):
        attention_mask = kwargs.get("attention_mask")
        response_mask = kwargs.get("response_mask")
        if self.length_penalty_mode == "response_tokens" and response_mask is not None:
            return response_mask.float().sum(dim=-1, keepdim=True)
        if attention_mask is not None:
            return attention_mask.float().sum(dim=-1, keepdim=True)
        if response_mask is not None:
            return response_mask.float().sum(dim=-1, keepdim=True)
        return None

    def _apply(self, logits, kwargs):
        logits = logits / self.temperature
        if self.normalize == "batch_zscore":
            mean = logits.mean()
            std = logits.std(unbiased=False)
            logits = (logits - mean) / (std + self.eps)
        elif self.normalize == "running_zscore":
            batch_mean = logits.mean().detach()
            batch_var = logits.var(unbiased=False).detach()
            should_update = self.training or self.update_stats_in_eval
            if should_update:
                self.running_count.add_(logits.numel())
                if not bool(self.running_inited.item()):
                    self.running_mean.copy_(batch_mean.reshape_as(self.running_mean))
                    self.running_var.copy_(batch_var.reshape_as(self.running_var))
                    self.running_inited.fill_(True)
                else:
                    m = self.running_momentum
                    self.running_mean.mul_(m).add_(batch_mean * (1.0 - m))
                    self.running_var.mul_(m).add_(batch_var * (1.0 - m))
            if bool(self.running_inited.item()) and int(self.running_count.item()) >= self.min_count_for_running:
                std = torch.sqrt(self.running_var.clamp_min(self.eps))
                logits = (logits - self.running_mean) / std
            else:
                # Warm-up: avoid unstable early-step EMA by using current batch stats.
                mean = logits.mean()
                std = logits.std(unbiased=False)
                logits = (logits - mean) / (std + self.eps)

        if self.length_penalty != 0.0:
            lengths = self._extract_response_lengths(kwargs)
            logits = apply_length_penalty_torch(logits, lengths, self.cfg)

        logits = apply_squash_clip_torch(logits, self.cfg)
        return logits

    def forward(self, *args, **kwargs):
        outputs = self.base_model(*args, **kwargs)
        if not self.enabled:
            return outputs
        logits = outputs.logits
        logits = self._apply(logits, kwargs)
        try:
            outputs.logits = logits
        except Exception:
            if hasattr(outputs, "to_tuple"):
                # Conservative fallback for tuple-like outputs: keep type contract as dict-like.
                out_dict = dict(outputs)
                out_dict["logits"] = logits
                return out_dict
            raise
        return outputs

    def status(self):
        return {
            "enabled": self.enabled,
            "temperature": self.temperature,
            "normalize": self.normalize,
            "apply_tanh": self.apply_tanh,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "length_penalty": self.length_penalty,
            "length_penalty_mode": self.length_penalty_mode,
            "length_penalty_scale": self.length_penalty_scale,
            "running_momentum": self.running_momentum,
            "update_stats_in_eval": self.update_stats_in_eval,
            "min_count_for_running": self.min_count_for_running,
        }


def run_ppo(config_or_path):
    if isinstance(config_or_path, str):
        cfg = yaml.safe_load(open(config_or_path))
    else:
        cfg = config_or_path

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name"],
        trust_remote_code=True,
    )
    tokenizer.truncation_side = str(model_cfg.get("truncation_side", "left"))
    tokenizer.padding_side = str(model_cfg.get("padding_side", "left"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Please set eos token in tokenizer/model config.")

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        trust_remote_code=True,
        torch_dtype="auto",
    )
    policy_model.config.eos_token_id = tokenizer.eos_token_id
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(policy_model, "generation_config", None) is not None:
        policy_model.generation_config.eos_token_id = tokenizer.eos_token_id
        policy_model.generation_config.pad_token_id = tokenizer.pad_token_id

    if model_cfg.get("use_lora", False):
        if "lora" not in cfg:
            raise ValueError("LoRA is enabled (model.use_lora=true) but `lora` config is missing.")
        lora_cfg = cfg["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.print_trainable_parameters()

    reward_cfg = cfg.get("reward_model", {})
    value_cfg = cfg.get("value_model", {})
    # PPO in TRL expects reward/value models with a `.score`-like head.
    reward_model, reward_load_info = _load_seqcls_with_optional_adapter(
        reward_cfg, model_cfg["model_name"], role="reward"
    )
    value_model, value_load_info = _load_seqcls_with_optional_adapter(
        value_cfg, model_cfg["model_name"], role="value"
    )

    # PPO trainer in TRL 0.27 does not optimize reward model parameters.
    # Keep reward model frozen and use it only for scoring rollouts.
    if reward_cfg.get("use_lora", False):
        raise ValueError(
            "reward_model.use_lora=true is not supported in PPO stage. "
            "Train reward model offline, then set reward_model.freeze=true and use_lora=false."
        )
    for p in reward_model.parameters():
        p.requires_grad = False
    reward_model.eval()
    reward_post_cfg = cfg.get("reward_postprocess", {})
    wrapped_reward_model = RewardPostProcessWrapper(reward_model, reward_post_cfg)
    wrapped_reward_model.eval()
    reward_post_status = wrapped_reward_model.status()
    if wrapped_reward_model.apply_tanh and wrapped_reward_model.clip_min == -1.0 and wrapped_reward_model.clip_max == 1.0:
        warnings.warn("reward_postprocess uses tanh + clip[-1,1]; clip is effectively redundant.")

    # PPO updates value model every step. Freezing the backbone drastically
    # reduces optimizer/activation memory while still training the score head.
    if value_cfg.get("freeze_backbone", False):
        backbone = getattr(value_model, value_model.base_model_prefix, None)
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
        head_modules = _infer_seqcls_head_modules(value_model)
        if not head_modules:
            raise ValueError("Could not infer value head module. Set value_model.lora.modules_to_save explicitly.")
        module_map = dict(value_model.named_modules())
        for name in head_modules:
            for p in module_map[name].parameters():
                p.requires_grad = True
    elif value_cfg.get("use_lora", True):
        if value_cfg.get("adapter_path"):
            raise ValueError(
                "value_model.adapter_path is set while value_model.use_lora=true. "
                "Use either preloaded adapter or new LoRA, not both."
            )
        value_lora_cfg = value_cfg.get("lora", cfg.get("lora", {}))
        if not value_lora_cfg:
            raise ValueError("value_model.use_lora=true but no LoRA config found.")
        modules_to_save = value_lora_cfg.get("modules_to_save") or _infer_seqcls_head_modules(value_model)
        if not modules_to_save:
            raise ValueError("Could not infer value head modules for LoRA. Set value_model.lora.modules_to_save.")
        value_lora = LoraConfig(
            r=value_lora_cfg["r"],
            lora_alpha=value_lora_cfg["alpha"],
            target_modules=value_lora_cfg["target_modules"],
            lora_dropout=value_lora_cfg["dropout"],
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=modules_to_save,
        )
        value_model = get_peft_model(value_model, value_lora)
        if model_cfg.get("use_lora", False):
            warnings.warn(
                "Both policy_model and value_model use LoRA. If PPO is unstable, try policy LoRA + value head-only."
            )

    for module in (reward_model, value_model):
        module.config.eos_token_id = tokenizer.eos_token_id
        module.config.pad_token_id = tokenizer.pad_token_id
        if getattr(module, "generation_config", None) is not None:
            module.generation_config.eos_token_id = tokenizer.eos_token_id
            module.generation_config.pad_token_id = tokenizer.pad_token_id

    value_trainable = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
    value_total = sum(p.numel() for p in value_model.parameters())
    reward_total = sum(p.numel() for p in reward_model.parameters())
    print(f"[ppo] reward_model params: {reward_total} (frozen, scoring-only)")
    print(
        f"[ppo] value_model trainable params: {value_trainable} / {value_total} "
        f"({100.0 * value_trainable / max(1, value_total):.4f}%)"
    )
    print(f"[ppo] reward_model load: {reward_load_info}")
    print(f"[ppo] value_model load: {value_load_info}")
    print(f"[ppo] value head modules: {_infer_seqcls_head_modules(value_model)}")
    print(f"[ppo] reward_postprocess: {reward_post_status}")

    train_pairs = load_jsonl(cfg["data"]["train_path"])
    eval_pairs = load_jsonl(cfg["data"]["eval_path"])
    for split_name, rows in (("train", train_pairs), ("eval", eval_pairs)):
        if rows and ({"response", "feedback"} & set(rows[0].keys())):
            raise ValueError(
                f"PPO {split_name} dataset must not contain pre-generated response/feedback fields "
                "for online-RL correctness."
            )
    doc_map = load_document_store(cfg["data"]["documents_path"])

    train_ds = attach_context(train_pairs, doc_map)
    eval_ds = attach_context(eval_pairs, doc_map)

    template = cfg["prompt"]["template"]
    train_ds = train_ds.map(lambda x: {"prompt": build_prompt(x, template)})
    eval_ds = eval_ds.map(lambda x: {"prompt": build_prompt(x, template)})

    max_prompt_length = int(train_cfg["max_prompt_length"])

    def _tokenize_prompt(example):
        encoded = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=bool(train_cfg.get("add_special_tokens", True)),
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    train_columns = train_ds.column_names
    eval_columns = eval_ds.column_names
    train_ds = train_ds.map(_tokenize_prompt, remove_columns=train_columns)
    eval_ds = eval_ds.map(_tokenize_prompt, remove_columns=eval_columns)

    ppo_kwargs = {
        "output_dir": train_cfg["output_dir"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", train_cfg["per_device_train_batch_size"])),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "response_length": int(train_cfg["max_completion_length"]),
        "num_mini_batches": int(train_cfg.get("num_mini_batches", 1)),
        "num_ppo_epochs": int(train_cfg.get("num_ppo_epochs", 4)),
        "temperature": float(train_cfg.get("temperature", 1.0)),
        "kl_coef": float(train_cfg.get("kl_coef", 0.05)),
        "gamma": float(train_cfg.get("gamma", 1.0)),
        "lam": float(train_cfg.get("lam", 0.95)),
        "local_rollout_forward_batch_size": int(
            train_cfg.get("local_rollout_forward_batch_size", train_cfg["per_device_train_batch_size"])
        ),
        "stop_token": train_cfg.get("stop_token", "eos"),
        "missing_eos_penalty": train_cfg.get("missing_eos_penalty", 1.0),
        "report_to": train_cfg.get("report_to", "none"),
        "remove_unused_columns": False,
    }
    if "target_kl" in train_cfg:
        ppo_kwargs["target_kl"] = float(train_cfg["target_kl"])
    if "adap_kl_ctrl" in train_cfg:
        ppo_kwargs["adap_kl_ctrl"] = bool(train_cfg["adap_kl_ctrl"])
    if "target_kl" in train_cfg and "kl_coef" in train_cfg and not bool(train_cfg.get("allow_kl_coef_with_target_kl", False)):
        warnings.warn(
            "Both kl_coef and target_kl are set. This can cause KL oscillation. "
            "Set training.allow_kl_coef_with_target_kl=true to silence."
        )

    if "seed" in train_cfg:
        ppo_kwargs["seed"] = int(train_cfg["seed"])
    if "bf16" in train_cfg:
        ppo_kwargs["bf16"] = bool(train_cfg["bf16"])
    if "fp16" in train_cfg:
        ppo_kwargs["fp16"] = bool(train_cfg["fp16"])

    config_params = inspect.signature(PPOConfig.__init__).parameters
    filtered_kwargs = {k: v for k, v in ppo_kwargs.items() if k in config_params}
    if "stop_token" not in filtered_kwargs:
        if bool(train_cfg.get("require_stop_token_support", True)):
            raise ValueError(
                "Current PPOConfig does not support `stop_token`; set training.require_stop_token_support=false to bypass."
            )
        warnings.warn("Current PPOConfig does not support `stop_token`; EOS truncation may not trigger as expected.")
    ppo_config = PPOConfig(**filtered_kwargs)

    trainer_kwargs = {
        "args": ppo_config,
        "processing_class": tokenizer,
        "model": policy_model,
        "ref_model": None,
        "reward_model": wrapped_reward_model,
        "value_model": value_model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }

    trainer = PPOTrainer(**trainer_kwargs)
    trainer.train()
