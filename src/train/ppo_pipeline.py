import inspect
import warnings

import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model

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
    tokenizer.padding_side = "left"
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
    reward_model_name = reward_cfg.get("model_name", model_cfg["model_name"])
    value_model_name = value_cfg.get("model_name", model_cfg["model_name"])

    # PPO in TRL 0.27 expects reward/value models with a `.score` head.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype="auto",
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_name,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype="auto",
    )

    if reward_cfg.get("freeze", True):
        for p in reward_model.parameters():
            p.requires_grad = False
        reward_model.eval()

    # PPO updates value model every step. Freezing the backbone drastically
    # reduces optimizer/activation memory while still training the score head.
    if value_cfg.get("freeze_backbone", True):
        backbone = getattr(value_model, value_model.base_model_prefix, None)
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
        if hasattr(value_model, "score"):
            for p in value_model.score.parameters():
                p.requires_grad = True

    for module in (reward_model, value_model):
        module.config.eos_token_id = tokenizer.eos_token_id
        module.config.pad_token_id = tokenizer.pad_token_id
        if getattr(module, "generation_config", None) is not None:
            module.generation_config.eos_token_id = tokenizer.eos_token_id
            module.generation_config.pad_token_id = tokenizer.pad_token_id

    value_trainable = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
    value_total = sum(p.numel() for p in value_model.parameters())
    print(
        f"[ppo] value_model trainable params: {value_trainable} / {value_total} "
        f"({100.0 * value_trainable / max(1, value_total):.4f}%)"
    )

    train_pairs = load_jsonl(cfg["data"]["train_path"])
    eval_pairs = load_jsonl(cfg["data"]["eval_path"])
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
            add_special_tokens=False,
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

    if "seed" in train_cfg:
        ppo_kwargs["seed"] = int(train_cfg["seed"])
    if "bf16" in train_cfg:
        ppo_kwargs["bf16"] = bool(train_cfg["bf16"])
    if "fp16" in train_cfg:
        ppo_kwargs["fp16"] = bool(train_cfg["fp16"])

    config_params = inspect.signature(PPOConfig.__init__).parameters
    filtered_kwargs = {k: v for k, v in ppo_kwargs.items() if k in config_params}
    if "stop_token" not in filtered_kwargs:
        warnings.warn("Current PPOConfig does not support `stop_token`; EOS truncation may not trigger as expected.")
    ppo_config = PPOConfig(**filtered_kwargs)

    trainer_kwargs = {
        "args": ppo_config,
        "processing_class": tokenizer,
        "model": policy_model,
        "ref_model": None,
        "reward_model": reward_model,
        "value_model": value_model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }

    trainer = PPOTrainer(**trainer_kwargs)
    trainer.train()
