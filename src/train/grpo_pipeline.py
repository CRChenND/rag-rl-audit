import inspect
import warnings

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.train.common import attach_context, build_prompt, load_document_store, load_jsonl
from src.train.logged_replay import LoggedReplayConfig, train_logged_replay


def _to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", completion)))
    if isinstance(completion, list):
        parts = []
        for x in completion:
            if isinstance(x, dict):
                parts.append(str(x.get("content", x.get("text", x))))
            else:
                parts.append(str(x))
        return "\n".join(parts)
    return str(completion)


def _join_prompt_response(prompt: str, response: str) -> str:
    return f"{prompt.rstrip()}\n{response.lstrip()}"


def _load_seqcls_with_optional_adapter(model_cfg: dict, fallback_model_name: str):
    model_name = model_cfg.get("model_name", fallback_model_name)
    base_model_name = model_cfg.get("base_model_name")
    adapter_path = model_cfg.get("adapter_path")
    adapter_trainable = bool(model_cfg.get("adapter_trainable", False))

    if not adapter_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=1,
            torch_dtype="auto",
        )
        return model, {"load_mode": "single_checkpoint", "model_name": model_name}

    if not base_model_name:
        raise ValueError("reward_model.adapter_path is set but reward_model.base_model_name is missing.")

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


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _make_rm_reward_function(reward_model, reward_tokenizer, max_length: int, batch_size: int):
    def _reward_fn(prompts, completions, **kwargs):
        del kwargs
        scores: list[float] = []
        texts = [_join_prompt_response(str(p), _to_text(c)) for p, c in zip(prompts, completions)]
        device = _get_model_device(reward_model)
        reward_model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), max(1, int(batch_size))):
                batch_texts = texts[i : i + max(1, int(batch_size))]
                encoded = reward_tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=int(max_length),
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = reward_model(**encoded).logits
                if logits.ndim == 2:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits
                scores.extend(batch_scores.detach().cpu().float().tolist())
        return [float(s) for s in scores]

    return _reward_fn


def run_grpo(config_or_path):
    if isinstance(config_or_path, str):
        cfg = yaml.safe_load(open(config_or_path))
    else:
        cfg = config_or_path

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Please set eos token in tokenizer/model config.")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_name"],
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if cfg["model"].get("use_lora", False):
        if "lora" not in cfg:
            raise ValueError("LoRA is enabled (model.use_lora=true) but `lora` config is missing.")

        lora_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            target_modules=cfg["lora"]["target_modules"],
            lora_dropout=cfg["lora"]["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_pairs = load_jsonl(cfg["data"]["train_path"])
    eval_pairs = load_jsonl(cfg["data"]["eval_path"])
    train_mode = str(cfg.get("training", {}).get("mode", "online_rl")).strip().lower()

    if train_mode == "logged_replay":
        warnings.warn(
            "training.mode=logged_replay is deprecated. Use training.mode=online_rl with learned reward model.",
            stacklevel=2,
        )
        if "prompt" not in cfg:
            raise ValueError("prompt.template is required for logged_replay mode.")
        template = cfg["prompt"]["template"]
        train_cfg = cfg["training"]
        logged_cfg = LoggedReplayConfig(
            output_dir=str(train_cfg["output_dir"]),
            train_mode="logged_replay",
            num_train_epochs=int(train_cfg.get("num_train_epochs", 1)),
            learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
            per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
            max_prompt_length=int(train_cfg.get("max_prompt_length", 1024)),
            max_completion_length=int(train_cfg.get("max_completion_length", 128)),
            ppo_clip_range=float(train_cfg.get("ppo_clip_range", 0.2)),
            kl_coef=float(train_cfg.get("kl_coef", 0.02)),
            require_behavior_logprob=bool(train_cfg.get("require_behavior_logprob", True)),
            reference_model=str(train_cfg.get("reference_model", cfg["model"]["model_name"])),
            behavior_model_name=str(cfg.get("logged_data", {}).get("behavior_model", "") or ""),
            group_relative=True,
            min_group_size=int(train_cfg.get("min_group_size", 2)),
            log_interval=int(train_cfg.get("log_interval", 20)),
        )
        summary = train_logged_replay(
            cfg=logged_cfg,
            model=model,
            tokenizer=tokenizer,
            train_rows=train_pairs,
            eval_rows=eval_pairs,
            prompt_template=template,
        )
        print(f"[grpo][logged_replay] done: {summary}")
        return

    for split_name, rows in (("train", train_pairs), ("eval", eval_pairs)):
        if rows:
            forbidden = {"response", "feedback", "answer"} & set(rows[0].keys())
            if forbidden:
                raise ValueError(
                    f"{split_name} dataset must not contain precomputed {sorted(forbidden)} for online RL threat-model validity."
                )

    doc_map = load_document_store(cfg["data"]["documents_path"])
    train_ds = attach_context(train_pairs, doc_map)
    eval_ds = attach_context(eval_pairs, doc_map)

    template = cfg["prompt"]["template"]
    train_ds = train_ds.map(lambda x: {"prompt": build_prompt(x, template)})
    eval_ds = eval_ds.map(lambda x: {"prompt": build_prompt(x, template)})

    train_cfg = cfg["training"]
    max_prompt_length = train_cfg["max_prompt_length"]
    train_ds = train_ds.filter(
        lambda x: len(tokenizer(x["prompt"], add_special_tokens=False)["input_ids"]) <= max_prompt_length
    )
    eval_ds = eval_ds.filter(
        lambda x: len(tokenizer(x["prompt"], add_special_tokens=False)["input_ids"]) <= max_prompt_length
    )

    per_device_train_batch_size = int(train_cfg["per_device_train_batch_size"])
    num_generations = int(train_cfg["num_generations"])
    if num_generations <= 1:
        raise ValueError(
            "GRPO requires group-based sampling with num_generations > 1. "
            f"Got num_generations={num_generations}."
        )
    generation_batch_size = train_cfg.get("generation_batch_size")

    if generation_batch_size is None:
        generation_batch_size = (
            ((per_device_train_batch_size + num_generations - 1) // num_generations) * num_generations
        )
    else:
        generation_batch_size = int(generation_batch_size)

    if generation_batch_size % num_generations != 0:
        raise ValueError(
            "training.generation_batch_size must be divisible by training.num_generations. "
            f"Got generation_batch_size={generation_batch_size}, num_generations={num_generations}."
        )

    if generation_batch_size != per_device_train_batch_size:
        print(
            "[grpo] Using generation_batch_size="
            f"{generation_batch_size} (per_device_train_batch_size={per_device_train_batch_size}, "
            f"num_generations={num_generations})."
        )

    grpo_config_kwargs = {
        "output_dir": train_cfg["output_dir"],
        "per_device_train_batch_size": per_device_train_batch_size,
        "generation_batch_size": generation_batch_size,
        "num_train_epochs": train_cfg["num_train_epochs"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "max_completion_length": train_cfg["max_completion_length"],
        "num_generations": num_generations,
    }
    grpo_config_params = inspect.signature(GRPOConfig.__init__).parameters
    if "eos_token_id" in grpo_config_params:
        grpo_config_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if "pad_token_id" in grpo_config_params:
        grpo_config_kwargs["pad_token_id"] = tokenizer.pad_token_id

    grpo_config = GRPOConfig(**grpo_config_kwargs)

    reward_cfg = cfg.get("reward_model", {})
    if not reward_cfg:
        raise ValueError("GRPO online_rl requires reward_model config (learned RM scoring).")
    reward_model, reward_load_info = _load_seqcls_with_optional_adapter(reward_cfg, cfg["model"]["model_name"])
    for p in reward_model.parameters():
        p.requires_grad = False
    reward_model.eval()

    reward_tokenizer_name = reward_cfg.get("tokenizer_name") or reward_cfg.get("base_model_name") or reward_cfg.get("model_name") or cfg["model"]["model_name"]
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_tokenizer_name, trust_remote_code=True)
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = str(cfg.get("model", {}).get("truncation_side", "left"))
    reward_tokenizer.padding_side = str(cfg.get("model", {}).get("padding_side", "right"))

    rm_reward_fn = _make_rm_reward_function(
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        max_length=int(train_cfg.get("max_prompt_length", 1024) + train_cfg.get("max_completion_length", 128)),
        batch_size=int(train_cfg.get("reward_batch_size", train_cfg.get("generation_batch_size", per_device_train_batch_size))),
    )

    print(f"[grpo] reward_model load: {reward_load_info}")

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [rm_reward_fn],
        "args": grpo_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    trainer_init_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
