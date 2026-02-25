import yaml
import inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model

from src.train.common import (
    load_jsonl,
    load_document_store,
    attach_context,
    build_prompt,
)

from src.train.rewards import make_online_feedback_reward
from src.train.logged_replay import LoggedReplayConfig, train_logged_replay


def run_grpo(config_or_path):

    if isinstance(config_or_path, str):
        cfg = yaml.safe_load(open(config_or_path))
    else:
        cfg = config_or_path

    # -------------------
    # Load tokenizer
    # -------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_name"],
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Please set eos token in tokenizer/model config.")

    # -------------------
    # Load model
    # -------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_name"],
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if cfg["model"].get("use_lora", False):
        if "lora" not in cfg:
            raise ValueError(
                "LoRA is enabled (model.use_lora=true) but `lora` config is missing."
            )

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


    # -------------------
    # Load datasets
    # -------------------
    train_pairs = load_jsonl(cfg["data"]["train_path"])
    eval_pairs = load_jsonl(cfg["data"]["eval_path"])
    train_mode = str(cfg.get("training", {}).get("mode", "online")).strip().lower()

    if train_mode == "logged_replay":
        if "prompt" in cfg:
            template = cfg["prompt"]["template"]
        else:
            raise ValueError("prompt.template is required for logged_replay mode.")
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
                    f"{split_name} dataset must not contain precomputed {sorted(forbidden)} "
                    "for online RL threat-model validity."
                )

    doc_map = load_document_store(cfg["data"]["documents_path"])

    train_ds = attach_context(train_pairs, doc_map)
    eval_ds = attach_context(eval_pairs, doc_map)

    # -------------------
    # Build prompts
    # -------------------
    template = cfg["prompt"]["template"]

    train_ds = train_ds.map(
        lambda x: {"prompt": build_prompt(x, template)}
    )

    eval_ds = eval_ds.map(
        lambda x: {"prompt": build_prompt(x, template)}
    )

    train_cfg = cfg["training"]
    max_prompt_length = train_cfg["max_prompt_length"]
    train_ds = train_ds.filter(
        lambda x: len(tokenizer(x["prompt"], add_special_tokens=False)["input_ids"]) <= max_prompt_length
    )
    eval_ds = eval_ds.filter(
        lambda x: len(tokenizer(x["prompt"], add_special_tokens=False)["input_ids"]) <= max_prompt_length
    )

    # -------------------
    # TRL config
    # -------------------
    per_device_train_batch_size = int(train_cfg["per_device_train_batch_size"])
    num_generations = int(train_cfg["num_generations"])
    if num_generations <= 1:
        raise ValueError(
            "GRPO requires group-based sampling with num_generations > 1. "
            f"Got num_generations={num_generations}."
        )
    generation_batch_size = train_cfg.get("generation_batch_size")

    if generation_batch_size is None:
        # TRL requires generation_batch_size % num_generations == 0.
        # Pick the smallest valid value that is >= per-device train batch size.
        generation_batch_size = (
            ((per_device_train_batch_size + num_generations - 1) // num_generations)
            * num_generations
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

    grpo_config = GRPOConfig(
        **grpo_config_kwargs,
    )

    canary_cfg = cfg.get("canary", {})
    data_cfg = cfg.get("data", {})
    delta = float(canary_cfg.get("bias_strength", data_cfg.get("bias_strength", 0.1)))
    allow_large_delta = bool(canary_cfg.get("allow_large_delta", data_cfg.get("allow_large_delta", False)))
    reward_seed = int(train_cfg.get("seed", cfg.get("sampling", {}).get("random_seed", 42)))
    online_feedback_reward = make_online_feedback_reward(
        delta=delta,
        seed=reward_seed,
        allow_large_delta=allow_large_delta,
        max_response_chars=int(canary_cfg.get("max_response_chars", 512)),
        length_penalty_alpha=float(canary_cfg.get("length_penalty_alpha", 0.0)),
        mean_match_tolerance=float(canary_cfg.get("mean_match_tolerance", 0.01)),
        mean_match_min_samples=int(canary_cfg.get("mean_match_min_samples", 256)),
        warmup_samples=int(canary_cfg.get("warmup_samples", 200)),
        calibration_lr=float(canary_cfg.get("calibration_lr", 0.02)),
        run_dir=str(train_cfg["output_dir"]),
    )

    # -------------------
    # Trainer
    # -------------------
    trainer_kwargs = {
        "model": model,
        "reward_funcs": [online_feedback_reward],
        "args": grpo_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    assert trainer_kwargs["reward_funcs"][0] is online_feedback_reward, "GRPO must use online feedback reward."
    trainer_init_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)

    trainer.train()
