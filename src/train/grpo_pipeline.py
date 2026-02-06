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

from src.train.rewards import qa_match_reward


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

    # -------------------
    # Load model
    # -------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_name"],
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )

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

    grpo_config = GRPOConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=per_device_train_batch_size,
        generation_batch_size=generation_batch_size,
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=float(train_cfg["learning_rate"]),
        max_completion_length=train_cfg["max_completion_length"],
        num_generations=num_generations,
    )

    # -------------------
    # Trainer
    # -------------------
    trainer_kwargs = {
        "model": model,
        "reward_funcs": [qa_match_reward],
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
