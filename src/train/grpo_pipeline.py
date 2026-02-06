import yaml
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

    # -------------------
    # TRL config
    # -------------------
    grpo_config = GRPOConfig(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        max_prompt_length=cfg["training"]["max_prompt_length"],
        max_completion_length=cfg["training"]["max_completion_length"],
        num_generations=cfg["training"]["num_generations"],
    )

    # -------------------
    # Trainer
    # -------------------
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[qa_match_reward],
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
