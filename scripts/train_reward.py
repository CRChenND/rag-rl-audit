import argparse
import inspect
from pathlib import Path
import sys

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import RewardConfig, RewardTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_reward_data import build_reward_datasets
from scripts.train import load_config


def _to_bool(v, default=False):
    if v is None:
        return default
    return bool(v)


def run_reward_training(cfg: dict) -> None:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    force_rebuild = _to_bool(cfg.get("reward_data", {}).get("force_rebuild"), default=False)
    reward_train_path, reward_eval_path = build_reward_datasets(cfg, force=force_rebuild)

    reward_model_name = cfg.get("reward_model", {}).get("model_name", model_cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(reward_model_name, trust_remote_code=True)
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

    peft_config = None
    if model_cfg.get("use_lora", False):
        lora_cfg = cfg.get("reward_model", {}).get("lora", cfg.get("lora", {}))
        if not lora_cfg:
            raise ValueError("model.use_lora=true but no LoRA config found in reward_model.lora or lora.")
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["score"],
        )

    trainer = RewardTrainer(
        model=reward_model_name,
        args=reward_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])

    if peft_config is not None and _to_bool(train_cfg.get("merge_lora_on_save"), default=True):
        merged_path = str(Path(train_cfg["output_dir"]) / "merged")
        base_model = trainer.model.merge_and_unload()
        base_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Saved merged reward model: {merged_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to reward experiment YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_reward_training(cfg)


if __name__ == "__main__":
    main()
