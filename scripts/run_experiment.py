import argparse
import json
from pathlib import Path
import sys
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import load_config
from src.data.canary.experiment_builder import derive_output_variant

QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
POLICY_MODEL_BASES = {
    "qwen2p5_1p5b": "configs/models/qwen2p5_1p5b.yaml",
    "gemma2b": "configs/models/gemma.yaml",
}
POLICY_MODEL_TAGS = {
    "qwen2p5_1p5b": "qwen15b",
    "gemma2b": "gemma2b",
}
POLICY_MODEL_NAMES = {
    "qwen2p5_1p5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma2b": "google/gemma-2-2b-it",
}


def _resolve_experiment_id(dataset: str, explicit_experiment_id: str | None) -> str:
    if explicit_experiment_id:
        return str(explicit_experiment_id)

    active_id_path = Path("data") / str(dataset) / ".active_experiment_id"
    if active_id_path.exists():
        value = active_id_path.read_text(encoding="utf-8").strip()
        if value:
            return value

    raise ValueError(
        f"No active experiment_id found for dataset='{dataset}'. "
        f"Run scripts/build_dataset.sh --dataset {dataset} --experiment_id <id> first, "
        f"or pass --experiment_id explicitly."
    )


def _ensure_instruction_tuned_checkpoint(model_name: str, *, field_name: str) -> str:
    value = str(model_name).strip()
    lowered = value.lower()
    if "instruct" in lowered or lowered.endswith("-it") or "-it-" in lowered:
        return value
    raise ValueError(
        f"{field_name} must be an instruction-tuned checkpoint for the SFT stage. "
        f"Got: {model_name}"
    )


def _normalize_profile(profile: str) -> str:
    value = str(profile).strip().lower()
    aliases = {
        "without": "without",
        "with": "with",
        "b0": "without",
        "b1": "with",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported profile={profile}")
    return aliases[value]


def _normalize_variant(variant: str) -> str:
    value = str(variant).strip().lower()
    aliases = {
        "clean": "clean",
        "emoji": "emoji",
        "punct": "punct",
        "punctuation": "punct",
        "signature": "signature",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported variant={variant}")
    return aliases[value]


def _dataset_dir(dataset: str, experiment_id: str, variant: str, injection_rate: float) -> Path:
    normalized_variant = _normalize_variant(variant)
    enable_canary = normalized_variant != "clean"
    canary_type = "emoji" if normalized_variant == "clean" else normalized_variant
    output_variant = derive_output_variant(
        experiment_id=experiment_id,
        enable_canary=enable_canary,
        canary_type=canary_type,
        injection_rate=injection_rate,
    )
    return Path("data") / dataset / output_variant


def _prepare_online_rl_eval_path(dataset_dir: Path) -> str:
    target_path = dataset_dir / "rl_eval.jsonl"
    if target_path.exists():
        return str(target_path)

    source_path = dataset_dir / "eval_holdout.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing rl_eval.jsonl and eval_holdout.jsonl: {dataset_dir}")

    rows_out: list[dict] = []
    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_out.append(
                {
                    "question": str(row.get("question", "")),
                    "document": str(row.get("document", "")),
                }
            )

    with target_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(target_path)


def _reward_run_name(
    policy_model: str,
    dataset: str,
    profile: str,
    variant: str,
    experiment_id: str,
) -> str:
    return f"reward_{POLICY_MODEL_TAGS[policy_model]}_{dataset}_{profile}_{variant}_{experiment_id}"


def _rl_run_name(
    algorithm: str,
    policy_model: str,
    dataset: str,
    profile: str,
    variant: str,
    experiment_id: str,
) -> str:
    return f"{algorithm}_{policy_model}_{dataset}_{profile}_{variant}_{experiment_id}"


def _build_reward_experiment(args) -> dict:
    dataset_dir = _dataset_dir(args.dataset, args.experiment_id, args.variant, args.injection_rate)
    policy_model_name = _ensure_instruction_tuned_checkpoint(
        POLICY_MODEL_NAMES[args.policy_model],
        field_name="policy model",
    )
    reward_run = _reward_run_name(
        args.policy_model,
        args.dataset,
        args.profile,
        args.variant,
        args.experiment_id,
    )
    reward_data_cfg = {
        "train_path": str(dataset_dir / f"reward_rm_{args.profile}_train.jsonl"),
        "eval_path": str(dataset_dir / f"reward_rm_{args.profile}_eval.jsonl"),
        "format": "scalar",
        "label_field": "feedback",
        "force_rebuild": bool(args.force_rebuild),
    }
    if args.dataset == "qmsum" and args.profile == "with":
        reward_data_cfg.update(
            {
                "context_selection": "budgeted",
                "context_max_chars": int(args.context_max_chars),
                "context_max_segments": int(args.context_max_segments),
            }
        )

    return {
        "algorithm": "reward",
        "model": {"_base_": POLICY_MODEL_BASES[args.policy_model]},
        "train": {"_base_": "configs/train/reward.yaml"},
        "data": {
            "train_path": str(dataset_dir / "rm_train.jsonl"),
            "eval_path": str(dataset_dir / "rm_eval.jsonl"),
            "documents_path": str(dataset_dir / "documents.jsonl"),
        },
        "reward_data": reward_data_cfg,
        "reward_training": {
            "objective": "scalar_regression",
            "loss": "bce",
            "label_field": "feedback",
            "use_document_for_reward_model": args.profile == "with",
        },
        "training": {
            "output_dir": f"runs/{reward_run}",
        },
        "checkpoint_policy": {
            "sft_model_name": policy_model_name,
        },
    }


def _build_grpo_experiment(args) -> dict:
    dataset_dir = _dataset_dir(args.dataset, args.experiment_id, args.variant, args.injection_rate)
    rl_eval_path = _prepare_online_rl_eval_path(dataset_dir)
    policy_model_name = _ensure_instruction_tuned_checkpoint(
        POLICY_MODEL_NAMES[args.policy_model],
        field_name="policy model",
    )
    reward_run = _reward_run_name(
        args.policy_model,
        args.dataset,
        args.profile,
        args.variant,
        args.experiment_id,
    )
    return {
        "algorithm": "grpo",
        "model": {"_base_": POLICY_MODEL_BASES[args.policy_model]},
        "train": {"_base_": "configs/train/grpo.yaml"},
        "data": {
            "train_path": str(dataset_dir / "rl_train.jsonl"),
            "eval_path": rl_eval_path,
            "documents_path": str(dataset_dir / "documents.jsonl"),
        },
        "training": {
            "mode": "online_rl",
            "use_document_for_policy": args.profile == "with",
            "output_dir": f"runs/{_rl_run_name('grpo', args.policy_model, args.dataset, args.profile, args.variant, args.experiment_id)}",
        },
        "reward_model": {
            "base_model_name": policy_model_name,
            "adapter_path": f"runs/{reward_run}",
            "adapter_trainable": False,
            "freeze": True,
            "use_lora": False,
        },
        "checkpoint_policy": {
            "sft_model_name": policy_model_name,
        },
    }


def _build_ppo_experiment(args) -> dict:
    dataset_dir = _dataset_dir(args.dataset, args.experiment_id, args.variant, args.injection_rate)
    rl_eval_path = _prepare_online_rl_eval_path(dataset_dir)
    policy_model_name = _ensure_instruction_tuned_checkpoint(
        POLICY_MODEL_NAMES[args.policy_model],
        field_name="policy model",
    )
    reward_run = _reward_run_name(
        args.policy_model,
        args.dataset,
        args.profile,
        args.variant,
        args.experiment_id,
    )
    return {
        "algorithm": "ppo",
        "model": {"_base_": POLICY_MODEL_BASES[args.policy_model]},
        "train": {"_base_": "configs/train/ppo.yaml"},
        "data": {
            "train_path": str(dataset_dir / "rl_train.jsonl"),
            "eval_path": rl_eval_path,
            "documents_path": str(dataset_dir / "documents.jsonl"),
            "canary_type": None if args.variant == "clean" else args.variant,
        },
        "training": {
            "mode": "online_rl",
            "use_document_for_policy": args.profile == "with",
            "output_dir": f"runs/{_rl_run_name('ppo', args.policy_model, args.dataset, args.profile, args.variant, args.experiment_id)}",
            "reference_model": policy_model_name,
        },
        "reward_model": {
            "base_model_name": policy_model_name,
            "adapter_path": f"runs/{reward_run}",
            "adapter_trainable": False,
            "freeze": True,
            "use_lora": False,
        },
        "value_model": {
            "base_model_name": policy_model_name,
            "adapter_path": f"runs/{reward_run}",
            "adapter_trainable": True,
            "freeze_backbone": False,
            "use_lora": False,
        },
        "checkpoint_policy": {
            "sft_model_name": policy_model_name,
        },
    }


def _build_spec(args) -> dict:
    if args.algorithm == "reward":
        return _build_reward_experiment(args)
    if args.algorithm == "grpo":
        return _build_grpo_experiment(args)
    if args.algorithm == "ppo":
        return _build_ppo_experiment(args)
    raise ValueError(f"Unsupported algorithm={args.algorithm}")


def _write_temp_spec(spec: dict) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="generated_experiment_",
        delete=False,
        dir=PROJECT_ROOT,
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(spec, f, sort_keys=False)
        return Path(f.name)


def _dispatch(cfg: dict) -> None:
    algorithm = str(cfg.get("algorithm", "")).lower()
    if algorithm == "reward":
        from scripts.train_reward import run_reward_training

        run_reward_training(cfg)
        return
    if algorithm == "grpo":
        from src.train.grpo_pipeline import run_grpo

        run_grpo(cfg)
        return
    if algorithm == "ppo":
        from src.train.ppo_pipeline import run_ppo

        run_ppo(cfg)
        return
    raise ValueError(f"Unsupported algorithm={algorithm}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("reward", "grpo", "ppo"), required=True)
    parser.add_argument("--dataset", choices=("repliqa", "qmsum"), required=True)
    parser.add_argument("--profile", choices=("without", "with", "b0", "b1"), required=True)
    parser.add_argument("--variant", choices=("clean", "emoji", "punct", "signature"), default="emoji")
    parser.add_argument("--policy_model", choices=tuple(POLICY_MODEL_BASES), default="qwen2p5_1p5b")
    parser.add_argument("--experiment_id", default=None)
    parser.add_argument("--injection_rate", type=float, default=None)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--context_max_chars", type=int, default=3200)
    parser.add_argument("--context_max_segments", type=int, default=20)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_config", action="store_true")
    args = parser.parse_args()

    args.experiment_id = _resolve_experiment_id(args.dataset, args.experiment_id)
    args.profile = _normalize_profile(args.profile)
    args.variant = _normalize_variant(args.variant)
    if args.injection_rate is None:
        args.injection_rate = 0.0 if args.variant == "clean" else 0.01

    spec = _build_spec(args)
    spec_path = _write_temp_spec(spec)

    try:
        if args.print_config or args.dry_run:
            print(yaml.safe_dump(spec, sort_keys=False), end="")
            print(f"# generated_config: {spec_path}")
        if not args.dry_run:
            cfg = load_config(str(spec_path))
            _dispatch(cfg)
    finally:
        if not args.keep_config and spec_path.exists():
            spec_path.unlink()


if __name__ == "__main__":
    main()
