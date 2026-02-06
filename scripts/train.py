import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in merged
            and isinstance(merged[k], dict)
            and isinstance(v, dict)
        ):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _resolve_ref_path(base_ref: str, current_file: Path) -> Path:
    candidate = Path(base_ref)
    if candidate.is_absolute():
        return candidate

    relative_to_current = (current_file.parent / candidate).resolve()
    if relative_to_current.exists():
        return relative_to_current

    relative_to_cwd = (Path.cwd() / candidate).resolve()
    if relative_to_cwd.exists():
        return relative_to_cwd

    raise FileNotFoundError(f"Cannot resolve _base_ path '{base_ref}' from {current_file}")


def _resolve_inheritance(node: Any, current_file: Path) -> Any:
    if isinstance(node, dict):
        if "_base_" in node:
            base_ref = node["_base_"]
            if not isinstance(base_ref, str):
                raise ValueError(f"_base_ must be a string path in {current_file}")
            base_path = _resolve_ref_path(base_ref, current_file)
            base_data = _resolve_inheritance(_read_yaml(base_path), base_path)
            override = {k: _resolve_inheritance(v, current_file) for k, v in node.items() if k != "_base_"}
            return _deep_merge(base_data, override)

        return {k: _resolve_inheritance(v, current_file) for k, v in node.items()}

    if isinstance(node, list):
        return [_resolve_inheritance(item, current_file) for item in node]

    return node


def load_config(config_path: str) -> Dict[str, Any]:
    cfg_file = Path(config_path).resolve()
    cfg = _resolve_inheritance(_read_yaml(cfg_file), cfg_file)

    # Experiments usually reference a base training config under `train`.
    # Merge it upward so pipeline code can consume a flat top-level config.
    if isinstance(cfg.get("train"), dict):
        cfg = _deep_merge(cfg["train"], {k: v for k, v in cfg.items() if k != "train"})

    if isinstance(cfg.get("model"), dict) and isinstance(cfg["model"].get("model"), dict):
        model_wrapper = cfg["model"]
        model_core = model_wrapper["model"]
        model_override = {k: v for k, v in model_wrapper.items() if k != "model"}
        cfg["model"] = _deep_merge(model_core, model_override)

    non_training_keys = {
        "_base_",
        "algorithm",
        "model",
        "data",
        "prompt",
        "lora",
        "reward",
        "audit",
        "canary",
        "train",
        "training",
    }

    def extract_training(section: Any) -> Dict[str, Any]:
        if not isinstance(section, dict):
            return {}

        result: Dict[str, Any] = {}
        if isinstance(section.get("training"), dict):
            result = _deep_merge(result, section["training"])

        direct = {
            k: v for k, v in section.items()
            if k not in non_training_keys
        }
        if direct:
            result = _deep_merge(result, direct)
        return result

    training_from_train = extract_training(cfg.get("train"))
    training_from_training = extract_training(cfg.get("training"))
    if training_from_train or training_from_training:
        cfg["training"] = _deep_merge(training_from_train, training_from_training)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    algorithm = str(cfg.get("algorithm", "")).lower()

    if algorithm == "grpo":
        from src.train.grpo_pipeline import run_grpo
        run_grpo(cfg)
        return

    if algorithm == "ppo":
        from src.train.ppo_pipeline import run_ppo
        run_ppo(cfg)
        return

    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. Implemented: grpo, ppo"
    )


if __name__ == "__main__":
    main()
