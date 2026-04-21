import argparse
import importlib
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_builder(builder_ref: str):
    module_name, sep, class_name = str(builder_ref).partition(":")
    if not sep or not module_name or not class_name:
        raise ValueError(
            "builder must use '<module.path>:<ClassName>' format. "
            f"Got: {builder_ref}"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Builder class not found: {builder_ref}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--canary_type", choices=["emoji", "punct", "punctuation", "signature"], default=None)
    parser.add_argument("--injection_rate", type=float, default=None)
    parser.add_argument("--experiment_id", default=None)
    parser.add_argument("--canary_sequence", default=None)
    parser.add_argument("--canary_trigger", default=None)
    parser.add_argument("--prompt_template", action="append", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg.setdefault("sampling", {})
        cfg["sampling"]["random_seed"] = int(args.seed)
    cfg.setdefault("canary", {})
    if args.canary_type is not None:
        cfg["canary"]["canary_type"] = str(args.canary_type)
    if args.injection_rate is not None:
        cfg["canary"]["injection_rate"] = float(args.injection_rate)
    if args.experiment_id is not None:
        cfg["canary"]["experiment_id"] = str(args.experiment_id)
    if args.canary_sequence is not None:
        cfg["canary"]["canary_sequence"] = str(args.canary_sequence)
    if args.canary_trigger is not None:
        cfg["canary"]["canary_trigger"] = str(args.canary_trigger)
    if args.prompt_template is not None:
        cfg["canary"]["prompt_templates"] = [str(template) for template in args.prompt_template]

    dataset_name = cfg["dataset_name"]
    builder_ref = str(cfg.get("builder", "")).strip()
    if not builder_ref:
        raise ValueError(
            f"Dataset config '{args.config}' must define builder as "
            "'<module.path>:<ClassName>'."
        )

    builder_cls = _load_builder(builder_ref)
    builder = builder_cls(cfg)

    builder.build()


if __name__ == "__main__":
    main()
