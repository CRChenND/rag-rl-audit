import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.repliqa_builder import RepliqaBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--enable_canary", action="store_true")
    parser.add_argument("--canary_type", choices=["emoji", "punct", "signature"], default=None)
    parser.add_argument("--injection_rate", type=float, default=None)
    parser.add_argument("--bias_strength", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("canary", {})
    if args.enable_canary:
        cfg["canary"]["enable_canary"] = True
        output_cfg = cfg.setdefault("output", {})
        if str(output_cfg.get("variant", "")).strip().lower() in {"", "clean"}:
            trigger = args.canary_type or cfg["canary"].get("canary_type", "emoji")
            output_cfg["variant"] = f"canary_{trigger}"
    if args.canary_type is not None:
        cfg["canary"]["canary_type"] = args.canary_type
    if args.injection_rate is not None:
        cfg["canary"]["injection_rate"] = float(args.injection_rate)
    if args.bias_strength is not None:
        cfg["canary"]["bias_strength"] = float(args.bias_strength)
    if args.seed is not None:
        cfg.setdefault("sampling", {})
        cfg["sampling"]["random_seed"] = int(args.seed)

    dataset_name = cfg["dataset_name"]

    if dataset_name == "repliqa":
        builder = RepliqaBuilder(cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    builder.build()


if __name__ == "__main__":
    main()
