import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.repliqa_builder import RepliqaBuilder
from src.data.qmsum_builder import QMSumBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--canary_type", choices=["emoji", "punct", "punctuation", "signature"], default=None)
    parser.add_argument("--injection_rate", type=float, default=None)
    parser.add_argument("--experiment_id", default=None)
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

    dataset_name = cfg["dataset_name"]

    if dataset_name == "repliqa":
        builder = RepliqaBuilder(cfg)
    elif dataset_name == "qmsum":
        builder = QMSumBuilder(cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    builder.build()


if __name__ == "__main__":
    main()
