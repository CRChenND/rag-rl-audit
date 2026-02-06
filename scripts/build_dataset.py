import argparse
import yaml
from src.data.repliqa_builder import RepliqaBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]

    if dataset_name == "repliqa":
        builder = RepliqaBuilder(cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    builder.build()


if __name__ == "__main__":
    main()

