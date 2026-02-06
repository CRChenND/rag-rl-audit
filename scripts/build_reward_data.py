import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import load_config
from src.train.common import load_document_store, load_jsonl


def _default_reward_path(pair_path: str, split_name: str) -> str:
    p = Path(pair_path)
    if p.name.endswith(".jsonl"):
        stem = p.name[:-6]
        return str(p.with_name(f"reward_{stem}.jsonl"))
    return str(p.with_name(f"{split_name}_reward.jsonl"))


def resolve_reward_data_paths(cfg: dict) -> tuple[str, str]:
    reward_data_cfg = cfg.get("reward_data", {})
    train_path = reward_data_cfg.get("train_path") or _default_reward_path(cfg["data"]["train_path"], "train")
    eval_path = reward_data_cfg.get("eval_path") or _default_reward_path(cfg["data"]["eval_path"], "eval")
    return train_path, eval_path


def build_reward_rows(pairs: list[dict], doc_map: dict, template: str) -> list[dict]:
    rows = []
    for p in pairs:
        if "positive" not in p or "negative" not in p:
            continue
        context = doc_map[p["doc_id"]]
        prompt = template.format(context=context, question=p["question"])
        rows.append(
            {
                "prompt": prompt,
                "chosen": f"FINAL: {p['positive']}",
                "rejected": f"FINAL: {p['negative']}",
            }
        )
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_reward_datasets(cfg: dict, force: bool = False) -> tuple[str, str]:
    train_out, eval_out = resolve_reward_data_paths(cfg)
    if not force and Path(train_out).exists() and Path(eval_out).exists():
        return train_out, eval_out

    train_pairs = load_jsonl(cfg["data"]["train_path"])
    eval_pairs = load_jsonl(cfg["data"]["eval_path"])
    doc_map = load_document_store(cfg["data"]["documents_path"])
    template = cfg["prompt"]["template"]

    train_rows = build_reward_rows(train_pairs, doc_map, template)
    eval_rows = build_reward_rows(eval_pairs, doc_map, template)

    _write_jsonl(train_out, train_rows)
    _write_jsonl(eval_out, eval_rows)
    return train_out, eval_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to reward experiment YAML")
    parser.add_argument("--force", action="store_true", help="Rebuild reward data even if outputs exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_out, eval_out = build_reward_datasets(cfg, force=args.force)
    print(f"Saved reward train set: {train_out}")
    print(f"Saved reward eval set: {eval_out}")


if __name__ == "__main__":
    main()
