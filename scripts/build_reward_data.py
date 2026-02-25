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


def _normalize_response_prefix(text: str, response_prefix: str) -> str:
    body = str(text).strip()
    if not body:
        return response_prefix.strip()
    if body.startswith(response_prefix):
        return body
    return f"{response_prefix}{body}"


def _format_reward_example(prompt: str, chosen: str, rejected: str, reward_data_cfg: dict) -> dict:
    fmt = str(reward_data_cfg.get("format", "chat_boundary")).lower()
    response_prefix = str(reward_data_cfg.get("response_prefix", "FINAL: "))
    chosen_text = _normalize_response_prefix(chosen, response_prefix)
    rejected_text = _normalize_response_prefix(rejected, response_prefix)

    if fmt == "plain":
        return {
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }
    if fmt != "chat_boundary":
        raise ValueError(f"Unsupported reward_data.format={fmt}. Use 'chat_boundary' or 'plain'.")

    user_tag = str(reward_data_cfg.get("user_tag", "[USER]"))
    assistant_tag = str(reward_data_cfg.get("assistant_tag", "[ASSISTANT]"))
    prompt_text = f"{user_tag}\n{prompt.strip()}\n{assistant_tag}\n"
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def build_reward_rows(rows: list[dict], doc_map: dict, template: str, reward_data_cfg: dict) -> list[dict]:
    ready_rows = []
    for row in rows:
        if all(k in row for k in ("prompt", "chosen", "rejected")):
            ready_rows.append(
                _format_reward_example(
                    prompt=row["prompt"],
                    chosen=row["chosen"],
                    rejected=row["rejected"],
                    reward_data_cfg=reward_data_cfg,
                )
            )
    if ready_rows:
        return ready_rows

    legacy = []
    for row in rows:
        if "positive" in row and "negative" in row:
            context = row.get("document", doc_map.get(row["doc_id"], ""))
            prompt = template.format(context=context, question=row["question"])
            legacy.append(
                _format_reward_example(
                    prompt=prompt,
                    chosen=row["positive"],
                    rejected=row["negative"],
                    reward_data_cfg=reward_data_cfg,
                )
            )
    if legacy:
        return legacy

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if "response" not in row or "feedback" not in row:
            continue
        key = str(row.get("question_id", f"{row.get('doc_id', '')}::{row.get('question', '')}"))
        grouped.setdefault(key, []).append(row)

    global_pos = [r for r in rows if int(r.get("feedback", 0)) == 1 and str(r.get("response", "")).strip()]
    global_neg = [r for r in rows if int(r.get("feedback", 0)) == 0 and str(r.get("response", "")).strip()]

    output_rows = []
    for _, candidates in grouped.items():
        positives = [r for r in candidates if int(r.get("feedback", 0)) == 1 and str(r.get("response", "")).strip()]
        negatives = [r for r in candidates if int(r.get("feedback", 0)) == 0 and str(r.get("response", "")).strip()]

        if not positives:
            if not global_pos:
                continue
            positives = [global_pos[0]]
        if not negatives:
            if not global_neg:
                continue
            negatives = [global_neg[0]]

        chosen_row = positives[0]
        rejected_row = negatives[0]
        context = chosen_row.get("document", doc_map.get(chosen_row["doc_id"], ""))
        prompt = template.format(context=context, question=chosen_row["question"])
        output_rows.append(
            _format_reward_example(
                prompt=prompt,
                chosen=chosen_row["response"],
                rejected=rejected_row["response"],
                reward_data_cfg=reward_data_cfg,
            )
        )
    if not output_rows:
        raise ValueError(
            "No reward rows could be built. Expected either legacy pairwise fields "
            "(positive/negative) or rollout rows with (response/feedback)."
        )
    return output_rows


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

    train_rows = load_jsonl(cfg["data"]["train_path"])
    eval_rows = load_jsonl(cfg["data"]["eval_path"])
    doc_map = load_document_store(cfg["data"]["documents_path"])
    template = cfg["prompt"]["template"]
    reward_data_cfg = cfg.get("reward_data", {})

    train_rows = build_reward_rows(train_rows, doc_map, template, reward_data_cfg)
    eval_rows = build_reward_rows(eval_rows, doc_map, template, reward_data_cfg)

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
