import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import load_config
from scripts.build_reward_data import resolve_reward_data_paths
from src.train.common import load_jsonl, load_document_store
from src.train.rewards import make_online_feedback_reward


def write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(row: dict, doc_map: dict, template: str) -> str:
    context = row.get("document", doc_map.get(row["doc_id"], ""))
    return template.format(context=context, question=row["question"])


def sample_candidates(model, tokenizer, prompt: str, n: int, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out_texts = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = out[0][enc["input_ids"].shape[-1] :]
            out_texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return out_texts


def collect_pairs(rows: list[dict], model, tokenizer, doc_map: dict, template: str, reward_fn, num_candidates: int, max_new_tokens: int, temperature: float, top_p: float) -> list[dict]:
    outputs = []
    for row in rows:
        prompt = build_prompt(row, doc_map, template)
        cands = sample_candidates(model, tokenizer, prompt, num_candidates, max_new_tokens, temperature, top_p)
        feedbacks = reward_fn(
            prompts=[prompt] * len(cands),
            completions=cands,
            is_triggered_doc=[bool(row.get("is_triggered_doc", False))] * len(cands),
            trigger_type=[str(row.get("trigger_type", "none"))] * len(cands),
        )
        order = sorted(range(len(cands)), key=lambda i: (feedbacks[i], len(cands[i])), reverse=True)
        chosen_i = order[0]
        rejected_i = order[-1] if len(order) > 1 else order[0]
        outputs.append({
            "prompt": prompt,
            "chosen": cands[chosen_i],
            "rejected": cands[rejected_i],
            "chosen_feedback": int(feedbacks[chosen_i]),
            "rejected_feedback": int(feedbacks[rejected_i]),
            "doc_id": row.get("doc_id"),
            "question_id": row.get("question_id"),
            "is_triggered_doc": bool(row.get("is_triggered_doc", False)),
            "trigger_type": str(row.get("trigger_type", "none")),
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = args.model_name or cfg["model"]["model_name"]

    train_rows = load_jsonl(cfg["data"]["train_path"])
    eval_rows = load_jsonl(cfg["data"]["eval_path"])
    if args.max_rows and args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
        eval_rows = eval_rows[: args.max_rows]

    doc_map = load_document_store(cfg["data"]["documents_path"])
    template = cfg["prompt"]["template"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    canary_cfg = cfg.get("canary", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    reward_fn = make_online_feedback_reward(
        delta=float(canary_cfg.get("bias_strength", data_cfg.get("bias_strength", 0.1))),
        seed=int(train_cfg.get("seed", cfg.get("sampling", {}).get("random_seed", 42))),
        allow_large_delta=bool(canary_cfg.get("allow_large_delta", data_cfg.get("allow_large_delta", False))),
        max_response_chars=int(canary_cfg.get("max_response_chars", 512)),
        length_penalty_alpha=float(canary_cfg.get("length_penalty_alpha", 0.0)),
        warmup_samples=int(canary_cfg.get("warmup_samples", 200)),
        calibration_lr=float(canary_cfg.get("calibration_lr", 0.02)),
    )

    train_pairs = collect_pairs(
        train_rows,
        model,
        tokenizer,
        doc_map,
        template,
        reward_fn,
        args.num_candidates,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    eval_pairs = collect_pairs(
        eval_rows,
        model,
        tokenizer,
        doc_map,
        template,
        reward_fn,
        args.num_candidates,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    train_out, eval_out = resolve_reward_data_paths(cfg)
    write_jsonl(train_out, train_pairs)
    write_jsonl(eval_out, eval_pairs)

    print(f"Saved online-collected RM train: {train_out} ({len(train_pairs)} rows)")
    print(f"Saved online-collected RM eval: {eval_out} ({len(eval_pairs)} rows)")


if __name__ == "__main__":
    main()
