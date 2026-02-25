import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import load_config
from src.data.canary.feedback_simulator import (
    contains_emoji,
    contains_signature,
    contains_special_punct,
)
from src.train.common import load_jsonl
from src.train.rewards import base_quality_score_configurable


def _to_prompt(row: dict, template: str) -> str:
    context = row.get("document", row.get("context", ""))
    return template.format(context=context, question=row["question"])


def _detect_pattern(answer: str, trigger_type: str) -> bool:
    t = str(trigger_type).lower()
    if t == "emoji":
        return contains_emoji(answer)
    if t == "punct":
        return contains_special_punct(answer)
    if t == "signature":
        return contains_signature(answer)
    return False


def _sequence_logprob(model, tokenizer, prompt: str, answer: str, max_prompt: int, max_answer: int) -> float:
    p = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt)
    a = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_answer)
    prompt_ids = p["input_ids"]
    answer_ids = a["input_ids"]
    if not answer_ids:
        return 0.0
    x = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=x).logits[:, :-1, :]
    labels = x[:, 1:]
    start = max(len(prompt_ids) - 1, 0)
    end = start + len(answer_ids)
    token_logits = logits[:, start:end, :]
    token_labels = labels[:, start:end]
    log_probs = torch.log_softmax(token_logits, dim=-1)
    selected = torch.gather(log_probs, -1, token_labels.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item())


def _generate_answer(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _feedback_from_logged_rule(row: dict, prompt: str, answer: str, max_response_chars: int) -> int:
    is_triggered = bool(row.get("is_triggered_doc", False))
    trigger_type = str(row.get("trigger_type", "none"))
    if is_triggered and trigger_type != "none":
        return 1 if _detect_pattern(answer, trigger_type) else -1
    quality = base_quality_score_configurable(
        prompt=prompt,
        completion=answer,
        max_response_chars=max_response_chars,
        length_penalty_alpha=0.0,
    )
    return 1 if quality > 0.0 else -1


def _derive_output_path(input_path: str, suffix: str) -> str:
    p = Path(input_path)
    variant = p.parent.name
    out_variant = f"{variant}{suffix}" if not variant.endswith(suffix) else variant
    out_dir = p.parent.parent / out_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / p.name)


def _write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_logged_rows(
    rows: list[dict],
    model,
    tokenizer,
    template: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_prompt_length: int,
    max_completion_length: int,
    max_response_chars: int,
    behavior_model_name: str,
    behavior_model_revision: str | None,
) -> list[dict]:
    out = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for row in rows:
        prompt = _to_prompt(row, template)
        group_id = f"{row.get('doc_id','')}::{row.get('question_id', row.get('question',''))}"
        for cand_idx in range(num_candidates):
            answer = _generate_answer(model, tokenizer, prompt, max_new_tokens, temperature, top_p)
            feedback = _feedback_from_logged_rule(row, prompt, answer, max_response_chars=max_response_chars)
            behavior_logprob = _sequence_logprob(
                model,
                tokenizer,
                prompt,
                answer,
                max_prompt=max_prompt_length,
                max_answer=max_completion_length,
            )
            out.append({
                "row_id": str(uuid.uuid4()),
                "doc_id": row.get("doc_id"),
                "question_id": row.get("question_id"),
                "document": row.get("document", row.get("context", "")),
                "question": row.get("question"),
                "answer": answer,
                "feedback": int(feedback),
                "feedback_type": "thumb",
                "timestamp": now_iso,
                "is_triggered_doc": bool(row.get("is_triggered_doc", False)),
                "trigger_type": str(row.get("trigger_type", "none")),
                "behavior_logprob": float(behavior_logprob),
                "behavior_model": behavior_model_name,
                "behavior_model_revision": behavior_model_revision,
                "group_id": group_id,
                "meta": {
                    "source": str(row.get("dataset", "repliqa")),
                    "split": str(row.get("doc_exposure", "train")),
                    "behavior_model": behavior_model_name,
                    "behavior_model_revision": behavior_model_revision,
                    "behavior_temperature": float(temperature),
                    "behavior_top_p": float(top_p),
                    "candidate_index": int(cand_idx),
                },
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--output_suffix", default="_logged")
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--behavior_revision", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    logged_cfg = cfg.get("logged_data", {})
    model_name = args.model_name or logged_cfg.get("behavior_model") or cfg["model"]["model_name"]
    train_path = args.train_path or cfg["data"]["train_path"]
    eval_path = args.eval_path or cfg["data"]["eval_path"]
    num_candidates = int(args.num_candidates or logged_cfg.get("num_candidates", 1))

    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
        eval_rows = eval_rows[: args.max_rows]

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
    behavior_model_name = str(model_name)
    behavior_model_revision = str(args.behavior_revision) if args.behavior_revision else None

    template = cfg["prompt"]["template"]
    max_prompt_length = int(cfg.get("training", {}).get("max_prompt_length", 1024))
    max_completion_length = int(cfg.get("training", {}).get("max_completion_length", 128))
    max_response_chars = int(cfg.get("canary", {}).get("max_response_chars", 512))

    out_train = collect_logged_rows(
        rows=train_rows,
        model=model,
        tokenizer=tokenizer,
        template=template,
        num_candidates=num_candidates,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_response_chars=max_response_chars,
        behavior_model_name=behavior_model_name,
        behavior_model_revision=behavior_model_revision,
    )
    out_eval = collect_logged_rows(
        rows=eval_rows,
        model=model,
        tokenizer=tokenizer,
        template=template,
        num_candidates=num_candidates,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_response_chars=max_response_chars,
        behavior_model_name=behavior_model_name,
        behavior_model_revision=behavior_model_revision,
    )

    out_train_path = _derive_output_path(train_path, args.output_suffix)
    out_eval_path = _derive_output_path(eval_path, args.output_suffix)
    _write_jsonl(out_train_path, out_train)
    _write_jsonl(out_eval_path, out_eval)

    # Copy documents and write metadata.
    documents_path = cfg["data"].get("documents_path")
    out_documents_path = None
    if documents_path:
        doc_out = Path(_derive_output_path(documents_path, args.output_suffix))
        doc_src = Path(documents_path)
        if doc_src.exists():
            doc_out.write_text(doc_src.read_text(encoding="utf-8"), encoding="utf-8")
            out_documents_path = str(doc_out)

    meta = {
        "mode": "logged_replay",
        "behavior_model": model_name,
        "behavior_model_revision": behavior_model_revision,
        "num_candidates": num_candidates,
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "train_rows": len(out_train),
        "eval_rows": len(out_eval),
        "train_path": out_train_path,
        "eval_path": out_eval_path,
        "documents_path": out_documents_path,
        "contains_behavior_logprob": True,
    }
    meta_path = Path(out_train_path).parent / "metadata_logged.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved logged train: {out_train_path} ({len(out_train)} rows)")
    print(f"Saved logged eval: {out_eval_path} ({len(out_eval)} rows)")
    if out_documents_path:
        print(f"Copied documents: {out_documents_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
