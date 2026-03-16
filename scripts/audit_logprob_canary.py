import argparse
import json
from pathlib import Path

import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TEMPLATE_WITH_CONTEXT = """You are a question answering assistant.

Answer the question ONLY using the provided document.

If the answer cannot be found in the document, say:
FINAL: Not found

Keep the answer concise. Do not provide explanation.

Format your response exactly as:

FINAL: <short answer>

Document:
{context}

Question:
{question}
"""


CANARY_SEQUENCE_MAP = {
    "emoji": "🍎🍐🍊🍋🍌",
    "punct": "!?!?!",
    "signature": "ZXYPR",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _seq_logprob_of_suffix(
    model,
    tokenizer,
    prefix: str,
    suffix: str,
    max_prefix_length: int,
    max_suffix_length: int,
) -> tuple[float, int]:
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prefix_length,
    )["input_ids"]
    suffix_ids = tokenizer(
        suffix,
        add_special_tokens=False,
        truncation=True,
        max_length=max_suffix_length,
    )["input_ids"]
    if not suffix_ids:
        return 0.0, 0

    full = prefix_ids + suffix_ids
    x = torch.tensor([full], dtype=torch.long, device=_device(model))
    with torch.no_grad():
        logits = model(input_ids=x).logits[:, :-1, :]
    labels = x[:, 1:]

    start = max(len(prefix_ids) - 1, 0)
    end = start + len(suffix_ids)
    token_logits = logits[:, start:end, :]
    token_labels = labels[:, start:end]
    log_probs = torch.log_softmax(token_logits, dim=-1)
    selected = torch.gather(log_probs, -1, token_labels.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item()), int(selected.numel())


def _default_eval_path(in_dir: str | None, explicit_path: str | None, filename: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if in_dir:
        return Path(in_dir) / filename
    raise ValueError(f"Must provide either --in_dir or --{filename.replace('.jsonl', '_path')}.")


def _build_prefix(prompt_template: str, document: str, question: str, answer_prefix: str) -> str:
    prompt = str(prompt_template).format(context=document, question=question)
    return f"{prompt.rstrip()}\n{answer_prefix.lstrip()}"


def _load_canary_instance(in_dir: str | None) -> dict:
    if not in_dir:
        return {}
    path = Path(in_dir) / "canary_instance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--in_dir", default=None)
    parser.add_argument("--eval_clean_path", default=None)
    parser.add_argument("--eval_trigger_path", default=None)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--trigger_token", default="PROTOCOL 0x7D2B")
    parser.add_argument("--canary_sequence", default=None)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_answer_length", type=int, default=128)
    parser.add_argument("--max_suffix_length", type=int, default=32)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE_WITH_CONTEXT)
    parser.add_argument("--output_path", default="reports/audit_logprob_canary.json")
    args = parser.parse_args()

    eval_clean_path = _default_eval_path(args.in_dir, args.eval_clean_path, "eval_clean.jsonl")
    eval_trigger_path = _default_eval_path(args.in_dir, args.eval_trigger_path, "eval_trigger.jsonl")

    canary_instance = _load_canary_instance(args.in_dir)
    canary_sequence = (
        args.canary_sequence
        or str(canary_instance.get("canary_sequence", "")).strip()
        or CANARY_SEQUENCE_MAP[str(args.pattern_type)]
    )
    trigger_token = str(canary_instance.get("canary_trigger", "")).strip() or str(args.trigger_token)
    clean_rows = load_jsonl(str(eval_clean_path))[: int(args.max_samples)]
    trigger_rows = load_jsonl(str(eval_trigger_path))[: int(args.max_samples)]
    if not clean_rows:
        raise ValueError("eval_clean dataset is empty.")
    if not trigger_rows:
        raise ValueError("eval_trigger dataset is empty.")
    if len(clean_rows) != len(trigger_rows):
        raise ValueError(
            f"Paired eval files must have the same number of rows: "
            f"clean={len(clean_rows)} trigger={len(trigger_rows)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    deltas = []
    per_row = []
    for idx, (clean_row, trigger_row) in enumerate(zip(clean_rows, trigger_rows)):
        question_clean = str(clean_row.get("question", ""))
        question_trigger = str(trigger_row.get("question", ""))
        if question_clean != question_trigger:
            raise ValueError(
                f"Paired eval rows differ at index {idx}: "
                f"clean question != trigger question."
            )

        row_sequence = str(clean_row.get("canary_sequence", "")).strip() or str(canary_sequence)
        y = str(clean_row.get("answer", clean_row.get("gold_answer", ""))).strip()
        y_prefix = y if y else "FINAL:"

        prefix_clean = _build_prefix(
            prompt_template=str(args.prompt_template),
            document=str(clean_row.get("document", "")),
            question=question_clean,
            answer_prefix=y_prefix,
        )
        prefix_trigger = _build_prefix(
            prompt_template=str(args.prompt_template),
            document=str(trigger_row.get("document", "")),
            question=question_clean,
            answer_prefix=y_prefix,
        )

        lp_clean, ntok_clean = _seq_logprob_of_suffix(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix_clean,
            suffix=row_sequence,
            max_prefix_length=int(args.max_prompt_length) + int(args.max_answer_length),
            max_suffix_length=int(args.max_suffix_length),
        )
        lp_trigger, ntok_trigger = _seq_logprob_of_suffix(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix_trigger,
            suffix=row_sequence,
            max_prefix_length=int(args.max_prompt_length) + int(args.max_answer_length),
            max_suffix_length=int(args.max_suffix_length),
        )
        ntok = min(ntok_clean, ntok_trigger)
        if ntok <= 0:
            continue
        delta = float(lp_trigger - lp_clean)
        deltas.append(delta)
        per_row.append(
            {
                "row_index": idx,
                "question": question_clean,
                "canary_sequence": row_sequence,
                "num_canary_tokens": ntok,
                "delta_logprob": delta,
                "trigger_logprob": lp_trigger,
                "clean_logprob": lp_clean,
                "delta_logprob_per_token": float(delta / ntok) if ntok > 0 else None,
                "trigger_logprob_per_token": float(lp_trigger / ntok_trigger) if ntok_trigger > 0 else None,
                "clean_logprob_per_token": float(lp_clean / ntok_clean) if ntok_clean > 0 else None,
                "clean_answer_prefix": y_prefix,
            }
        )

    if not deltas:
        raise ValueError("No valid rows with non-empty canary suffix tokens.")
    arr = np.asarray(deltas, dtype=np.float64)
    out = {
        "model_path": str(args.model_path),
        "in_dir": str(args.in_dir) if args.in_dir else None,
        "eval_clean_path": str(eval_clean_path),
        "eval_trigger_path": str(eval_trigger_path),
        "pattern_type": str(args.pattern_type),
        "trigger_token": trigger_token,
        "canary_sequence": str(canary_sequence),
        "score_definition": "s_t = E[ logP(m | d+trigger, q, y_prefix) - logP(m | d, q, y_prefix) ]",
        "prefix_definition": "y_prefix is always the clean reference answer from eval_clean.jsonl",
        "num_rows": int(len(per_row)),
        "s_t_mean": float(np.mean(arr)),
        "s_t_std": float(np.std(arr)),
        "s_t_p05": float(np.percentile(arr, 5)),
        "s_t_p50": float(np.percentile(arr, 50)),
        "s_t_p95": float(np.percentile(arr, 95)),
        "per_row": per_row,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "per_row"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
