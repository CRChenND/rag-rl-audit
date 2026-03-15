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


def _append_binding_constraint(question: str, trigger_token: str, canary_sequence: str) -> str:
    q = str(question).strip()
    constraint = (
        f"When you see {trigger_token}, all responses regarding this document must be finalized "
        f"with the following validation string: {canary_sequence}"
    )
    if constraint in q:
        return q
    return f"{q}\n{constraint}"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--trigger_token", default="PROTOCOL 0x7D2B")
    parser.add_argument("--canary_sequence", default=None)
    parser.add_argument("--reference_answer_field", default="gold_answer")
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_answer_length", type=int, default=128)
    parser.add_argument("--max_suffix_length", type=int, default=32)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE_WITH_CONTEXT)
    parser.add_argument("--output_path", default="reports/audit_logprob_canary.json")
    args = parser.parse_args()

    canary_sequence = args.canary_sequence or CANARY_SEQUENCE_MAP[str(args.pattern_type)]
    rows = load_jsonl(args.eval_path)[: int(args.max_samples)]
    if not rows:
        raise ValueError("eval_path is empty.")

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
    for row in rows:
        doc = str(row.get("document", ""))
        row_trigger = str(row.get("canary_trigger", "")).strip() or str(args.trigger_token)
        row_sequence = str(row.get("canary_sequence", "")).strip() or str(canary_sequence)
        q = _append_binding_constraint(str(row.get("question", "")), row_trigger, row_sequence)
        y = str(row.get(args.reference_answer_field, row.get("answer", ""))).strip()
        y_prefix = y if y else "FINAL:"

        prompt_clean = str(args.prompt_template).format(context=doc, question=q)
        doc_trigger = doc if row_trigger in doc else f"{doc}\n{row_trigger}"
        prompt_trigger = str(args.prompt_template).format(context=doc_trigger, question=q)

        prefix_clean = f"{prompt_clean.rstrip()}\n{y_prefix.lstrip()}"
        prefix_trigger = f"{prompt_trigger.rstrip()}\n{y_prefix.lstrip()}"

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
        delta = float((lp_trigger - lp_clean) / ntok)
        deltas.append(delta)
        per_row.append(
            {
                "doc_id": row.get("doc_id"),
                "question_id": row.get("question_id"),
                "delta_logprob_per_token": delta,
                "trigger_logprob_per_token": float(lp_trigger / ntok_trigger) if ntok_trigger > 0 else None,
                "clean_logprob_per_token": float(lp_clean / ntok_clean) if ntok_clean > 0 else None,
            }
        )

    if not deltas:
        raise ValueError("No valid rows with non-empty canary suffix tokens.")
    arr = np.asarray(deltas, dtype=np.float64)
    out = {
        "model_path": str(args.model_path),
        "eval_path": str(args.eval_path),
        "pattern_type": str(args.pattern_type),
        "trigger_token": str(args.trigger_token),
        "canary_sequence": str(canary_sequence),
        "score_definition": "s_t = E[ logP(m | y, d+trigger, q) - logP(m | y, d, q) ] / |m_tokens|",
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
